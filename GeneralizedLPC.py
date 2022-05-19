import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp

def taylor_operator(n=3, dt=1, t=0):
    """ Embedding operator for generalized Taylor series expansion """

    s = np.round((t)/dt)
    k = np.asarray(range(1,n+1,1)) + np.trunc(s - (n + 1)/2)
    x = s - np.min(k) + 1

    # inverse embedding operator T: sequence = T*embedding
    T = np.zeros([n,n])
    for i in range(1, n+1, 1):
        for j in range(1, n+1, 1):
            T[i-1,j-1] = ((i-x)*dt)**(j-1) / np.prod(np.asarray(range(1,j,1)))

    # embedding operator E: embedding = E*sequence
    E = np.linalg.inv(np.matrix(T))

    return torch.from_numpy(T).unsqueeze(-1).float(), torch.from_numpy(E).unsqueeze(1).float()


def to_tensor(vector):
    X = np.expand_dims(vector, axis=0)
    X = torch.tensor(np.expand_dims(X, axis=0))
    return X


def embed_audio(audio, GEN_COORDS, dt, t, kernel_size, length, b_size, input_units, plot=True, use_gc=True):

    # create embedding operator (1D convolution kernel): sequence -> embedding
    T, E = taylor_operator(n=GEN_COORDS, dt=dt, t=t)
    conv_E = torch.nn.Conv1d(1, GEN_COORDS, GEN_COORDS, stride=1, padding="valid", bias=False)  # same padding creates inaccuracies
    conv_E.weight = torch.nn.Parameter(E)

    # embed audio in generalized coordinates
    seq = torch.tensor(audio).unsqueeze(0).unsqueeze(0)
    seq = seq.reshape([length, b_size, input_units]).transpose(1,-1).transpose(0,-1).reshape([b_size*input_units,1,length])
    embedding = conv_E(seq)
    length = embedding.shape[-1]  # length might change from padding
    embedding = embedding.reshape([input_units*b_size,GEN_COORDS,length]).permute([2,0,1])

    Y0_gc = torch.transpose(embedding, 0, 2)[0:1][:,:,:-kernel_size].detach() # y

    YGC_in = []
    for gc in range(1, GEN_COORDS):
        YGC_in.append(torch.transpose(embedding, 0, 2)[gc:gc+1][:,:,:-1].detach())

    YGC_in_padded = []
    for gc in range(1, GEN_COORDS-1):
        YGC_in_padded.append(torch.transpose(embedding, 0, 2)[gc:gc+1][:,:,:-kernel_size].detach())

    if plot:
        print("Embedding", embedding.shape)
        plt.plot(embedding.detach()[:100,0,0], label="audio GC 0")
        plt.plot(embedding.detach()[:100,0,1], label="audio GC 1")
        if gc > 2: plt.plot(embedding.detach()[:100,0,2], label="audio GC 2")
        plt.plot((seq.detach().numpy()[0,0,1:]-seq.detach().numpy()[0,0,:-1])[:100], label="audio(t)-audio(t-1)", linestyle="dashed")
        plt.legend(); plt.xlabel("Samples"); plt.ylabel("Magnitude");
        plt.title("Audio in generalized coordinates");
        plt.savefig("./gen_coords_compare.png"); plt.show()

    return Y0_gc, YGC_in, YGC_in_padded


class PCNet(nn.Module):
    def __init__(self, kernel_size=12, channels=1, activation=torch.nn.Identity()):#channels=1, activation=torch.nn.Identity()):
        super(PCNet, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        self.B = nn.Sequential(
            nn.Conv1d(in_channels=self.channels, out_channels=self.channels, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size-1), bias=False))
        self.C = nn.Sequential(nn.Conv1d(in_channels=self.channels, out_channels=1, kernel_size=1, stride=1, bias=False))
        self.x0 = None
        self.act = activation

        #torch.nn.init.xavier_uniform_(self.B[0].weight, 0.1)
        #torch.nn.init.xavier_uniform_(self.C[0].weight, 0.1)

    def forward(self, causes, higher_order_cause=False):
        self.hiddens = []
        self.outs = []

        #dynamical prediction
        if higher_order_cause: # predict from v --> x', v' --> x'', ...
            for v in causes:
                self.hiddens.append(self.act(self.B(v)))  # todo channels
        else: # predict from v --> x', x' --> x'', ..., i.e. hidden state = cause
            self.hiddens.append(self.act(self.B(causes[0].repeat(1,self.channels,1))))  # v --> x'
            for _ in causes[1:]: # x' --> x'', x'' --> x''', ...
                self.hiddens.append(self.act(self.B(self.hiddens[-1][:,:,:-self.kernel_size+1])))  # todo padding / kernel size

        # make sure state is initialised
        if self.x0 is None:
            self.x0 = torch.zeros_like(self.hiddens[0]).requires_grad_()
            self.opt_x0 = torch.optim.SGD([self.x0], lr=1.)

        # hierarchical prediction
        #self.y0 = self.C(self.x0)  # x --> y
        self.y0 = self.C(causes[0].repeat(1,self.channels,1))  # x --> y
        for x in self.hiddens:
            self.outs.append(self.C(x))
        return self.y0, self.outs #, self.x1_pred


def run(inputs, YGC_in, use_gc, kernel_size, epochs=3000, lr=0.01, optimiser=torch.optim.Adam,
        print=False, identity_output=False, loss_fn = nn.MSELoss(reduction='sum')):
    model = PCNet(kernel_size=kernel_size)
    optimizer = optimiser(model.parameters(), lr=lr) # 0.001 for SGD

    for epoch in range(epochs):

        if identity_output: # don't optimize output weights
            torch.nn.init.constant_(model.C[0].weight, 1)

        Y0pred, YGC = model(inputs)

        loss = 0
        for ygc, target in zip(YGC, YGC_in): # todo
            loss += loss_fn(ygc, target.detach()) # prediction error on y'
            if not use_gc: break

        if False:
            loss += loss_fn(Y0pred, inputs[0].detach())  # prediction error on y'

        if print:
            if epoch % 1000 == 0: print(epoch, loss.item())

        optimizer.zero_grad(); #model.opt_x0.zero_grad()
        loss.backward(); #loss0.backward()

        if identity_output: model.C[0].weight.grad.zero_()  # don't optimize output weights

        optimizer.step(); #model.opt_x0.step()

    return model


def evaluate(model, inputs, use_gc, Y0_gc, kernel_size, highest_order=3, plot=True, print_result=False):
    pred_length = inputs[0].shape[-1]
    pred = model(inputs)

    if not use_gc:
        predictions = Y0_gc[0,0,:]+pred[1][0][0,0,:pred_length]
    else:

        predictions = Y0_gc[0,0,:].clone().detach()
        for gc, pred_gc in enumerate(pred[1][:highest_order]):
            predictions = predictions.clone().detach() + pred_gc[0, 0, :pred_length].clone().detach()

    start = 0 #kernel_size # todo

    # the absolute prediction error between target audio and input audio
    error = predictions[start:pred_length-1] - Y0_gc[0,0,start+1:pred_length]
    error[:kernel_size] = 0.
    err = sum(error**2)/max(Y0_gc[0,0,start+1:pred_length].shape)
    sigpow = sum(Y0_gc[0,0,start+1:pred_length]**2)/max(Y0_gc[0,0,start+1:pred_length].shape)

    # the first order prediction error between changes in audio samples and predicted changes
    #true_velocity = (Y0_gc[0, 0, start + 1:pred_length] - Y0_gc[0, 0, start:pred_length-1])
    #error_first_order = true_velocity - pred[1][0][0,0,start:pred_length-1]
    #error_first_order[:kernel_size] = 0.
    #err_first_order = sum(error_first_order**2)/max(Y0_gc[0,0,start+1:pred_length].shape)

    if print_result:
        print("Mean squared prediction error=", err)
        print("Mean signal power=", sigpow)
        print("Signal to Error Power Ratio:", sigpow/err)

        #print("Mean squared prediction error (first order)=", err)
        #print("Signal to Error Power Ratio (first order):", sigpow/err_first_order)

    if plot:

        plt.figure(figsize=(10,8)); end = 60
        plt.plot(pred[0][0,0,:end].detach(), label=f"Predicted y GC{0}")
        for gc, ygc in enumerate(pred[1]):
            plt.plot(ygc[0,0,:end].detach(), label=f"Predicted y GC{gc+1}")
        plt.legend(); plt.show()

        plt.figure(figsize=(10,8))
        plt.plot(np.array(Y0_gc[0,0,1:pred_length]))
        plt.plot(predictions.detach().numpy())
        plt.legend(('Audio','Prediction')); plt.title('Audio and prediction')
        plt.xlabel('Sample'); plt.ylabel('Magnitude');
        plt.savefig("./audio_prediction.png"); plt.show()

        plt.figure(figsize=(10,8))
        plt.plot(np.array(Y0_gc[0,0,1:pred_length]))
        #plt.plot(predictions[:-1].detach().numpy()-np.array(Y0_gc[0,0,1:pred_length]))
        plt.plot(error.detach())
        plt.legend(('Audio','Prediction Error')); plt.title('Audio and Prediction Error')
        plt.xlabel('Sample'); plt.ylabel('Magnitude');
        plt.savefig("./audio_error.png"); plt.show()

        end = 120
        plt.figure(figsize=(10,8))
        plt.plot(np.array(Y0_gc[0,0,1:end]))
        plt.plot(predictions.detach().numpy()[:end])
        plt.legend(('Audio','Prediction')); plt.title('Audio and prediction')
        plt.xlabel('Sample'); plt.ylabel('Magnitude');
        plt.savefig("./audio_prediction_zoom.png"); plt.show()

        plt.figure(figsize=(10,8))
        plt.plot(np.array(Y0_gc[0,0,1:end]))
        plt.plot(error.detach()[:end])
        plt.legend(('Audio','Prediction Error'))
        plt.title('Audio and Prediction Error')
        plt.xlabel('Sample'); plt.ylabel('Magnitude');
        plt.savefig("./audio_error_zoom.png"); plt.show()

    return predictions, err, sigpow, sigpow/err


def closed_form(x, LENGTH = 10000, KERNEL_SIZE=12, plot=False, print_result=False):
    """ Closed from Wiener-Hopf solution for Linear prediction """

    # Normalize Audio
    #x/=np.abs(x).max()

    # audio as column
    x=np.matrix(x,dtype=float).T

    # matrix A
    A=np.matrix(np.zeros((LENGTH,KERNEL_SIZE)));
    for m in range(0,LENGTH):
        A[m, :]=np.flipud(x[m+np.arange(KERNEL_SIZE)]).T

    # target signal
    d = x[np.arange(KERNEL_SIZE,LENGTH+KERNEL_SIZE)]

    # prediction filter
    h = np.linalg.inv(A.T*A) * A.T * d;

    hpred = np.vstack([0, (h)])
    if False: print("Prediction filter input --> predicted input", hpred)

    # prediction
    xpred = sp.lfilter(np.array(hpred.T)[0],1,np.array(x.T)[0])

    hperr = np.vstack([1, -(h)])
    if False: print("Prediction error filter input --> prediction error", hperr)

    # prediction error
    e = sp.lfilter(np.array(hperr.T)[0],1,np.array(x.T)[0]);
    e=np.matrix(e)

    # zero out prediction error for first KERNEL_SIZE samples
    e[0,:KERNEL_SIZE] = 0.

    # mean prediction error
    MSE = e*e.T/max(np.shape(e))

    # mean signal power
    MSP = x.T*x/max(np.shape(x))

    # signal power to error ratio
    SEPR = MSP/MSE

    # reconstruction
    xrec = sp.lfilter([1],np.array(hperr.T)[0], np.array(e)[0]);

    if print_result:
        print("MSE: ", MSE[0,0], "MSP: ", MSP[0,0], "SEPR: ", SEPR[0,0])

    if plot:
        plt.figure(figsize=(10,8))
        plt.plot((h))
        plt.xlabel('Sample'); plt.ylabel('Value')
        plt.title('Impulse Response')
        plt.grid(); plt.show()

        plt.figure(figsize=(10,8))
        plt.plot(x);
        plt.plot(xpred,'red')
        plt.legend(('Original','Predicted'), loc='upper right')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.title('Audio and prediction')
        plt.grid(); plt.show()

        plt.figure(figsize=(10,8))
        plt.plot(x)
        plt.plot(e.T,'r')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.title('Audio and prediction error')
        plt.legend(('Original', 'Prediction Error'), loc='upper right')
        plt.grid(); plt.show()

        plt.figure(figsize=(10,8))
        plt.plot(x,'b')
        plt.plot(xrec,'r')
        plt.legend(('Original', 'Reconstructed'), loc='upper right')
        plt.title('Audio and reconstruction')
        plt.grid(); plt.show()

    return SEPR[0,0], xrec, e




"""
Differentiable Generalized Predictive Coding
André Ofner 2021
"""

import torch
from tools import *
from torch.optim import SGD
from torch.nn import Sequential, Linear, Tanh, ConvTranspose2d

class Model(torch.nn.Module):
    """ Hierarchical dynamical predictive coding model """
    def __init__(self,sizes,act,act_d=None,dynamical=False,var=1,covar=1000000, dim=[],lr_w=None,lr_sl=None,lr_sh=None,sr=[]):
        super(Model,self).__init__()
        self.layers,self.layers_d,self.layers_p,self.covar,self.lr = [],[],[],[],[]  # weights,dynamical weights,precision weights,covariance
        self.dynamical,self.sizes,self.dim = dynamical,sizes,np.asarray(dim)  # layer type,state sizes,state channels
        self.covar,self.sr = [torch.zeros(1) for _ in sizes],sr # precision,sampling rate per layer
        self.n_cnn = ((self.dim > 1).sum())//2 + 1 # number of CNN layers,if present
        self.state_sizes = np.sqrt(self.sizes/self.dim).astype(int)[:self.n_cnn*2]
        if lr_w is None: lr_w = [0. for _ in self.sizes] # default weights learning rate
        if lr_sh is None: lr_sh = [1. for _ in self.sizes] # default lower state learning rate
        if lr_sl is None: lr_sl = [1. for _ in self.sizes] # default higher state learning rate

        for i in range(0,len(sizes),2): # create hierarchical CNN layers
            if not dynamical:
                net = ConvTranspose2d(dim[i+1],dim[i],2,stride=2,bias=False) if dim[i+1] > 1 else Linear(sizes[i+1],sizes[i],False)
                self.layers.append(Sequential(act, net))
                self.layers_d.append(Model(sizes[i:],act_d,dynamical=True,dim=dim[i:],sr=[sr[i] for _ in sizes[i:]])) # dynamical weights
            else:
                self.layers.append(Sequential(act,Linear((sizes[i+1]),(sizes[i]),False))) # dense dynamical layers
            self.lr.append([lr_sh[i],lr_sl[i],0,lr_w[i],0.1])  # learning rate per layer # todo dynamical layer lr!

        [self.currState,self.lastState] = [self.init_states(mixed=(self.n_cnn > 0)) for _ in range(2)] # initialise states
        if not dynamical and PRECISION: # initialise precision # todo dense layers
            self.covar = [torch.eye(cs.shape[-1]**2).repeat([B_SIZE * cs.shape[1],1,1]) * (var - covar) + covar for cs in self.currState[:(len(sizes)//2)+1]]
            self.covar += [torch.eye(cs.shape[-1]).repeat([B_SIZE * cs.shape[1],1,1]) * (var - covar) + covar for cs in self.currState[len(sizes)//2+1:]]

    def init_states(self,mixed=False):
        """ Create state priors for hierarchical and dynamical layers"""
        if self.dynamical:
            return self.predict(torch.ones([B_SIZE,1,self.sizes[-1]]).float(),prior=True)
        elif mixed: # CNN and dense layers (hierarchical model)
            states_cnn = self.predict(torch.ones([B_SIZE,self.dim[self.n_cnn*2-1],self.state_sizes[-1],self.state_sizes[-1]]).float(),prior=True,layers=self.layers[:self.n_cnn]) #todo
            states_dense = self.predict(torch.ones([B_SIZE,1,self.sizes[-1]]).float(),prior=True,layers=self.layers[self.n_cnn:])
            return states_cnn + states_dense[1:]
        elif self.n_cnn > 1: # only CNN layers (hierarchical model)
            return self.predict(torch.ones([B_SIZE,self.dim[-1],self.sizes[-1],self.sizes[-1]]).float(),prior=True)
        else: # dense layers (dynamical model)
            return self.predict(torch.ones([B_SIZE,1,self.sizes[-1]]).float(),prior=True)

    def parameters(self,l,dynamical=False):
        """ Parameters and learning rates per layer.
        Note that regularization through top-down predictability affects learning and inference."""
        params = [{'params': list(self.layers[l].parameters()),'lr': self.lr[l][3]},# top-down weights (l) --> prediction accuracy
                  {'params': [self.currState[l+1].requires_grad_()],'lr': self.lr[l][0]},# higher state (l+1,t) --> prediction accuracy
                  {'params': [self.currState[l].requires_grad_()],'lr': self.lr[l][1]},# lower state (l,t-dt) --> regularization (top-down predictability)
                  {'params': [self.lastState[l].requires_grad_()],'lr': self.lr[l][2]},# lower state (l,t) --> regularization (top-down predictability)
                  {'params': [self.covar[l].requires_grad_()],'lr': self.lr[l][4]}]  # covariance (l) --> learning rate
        return params[:-1] if dynamical else params

    def predict(self,target=None,states=None,prior=False,layers=None):
        """ Backward prediction through all layers. Optionally initialises prior states"""
        states = [target] if states is None else [states[-1]]
        if prior: [torch.nn.init.xavier_uniform_(states[-1][b],gain=1) for b in range(B_SIZE)]
        if layers is None: layers = self.layers
        for w in list(reversed(layers)):
            states.append(w(states[-1]).detach())
            if prior: [torch.nn.init.xavier_uniform_(states[-1][b],gain=1) for b in range(B_SIZE)]
        return list(reversed(states))

    def predict_mixed(self,target=None):
        """ Backward pass through CNN,dense or mixed models """
        try: # mixed model
            states_dense = self.predict(target,layers=self.layers[self.n_cnn:])
            dense_pred = states_dense[0].reshape([B_SIZE,self.dim[self.n_cnn*2-1],self.state_sizes[-1],self.state_sizes[-1]])
            return self.predict(dense_pred,layers=self.layers[:self.n_cnn])
        except: # DNN or CNN model
            return self.predict(target=target)

    def freeze(self,params):
        """ Disable optimisation of weights. State inference remains active. """
        for param in params:
            for lr in self.lr: lr[param] = 0.  # freeze hierarchical weights
            if not self.dynamical:
                for l_d in self.layers_d: l_d.freeze(params)  # freeze dynamical weights


def GPC(m,l,dynamical=False):
    """ Layer-wise Generalized Predictive Coding optimizer"""
    if dynamical: # assign sampling rates to dynamical states (= hidden states,i.e. not seen by hierarchical predictions)
        m.currState[l+1] = torch.cat([m.currState[l+1][:,:,:-1].detach(),torch.tensor([m.sr[l]]).repeat([B_SIZE,1,1])],dim=-1)
    else: # all dynamical states of a hierarchical layer contain the same sampling rate
        for l_d in range(len(m.layers_d[l].currState)):
            m.layers_d[l].currState[l_d] = torch.cat([m.layers_d[l].currState[l_d][:,:,:-1].detach(),torch.tensor([m.sr[l]]).repeat([B_SIZE,1,1])],dim=-1)
    opt = SGD(m.parameters(l,dynamical)) # create this layer's SGD optimizer
    opt.zero_grad() # reset gradients
    pred = m.layers[l].forward(m.currState[l+1].requires_grad_())  # prediction from higher layer
    if dynamical:  # predict state change
        error = (m.currState[l].detach() - m.lastState[l].requires_grad_()).flatten(1) - pred.flatten(1)
    else:  # predict state + state change
        if TRANSITION: pred = pred + m.layers_d[l].layers[0].forward(m.layers_d[l].currState[1].requires_grad_()).reshape(pred.shape)  # predicted state + predicted state transition
        error = m.currState[l].requires_grad_() - pred.reshape(m.currState[l].shape) # hierarchical-dynamical error
    error = error.reshape([B_SIZE * error.shape[1],-1]).unsqueeze(-1) # move CNN channels to batch dimension
    if not dynamical and PRECISION: # compute this layer's free energy
        F = torch.mean(torch.abs(error) * torch.abs(torch.matmul(m.covar[l]**-1,error)),dim=[1,2])
    else:
        F = torch.mean(torch.abs(error)**2,dim=[1,2]) # without precision weighting
    F.backward(gradient=torch.ones_like(F))  # loss per batch element (not scalar)
    opt.step() # update all variables of this layer
    return pred.detach().numpy(),error

UPDATES, SCALE, B_SIZE,IMAGE_SIZE = 100, 3, 16, 16*16  # model updates, relative layer updates, batch size, input size
ACTIONS = [1 for i in range(20)]  # actions in Moving MNIST (1 for next frame. see tools.py for spatial movement)
TRANSITION = False # first order transition model
DYNAMICAL = False  # higher order transition derivatives (generalized coordinates)
PRECISION = False # use precision estimation
IMG_NOISE = 0.5  # gaussian noise on inputs todo scaling

if __name__ == '__main__':
    for env_id,env_name in enumerate(['Mnist-Train-v0','Mnist-Test-v0']): # train set,test set
        env = gym.make(env_name)  # Moving MNIST gym environment
        env.reset()
        ch, ch2, ch3 = 32, 64, 128
        PCN = Model([1*16*16,ch*8*8,ch*8*8,ch2*4*4,ch2*4*4,ch3*2*2],# state sizes
                    Tanh(),Tanh(),# hierarchical & dynamical activation
                    lr_w=np.asarray([1,1,1,1,1,1])*.001,# weights lr
                    lr_sl=np.asarray([0,1,1,1,1,1])*1, # lower state lr
                    lr_sh=np.asarray([1,1,1,1,1,1])*10, # higher state lr
                    dim=[1,ch,ch,ch2,ch2,ch3],# state channels (use 1 for dense layers)
                    sr=[2,2,1,1,1,1]) # sampling interval
        [err_h,err_t,preds_h,preds_t,preds_g],inputs = [[[] for _ in PCN.layers] for _ in range(5)],[[]]  # visualization
        print("States:"),[print(str(s.shape)) for s in PCN.currState], print("Weights:"),print(PCN.layers)

        for a_id, action in enumerate(ACTIONS): # passive perception task, i.e. model has no control
            for i in range(int(PCN.sr[0])): # sample data observations
                obs,rew,done,_ = env.step([action for b in range(B_SIZE)])  # step environment
            input = ((torch.Tensor(obs['agent_image'])).reshape([B_SIZE,-1,64 ** 2]) / 255 + 0.1) * 0.8  # get observation
            input = torch.nn.MaxPool2d(2,stride=4)(input.reshape([B_SIZE,-1,64,64])).reshape([B_SIZE,-1,IMAGE_SIZE])  # optionally reduce input size

            #[PCN.currState,PCN.lastState] = [PCN.init_states(mixed=(PCN.n_cnn > 0)) for _ in range(2)] # optionally re-initialise states for each input
            PCN.currState[0] = torch.tensor(input.detach().float()).reshape([B_SIZE, 1, -1])  # feed data

            for update in range(UPDATES):  # update model
                for l_h in range(len(PCN.layers)):  # update each hierarchical layer
                    if env_id == 1: PCN.freeze([3])  # freeze weights when testing
                    for i in range(l_h*SCALE+1): p_h,e_h = GPC(PCN,l=l_h)  # step hierarchical variables
                    if update == UPDATES - 1: PCN.lastState[l_h] = PCN.currState[l_h].clone().detach()  # memorize last state

                    for l_d in range(1,len(PCN.layers_d[l_h].layers),1):  # update dynamical layers
                        if DYNAMICAL:
                            p_d,e_t = GPC(PCN.layers_d[l_h],l=l_d,dynamical=True)  # step higher order dynamical variables
                            if update == UPDATES - 1: PCN.layers_d[l_h].lastState[l_d] = PCN.layers_d[l_h].currState[l_d].clone().detach()  # memorize

                        if update == UPDATES - 1 and l_h == 0 and l_d == 1:  # visualization
                            for d,i in zip([inputs[0],preds_h[l_h],err_h[l_h]],[input[:1],p_h[:1],e_h[:1].detach()]): d.append(i)  # hierarchical
                            if DYNAMICAL: preds_t[l_d].append(p_d[:1]),err_t[l_d].append(e_t[:1].detach())  # dynamical
                            p_g = PCN.predict_mixed(target=PCN.currState[-1])[0]
                            preds_g[l_h].append(p_g[0]) # prediction from target state
                            if a_id == len(ACTIONS)-1: # visualize batch
                                plot_batch(input, title=str(env_name)+"input")
                                plot_batch(p_h, title=str(env_name)+"pred_h")
                                plot_batch(p_g, title=str(env_name)+"pred_g")

        for s,t in zip([preds_h,inputs,preds_g,err_h][:3],['p_h','ins','p_g','e_h'][:3]):  # generate videos
            sequence_video(s,t,scale=255,env_name=str(env_name))
        env.close()

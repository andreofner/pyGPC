"""
Differentiable Generalized Predictive Coding
André Ofner 2021

MNIST classification with a static predictive coding model
"""

from tools import *
import torch.nn.init
from torch.optim import SGD
from torch.nn import Sequential, Linear
import torchvision
import copy

class Model(torch.nn.Module):
    """ Hierarchical dynamical predictive coding model """
    def __init__(self, sizes, act, dim= [], sr= [],
                 lr_w=None, lr_sl=None, lr_sh=None, lr_p=None, lr_w_d=None,
                 dynamical=False):
        super(Model, self).__init__()

        self.layers = []  # hierarchical weights
        self.layers_d = []  # transition weights
        self.lr = []  # learning rate per layer
        self.dynamical = dynamical  # model type
        self.sizes = sizes  # hierarchical state sizes
        self.sr = sr  # sampling rate per layer
        self.lr_names = ["Higher state", "State", "Weights", "Precision", "Dynamical weights"]

        for i in range(0, len(self.sizes ), 2):
            # create hierarchical layers
            net = Linear(self.sizes[i+1]*2, self.sizes[i], bias=False) # todo hidden size
            self.layers.append(Sequential(act, net, torch.nn.Dropout(0.0)))

            # create dynamical layers
            net2 = Linear(self.sizes[i+1]*2, self.sizes[i], bias=False) # todo hidden size
            self.layers_d.append(Sequential(act, net2, torch.nn.Dropout(0.0)))
            self.lr.append([lr_sh[i], lr_sl[i], lr_w[i], lr_p[i], lr_w_d[i]])  # learning rate per layer

    def params(self, l):
        """ Parameters and learning rates per layer. Top-down predictability regularizes learning and inference."""
        return [{'params': [self.curr_cause[l + 1]], 'lr': self.lr[l][0]},  # higher state
                      {'params': [self.curr_hidden[l + 1]], 'lr': self.lr[l][0]},  # higher state
                      {'params': [self.curr_cause[l]], 'lr': self.lr[l][1]},  # lower state
                      {'params': [self.curr_hidden[l]], 'lr': self.lr[l][1]}] + self.params_weights(l) # lower hidden state

    def params_covar(self, l):
        """ Covariance parameters and learning rates per layer """
        return [{'params': [self.covar_slow[l]], 'lr': self.lr[l][3]}, # slow cause state covariance
                {'params': [self.covar_hidden_slow[l]], 'lr': self.lr[l][3]},  # slow hidden state covariance
                {'params': [self.covar[l]], 'lr': self.lr[l][3]},  # fast cause state covariance
                {'params': [self.covar_hidden[l]], 'lr': self.lr[l][3]}]  # fast hidden state covariance

    def params_weights(self, l):
        try:
            return [{'params': [self.layers[l][1].weight.requires_grad_()], 'lr': self.lr[l][2]}, # hierarchical weights
                      {'params': [self.layers_d[l + 1][1].weight.requires_grad_()], 'lr': self.lr[l][4]}] # dynamical weights
        except: # todo fix dynamical
            return [{'params': [self.layers[l][1].weight.requires_grad_()], 'lr': self.lr[l][2]}] # hierarchical weights

    def create_states(self, batch_size, initialiser=torch.nn.init.xavier_uniform_):
        """ Initialises concatenated cause and hidden states """
        states = []
        for layer in list(reversed(self.layers)):
                states.append(torch.zeros([batch_size, 1, layer[1].weight.shape[1]//2]).requires_grad_())
                [initialiser(states[-1][b]) for b in range(batch_size)]
        states.append(torch.zeros([batch_size, 1, self.layers[0][1].weight.shape[0]]).requires_grad_())
        [initialiser(states[-1][b]) for b in range(batch_size)]
        self.curr_cause = list(reversed(states))  # cause states
        self.curr_hidden = copy.deepcopy(list(reversed(states)))  # hidden states

    def create_covar(self):
        # initialise precision for inference (within datapoint)
        self.covar = [torch.zeros(1) for _ in self.curr_cause + [None]]  # cause states precision
        self.covar_hidden = [torch.zeros(1) for _ in self.curr_cause + [None]]  # hidden states precision

        # initialise precision for learning (between datapoints)
        self.covar_slow = [torch.zeros(1) for _ in self.curr_cause + [None]]  # cause states precision
        self.covar_hidden_slow = [torch.zeros(1) for _ in self.curr_cause + [None]]  # hidden states precision

    def freeze(self, params):
        """ Stops optimisation of weights. State inference remains active. """
        for param in params:
            for lr in self.lr: lr[param] = 0.  # freeze hierarchical weights

    def cause(self, l):
        """ Returns a layer's cause state """
        return self.curr_cause[l].requires_grad_()

    def hidden(self, l):
        """ Returns a layer's hidden state """
        if l == 0:
            return self.cause(l) # inputs have no hidden states
        return self.curr_hidden[l].requires_grad_()

    def state(self, l):
        """ Concatenated hidden & cause states """
        if l == 0:
            return self.cause(l) # inputs have no hidden states
        state = torch.cat([self.cause(l), self.hidden(l)], dim=-1)
        return state

    def mask_state_grads(self, l, split=4, cause=False):
        """ Set all state gradients to zero except for those used for the prediction.
            Higher split means less units for outgoing prediction, split=0 disables all units."""

        # set gradients of states that are not used for outgoing prediction to zero
        if cause:
            state = self.cause(l=l)
        else:
            state = self.hidden(l=l)
        if split < 1:
            split_size = 0 # zero out all gradients
        else:
            split_size = (state.shape[-1] // split)
        if state.grad is not None:
            state.grad[:, split_size:state.shape[-1]].zero_()

    def mask_weights_grads(self, l):
        """ Set all weights gradients to zero except for those used for the prediction.
        Assumes mask_state_grads() was already called. """

        if l > 0:
            state_grad = torch.concat([self.cause(l=l).grad, self.hidden(l=l).grad], -1)
            # multiply binary mask of state gradients with weights
            PCN.layers[l-1][1].weight.grad = PCN.layers[l-1][1].weight.grad * (state_grad[0]>0)*1.

def predict(m, l, keep_states=True):
    """ Prediction through the cause states of the entire network."""
    global keep_cause
    global keep_hidden

    # make sure model stays unchanged if requested
    if keep_states and l == len(m.layers)-1:
        keep_cause = [c.clone().detach() for c in m.curr_cause]
        keep_hidden = [c.clone().detach() for c in m.curr_hidden]

    # get cause prediction from higher layer
    higher_state = m.state(l + 1)
    higher_state = higher_state.requires_grad_()

    # replace cause with top-down prediction
    pred = m.layers[l].forward(higher_state)
    m.curr_cause[l] = pred.reshape(m.curr_cause[l].shape).detach().requires_grad_()

    # continue in lower layer
    if l > 0:
        return predict(m, l-1, keep_states=keep_states)
    else:
        input_prediction = m.curr_cause[0].clone().detach().numpy()
        if keep_states:
            m.curr_cause = keep_cause
            m.curr_hidden = keep_hidden
        return input_prediction

def feed_target(PCN, batch_size):
    """ Feed target to model """
    torch.nn.init.normal_(PCN.curr_cause[-1])
    if not (TEST_TARGET and env_id > 0):  # evaluate target reconstruction
        PCN.curr_cause[-1] = torch.tensor(input.flip(-1).detach().float()).reshape(
            [batch_size, 1, -1])  # e.g. mirrored input image as target

def feed_observation(PCN, batch_size):
    """ Feed observation to model """
    torch.nn.init.normal_(PCN.curr_cause[0])
    if not (TEST_INPUT and env_id > 0):  # evaluate input reconstruction
        PCN.curr_cause[0] = torch.zeros_like(PCN.curr_cause[0])
        for b in range(batch_size):
            for c in range(0,IMAGE_SIZE,16):
                PCN.curr_cause[0][b, :, c+target[b]] = 1  # e.g. one hot digit as input

def initialise(PCN, test=False):
    """ Initialise states and learning rates"""
    for l in range(len(PCN.layers)):
        torch.nn.init.xavier_uniform_(PCN.curr_cause[l])
        torch.nn.init.xavier_uniform_(PCN.curr_hidden[l])
        if test:  # change learning rates for testing
            PCN.lr[l][2] = 0  # hierarchical weights
            PCN.lr[l][4] = 0  # dynamical weights
            if TEST_INPUT: # evaluate input reconstruction
                PCN.lr[l][0] = 0.0 # higher state
            elif TEST_TARGET: # evaluate target reconstruction
                PCN.lr[l][1] = 0.0 # state

def batch_accuracy(pred_g, target, batch_size):
    """ Computes classification results for a batch"""
    pred_classes = pred_g.reshape([batch_size, 1, 10]).mean(-2).argmax(-1)
    correct = (torch.tensor(pred_classes) == target).sum().detach().numpy()
    accuracy = 100. * correct / batch_size
    return accuracy, pred_classes, correct

def GPC(m, l, infer_precision=False, optimize=True, var_prior=1, covar_prior=10000000, transition=False, learn=False):
    """ Layer-wise Generalized Predictive Coding optimizer """

    batch_size = m.state(l).shape[0]
    B_SIZE_PRECISION_SLOW = batch_size

    if learn:  # weights updates
        params = m.params_weights(l)   # weights learning
    else:  # inference or precision updates
        params = m.params_covar(l) if infer_precision else m.params(l)

    opt = SGD(params)
    opt.zero_grad()

    if transition:
        try:
            # transition the hidden to generate its prior prediction
            PCN.curr_hidden[l+1] = m.layers_d[l+1].forward(m.state(l+1).detach()).detach()

            # remember the prior hidden state to be able to predict its change
            current_hidden_prior = PCN.curr_hidden[l+1].detach()
        except:
            transition = False  # reached highest layer

    # cause state prediction error
    higher_state = m.state(l+1)
    pred = m.layers[l].forward(higher_state)

    if l == 0:  # negative log likelihood for multi-class classification
        target = (m.curr_cause[l].requires_grad_()).argmax(-1).squeeze()
        output = pred.reshape([batch_size, -1])
        error = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(output), target, reduction='none') # , reduction='mean'
    else:  # mean error
        error = (m.curr_cause[l].requires_grad_()) - pred.reshape(m.curr_cause[l].shape) # predict lower cause

    error = error.reshape([1, -1]).unsqueeze(-1)

    # initialise fast and slow cause state precision
    if m.covar_slow[l].shape[-1] <= 1:  # initialise prior cause state (co-)variance
        m.covar_slow[l] = torch.eye(error.shape[-2]).unsqueeze(0).repeat([B_SIZE_PRECISION_SLOW, 1, 1]) * (
                    var_prior - covar_prior) + covar_prior
    if m.covar[l].shape[-1] <= 1:  # initialise prior cause state (co-)variance
        m.covar[l] = torch.eye(error.shape[-2]).unsqueeze(0).repeat([batch_size, 1, 1]) * (
                    var_prior - covar_prior) + covar_prior

    if learn:
        if l >0:
            F = torch.abs(error) * torch.abs(torch.matmul(m.covar_slow[l].requires_grad_() ** -1, torch.abs(error))) #torch.abs(error) *
        else:
            F = torch.abs(error)
    else:
        if l >0:
            F = torch.abs(error) * torch.abs(torch.matmul(m.covar[l].requires_grad_() ** -1, torch.abs(error))) #torch.abs(error) *
        else:
            F = torch.abs(error)

    if optimize: # optimize hierarchical prediction
        F.backward(gradient=torch.ones_like(F))  # loss per batch element (not scalar)
        m.mask_state_grads(l+1, split=CAUSE_SPLIT, cause=True)  # part of cause states used for outgoing prediction
        m.mask_state_grads(l+1, split=HIDDEN_SPLIT)  # part of hidden states used for outgoing prediction
        m.mask_weights_grads(l+1)
        opt.step()

    if infer_precision: # update precision decay
        if learn:
            m.covar_slow[l] = (m.covar_slow[l] - m.lr[l][3] * m.covar_slow[l]**-1).detach()  # cause state variance decay
            m.covar_hidden_slow[l] = (m.covar_hidden_slow[l] - m.lr[l][3] * m.covar_hidden_slow[l]**-1).detach()  # hidden state variance decay
        else:
            m.covar[l] = (m.covar[l] - m.lr[l][3] * m.covar[l]**-1).detach()  # cause state variance decay
            m.covar_hidden[l] = (m.covar_hidden[l] - m.lr[l][3] * m.covar_hidden[l]**-1).detach()  # hidden state variance decay

    if optimize and transition: # optimize dynamical prediction

        # hidden state prediction error
        current_state = m.state(l+1)
        current_state = current_state.requires_grad_()
        pred_t = m.layers_d[l+1].forward(current_state)
        error_t = current_hidden_prior - pred_t.reshape(current_hidden_prior.shape)
        error_t = error_t.reshape([batch_size, -1]).unsqueeze(-1)

        # initialise fast and slow hidden state precision
        if m.covar_hidden_slow[l + 1].shape[-1] <= 1:  # initialise prior hidden state (co-)variance
            m.covar_hidden_slow[l + 1] = torch.eye(error_t.shape[-2]).unsqueeze(0).repeat([B_SIZE_PRECISION_SLOW, 1, 1]) * (
                        var_prior - covar_prior) + covar_prior
        if m.covar_hidden[l + 1].shape[-1] <= 1:  # initialise prior hidden state (co-)variance
            m.covar_hidden[l + 1] = torch.eye(error_t.shape[-2]).unsqueeze(0).repeat([batch_size, 1, 1]) * (
                        var_prior - covar_prior) + covar_prior

        if learn:
            F = torch.abs(error) * torch.abs(torch.matmul(m.covar_hidden_slow[l + 1].requires_grad_() ** -1, torch.abs(error_t))) # torch.abs(error_t) *
        else:
            F = torch.abs(error) * torch.abs(torch.matmul(m.covar_hidden[l+1].requires_grad_() ** -1, torch.abs(error_t))) # torch.abs(error_t) *

        F.backward(gradient=torch.ones_like(F))  # loss per batch element (not scalar)
        m.mask_grads(l+1, split=CAUSE_SPLIT, cause=True)  # part of higher cause state to predict from (0 means None)
        m.mask_grads(l+1, split=HIDDEN_SPLIT)  # (state size / split) hidden units are used for outgoing prediction
        opt.step()

    return pred.detach().numpy(), error, torch.zeros_like(pred)


""" Network settings"""
UPDATES, B_SIZE, B_SIZE_TEST, IMAGE_SIZE = 100, 128, 512, 10  # model updates, batch size, input size
CONVERGED_INFER = .8  # prediction error threshold to stop inference
PRECISION = True
STATE_SIZES = [10, 28*28]  # (output size, input size) per layer
CAUSE_SPLIT, HIDDEN_SPLIT = 1, 0 # causes & hidden states used for outgoing prediction

""" Experiment settings"""
TEST_TARGET = False  # test dataset: reconstruct target given input (generative setting)
TEST_INPUT = True  # test dataset: reconstruct input given target (discriminative setting)
DM_THRESHOLD = 1 # allowed error fluctuation for delayed updates (see https://www.nature.com/articles/s42003-021-02994-2)

if __name__ == '__main__':
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./', train=True, download=False,
                                                                          transform=torchvision.transforms.Compose(
                                                                              [torchvision.transforms.ToTensor(),
                                                                                  torchvision.transforms.Normalize(
                                                                                      (0.1307,), (0.3081,))])),
        batch_size=B_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./', train=False, download=False,
                                                                         transform=torchvision.transforms.Compose(
                                                                             [torchvision.transforms.ToTensor(),
                                                                                 torchvision.transforms.Normalize(
                                                                                     (0.1307,), (0.3081,))])),
        batch_size=B_SIZE_TEST, shuffle=True)

    PCN = Model(STATE_SIZES, act=torch.nn.Identity(),  # hierarchical activation
                lr_sh=[0 for i in range(len(STATE_SIZES)-2)]+[0],  # higher state learning rate
                lr_sl=[0] + [0 for i in range(len(STATE_SIZES) - 1)],  # state learning rate
                lr_w=[0.0001 for i in range(len(STATE_SIZES))],  # hierarchical weights learning rate
                lr_w_d=[0 for i in range(len(STATE_SIZES))],  # dynamical weights learning rate
                lr_p=[0.00 for i in range(len(STATE_SIZES))],  # hierarchical & dynamical precision learning rate
                sr=[1 for i in range(14)])  # sampling interval (skipped observations in lower layer)

    for env_id, env_name in enumerate(['Train', 'Test'][:2]):

        if env_id == 0: # train
            batch_size = B_SIZE
            DATAPOINTS = train_loader.dataset.train_data.shape[0] // batch_size  # number of datapoints
            data_loader = train_loader
        else: # test
            batch_size = B_SIZE_TEST
            data_loader = test_loader
            DATAPOINTS = 1

        # create and initialise states and learning rates
        PCN.create_states(batch_size=batch_size)
        PCN.create_covar()
        initialise(PCN, test=env_id > 0)

        print("Layers:", PCN.layers)
        print("Cause states :", [list(c.shape) for c in PCN.curr_cause])
        print("Hidden states:", [list(c.shape) for c in PCN.curr_hidden])

        # set up delayed learning
        if DM_THRESHOLD is not None:
            lowest_error = np.asarray([100.]) # threshold for delayed weights modulation
            last_PCN = copy.deepcopy(PCN) # weights before last update

        for a_id, (data, target) in enumerate(data_loader):

            """ Check progress and preprocess input """
            if a_id >= DATAPOINTS: break

            # select how much to log and visualize
            if a_id == 0:
                [err_h, err_t, preds_h, preds_t, preds_g, cause_l0, preds_gt], inputs = [[[] for _ in PCN.layers] for _ in range(7)], [[]]
                raw_errors, variances_slow, datapoints, total_updates = [], None, [], 0
                precisions, precisions_slow, precisions_hidden_slow, precisions_hidden = [], [], [], []

            converged = False
            input = data.reshape([batch_size, 1, -1])

            if DM_THRESHOLD is not None: # switch on delayed updates of weights
                """ Prior prediction """
                feed_target(PCN, batch_size=batch_size)
                pred_g_prior = predict(PCN, l=len(PCN.layers)-1, keep_states=False)
                feed_observation(PCN, batch_size=batch_size)
                _, e_h_prior, _ = GPC(PCN, l=0, infer_precision=True, optimize=False, transition=False, learn=False)
                accuracy_prior, pred_classes_prior, correct_prior = batch_accuracy(pred_g_prior, target, batch_size=batch_size)

                is_nan = e_h_prior.mean().isnan()
                e_h_prior = e_h_prior.clone().detach().abs().mean().numpy()

                """ Delayed modulation of last weight update (active inference experiments)
                This is not used for now, it will accept all weights updates """
                if e_h_prior <= lowest_error + DM_THRESHOLD and not is_nan:  # keep improved weights todo running average? todo threshold/precision weighting
                    lowest_error = e_h_prior
                    last_PCN = copy.deepcopy(PCN)
                    PCN.covar_slow = [v.clone().detach().requires_grad_() for v in PCN.covar] # set slow precision to fast precision if prior prediction improved
                    PCN.covar_hidden_slow = [v.detach().requires_grad_() for v in PCN.covar_hidden]
                else: # reject last weights update
                    PCN = last_PCN

            """ Reset fast precision to learned precision prior """
            PCN.covar = [v.clone().detach().requires_grad_() for v in PCN.covar_slow]
            PCN.covar_hidden = [v.detach().requires_grad_() for v in PCN.covar_hidden_slow]

            """ Optionally predict through entire network first. """
            if True:
                feed_target(PCN, batch_size=batch_size)
                predict(PCN, l=len(PCN.layers)-1, keep_states=False)

            """ Feed target ("deepest prior" or "cause") and input ("outcome") """
            feed_target(PCN, batch_size=batch_size)
            feed_observation(PCN, batch_size=batch_size)

            """  Optimise hierarchical layers in parallel """
            for update in range(UPDATES):
                if converged: break
                if env_id == 1: converged = True # no inference or learning during testing

                total_updates += 1

                # update hierarchical layers
                for l_h in reversed(range(len(PCN.layers))):

                    # compute prior accuracy and precision
                    if update == 0:
                        # update slow precision (learning)
                        if PRECISION: GPC(PCN, l=l_h, infer_precision=True, learn=True)

                    # optimise states & fast precision (inference)
                    p_h, e_h, p_t = GPC(PCN, l=l_h, transition=False) # (l_h > 0 )  # step hierarchical variables
                    if PRECISION: GPC(PCN, l=l_h, infer_precision=True)  # update fast precision (inference, within datapoint)

                    if l_h == 0:
                        # log precision and error
                        try:
                            precisions.append(np.diag(np.array(PCN.covar[0].mean(dim=0).detach())).mean())
                            precisions_slow.append(np.diag(np.array(PCN.covar_slow[0].mean(dim=0).detach())).mean())
                            # todo log hidden state precision
                        except:
                            pass

                        raw_errors.append(e_h[0].mean(dim=0).detach().numpy().mean())

                        # check for convergence
                        if env_id == 0:
                            converged = e_h[0].mean(dim=0).mean().abs() < CONVERGED_INFER

                    """ Collect results and visualize"""
                    if (update == UPDATES - 1 and l_h == 0) or (converged and l_h == 0):
                        for d, i in zip([inputs[0], preds_h[l_h], err_h[l_h]], [input[:1], p_h[:1], e_h[:1].detach()]): d.append(i)  # hierarchical
                        cause_l0[l_h].append(PCN.curr_cause[0].detach().numpy()) # first cause (input layer)
                        pred_g = predict(PCN, l=len(PCN.layers)-1, keep_states=True)
                        preds_g[l_h].append(pred_g)  # predict from deepest cause
                        preds_gt[l_h].append(PCN.curr_cause[-1].detach().numpy())  # deepest cause (target layer)

                        # compute posterior classification accuracy
                        accuracy, pred_classes, correct = batch_accuracy(pred_g, target, batch_size=batch_size)

                        if a_id == DATAPOINTS - 1 and False:
                            for d, n in zip([cause_l0[0][-1], p_h, preds_g[0][-1], preds_gt[0][-1]], ["input", "pred_h", "pred_g", "pred_gt"]):
                                plot_batch(d, title=str(env_name) + n, targets=target, predictions=torch.tensor(pred_classes))
                        datapoints.append(total_updates)

                        try: # print prior prediction accuracy if computed
                            print("Batch:", a_id+1, "|", DATAPOINTS, "\t Update:", update+1, "|", UPDATES,
                                  "\t Prior Acc:", accuracy_prior.round(decimals=1),
                                  "\t Posterior Acc:", accuracy.round(decimals=1),
                                  #"\t Prior Prec:", precision_prior.round(decimals=1),
                                  "\t Best error:", lowest_error.round(decimals=3))
                        except: # print only posterior accuracy
                            print(a_id + 1, "|", DATAPOINTS, "\t Update", update + 1, "|", UPDATES,
                                  "\t Posterior Acc:", accuracy.round(decimals=1) )

        # visualize results
        plot_thumbnails([precisions_slow, precisions], ["Cause state precision (learning)", "Cause state precision (inference)"],
                        errors=raw_errors, inputs=None, datapoints=datapoints, threshold=0.2, img_s=2);
        print(env_name, " done")




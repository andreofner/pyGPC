"""
Differentiable Generalized Predictive Coding
André Ofner 2021
MNIST classification with a static predictive coding model
"""
import matplotlib.pyplot as plt

from tools import *
import torch.nn.init
from torch.optim import SGD
from torch.nn import Sequential, Linear
import copy
plt.style.use(['seaborn-paper'])

class Model(torch.nn.Module):
    """ Hierarchical dynamical predictive coding model """
    def __init__(self, sizes, act, sr=[], dynamical=False,
                 lr_w=None, lr_sl=None, lr_sh=None, lr_p=None, lr_w_d=None):
        super(Model, self).__init__()

        self.layers = []  # hierarchical weights
        self.layers_d = []  # transition weights
        self.lr = []  # learning rate per layer
        self.dynamical = dynamical  # model type
        self.sizes = sizes  # hierarchical state sizes
        self.sr = sr  # sampling rate per layer
        self.lr_names = ["Higher state", "State", "Weights", "Precision", "Dynamical weights"]
        self.initialised = [False for _ in sizes] # inferred covariance gets initialised at first update
        self.initialised_slow = [False for _ in sizes] # learned covariance gets initialised at first update

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
        # precision for inference (within datapoint)
        self.covar = [torch.zeros(1) for _ in self.curr_cause]  # cause states precision
        self.covar_hidden = [torch.zeros(1) for _ in self.curr_cause]  # hidden states precision

        # precision for learning (between datapoints)
        self.covar_slow = [torch.zeros(1) for _ in self.curr_cause]  # cause states precision
        self.covar_hidden_slow = [torch.zeros(1) for _ in self.curr_cause]  # hidden states precision

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
            self.layers[l-1][1].weight.grad = self.layers[l-1][1].weight.grad * (state_grad[0]>0)*1.

def predict(m, l, keep_states=True, keep_cause=None, keep_hidden=None):
    """ Prediction through the cause states of the entire network."""

    # make sure model stays unchanged if requested
    if keep_states and keep_cause is None:
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
        return predict(m, l-1, keep_states=keep_states, keep_cause=keep_cause, keep_hidden=keep_hidden)
    else: # return prediction in lowest layer
        input_prediction = m.curr_cause[0].clone().detach().numpy()
        if keep_states:
            m.curr_cause = keep_cause
            m.curr_hidden = keep_hidden
        return input_prediction

def GPC(m, l, infer_precision=False, optimize=True,
        var_prior=2, covar_prior=10000000, transition=False, learn=False,
        full_covar=False):
    """ Layer-wise Generalized Predictive Coding optimizer """

    batch_size = m.state(l).shape[0]
    B_SIZE_PRECISION_SLOW = 1

    if learn:  # weights updates
        params = m.params_weights(l)   # weights learning
    else:  # inference or precision updates
        params = m.params_covar(l) if infer_precision else m.params(l)

    opt = SGD(params)
    opt.zero_grad()

    if transition:
        try:
            # transition the hidden to generate its prior prediction
            m.curr_hidden[l+1] = m.layers_d[l+1].forward(m.state(l+1).detach()).detach()

            # remember the prior hidden state to be able to predict its change
            current_hidden_prior = m.curr_hidden[l+1].detach()
        except:
            transition = False  # reached highest layer

    # cause state prediction error
    higher_state = m.state(l+1)
    pred = m.layers[l].forward(higher_state)

    if False and l == 0: # optionally use negative log likelihood for multi-class classification
        target = (m.curr_cause[l].requires_grad_()).argmax(-1).squeeze()
        output = pred.reshape([batch_size, -1])
        error = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(output), target, reduction='mean')  # , reduction='mean'
        error = error.unsqueeze(-1).unsqueeze(-1)
    else:
        target = (m.curr_cause[l].requires_grad_())
        output = pred.reshape(target.shape)
        error = output - target  # predict lower cause
        error = error.abs().unsqueeze(1)

    error += 1  # keeps error from getting too small from squaring

    if learn and not infer_precision:
        error = error.sum().unsqueeze(-1)  # aggregate over batch
    elif learn and not infer_precision:
        error = error.sum().unsqueeze(-1)  # aggregate over batch
    else:
        if full_covar:  # estimate variance and covariance
            error = error.squeeze() # don't aggregate over batch
        else:  # estimate only variance
            error = error.squeeze().mean(-1).unsqueeze(-1).unsqueeze(-1) # don't aggregate over batch

    # initialise fast and slow cause state precision
    if not m.initialised_slow[l] and learn:
        if m.covar_slow[l].shape[-1] <= 1:  # initialise prior cause state (co-)variance
            m.covar_slow[l] = torch.eye(error.shape[-1]).unsqueeze(0).repeat([B_SIZE_PRECISION_SLOW, 1, 1]) * (
                        var_prior - covar_prior) + covar_prior
        m.initialised_slow[l] = True

    if not m.initialised[l] and not learn:
        if m.covar[l].shape[-1] <= 1:  # initialise prior cause state (co-)variance
            m.covar[l] = torch.eye(error.shape[-1]).repeat([batch_size, 1, 1]) * (
                        var_prior - covar_prior) + covar_prior
        m.initialised[l] = True

    if False:
        print("l", l, "err", error.shape, "cov slow", m.covar_slow[l].shape, "cov", m.covar[l].shape)

    if learn:
        F = error * torch.matmul(m.covar_slow[l].requires_grad_() ** -1, error)
    else:
        if full_covar:
            F = error.unsqueeze(-1) * torch.matmul(m.covar[l].requires_grad_() ** -1, error.unsqueeze(-1))
        else:
            F = error * torch.matmul(m.covar[l].requires_grad_() ** -1, error)

    if optimize: # optimize hierarchical prediction
        F.backward(gradient=torch.ones_like(F))
        m.mask_state_grads(l+1, split=CAUSE_SPLIT, cause=True)  # part of cause states used for outgoing prediction
        m.mask_state_grads(l+1, split=HIDDEN_SPLIT)  # part of hidden states used for outgoing prediction
        m.mask_weights_grads(l+1)
        opt.step()

    if infer_precision: # update precision decay
        # todo precision dependent decay
        # todo keep each covariance unit separately >= 1
        if learn:
            if m.initialised_slow and m.covar_slow[l].diagonal().min() >= 1 :
                m.covar_slow[l] = (m.covar_slow[l] - m.lr[l][3] * m.covar_slow[l]**-1).detach()  # cause state variance decay
                m.covar_hidden_slow[l] = (m.covar_hidden_slow[l] - m.lr[l][3] * m.covar_hidden_slow[l]**-1).detach()  # hidden state variance decay
        else:
            if m.initialised and m.covar[l].diagonal().min() >= 1 :
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

        error = torch.abs(error)
        if learn:
            F = error * torch.matmul(m.covar_slow[l].requires_grad_() ** -1, error)
        else:
            F = error * torch.matmul(m.covar[l].requires_grad_() ** -1, error)

        F.sum().backward()
        m.mask_grads(l+1, split=CAUSE_SPLIT, cause=True)  # part of higher cause state to predict from (0 means None)
        m.mask_grads(l+1, split=HIDDEN_SPLIT)  # (state size / split) hidden units are used for outgoing prediction
        opt.step()

    return pred.detach().numpy(), error, torch.zeros_like(pred)

def feed_target_classification(PCN, input, batch_size, test):
    """ Feed target to model """
    torch.nn.init.xavier_uniform_(PCN.curr_cause[-1])
    if not (TEST_TARGET and test):  # evaluate target reconstruction
        PCN.curr_cause[-1] = torch.tensor(input.flip(-1).detach().float()).reshape(
            [batch_size, 1, -1])  # e.g. mirrored input image as target

def feed_observation_classification(PCN, target, batch_size, test):
    """ Feed observation to model """
    torch.nn.init.xavier_uniform_(PCN.curr_cause[0])
    if not (TEST_INPUT and test):  # evaluate input reconstruction
        PCN.curr_cause[0] = torch.zeros_like(PCN.curr_cause[0])
        for b in range(batch_size):
            for c in range(0,IMAGE_SIZE,16):
                PCN.curr_cause[0][b, :, c+target[b]] = 1  # e.g. one hot digit as input

def feed_target_image(PCN, input, batch_size, test):
    """ Feed target image to model """
    torch.nn.init.xavier_uniform_(PCN.curr_cause[-1])
    if not (TEST_TARGET and test):  # evaluate target reconstruction
        PCN.curr_cause[-1] = torch.tensor(input.flip(-1).detach().float()).reshape(
            [batch_size, 1, -1])  # e.g. mirrored input image as target

def feed_target(PCN, target, batch_size, test):
    """ Feed class target to model """
    torch.nn.init.xavier_uniform_(PCN.curr_cause[-1])
    if not (TEST_TARGET and test):  # evaluate target reconstruction
        PCN.curr_cause[-1] = torch.zeros_like(PCN.curr_cause[-1])
        for b in range(batch_size):
                PCN.curr_cause[-1][b, :, target[b]] = 1  # e.g. one hot digit as input

def feed_observation(PCN, input, batch_size, test):
    """ Feed image observation to model """
    torch.nn.init.xavier_uniform_(PCN.curr_cause[0])
    if not (TEST_INPUT and test):  # evaluate input reconstruction
        PCN.curr_cause[0] = torch.tensor(input.detach().float()).reshape(
            [batch_size, 1, -1])

def initialise(PCN, test=False):
    """ Initialise states and learning rates"""
    for l in range(len(PCN.layers)+1):
        PCN.curr_cause[l] = torch.ones_like(PCN.curr_cause[l]) # recover from nans..
        PCN.curr_hidden[l] = torch.ones_like(PCN.curr_hidden[l]) # recover from nans..
        torch.nn.init.xavier_uniform_(PCN.curr_cause[l],1)
        torch.nn.init.xavier_uniform_(PCN.curr_hidden[l],1)

    for l in range(len(PCN.layers)):
        if test:  # change learning rates for testing
            PCN.lr[l][2] = 0  # hierarchical weights lr
            PCN.lr[l][4] = 0  # dynamical weights lr
            if TEST_INPUT:  # evaluate input reconstruction
                PCN.lr[l][0] = 0.0  # higher state lr
            elif TEST_TARGET:  # evaluate target reconstruction
                #PCN.lr[l][0] = 0.01  # higher state lr # todo value
                PCN.lr[0][1] = 0.0  # state lr input layer
    return PCN

def batch_accuracy(pred_g, target, batch_size):
    """ Computes classification results for a batch"""
    pred_classes = pred_g.detach().reshape([batch_size, 1, 10]).mean(-2).argmax(-1)
    correct = (torch.tensor(pred_classes) == target).sum().detach().numpy()
    accuracy = 100. * correct / batch_size
    return accuracy, pred_classes, correct


""" Network settings"""
UPDATES, B_SIZE, B_SIZE_TEST, IMAGE_SIZE = 100, 128, 128, 28*28  # model updates, batch size, input size
CONVERGED_INFER = 1.1  # prediction error threshold to stop inference
SIZES = [28*28, 256,256, 128,128, 10]  # (output size, input size) per layer
CAUSE_SPLIT, HIDDEN_SPLIT = 1, 0  # percentage of causes & hidden states used for outgoing prediction
PREDICT_FIRST = True  # propagate prediction through entire network before computing errors
PRECISION = True  # estimate fast (inference, within datapoint) and slow precision (learning, between datapoint)

""" Experiment settings"""
TEST_TARGET = True  # test dataset: reconstruct target given input (generative setting)
TEST_INPUT = False  # test dataset: reconstruct input given target (discriminative setting)
DM_THRESHOLD = None  # allowed error fluctuation for delayed updates (see nature.com/articles/s42003-021-02994-2). None means inactive

""" Plot & Print settings"""
SUMMARY = True
PLOT = True

def run(UPDATES, PCN=None, test=False, DATAPOINTS=50):
    if PCN is None:
        PCN = Model(SIZES, act=torch.nn.Identity(),  # activation function of hierarchical weights
                    lr_sh=[10 for _ in range(len(SIZES)-2)]+[0],  # higher state learning rate
                    lr_sl=[0] + [1 for _ in range(len(SIZES))],  # state learning rate
                    lr_w=[.01 for _ in range(len(SIZES))],  # hierarchical weights learning rate
                    lr_w_d=[0 for _ in range(len(SIZES))],  # dynamical weights learning rate
                    lr_p=[.01 for _ in range(len(SIZES))],  # hierarchical & dynamical precision learning rate
                    sr=[1 for _ in range(14)])  # sampling interval (skipped observations in lower layer)

    if test:
        env_id = 1; env_name = 'Test';
    else:
        env_id = 0; env_name = 'Train';

    if env_id == 0: # train
        batch_size = B_SIZE
        data_loader = torchvision_mnist(B_SIZE, train=True)
        if DATAPOINTS is None:
            DATAPOINTS = data_loader.dataset.train_data.shape[0] // batch_size
        PCN.create_states(batch_size=batch_size)
        PCN.create_covar() # create covariance
    else: # test
        batch_size = B_SIZE_TEST
        data_loader = torchvision_mnist(B_SIZE_TEST, train=False)
        DATAPOINTS = 1
        PCN.initialised = [False for _ in PCN.initialised]
        PCN.create_states(batch_size=batch_size)

    # create and initialise states and learning rates
    initialise(PCN, test=env_id > 0)

    if SUMMARY:
        print("Layers:", PCN.layers)
        print("Cause states :", [list(c.shape) for c in PCN.curr_cause])
        print("Hidden states:", [list(c.shape) for c in PCN.curr_hidden])
        print("Covariance learning:", [list(c.shape) for c in PCN.covar_slow])
        print("Covariance inference:", [list(c.shape) for c in PCN.covar])
        print("Learning rates:", [list(c) for c in PCN.lr])

    # set up delayed learning
    if DM_THRESHOLD is not None:
        lowest_error = np.asarray([100.]) # threshold for delayed weights modulation
        last_PCN = copy.deepcopy(PCN) # weights before last update

    for a_id, (data, target_) in enumerate(data_loader):

        """ Check progress """
        if a_id >= DATAPOINTS: break
        converged = False
        input = data.reshape([batch_size, 1, -1])
        target = target_

        """ select how much to log and visualize """
        if a_id == 0:
            [err_h, preds_h, preds_g, cause_l0, preds_gt], inputs = [[[] for _ in PCN.layers] for _ in range(5)], [[]]
            variances_slow, datapoints, total_updates = None, [], 0
            raw_errors, precisions, precisions_slow, precisions_hidden_slow, precisions_hidden = [[[] for _ in PCN.layers] for _ in range(5)]

        """ Prior prediction """
        if False:
            feed_target(PCN, batch_size=batch_size, target=target, test=env_id>0)
            pred_g_prior = predict(PCN, l=len(PCN.layers)-1, keep_states=True)
            feed_observation(PCN, batch_size=batch_size, input=input, test=env_id > 0)
            _, e_h_prior, _ = GPC(PCN, l=0, infer_precision=True, optimize=False, transition=False, learn=False, full_covar=test)
            #accuracy_prior, pred_classes_prior, correct_prior = batch_accuracy(PCN.curr_cause[-1], target, batch_size=batch_size)

        if DM_THRESHOLD is not None:  # switch on delayed updates of weights
            """ Delayed modulation of last weight update (active inference experiments) 
            Inactive when DM_THRESHOLD is None or very large """
            is_nan = e_h_prior.mean().isnan()
            e_h_prior = e_h_prior.clone().detach().abs().mean().numpy()
            if e_h_prior <= lowest_error + DM_THRESHOLD and not is_nan:  # keep improved weights todo running average? todo threshold/precision weighting
                lowest_error = e_h_prior
                last_PCN = copy.deepcopy(PCN)
            else: # reject last weights update
                PCN = last_PCN

        """ Initialise fast precision with learned precision prior """
        if PRECISION and not test:  # during testing, we might have different covar shapes
            PCN.covar = [(PCN.covar[l]*0 + v).detach().requires_grad_() for l,v in enumerate(PCN.covar_slow)]
            PCN.covar_hidden = [(PCN.covar_hidden[l]*0 + v).detach().requires_grad_() for l,v in enumerate(PCN.covar_hidden_slow)]
            #PCN.initialised = [False for _ in PCN.layers] # ignore learned precision

        """ Optionally predict through entire network first. """
        if PREDICT_FIRST:
            feed_target(PCN, batch_size=batch_size, target=target, test=env_id>0)
            predict(PCN, l=len(PCN.layers)-1, keep_states=False)

        """ Feed target ("deepest prior" or "cause") and input ("outcome") """
        feed_target(PCN, batch_size=batch_size, target=target, test=env_id>0)
        feed_observation(PCN, batch_size=batch_size, input=input, test=env_id>0)

        """  Optimise hierarchical layers in parallel """
        for update in range(UPDATES):
            if converged: break
            total_updates += 1

            # update hierarchical layers
            for l_h in reversed(range(len(PCN.layers))): # todo

                # inference: states & fast precision
                if PRECISION: GPC(PCN, l=l_h, infer_precision=True, full_covar=test)  # update fast precision (inference, within datapoint)
                p_h, e_h, p_t = GPC(PCN, l=l_h, transition=False, full_covar=test) # (l_h > 0 )  # step hierarchical variables

                # learning: update slow precision and weights
                if PRECISION and update == 0 and not test:
                    GPC(PCN, l=l_h, infer_precision=True, learn=True, full_covar=test)

                # log precision and error at selected layer
                precisions[l_h].append(np.diag(np.array(PCN.covar[l_h].mean(dim=0).detach())).mean())
                precisions_slow[l_h].append(np.diag(np.array(PCN.covar_slow[l_h].mean(dim=0).detach())).mean())
                raw_errors[l_h].append(e_h[0].mean(dim=0).detach().numpy().mean())

                # check for convergence on input layer
                if env_id == 0 and l_h == 0:
                    converged = e_h[0].mean(dim=0).mean().abs() < CONVERGED_INFER

                """ Collect results and visualize"""
                if (update == UPDATES - 1 and l_h == 0) or (converged and l_h == 0):
                    for d, i in zip([inputs[0], preds_h[l_h], err_h[l_h]], [input[:1], p_h[:1], e_h[:1].detach()]): d.append(i)  # hierarchical
                    cause_l0[l_h].append(PCN.curr_cause[0].detach().numpy()) # first cause (input layer)
                    pred_g = predict(PCN, l=len(PCN.layers)-1, keep_states=True)
                    preds_g[l_h].append(pred_g)  # predict from deepest cause
                    preds_gt[l_h].append(PCN.curr_cause[-1].detach().numpy())  # deepest cause (target layer)

                    # compute posterior classification accuracy
                    accuracy, pred_classes, correct = batch_accuracy(PCN.curr_cause[-1], target, batch_size=batch_size)
                    if a_id == DATAPOINTS - 1 and False:
                        for d, n in zip([cause_l0[0][-1], p_h, preds_g[0][-1], preds_gt[0][-1]], ["input", "pred_h", "pred_g", "pred_gt"]):
                            plot_batch(d, title=str(env_name) + n, targets=target, predictions=torch.tensor(pred_classes))
                    datapoints.append(total_updates)

                    print(a_id + 1, "|", DATAPOINTS, "\t Update", update + 1, "|", UPDATES, "Error", e_h[0].mean().detach().numpy().round(3))

    if PLOT and env_id == 0:

        # visualize results
        plot_thumbnails([precisions_slow[0], precisions[0]], ["Cause state precision (learning)", "Cause state precision (inference)"],
                        errors=raw_errors[0], inputs=None, datapoints=datapoints, threshold=0.2, img_s=2, l=1);
        try:  # visualize second layer if available
            plot_thumbnails([precisions_slow[1], precisions[1]], ["Cause state precision (learning)", "Cause state precision (inference)"],
                            errors=raw_errors[1], inputs=None, datapoints=datapoints, threshold=0.2, img_s=2, l=2);
        except:
            pass
    return PCN, input, target, pred_g

if __name__ == '__main__':

    """
    Train a static predictive coding model on MNIST
    During training only the mean of the variance is estimated for each batch element
    """


    """ Train """
    # train weights on <DATAPOINTS> batches. <UPDATES> inference steps per DATAPOINTS.
    # each inference step refines a) the prior prediction and b) the estimated cause state for the current batch
    PCN, input, target, pred_g = run(UPDATES=20, DATAPOINTS=100, PCN=None, test=False)  # around 50 batches are enough to see meaningful results
    visualize_multilayer_generation(PCN, input, target, title="Hierarchical prediction (train set)")

    """ Test """
    #  <UPDATES> inference steps on a single batch
    # the classification accuracy increases with increasing <UPDATES>
    # when high accuracy classification or prediction is relevant, set a high value <UPDATES> and stop when converged, e.g. precision=1
    PCN = initialise(PCN, test=True)

    # manually initialise inference precision 
    for l,_ in enumerate(PCN.layers):
        PCN.covar[l] = torch.eye(PCN.curr_cause[l].shape[-1]).repeat([B_SIZE_TEST, 1, 1]) * (1 - 1000) + 1000
    PCN.initialised = [True for _ in PCN.layers]

    for l, _ in enumerate(PCN.layers):
        PCN.lr[l][0] = .01  # learning rate higher state
        PCN.lr[l][1] = .01  # learning rate lower state
        PCN.lr[l][2] = 0  # learning rate hierarchical weights
        PCN.lr[l][3] = .1  # learning rate precision
        PCN.lr[l][4] = 0  # learning rate dynamical weights
    PCN, input, target, _ = run(UPDATES=10, PCN=PCN, test=True) # 10-100 updates are enough to see meaningful results
    print("Test accuracy:", batch_accuracy(PCN.curr_cause[-1], target, B_SIZE_TEST)[0])
    visualize_multilayer_generation(PCN, input, target, title="Hierarchical prediction (test set)")

    for example in range(1):
        visualize_precision(PCN, batch_id=example)

    # visualize the covariance matrix (inverse of precision) of the first layer
    plt.imshow(PCN.covar[0][example].detach().reshape([28*28,28*28])**-1)
    plt.colorbar()
    plt.show()
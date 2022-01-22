"""
Differentiable Generalized Predictive Coding
André Ofner 2021
"""

import torch
from tools import *
from torch.optim import SGD
from torch.nn import Sequential, Linear, ELU, ConvTranspose2d

class Model(torch.nn.Module):
    """ Hierarchical dynamical predictive coding model """

    def __init__(self, sizes, act, act_d=None, dynamical=False,
                 dim=[], sr=[], lr_w=None, lr_sl=None, lr_sh=None, lr_prev=None, lr_g=None,
                 gen_coord=1):
        super(Model, self).__init__()

        self.layers = [] # hierarchical+dynamical weights
        self.layers_c = [] # needed for state initialisation
        self.layers_d = []  # dynamical weights
        self.lr = []  # learning rate per layer
        self.dynamical = dynamical  # model type
        self.sizes = sizes  # hierarchical state sizes
        self.dim = np.asarray(dim)  # state channels
        self.sr = sr  # sampling rate per layer
        self.gen_coord = gen_coord # number of higher order transition derivatives (generalised coordinates)
        self.n_cnn = ((self.dim > 1).sum()) // 2 + 1  # number of CNN layers (optional)
        self.state_sizes = np.sqrt(self.sizes / self.dim).astype(int)[:self.n_cnn * 2]  # for CNN layers (optional)

        # default learning rates
        if lr_w is None: lr_w = [0. for _ in self.sizes]  # weights
        if lr_sh is None: lr_sh = [1. for _ in self.sizes]  # higher state
        if lr_sl is None: lr_sl = [1. for _ in self.sizes]  # lower state
        if lr_prev is None: lr_prev = [lr_sl[l] for l in range(len(lr_sl))]  # last lower state
        if lr_g is None: lr_g = [.1 for _ in self.sizes]  # state gain

        # create layers
        for i in range(0, len(self.sizes ), 2):
            if not dynamical:
                net_h = ConvTranspose2d(dim[i + 1], dim[i], 2, stride=2, bias=False) if dim[i + 1] > 1 else Linear(
                    self.sizes[i+1], self.sizes[i], False)
                net = ConvTranspose2d(dim[i + 1]*2, dim[i], 2, stride=2, bias=False) if dim[i + 1] > 1 else Linear(
                    self.sizes[i+1]*2, self.sizes[i], False) # todo hidden size

                # dense or convolutional hierarchical weights
                self.layers.append(Sequential(act, net_h, torch.nn.Dropout(0.0)))
                self.layers_c.append(Sequential(act, net, torch.nn.Dropout(0.0)))

                # dynamical weights # todo hidden size
                if i == 0: # input layer has no hidden units
                    dynamical_ch = dim[i] # dynamical_dim=1 for dense (single state channel) dynamical layers
                    self.layers_d.append(Model([self.sizes[i] for _ in range(gen_coord*2)], act_d, dynamical=True,
                                               dim=[dynamical_ch for c in range(gen_coord * 2)],
                                               sr=[sr[i] for _ in self.sizes[i:]]))
                else:
                    ratio = [self.sizes[i], self.sizes[i]*2] # todo hidden size
                    self.layers_d.append(Model(ratio, act_d, dynamical=True,
                                               dim=[1 for _ in range(gen_coord*2)], sr=[sr[i] for _ in self.sizes[i:]]))
            else:
                # dynamical layers
                net_d = Sequential(act, ConvTranspose2d(dim[i+1], dim[i], 2, stride=2, bias=False)) if dim[i + 1] > 1 else \
                    Sequential(act, Linear((self.sizes[i + 1]), (self.sizes[i]), False))
                self.layers.append(net_d)  # dynamical layers
            self.lr.append([lr_sh[i], lr_sl[i], lr_prev[i], lr_w[i], .1, lr_g[i]])  # learning rate per layer

        # initialise states, state gain and precision
        [self.curr_cause, self.last_cause] = [self.init_states(mixed=(self.n_cnn > 0)) for _ in range(2)] # states at time t & t-dt
        [self.c_curr_gain, self.c_last_gain] = [self.init_states(mixed=(self.n_cnn > 0)) for _ in range(2)] # state gain at time t & t-dt
        self.covar = [torch.zeros(1) for _ in self.layers + [None]]  # inverse precision

        # initialise memory states, state gain
        [self.curr_hidden, self.last_hidden] = [self.init_states(mixed=(self.n_cnn > 0)) for _ in range(2)]  # states at time t & t-dt
        [self.h_curr_gain, self.h_last_gain] = [self.init_states(mixed=(self.n_cnn > 0)) for _ in range(2)]  # state gain at time t & t-dt
        self.covar_hidden = [torch.zeros(1) for _ in self.layers + [None]]  # inverse precision

        if not dynamical:
            self.layers = self.layers_c  # keep only layers with full state as input
            del self.layers_c # todo improve state initialisation

    def init_states(self, mixed=False):
        """ Create state priors for hierarchical and dynamical layers"""
        if self.dynamical:
            return self.predict(torch.ones([B_SIZE, 1, self.sizes[-1]]).float(), prior=True)
        elif mixed:  # CNN and dense layers
            states_cnn = self.predict(
                torch.ones([B_SIZE, self.dim[self.n_cnn * 2 - 1], self.state_sizes[-1], self.state_sizes[-1]]).float(),
                prior=True, layers=self.layers[:self.n_cnn])
            states_dense = self.predict(torch.ones([B_SIZE, 1, self.sizes[-1]]).float(), prior=True, layers=self.layers[self.n_cnn:])
            return states_cnn + states_dense[1:]
        elif self.n_cnn > 1:  # CNN layers
            return self.predict(torch.ones([B_SIZE, self.dim[-1], self.sizes[-1], self.sizes[-1]]).float(), prior=True)
        else:  # dense layers
            return self.predict(torch.ones([B_SIZE, 1, self.sizes[-1]]).float(), prior=True)

    def params(self, l, deepest_layer=False, dynamical=False):  # todo optimize last_gain
        """ Parameters and learning rates per layer. Top-down predictability regularizes learning and inference."""
        l_h = l if deepest_layer else l + 1
        if not dynamical:
            params = [{'params': [self.layers[l][1].weight.requires_grad_()], 'lr': self.lr[l][3]}, # top-down weights (l)
                      {'params': [self.curr_cause[l + 1]], 'lr': self.lr[l][0]},  # higher state (l+1,t)
                      {'params': [self.curr_hidden[l + 1]], 'lr': self.lr[l][0]},  # higher state (l+1,t)
                      {'params': [self.curr_cause[l]], 'lr': self.lr[l][1]},  # lower state (l,t-dt)
                      {'params': [self.curr_hidden[l]], 'lr': self.lr[l][1]},  # lower hidden state (l,t-dt)
                      {'params': [self.last_cause[l]], 'lr': self.lr[l][2]},  # lower state (l,t)
                      {'params': [self.last_hidden[l]], 'lr': self.lr[l][2]},  # lower hidden state (l,t)
                      {'params': [self.c_curr_gain[l]], 'lr': self.lr[l][5]},  # state gain (l)
                      {'params': [self.h_curr_gain[l]], 'lr': self.lr[l][5]},  # state gain (l)
                      {'params': [self.c_curr_gain[l+1]], 'lr': self.lr[l_h][5]},  # state gain (l)
                      {'params': [self.h_curr_gain[l+1]], 'lr': self.lr[l_h][5]}]  # state gain (l)
            return params+self.layers_d[l].params(l=0, dynamical=True)
        else:
            params = [{'params': [self.layers[l][1].weight.requires_grad_()], 'lr': self.lr[l][3]}, # top-down weights (l)
                      {'params': [self.curr_cause[l + 1]], 'lr': self.lr[l][0]},  # higher state (l+1,t)
                      {'params': [self.curr_hidden[l + 1]], 'lr': self.lr[l][0]},  # higher state (l+1,t)
                      {'params': [self.curr_cause[l]], 'lr': self.lr[l][1]},  # lower state (l,t-dt)
                      {'params': [self.curr_hidden[l]], 'lr': self.lr[l][1]},  # lower state (l,t-dt)
                      {'params': [self.last_cause[l]], 'lr': self.lr[l][2]},  # lower state (l,t)
                      {'params': [self.last_hidden[l]], 'lr': self.lr[l][2]},  # lower state (l,t)
                      {'params': [self.c_curr_gain[l]], 'lr': self.lr[l][5]},  # state gain (l)
                      {'params': [self.h_curr_gain[l]], 'lr': self.lr[l][5]}]  # state gain (l)
            return params

    def params_covar(self, l):
        """ Covariance parameters and learning rates per layer """
        params = [{'params': [self.covar[l]], 'lr': self.lr[l][4]}]  # covariance (l) - learning rate
        return params

    def predict(self, target=None, states=None, prior=False, layers=None, state_gains=None):
        """ Backward prediction through all layers. Optionally initialises prior states"""
        states = [target] if states is None else [states[-1]]
        if prior:
            [torch.nn.init.torch.nn.init.normal_(states[-1][b]) for b in range(B_SIZE)]
        if layers is None:
            layers = self.layers
        if state_gains is None and not prior:
            state_gains = self.c_curr_gain
        elif prior:
            state_gains = [None for _ in self.layers]
        for w, g in zip(list(reversed(layers)), list(reversed(state_gains))):
            if not prior:
                states.append(w(states[-1] * g).detach())
            else:
                states.append(w(states[-1]).detach())
                [torch.nn.init.torch.nn.init.normal_(states[-1][b]) for b in range(B_SIZE)]
        return list(reversed(states))

    def freeze(self, params):
        """ Stops optimisation of weights. State inference remains active. """
        for param in params:
            for lr in self.lr: lr[param] = 0.  # freeze hierarchical weights
            if not self.dynamical:
                for l_d in self.layers_d: l_d.freeze(params)  # freeze dynamical weights

    def cause(self, l, last=False):
        """ Returns a layer's cause state with gain """
        state = self.last_cause[l] if last else self.curr_cause[l]
        gain = self.c_last_gain[l] if last else self.c_curr_gain[l]
        return state, gain

    def hidden(self, l, last=False):
        """ Returns a layer's hidden state with gain """
        if l == 0: return self.cause(l, last=False) # inputs have no hidden states
        state = self.last_hidden[l] if last else self.curr_hidden[l]
        gain = self.h_last_gain[l] if last else self.h_curr_gain[l]
        return state, gain

    def state(self, l, last=False):
        """ Concatenated hidden & cause states """
        if l == 0: return self.cause(l, last=False) # inputs have no hidden states
        state_c = self.cause(l, last=last)
        state_h = self.hidden(l, last=last)
        dim = 1 if self.dim[2*l-1] > 1 else -1 # single or multichannel (CNN)
        state = torch.cat([state_c[0], state_h[0]], dim=dim)
        gain = torch.cat([state_c[1], state_h[1]], dim=dim)
        return state, gain

    def mask_grads(self, l, last=False, split=4, cause=False):
        """ Set all state gradients to zero except for those used for the prediction.
            Use split = 0 to zero out all gradients. """
        if cause:
            state, gain = self.cause(l=l, last=last)
        else:
            state, gain = self.hidden(l=l, last=last)
        dim = 1 if self.dim[2 * l - 1] > 1 else -1  # single or multichannel (CNN)
        if split < 1:
            split_size = 0 # zero out all gradients
        else:
            split_size = (state.shape[dim] // split)
        if state.grad is not None:
            if dim == 1: # single state channel
                state.grad[:, split_size:state.shape[dim]].zero_()
                if gain.grad is not None:
                    gain.grad[:, split_size:state.shape[dim]].zero_()
            else: # multiple state channels (CNN)
                state.grad[..., :split_size].zero_()
                if gain.grad is not None:
                    gain.grad[..., split_size:state.shape[dim]].zero_()


def transition_state(m, l, last_state):
    """ Applies the learned transition dynamics to a provided state """
    try:
        m.curr_hidden[l] = m.layers_d[l].layers[0].requires_grad_().forward(last_state[0].requires_grad_()).reshape(
            m.curr_hidden[l].shape).clone().detach().requires_grad_()  # predict from last hidden before gain
    except:  # layers have different amounts of channels
        m.curr_hidden[l] = m.layers_d[l].layers[0].requires_grad_().forward(
            last_state[0].requires_grad_().flatten(start_dim=1)).reshape(
            m.curr_hidden[l].shape).clone().detach().requires_grad_()  # predict from last hidden before gain

def predict(m, l, transition=False, transition_lower=True):
    """
    A) Prediction through the cause states of the entire network, without transitioning.
    B) Transition (current layer or all layers) and predict lower causes.
    """

    if transition: # transition higher layer
        transition_state(m, l + 1, m.state(l + 1, last=True))

    # get cause prediction from higher layer
    higher_state = m.state(l + 1)
    higher_state = higher_state[0].requires_grad_() * higher_state[1].requires_grad_().reshape(
        higher_state[0].shape)

    # replace cause with top-down prediction
    pred = m.layers[l].forward(higher_state)
    m.curr_cause[l] = pred.reshape(m.curr_cause[l].shape).detach().requires_grad_()

    # predict in next lower layer
    if l > 0:
        return predict(m, l-1, transition=transition_lower)
    else:
        return m.curr_cause[0].clone().detach().numpy()


def GPC(m, l, dynamical=False, infer_precision=False, var_prior=7, covar_prior=100000000):
    """ Layer-wise Generalized Predictive Coding optimizer """

    # create optimizer and reset gradients
    if infer_precision:
        params = m.params_covar(l)
    else:
        params = m.params(l, dynamical=dynamical, deepest_layer=(l == len(m.layers) - 1))
    opt = SGD(params)
    opt.zero_grad()

    # apply state gain to relevant states for this layer's update
    higher_state = m.state(l+1)  # split: how much of the state is used for the prediction
    higher_state = higher_state[0].requires_grad_() * higher_state[1].requires_grad_().reshape(higher_state[0].shape)  # higher state
    hidden = m.hidden(l)[0].requires_grad_() * m.hidden(l)[1].requires_grad_().reshape(m.hidden(l)[0].shape)  # current hidden * hidden gain
    last_hidden = m.hidden(l, last=True)[0].requires_grad_() * m.hidden(l, last=True)[1].requires_grad_().reshape(m.hidden(l, last=True)[0].shape) # last hidden * hidden gain
    last_state = m.state(l, last=True) # last

    # prediction from higher layer's current cause and hidden state
    pred = m.layers[l].forward(higher_state)

    if dynamical:  # higher order state change error (generalised coordinates)
        error = (hidden.detach() - last_hidden).flatten(1) - pred.flatten(1)
    else:  # cause prediction error & hidden transition error
        error = (m.curr_cause[l].requires_grad_()) - pred.reshape(m.curr_cause[l].shape) # predict lower cause before gain
        error = error.reshape([B_SIZE, -1]).unsqueeze(-1) # independent precision for each CNN channel

        if m.covar[l].shape[-1] <= 1:  # initialise prior cause state (co-)variance
            m.covar[l] = torch.eye(error.shape[-2]).unsqueeze(0).repeat([B_SIZE_PREC, 1, 1]) * (var_prior - covar_prior) + covar_prior

        F = torch.abs(error) * torch.abs(torch.matmul(m.covar[l].requires_grad_() ** -1, torch.abs(error)))

        if TRANSITION:
            try:
                pred_trans = m.layers_d[l].layers[0].forward(last_state[0].requires_grad_()) # predict from last hidden before gain
            except:  # layers differ in channels
                pred_trans = m.layers_d[l].layers[0].forward(last_state[0].requires_grad_().flatten(start_dim=1)) # predict from last hidden before gain
            error_trans = ((m.curr_hidden[l].requires_grad_()) - pred_trans.reshape(m.curr_hidden[l].shape)).flatten(start_dim=1) # predict current hidden before gain
            error_trans = error_trans.reshape([B_SIZE, -1]).unsqueeze(-1) # independent precision for each CNN channel

            if m.covar_hidden[l].shape[-1] <= 1:  # initialise prior hidden state (co-)variance
                m.covar_hidden[l] = torch.eye(error_trans.shape[-2]).unsqueeze(0).repeat([B_SIZE_PREC, 1, 1]) * (var_prior - covar_prior) + covar_prior

            F = F + torch.abs(error_trans) * torch.abs(torch.matmul(m.covar_hidden[l].requires_grad_() ** -1, torch.abs(error_trans)))

    F.backward(gradient=torch.ones_like(F))  # loss per batch element (not scalar)
    m.mask_grads(l+1, split=0, cause=True) # part of higher cause state to predict from (0 means None)
    m.mask_grads(l+1, split=2)  # use < state size / split > units to predict
    opt.step()
    if infer_precision:
        m.covar[l] = (m.covar[l] - m.lr[l][4] * m.covar[l]**-1).detach()  # cause state variance decay
        m.covar_hidden[l] = (m.covar_hidden[l] - m.lr[l][4] * m.covar_hidden[l]**-1).detach()  # hidden state variance decay
    return pred.detach().numpy(), error


UPDATES, SCALE, B_SIZE, IMAGE_SIZE = 10, 0, 1, 16*16  # model updates, relative layer updates, batch size, input size
ACTIONS = [1 for i in range(20)]  # actions in Moving MNIST (1 for next frame. see tools.py for spatial movement)
TRANSITION, DYNAMICAL = True, False  # first order transition model, higher order derivatives (generalized coordinates)
PRECISION = True  # use precision estimation
IMG_NOISE = 0.0  # gaussian noise on inputs
B_SIZE_PREC = 1 # either 1 or B_SIZE (batch mean / learning or per batch element / inference)
CONVERGENCE_TRESHOLD = .1 # inference stops at this error threshold

if __name__ == '__main__':
    for env_id, env_name in enumerate(['Mnist-Train-v0']):  #  'Mnist-Test-v0'
        # create Moving MNIST gym environment
        env = gym.make(env_name); env.reset()

        # create model and print summary
        ch, ch2, ch3 = 64, 64, 128  # CNN channels
        PCN = Model([1 * 16 * 16, ch*8*8, ch*8*8, ch2*4*4, ch2*4*4, ch3*2*2, ch3*2*2, ch3*1*1, ch3*1*1, 64, 64, 4],  # state sizes
            ELU(), ELU(),  # hierarchical & dynamical activation
            lr_w=np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]) * 0.0, # weights lr
            lr_sl=np.asarray([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) * 0,  # lower state lr
            lr_sh=np.asarray([1, 1, 1, 1, 1, 1, 1, 1, .1, .1, .1, .1]) * .1,  # higher state lr
            lr_g=np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) * 1,  # state gain lr
            dim=[1, ch, ch, ch2, ch2, ch3, ch3, ch3, 1, 1, 1, 1],  # state channels (use 1 for dense layers)
            sr=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # sampling interval (skipped observations in lower layer)
        model_sizes_summary(PCN)

        for a_id, action in enumerate(ACTIONS):  # passive perception task (model has no control)

            # select much history to log and visualize
            if a_id % 20 == 0 and a_id < len(ACTIONS):
                [err_h, err_t, preds_h, preds_t, preds_g, preds_gt], inputs = [[[] for _ in PCN.layers] for _ in range(6)], [[]]
                precisions, raw_errors, variances_slow, datapoints, total_updates = [], [], None, [], 0

            # get observation and preprocess
            for i in range(int(PCN.sr[0])):  # skip observations according to data layer's sample rate
                obs, rew, done, _ = env.step([action for b in range(B_SIZE)])  # step environment
            input = (torch.Tensor(obs['agent_image'])).reshape([B_SIZE, -1, 64 ** 2]) / 25
            input = torch.nn.MaxPool2d(4, stride=4)(input.reshape([B_SIZE, -1, 64, 64]))  # reduce input size
            PCN.curr_cause[0] = torch.tensor(input.detach().float()).reshape([B_SIZE, 1, -1])  # feed to model
            PCN.c_curr_gain[0] = torch.ones_like(PCN.c_curr_gain[0])  # feed to model

            converged = False
            for update in range(UPDATES):
                total_updates += 1

                # update hierarchical layers
                for l_h in reversed(range(len(PCN.layers))):

                    # freeze weights when testing
                    if env_id == 1: PCN.freeze([3])

                    # step hierarchical variables
                    for i in range(1): # how often to update layers relative to each other
                        p_h, e_h = GPC(PCN, l=l_h)
                        GPC(PCN, l=l_h, infer_precision=True);  # update precision (precision inference)
                        if l_h == 0:
                            converged = e_h[0].mean(dim=0).mean().abs() < CONVERGENCE_TRESHOLD
                            precisions.append(np.diag(np.array(PCN.covar[0].mean(dim=0).detach())).mean())
                            raw_errors.append(e_h[0].mean(dim=0).detach().numpy().mean())

                    # update dynamical layers
                    for l_d in range(1, len(PCN.layers_d[l_h].layers), 1):
                        if DYNAMICAL:
                            p_d, e_t = GPC(PCN.layers_d[l_h], l=l_d, dynamical=True)  # step higher order dynamical variables
                            if update == UPDATES - 1: PCN.layers_d[l_h].last_cause[l_d] = PCN.layers_d[l_h].curr_cause[
                                l_d].clone().detach()  # memorize

                    if (update == UPDATES - 1 and l_h == 0) or (converged and l_h == 0):
                        # collect results
                        for d, i in zip([inputs[0], preds_h[l_h], err_h[l_h]], [input[:1], p_h[:1], e_h[:1].detach()]): d.append(i)  # hierarchical
                        if DYNAMICAL: preds_t[l_d].append(p_d[:1]), err_t[l_d].append(e_t[:1].detach())  # dynamical
                        preds_g[l_h].append(predict(PCN, l=len(PCN.layers)-1, transition=False))  # predict from deepest cause
                        preds_gt[l_h].append(predict(PCN, l=len(PCN.layers)-2, transition=True))  # transition and predict from deepest cause

                        # visualize batch
                        if a_id == len(ACTIONS) - 1:
                            for d, n in zip([input, p_h, preds_g[0][-1], preds_gt[0][-1]], ["input", "pred_h", "pred_g", "pred_gt"]):
                                plot_batch(d, title=str(env_name) + n)

                    # memorize states and precision
                    if update == UPDATES - 1 or converged:
                        if l_h == 0: datapoints.append(total_updates)
                        PCN.last_cause[l_h] = PCN.curr_cause[l_h].clone().detach()  # memorize hierarchical state
                        if variances_slow is not None:
                            PCN.covar = variances_slow
                            PCN.covar_hidden = variances_slow_hidden
                        _, raw_error = GPC(PCN, l=l_h, infer_precision=True);  # precision per datapoint
                        variances_slow = [c.clone().detach() for c in PCN.covar]  # variance per datapoint (learning)
                        variances_slow_hidden = [c.clone().detach() for c in PCN.covar_hidden]  # variance per datapoints (learning)
                if converged: break
        env.close()

        """ Print summaries """
        model_sizes_summary(PCN);
        print_layer_variances(PCN,0);

        """ Visualize """
        generate_videos(preds_h, inputs, preds_g, preds_gt, err_h, env_name, nr_videos=4, scale=25);
        plot_thumbnails(precisions, errors=raw_errors, inputs=inputs, datapoints=datapoints, threshold=0.2, img_s=2);
        visualize_covariance_matrix(PCN, title=env_name, skip_l=1);

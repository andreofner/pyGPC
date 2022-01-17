"""
Differentiable Generalized Predictive Coding
André Ofner 2021
"""
import torch
from tools import *
from torch.optim import SGD
from torch.nn import Sequential, Linear, ELU, Identity, ConvTranspose2d

class Model(torch.nn.Module):
    """ Hierarchical dynamical predictive coding model """

    def __init__(self, sizes, act, act_d=None, dynamical=False, var=1, covar=1000000, dim=[], sr=[], lr_w=None,
                 lr_sl=None, lr_sh=None, lr_g=None):
        super(Model, self).__init__()

        # model variables
        self.layers = []  # weights
        self.layers_d = []  # dynamical weights
        self.lr = []  # learning rate per layer
        self.dynamical = dynamical  # model type
        self.sizes = sizes  # state sizes
        self.dim = np.asarray(dim)  # state channels
        self.sr = sr  # sampling rate per layer

        # optionally use CNN layers
        self.n_cnn = ((self.dim > 1).sum()) // 2 + 1  # number of CNN layers, if present
        self.state_sizes = np.sqrt(self.sizes / self.dim).astype(int)[:self.n_cnn * 2]  # cnn state sizes, if present

        # default learning rates
        if lr_w is None: lr_w = [0. for _ in self.sizes]  # weights
        if lr_sh is None: lr_sh = [1. for _ in self.sizes]  # state
        if lr_sl is None: lr_sl = [1. for _ in self.sizes]  # higher state
        if lr_g is None: lr_g = [.1 for _ in self.sizes]  # state gain (multiplicative bias)

        # create layers
        for i in range(0, len(sizes), 2):
            if not dynamical:
                net = ConvTranspose2d(dim[i + 1], dim[i], 2, stride=2, bias=False) if dim[i + 1] > 1 else Linear(
                    sizes[i + 1], sizes[i], False)
                self.layers.append(Sequential(act, net, torch.nn.Dropout(0.0)))  # CNN or dense hierarchical weights
                self.layers_d.append(Model(sizes[i:], act_d, dynamical=True, dim=dim[i:], sr=[sr[i] for _ in sizes[i:]]))  # dynamical weights
            else:
                self.layers.append(Sequential(act, Linear((sizes[i + 1]), (sizes[i]), False)))  # dense dynamical layers
            self.lr.append([lr_sh[i], lr_sl[i], 0, lr_w[i], .1, lr_g[i]])  # learning rate per layer

        # initialise states, state gain and precision (inverse variance)
        [self.currState, self.lastState] = [self.init_states(mixed=(self.n_cnn > 0)) for _ in range(2)]
        [self.curr_gain, self.last_gain] = [self.init_states(mixed=(self.n_cnn > 0)) for _ in range(2)]
        self.covar = [torch.zeros(1) for _ in self.layers + [None]]  # precision

    def init_states(self, mixed=False):
        """ Create state priors for hierarchical and dynamical layers"""
        if self.dynamical:
            return self.predict(torch.ones([B_SIZE, 1, self.sizes[-1]]).float(), prior=True)
        elif mixed:  # CNN and dense layers
            states_cnn = self.predict(
                torch.ones([B_SIZE, self.dim[self.n_cnn * 2 - 1], self.state_sizes[-1], self.state_sizes[-1]]).float(),
                prior=True, layers=self.layers[:self.n_cnn])
            states_dense = self.predict(torch.ones([B_SIZE, 1, self.sizes[-1]]).float(), prior=True,layers=self.layers[self.n_cnn:])
            return states_cnn + states_dense[1:]
        elif self.n_cnn > 1:  # CNN layers
            return self.predict(torch.ones([B_SIZE, self.dim[-1], self.sizes[-1], self.sizes[-1]]).float(), prior=True)
        else:  # dense layers
            return self.predict(torch.ones([B_SIZE, 1, self.sizes[-1]]).float(), prior=True)

    def params(self, l, deepest_layer=False, dynamical=False):
        """ Parameters and learning rates per layer.
        Regularization through top-down predictability affects learning and inference."""
        l_h = l if deepest_layer else l + 1
        params = [{'params': [self.layers[l][1].weight.requires_grad_()], 'lr': self.lr[l][3]},  # top-down weights (l): accuracy
                  {'params': [self.currState[l + 1]], 'lr': self.lr[l][0]},  # higher state (l+1,t): accuracy
                  {'params': [self.currState[l]], 'lr': self.lr[l][1]},  # lower state (l,t-dt): regularization
                  {'params': [self.lastState[l]], 'lr': self.lr[l][2]},  # lower state (l,t): regularization
                  {'params': [self.curr_gain[l]], 'lr': self.lr[l][5]},  # state gain (l): accuracy
                  {'params': [self.curr_gain[l + 1]], 'lr': self.lr[l_h][5]}]  # state gain (l): accuracy
        return params[:-1] if dynamical else params  # todo dynamical layers without covar

    def params_covar(self, l):
        """ Covariance parameters and learning rates per layer """
        params = [{'params': [self.covar[l]], 'lr': self.lr[l][4]}]  # covariance (l) - learning rate
        return params  # todo dynamical layers without covar

    def predict(self, target=None, states=None, prior=False, layers=None, state_gains=None):
        """ Backward prediction through all layers. Optionally initialises prior states"""
        states = [target] if states is None else [states[-1]]
        if prior: [torch.nn.init.torch.nn.init.normal_(states[-1][b]) for b in range(B_SIZE)]
        if layers is None: layers = self.layers
        if state_gains is None and not prior: state_gains = self.curr_gain
        if prior: state_gains = [None for _ in self.layers]
        for w, g in zip(list(reversed(layers)), list(reversed(state_gains))):
            if not prior:
                states.append(w(states[-1] * g).detach())
            else:
                states.append(w(states[-1]).detach())
                [torch.nn.init.torch.nn.init.normal_(states[-1][b]) for b in range(B_SIZE)]
        return list(reversed(states))

    def predict_mixed(self, target=None):
        """ Backward pass through CNN, dense or mixed models """
        try:  # mixed model
            preds_d = self.predict(target, layers=self.layers[self.n_cnn:], state_gains=self.curr_gain[self.n_cnn:])
            pred_d = preds_d[0].reshape(
                [B_SIZE, self.dim[self.n_cnn * 2 - 1], self.state_sizes[-1], self.state_sizes[-1]])
            return self.predict(pred_d, layers=self.layers[:self.n_cnn], state_gains=self.curr_gain[:self.n_cnn + 1])
        except:  # DNN or CNN model
            return self.predict(target=target)

    def freeze(self, params):
        """ Stop optimisation of weights. State inference remains active. """
        for param in params:
            for lr in self.lr: lr[param] = 0.  # freeze hierarchical weights
            if not self.dynamical:
                for l_d in self.layers_d: l_d.freeze(params)  # freeze dynamical weights


def GPC(m, l, dynamical=False, infer_precision=False):
    """ Layer-wise Generalized Predictive Coding optimizer """

    if not infer_precision:
        opt = SGD(m.params(l, dynamical=dynamical, deepest_layer=(l == len(m.layers) - 1)))
    else:
        opt = SGD(m.params_covar(l))
    opt.zero_grad()  # reset gradients

    # prediction from higher layer prediction_l = weights_l( state_(l+1) * state_gain_(l+1) )
    pred = m.layers[l].forward(m.currState[l + 1].requires_grad_() * m.curr_gain[l + 1].requires_grad_())

    # compute layer's prediction error
    if dynamical:  # state change error
        error = (m.currState[l].detach() - m.lastState[l].requires_grad_()).flatten(1) - pred.flatten(1)
    else:  # state & state change error
        if TRANSITION:
            pred_change = m.layers_d[l].layers[0].forward((m.layers_d[l].currState[1].requires_grad_()))
            pred = pred + pred_change.reshape(pred.shape)
        error = (m.currState[l].requires_grad_()) - pred.reshape(m.currState[l].shape)

    # move CNN channels to batch dimension so that each channel tracks precision independently
    error = error.reshape([B_SIZE, -1]).unsqueeze(-1)

    if m.covar[l].shape[-1] <= 1:
        var_prior, covar_prior = 7, 100000000  # (co-)variance prior (hyperparameter)
        m.covar[l] = torch.eye(error.shape[-2]).unsqueeze(0).repeat([BATCH_SIZE_PRECISION, 1, 1]) * (var_prior - covar_prior) + covar_prior

    # compute this layer's free energy
    F = torch.abs(error) * torch.abs(torch.matmul(m.covar[l].requires_grad_() ** -1, torch.abs(error)))
    F.backward(gradient=torch.ones_like(F))  # loss per batch element (not scalar)
    opt.step()
    if infer_precision:
        m.covar[l] = (m.covar[l] - m.lr[l][4] * m.covar[l]**-1).detach()  # variance decay
    return pred.detach().numpy(), error

UPDATES, SCALE, B_SIZE, IMAGE_SIZE = 50, 0, 1, 16*16  # model updates, relative layer updates, batch size, input size
ACTIONS = [1 for i in range(19)]  # actions in Moving MNIST (1 for next frame. see tools.py for spatial movement)
TRANSITION, DYNAMICAL = False, False  # first order transition model, higher order derivatives (generalized coordinates)
PRECISION = True  # use precision estimation
IMG_NOISE = 0.0  # gaussian noise on inputs
BATCH_SIZE_PRECISION = 1 # either 1 or B_SIZE (batch mean / learning or per batch element / inference)
CONVERGENCE_TRESHOLD = .01 # criterion to stop optimizing a datapoint

if __name__ == '__main__':
    for env_id, env_name in enumerate(['Mnist-Train-v0']):  #  'Mnist-Test-v0'

        # create Moving MNIST gym environment
        env = gym.make(env_name); env.reset()

        # create model
        ch, ch2, ch3 = 8, 8, 8  # CNN channels
        PCN = Model([1 * 16 * 16, ch*8*8, ch*8*8, ch2*4*4, ch2*4*4, ch3*2*2, ch3*2*2, ch3*1*1, ch3*1*1, 32, 32, 32],  # state sizes
            ELU(), Identity(),  # hierarchical & dynamical activation
            lr_w=np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]) * 0.0001,  # 1,  # weights lr
            lr_sl=np.asarray([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) * .1,  # lower state lr
            lr_sh=np.asarray([1, 1, 1, 1, 1, 1, 1, 1, .1, .1, .1, .1]) * .1,  # higher state lr
            lr_g=np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) * 1,  # state gain lr
            dim=[1, ch, ch, ch2, ch2, ch3, ch3, ch3, 1, 1, 1, 1],  # state channels (use 1 for dense layers)
            sr=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # sampling interval (skipped observations in lower layer)

        # visualization
        [err_h, err_t, preds_h, preds_t, preds_g], inputs = [[[] for _ in PCN.layers] for _ in range(5)], [[]]
        precisions, raw_errors, variances_slow = [], [], None

        for a_id, action in enumerate(ACTIONS):  # passive perception task (model has no control)
            converged = False

            # get observation and preprocess
            for i in range(int(PCN.sr[0])):  # skip observations according to data layer's sample rate
                obs, rew, done, _ = env.step([action for b in range(B_SIZE)])  # step environment
            input = (torch.Tensor(obs['agent_image'])).reshape([B_SIZE, -1, 64 ** 2]) / 25
            input = torch.nn.MaxPool2d(4, stride=4)(input.reshape([B_SIZE, -1, 64, 64]))  # reduce input size
            PCN.currState[0] = torch.tensor(input.detach().float()).reshape([B_SIZE, 1, -1])  # feed to model

            for update in range(UPDATES):

                # update hierarchical layers
                layers_h = reversed(range(len(PCN.layers)))

                for l_h in layers_h:
                    # freeze weights when testing
                    if env_id == 1: PCN.freeze([3])

                    # step hierarchical variables
                    for i in range(l_h * SCALE + 1): # how often to update layers relative to each other
                        p_h, e_h = GPC(PCN, l=l_h)
                        GPC(PCN, l=l_h, infer_precision=True);  # update precision (precision inference)
                        if l_h == 0:
                            converged = e_h[0].mean(dim=0).mean().abs() < CONVERGENCE_TRESHOLD  # todo convergence criterium
                            precisions.append(np.diag(np.array(PCN.covar[0].mean(dim=0).detach())).mean())
                            raw_errors.append(e_h[0].mean(dim=0).detach().numpy().mean())

                    # update dynamical layers
                    for l_d in range(1, len(PCN.layers_d[l_h].layers), 1):
                        if DYNAMICAL:
                            p_d, e_t = GPC(PCN.layers_d[l_h], l=l_d,
                                           dynamical=True)  # step higher order dynamical variables
                            if update == UPDATES - 1: PCN.layers_d[l_h].lastState[l_d] = PCN.layers_d[l_h].currState[
                                l_d].clone().detach()  # memorize

                        # visualization
                        if (update == UPDATES - 1 and l_h == 0 and l_d == 1) or (converged and l_h == 0 and l_d == 1):
                            for d, i in zip([inputs[0], preds_h[l_h], err_h[l_h]], [input[:1], p_h[:1], e_h[:1].detach()]): d.append(i)  # hierarchical
                            if DYNAMICAL: preds_t[l_d].append(p_d[:1]), err_t[l_d].append(e_t[:1].detach())  # dynamical
                            p_g = PCN.predict_mixed(target=PCN.currState[-1])[0]
                            preds_g[l_h].append(p_g[0])  # prediction from target state
                            if a_id == len(ACTIONS) - 1 :  # visualize batch
                                for d, n in zip([input, p_h, p_g], ["input", "pred_h", "pred_g"]):
                                    plot_batch(d, title=str(env_name) + n)

                    # memorize last state mean and precision
                    if update == UPDATES - 1 or converged:
                        PCN.lastState[l_h] = PCN.currState[l_h].clone().detach()
                        if variances_slow is not None: PCN.covar = variances_slow
                        _, raw_error = GPC(PCN, l=l_h, infer_precision=True); # update precision once per datapoint
                        variances_slow = [c.clone().detach() for c in PCN.covar] # track variance over datapoints (precision learning)
                if converged: break
        env.close()

        """ Visualization """
        generate_videos(preds_h, inputs, preds_g, err_h, env_name, nr_videos=3, scale=25);
        model_sizes_summary(PCN);
        visualize_covariance_matrix(PCN, title=env_name);
        plot_variance_updates(precisions, errors=raw_errors);
        print_layer_variances(PCN,0, title="Posterior");

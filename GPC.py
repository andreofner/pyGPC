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
    def __init__(self, sizes, act, act_d=None, dynamical=False, var=.1, covar=100000, dim=64, lr_w=0, sr=[]):
        super(Model, self).__init__()
        self.layers, self.layers_d, self.layers_p, self.covar, self.lr = [], [], [], [], []  # weights, dynamical weights, precision weights, covariance
        self.dynamical, self.sizes, self.dim = dynamical, sizes, dim  # layer type, state sizes, state channels
        self.covar = [torch.zeros(1) for _ in sizes]
        self.sr = sr # sampling rate of each layer
        for i in range(0, len(sizes), 2):  # create hierarchical layers
            if not dynamical:
                out_dim = 1 if i == 0 else dim  # single channel in output layer
                self.layers.append(Sequential(ConvTranspose2d(dim, out_dim, 2, stride=2, bias=False), act))
                self.layers_d.append(Model(sizes[i:], act_d, dynamical=True, dim=out_dim, sr=[sr[i] for _ in sizes[i:]])) # dynamical weights
            else:
                self.layers.append( Sequential(act, Linear((sizes[i + 1] ** 2) * self.dim, (sizes[i] ** 2) * self.dim, False)))
            self.lr.append([1, 1, 0, lr_w, 0.1])  # LR hidden layer
            if i == 0: self.lr[-1] = [1, 0, 0, lr_w, 0.1]  # LR output layer
        self.initialise_states(dynamical=dynamical)
        if not dynamical:
            self.covar = [torch.eye(cs.shape[-1] ** 2).repeat([B_SIZE * cs.shape[1], 1, 1]) * (var - covar) + covar for cs in self.currState]

    def initialise_states(self, dynamical=False):
        """ Create state priors for hierarchical and dynamical layers"""
        if dynamical:
            self.lastState = self.predict(torch.ones([B_SIZE, 1, self.sizes[-1] ** 2 * self.dim]).float(), prior=True)  # prior states at time t-dt_l
            self.currState = self.predict(torch.ones([B_SIZE, 1, self.sizes[-1] ** 2 * self.dim]).float(), prior=True)  # prior states at time t
        else:
            self.lastState = self.predict(torch.ones([B_SIZE, self.dim, self.sizes[-1], self.sizes[-1]]).float(), prior=True)  # prior states at time t-dt_l
            self.currState = self.predict(torch.ones([B_SIZE, self.dim, self.sizes[-1], self.sizes[-1]]).float(), prior=True)  # prior states at time t

    def parameters(self, l, dynamical=False):
        """ Parameters and learning rates per layer """
        params = [{'params': list(self.layers[l].parameters()), 'lr': self.lr[l][3]}, # top-down weights (l) --> prediction accuracy
                  {'params': [self.currState[l + 1].requires_grad_()], 'lr': self.lr[l][0]}, # higher state (l+1, t) --> prediction accuracy
                  {'params': [self.currState[l].requires_grad_()], 'lr': self.lr[l][1]}, # lower state (l, t-dt) --> regularization (top-down predictability)
                  {'params': [self.lastState[l].requires_grad_()], 'lr': self.lr[l][2]}, # lower state (l+1, t) --> regularization (top-down predictability)
                  {'params': [self.covar[l].requires_grad_()], 'lr': self.lr[l][4]}]  # covariance (l) --> learning rate
        return params[:-1] if dynamical else params

    def predict(self, target=None, states=None, prior=False, layers=None):
        """ Backward prediction through all layers. Optionally initialises prior states"""
        states = [target] if states is None else [states[-1]]
        if prior: [torch.nn.init.xavier_uniform_(states[-1][b], gain=1) for b in range(B_SIZE)]
        if layers is None: layers = self.layers
        for w in list(reversed(layers)):
            states.append(w(states[-1]).detach())
            if prior: [torch.nn.init.xavier_uniform_(states[-1][b], gain=1) for b in range(B_SIZE)]
        return list(reversed(states))

    def freeze(self, params):
        """ Disable optimisation of weights. State inference remains active. """
        for param in params:
            for lr in self.lr: lr[param] = 0.  # freeze hierarchical weights
            if not self.dynamical:
                for l_d in self.layers_d: l_d.freeze(params)  # freeze dynamical weights


def GPC(m, l, dynamical=False):
    """ Layer-wise Generalized Predictive Coding optimizer"""

    if dynamical: # assign sampling rates to dynamical states (= hidden states, i.e. not seen by hierarchical predictions)
        m.currState[l+1] = torch.cat([m.currState[l+1][:,:,:-1].detach(), torch.tensor([m.sr[l]]).repeat([B_SIZE,1,1])], dim=-1)
    else: # all dynamical states of a hierarchical layer contain the same known sampling rate
        for l_d in range(len(m.layers_d[l].currState)):
            m.layers_d[l].currState[l_d] = torch.cat([m.layers_d[l].currState[l_d][:,:,:-1].detach(), torch.tensor([m.sr[l]]).repeat([B_SIZE,1,1])], dim=-1)


    opt = SGD(m.parameters(l, dynamical)) # create this layer's SGD optimizer
    opt.zero_grad()  # reset gradients
    pred = m.layers[l].forward(m.currState[l+1].requires_grad_())  # prediction from higher layer
    if dynamical:  # predict state change
        error = (m.currState[l].detach() - m.lastState[l].requires_grad_()).flatten(1) - pred.flatten(1)
    else:  # predict state + state change
        if TRANSITION: pred = pred + m.layers_d[l].layers[0].forward(m.layers_d[l].currState[1].requires_grad_()).reshape(pred.shape)  # predicted state + predicted state transition
        error = m.currState[l].requires_grad_() - pred.reshape(m.currState[l].shape) # hierarchical-dynamical error
    error = error.reshape([B_SIZE * error.shape[1], -1]).unsqueeze(-1)
    F = torch.mean(torch.abs(error) * torch.abs(torch.matmul(m.covar[l] ** -1, error)), dim=[1, 2])
    F.backward(gradient=torch.ones_like(F))  # loss per batch element (not scalar)
    opt.step() # update all variables of this layer in parallel
    return pred.detach().numpy(), error


UPDATES, B_SIZE, IMAGE_SIZE = 200, 4, 16 * 16  # model updates, batch size, input size
ACTIONS = [1 for i in range(1)]  # actions in Moving MNIST
TRANSITION = True # first order transition model
DYNAMICAL = True  # higher order transition derivatives (generalized coordinates)
IMG_NOISE = 0.5  # gaussian noise on inputs todo scaling

if __name__ == '__main__':

    for weights_lr in [0]:  # pure inference, learning and inference
        for env_id, env_name in enumerate(['Mnist-Train-v0', 'Mnist-Test-v0']): # train set, test set
            env = gym.make(env_name)  # Moving MNIST gym environment
            env.reset()
            PCN = Model([16, 8, 8, 4], Tanh(), Tanh(), lr_w=weights_lr, sr=[2.,2.,2.])  # create model
            [err_h, err_t, preds_h, preds_t, preds_g], inputs = [[[] for _ in PCN.layers] for _ in range(5)], [[]]  # visualization

            # train model
            for i, action in enumerate(ACTIONS):
                for i in range(int(PCN.sr[0])): # sample data observations
                    obs, rew, done, _ = env.step([action for b in range(B_SIZE)])  # step environment
                input = ((torch.Tensor(obs['agent_image'])).reshape([B_SIZE, -1, 64 ** 2]) / 255 + 0.1) * 0.8  # get observation
                input = torch.nn.MaxPool2d(2, stride=4)(input.reshape([B_SIZE, -1, 64, 64])).reshape([B_SIZE, -1, IMAGE_SIZE])  # optionally reduce input size
                PCN.initialise_states()  # create prior states
                PCN.currState[0] = torch.tensor(input.detach().float()).reshape([B_SIZE, 1, -1])  # feed data

                for update in range(UPDATES):  # update model
                    for l_h in reversed(range(len(PCN.layers))):  # update each hierarchical layer
                        if env_id == 1: PCN.freeze([3])  # freeze weights for testing
                        p_h, e_h = GPC(PCN, l=l_h)  # step hierarchical variables
                        if update == UPDATES - 1: PCN.lastState[l_h] = PCN.currState[l_h].clone().detach()  # memorize last state

                        for l_d in range(1, len(PCN.layers_d[l_h].layers), 1):  # update dynamical layers
                            if DYNAMICAL:
                                p_d, e_t = GPC(PCN.layers_d[l_h], l=l_d, dynamical=True)  # step higher order dynamical variables
                                if update == UPDATES - 1: PCN.layers_d[l_h].lastState[l_d] = \
                                PCN.layers_d[l_h].currState[l_d].clone().detach()  # memorize

                            if update == UPDATES - 1 and l_h == 0 and l_d == 1:  # visualization
                                for d, i in zip([inputs[0], preds_h[l_h], err_h[l_h]], [input[:1], p_h[:1], e_h[:1].detach()]): d.append(i)  # hierarchical
                                if DYNAMICAL: preds_t[l_d].append(p_d[:1]), err_t[l_d].append(e_t[:1].detach())  # dynamical
                                preds_g[l_h].append(PCN.predict(states=PCN.currState, layers=PCN.layers)[0][0])  # prediction from target state

            for s, t in zip([preds_h, inputs, preds_g, err_h][:3], ['p_h', 'p_g', 'ins', 'e_h'][:2]):  # generate videos
                sequence_video(s, t, scale=255, env_name=str(env_name) + str(weights_lr))
            env.close()

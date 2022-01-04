"""
Differentiable Generalized Predictive Coding
André Ofner 2021
"""

import torch
from tools import *
from torch.optim import SGD
from torch.nn import Linear, Sequential, Tanh

class Model(torch.nn.Module):
    """ Hierarchical dynamical predictive coding model """
    def __init__(self, sizes, act, act_d, dynamical=False, var=1, covar=10, hidden_size=512):
        super(Model, self).__init__()
        self.layers, self.layers_d, self.layers_p, self.covar, self.lr = [],[],[],[],[] # weights, dynamical weights, precision weights, covariance
        for i in range(0, len(sizes), 2): # create hierarchical layers
            self.layers.append(Sequential(Linear(sizes[i+1], hidden_size, False), act, # hidden weights
                Linear(hidden_size, sizes[i], False), act)) # output weights
            self.lr.append([1, 1, 0, .1, 1]) # LR hidden layer
            if i == 0: self.lr[-1] = [1, 0, 0, .1, 1] # LR output layer
            if not dynamical: # each hierarchical layer l gets (hierarchical layers - l) dynamical layers
                self.layers_d.append(Model(sizes[i:], act_d, None, dynamical=True)) # dynamical weights
            self.covar.append((torch.eye(sizes[i])*(var-covar)+covar).repeat([B_SIZE, 1, 1])) # prior covariance
        self.lastState = self.predict(torch.ones([B_SIZE, 1, sizes[-1]]).float(), prior=True) # prior states at time t-dt_l
        self.currState = self.predict(torch.ones([B_SIZE, 1, sizes[-1]]).float(), prior=True) # prior states at time t

    def parameters(self, l):
        """ Parameters and learning rates per layer """
        return [{'params': [self.currState[l+1].requires_grad_()], 'lr': self.lr[l][0]},  # higher state (l+1, t) --> prediction accuracy
                {'params': [self.lastState[l].requires_grad_()], 'lr': self.lr[l][1]},  # lower state (l, t-dt) --> regularization (top-down predictability)
                {'params': [self.currState[l].requires_grad_()], 'lr': self.lr[l][2]},  # lower state (l+1, t) --> regularization (top-down predictability)
                {'params': list(self.layers[l].parameters()), 'lr': self.lr[l][3]},  # top-down weights (l) --> prediction accuracy
                {'params': [self.covar[l].requires_grad_()], 'lr': self.lr[l][4]}]  # covariance (l) --> learning rate

    def predict(self, target=None, states=None, prior=False, layers=None):
        """ Backward prediction through all layers. Optionally initialises prior states"""
        states = [target] if states is None else [states[-1]]
        if prior: [torch.nn.init.xavier_uniform_(states[-1][b]) for b in range(B_SIZE)]
        if layers is None: layers = self.layers
        for w in list(reversed(layers)):
            states.append(w(states[-1]).detach())
            if prior: [torch.nn.init.xavier_uniform_(states[-1][b]) for b in range(B_SIZE)]
        return list(reversed(states))

def GPC(m, l, dynamical=False):
    """ Layer-wise Generalized Predictive Coding optimizer"""
    opt = SGD(m.parameters(l)) # parameter optimizer
    opt.zero_grad() # reset gradients
    pred = m.layers[l].forward(m.currState[l+1].requires_grad_()) # prediction from higher layer
    if dynamical: # predict state change
        error = (m.currState[l].requires_grad_()-m.lastState[l].requires_grad_()) - pred
    else:  # predict state + state change
        pred = pred + m.layers_d[l].layers[0].forward(m.layers_d[l].currState[1].requires_grad_())
        error = m.currState[l].requires_grad_() - pred  # hierarchical-dynamical error
    F = torch.mean(torch.abs(error)*torch.abs(torch.matmul(error, m.covar[l]**-1)), dim=[1,2]) # error * weighted error
    F.backward(gradient=torch.ones_like(F)) # loss per batch element (not scalar)
    opt.step()
    return pred.detach().numpy(), error

UPDATES, B_SIZE, IMAGE_SIZE = 10, 8, 16*16 # model updates, batch size, input size
ACTIONS = [1 for i in range(100)]  # actions in Moving MNIST
DYNAMICAL = True # higher order derivatives (generalized coordinates)
IMG_NOISE = 0.8 # gaussian noise on inputs todo scaling

if __name__ == '__main__':

    # create dataset
    env = gym.make('Mnist-s1-v0') # Moving MNIST gym environment (passive perception)
    env.reset()

    # create model
    l_sizes = [IMAGE_SIZE,64, 64,16, 16,4]  # hierarchical layer sizes
    PCN = Model(l_sizes, Tanh(), Tanh()) # create model
    [err_h, err_t, preds_h, preds_t, preds_g], inputs = [[[] for _ in PCN.layers] for _ in range(5)], [[]] # visualization

    # train model
    for i, action in enumerate(ACTIONS):

        # preprocess new input
        for i in range(1): obs, rew, done, _ = env.step([action for b in range(B_SIZE)])
        input = (torch.Tensor(obs['agent_image'])).reshape([B_SIZE, -1, 64**2]) / 255 # get observation
        input = torch.nn.MaxPool2d(2, stride=4)(input.reshape([B_SIZE, -1, 64, 64])).reshape([B_SIZE, -1, IMAGE_SIZE]) # optionally reduce input size
        PCN.currState[0] = torch.tensor(input.detach().float()).reshape([B_SIZE, 1, -1])  # feed data

        # update model
        for update in range(UPDATES):

            # update each hierarchical layer
            for l_h in range(len(PCN.layers)):
                p_h, e_h = GPC(PCN, l=l_h) # step variables
                if update == UPDATES-1: PCN.lastState[l_h] = PCN.currState[l_h].clone().detach() # memorize last state
                for l_d in range(1, len(PCN.layers_d[l_h].layers), 1): # update dynamical layers

                    # predict higher order derivatives of the transition function in each hierarchical layer
                    if DYNAMICAL:
                        p_d, e_t = GPC(PCN.layers_d[l_h], l=l_d, dynamical=True) # step variables
                        if update == UPDATES-1: PCN.layers_d[l_h].lastState[l_d] = PCN.layers_d[l_h].currState[l_d].clone().detach() # memorize
                    
                    # visualization
                    if update == UPDATES-1 and l_h == 0 and l_d == 1: 
                        for d, i in zip([inputs[0],preds_h[l_h], err_h[l_h]], [input[:1], p_h[:1], e_h[:1].detach()/255]): d.append(i) # hierarchical
                        if DYNAMICAL: preds_t[l_d].append(p_d[:1]), err_t[l_d].append(e_t[:1].detach()) # dynamical
                        preds_g[l_h].append(PCN.predict(states=PCN.currState, layers=PCN.layers)[0][0])  # prediction from target state

    # generate videos
    for s, t in zip([preds_h, inputs, preds_g, err_h],['p_h','ins','p_g','e_h']): sequence_video(s, t, scale=255)

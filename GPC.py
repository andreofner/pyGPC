"""
Differentiable Generalized Predictive Coding
André Ofner 2021
"""

import torch
from tools import *
from torch.optim import SGD
from torch.nn import Linear, Sequential
from torch.nn.init import xavier_uniform_

class Model(torch.nn.Module):
    """ Predictive Coding model with hierarchical and dynamical layers """
    def __init__(self, sizes, bias, act, bias_d=False, act_d=[], dynamical=False, var=1, covar=1):
        super(Model, self).__init__()
        self.layers, self.layers_d, self.layers_p, self.covar = [],[],[],[] # weights, dynamical weights, precision weights, covariance
        self.lr = [10, 1, 0, 0.0, 0] # learning rate: state (l+1), state, state (past), weights, variance
        for i in range(0, len(sizes)-1, 2): # create hierarchical layers
            self.layers.append(Sequential(Linear(sizes[i + 1], sizes[i], bias), act[i//2]))
            if not dynamical: # multiple dynamical layers per hierarchical layer
                self.layers_d.append(Model(sizes[i:], bias_d, act_d[i//2:], dynamical=True)) # dynamical weights
            self.covar.append((torch.eye(sizes[i])*(var-covar)+covar).repeat([B_SIZE, 1, 1])) # prior covariance
        self.lastState = self.predict(torch.ones([B_SIZE, 1, sizes[-1]]).float(), prior=True) # prior states at time t-dt_l
        self.currState = self.predict(torch.ones([B_SIZE, 1, sizes[-1]]).float(), prior=True) # prior states at time t

    def parameters(self, l):
        """ Parameters and learning rates per layer """
        return [{'params': [self.currState[l+1].requires_grad_()], 'lr': self.lr[0]},  # higher state (l+1, t)
                {'params': [self.lastState[l].requires_grad_()], 'lr': self.lr[1]},  # lower state (l, t-dt)
                {'params': [self.currState[l].requires_grad_()], 'lr': self.lr[2]},  # lower state (l+1, t)
                {'params': list(self.layers[l].parameters()), 'lr': self.lr[3]},  # top-down weights (l)
                {'params': [self.covar[l].requires_grad_()], 'lr': self.lr[4]}]  # covariance (l)

    def predict(self, target=None, states=None, prior=False, prior_gain=1):
        """ Backward prediction pass. Optionally initialises prior states"""
        states = [target] if states is None else [states[-1]]
        if prior: xavier_uniform_(states[-1], gain=prior_gain)
        for w in list(reversed(self.layers)):
            states.append(w(states[-1]).detach())
            if prior: xavier_uniform_(states[-1], gain=prior_gain)
        return list(reversed(states))


def GPC(m, l, dynamical=False):
    """ Generalized Predictive Coding optimizer (layer-wise) """
    opt = SGD(m.parameters(l)) # parameter optimizer
    opt.zero_grad() # reset gradients
    pred = m.layers[l].forward(m.currState[l+1].requires_grad_()) # prediction from higher layer
    if dynamical: # predict state change
        error = (m.currState[l].requires_grad_()-m.lastState[l].requires_grad_()) - pred
    else:  # predict state + state change
        pred = pred + m.layers_d[l].layers[0].forward(m.layers_d[l].currState[1].requires_grad_())
        error = m.currState[l].requires_grad_() - pred
    error_weighted = torch.matmul(m.covar[l]**-1, torch.reshape(error, [B_SIZE, -1, 1])) # weighted error
    F = torch.mean(torch.abs(error)*torch.abs(error_weighted), dim=[1,2]) # total error
    F.backward(gradient=torch.ones_like(F)) # loss per batch element (i.e. vector, not scalar)
    opt.step()
    if l == 0 and not dynamical: print("Free Energy Layer 1: ", F.detach().numpy().mean())
    return pred.detach().numpy(), error_weighted

UPDATES = 10  # model updates per input
ACTIONS = [1 for i in range(100)]  # actions in Moving MNIST
B_SIZE = 1  # batch size
IMAGE_SIZE = 16*16  # image size after preprocessing
DYNAMICAL = False

if __name__ == '__main__':
    # load Moving MNIST gym environment (passive perception task)
    env = gym.make('Mnist-s1-v0')
    env.reset()

    # create model
    l_sizes = [IMAGE_SIZE, 1024,1024, 512,512, 256]  # hierarchical layer sizes
    act_h = [torch.nn.Sigmoid()] + [torch.nn.Sigmoid() for l in l_sizes[1::2]]  # hierarchical activations
    act_d = [torch.nn.Sigmoid()] + [torch.nn.Sigmoid() for l in l_sizes[1::2]]  # dynamical activations
    PCN = Model(sizes=l_sizes, bias=False, act=act_h, bias_d=False, act_d=act_d)
    [err_h, err_t, preds_h, preds_t], inputs = [[[] for _ in PCN.layers] for _ in range(4)], [[]] # visualization

    # iterate over time steps
    for i, action in enumerate(ACTIONS):
        PCN.currState = PCN.predict(target=PCN.currState[-1], prior=True) # initialise states
        PCN.lastState = PCN.predict(target=PCN.currState[-1], prior=True) # initialise states
        for i in range(1): obs, rew, done, _ = env.step([action for b in range(B_SIZE)])
        input = (torch.Tensor(obs['agent_image'])).reshape([B_SIZE, -1, 64**2]) # get observation
        if True:  # optionally reduce input size
            input = input.reshape([B_SIZE, -1, 64, 64]) / 255
            input = torch.nn.MaxPool2d((2, 2), stride=(4, 4))(input).reshape([B_SIZE, -1, IMAGE_SIZE])
        for update in range(UPDATES):
            if update == UPDATES-1: PCN.lr[-2] = 0.000001 # todo remove fixed prediction assumption for weights
            for l_h in range(len(PCN.layers) - 1): # update hierarchical layers
                PCN.currState[0] = torch.tensor(input.detach().float()).reshape([B_SIZE,1,-1])  # feed data
                p_h, e_h = GPC(PCN, l=l_h) # step variables
                PCN.lastState[l_h] = PCN.currState[l_h].clone().detach() # memorize last state
                for l_d in range(len(PCN.layers_d[l_h].layers) - 1): # update dynamical layers
                    if DYNAMICAL:
                        PCN.currState[0] = torch.tensor(input.detach().float().squeeze()).unsqueeze(1)  # feed data
                        p_d, e_t = GPC(PCN.layers_d[l_h], l=l_d, dynamical=True) # step variables
                        PCN.layers_d[l_h].lastState[l_d] = PCN.layers_d[l_h].currState[l_d].clone().detach() # memorize
                    if update == UPDATES - 1 and l_h == 0 and l_d == 0: # visualization
                        inputs[0].append(input[:1])
                        preds_h[l_h].append(p_h[:1]), err_h[l_h].append(e_h[:1].detach())
                        if DYNAMICAL: [l_d].append(p_h[:1] + p_d[:1]), err_t[l_d].append(e_t[:1].detach())

    """ Plotting """
    for s, t in zip([preds_h, inputs, preds_t, err_t, err_h][:2], ['p_h', 'ins', 'p_d', 'e_d', 'e_h'][:2]):
        sequence_video(s, t, scale=255)




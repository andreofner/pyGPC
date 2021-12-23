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
      """ PCN model with hierarchical and dynamical weights """
      def __init__(self, sizes, bias, act, bias_d=True, act_d=[], dynamical=False, var=1., covar=.1, prior=1):
            super(Model, self).__init__()
            self.layers = [] # hierarchical weights
            self.prec = [] # state precision
            self.layers_d = [] # dynamical weights
            for i in range(0, len(sizes)-1, 2): # hierarchical layers
                  self.layers.append(Sequential(Linear(sizes[i+1], sizes[i], bias), act[int(i/2)]))
                  if not dynamical: # each hierarchical layer gets a dynamical model
                        self.layers_d.append(Model(sizes[i:], bias_d, act_d, dynamical=True))
                  self.prec.append((torch.eye(sizes[i])*(var-covar)+covar).repeat([B_SIZE,1,1])) # prior precision
            self.target = torch.ones([B_SIZE,1,sizes[-1]])*prior # prior target state
            self.lastState = self.predict(target=self.target.float(), states=None, prior=True) # prior states at time t
            self.currState = self.predict(target=self.target.float(), states=None, prior=True) # prior states at time t+dt

      def w_h(self, l): return self.layers[l] # get layer

      def parameters(self, l):
            """ Parameters and learning rates to update a layer """
            return [{'params': [self.lastState[l].requires_grad_()], 'lr': 0.01}, # state l, t
            {'params': [self.currState[l].requires_grad_()], 'lr': 0.01}, # state l+1, t+dt
            {'params': [self.currState[l+1].requires_grad_()], 'lr': 0.1}, # state l+1, t
            {'params': list(self.w_h(l=l).parameters()), 'lr': 0.01}, # top-down weights l
            {'params': [self.prec[l].requires_grad_()], 'lr': 0.1}]  # precision ls

      def predict(self, target=None, states=None, prior=False):
            """ Backward pass through layers """
            states = [target] if states is None else [states[-1]]
            if not prior: # normal backward prediction pass
                  for w in list(reversed(self.layers)): states.append(w(states[-1]).detach())
            else: # generate random prior states in each layer
                  for w in list(reversed(self.layers)):
                        states.append(xavier_uniform_(torch.ones_like(w(states[-1])), gain=1000).detach())
            return list(reversed(states))


def GPC(m, l, loss=torch.abs, dynamical=False):
      """ Generalized Predictive Coding optimizer """

      m_d = m.layers_d[l] # dynamical layers of this hierarchical layer
      opt = SGD(m.parameters(l)+m_d.parameters(l), lr=0., momentum=0.) # optimizer
      opt.zero_grad()  # reset gradients

      if dynamical: # dynamical prediction
            pred = m_d.w_h(l=l).forward(m_d.currState[l+1].requires_grad_())  # dynamical top-down prediction
            if l==0: m_d.currState[l] = m.currState[l].clone().detach().requires_grad_() # lowest dynamical layer predicts the cause
            error = loss((m_d.currState[l].requires_grad_()-m_d.lastState[l].requires_grad_()) - pred)  # dynamical top-down PE (predicted change/grad)
            if LEARN_PRECISION: torch.matmul(m_d.prec[l].requires_grad_()**-1, torch.reshape(error, [B_SIZE, -1, 1]))
      else: # hierarchical prediction
            pred = m.w_h(l=l).forward((m.currState[l+1].requires_grad_())) # hierarchical prediction (from cause state)
            error = loss((m.currState[l].requires_grad_() - pred)) # hierarchical PE
            if LEARN_PRECISION: torch.matmul(m.prec[l].requires_grad_()**-1, torch.reshape(error, [B_SIZE, -1, 1]))

      torch.mean(error).backward() # compute gradients
      opt.step() # step variables
      return pred.detach().numpy(), error

UPDATES = 100 # model updates per input
ACTIONS = [1 for i in range(20)] # actions in Moving MNIST
B_SIZE = 4 # batch of agents
IMAGE_SIZE = 32*32 # image size after preprocessing
LEARN_PRECISION = True

if __name__ == '__main__':
      # Moving MNIST in OpenAI gym
      env = gym.make('Mnist-s1-v0')
      env.reset()

      # hyperparameters
      l_sizes = [IMAGE_SIZE,256, 256,128, 128,64] # hierarchical layer sizes
      act_h = [torch.nn.Identity()] + [torch.nn.Identity() for l in l_sizes[1::2]] # hierarchical activations
      act_d = [torch.nn.Identity()] + [torch.nn.Identity() for l in l_sizes[1::2]] # dynamical activations

      # create model
      PCN = Model(sizes=l_sizes, bias=False, act=act_h, bias_d=False, act_d=act_d)

      # visualization
      [err_h, err_t, preds_h, preds_t], inputs = [[[] for _ in PCN.layers] for _ in range(4)], [[]]

      for i, action in enumerate(ACTIONS): # iterate over time steps
            obs, rew, done, _ = env.step([action for b in range(B_SIZE)])
            input = (torch.Tensor(obs['agent_image'])).reshape([B_SIZE, -1, IMAGE_SIZE])

            if True: # reduce size
                  input = input.reshape([B_SIZE, -1, 64, 64])
                  input = torch.nn.MaxPool2d((2,2), stride=(2,2))(input).reshape([B_SIZE, -1, IMAGE_SIZE])

            # update model
            for update in range(UPDATES):
                  currState = PCN.predict(states=PCN.currState) # hierarchical prediction
                  for layer_h in range(len(PCN.layers)-1): # update hierarchical layers
                        PCN.currState[0] = torch.tensor(input.detach().float().squeeze()).unsqueeze(1) # feed data
                        p_h, e_h = GPC(PCN, l=layer_h) # update hierarchical variables
                        for layer_d in range(len(PCN.layers_d[layer_h].layers)-1):
                              p_d, e_t = GPC(PCN, l=layer_d, dynamical=True) # update dynamical variables
                              if update == UPDATES-1 and layer_h == 0 and layer_d==0: # visualization
                                    inputs[0].append(input[:1]) # observed data
                                    preds_h[layer_h].append(p_h[:1]) # lowest hierarchical prediction
                                    err_h[layer_h].append(e_h[:1].detach()) # lowest hierarchical error
                                    preds_t[layer_d].append(p_h[:1]+p_d[:1]) # cause + change of cause
                                    err_t[layer_d].append(e_t[:1].detach()) # lowest dynamical error

            # memorize last state in all layers
            for i in range(len(PCN.layers)):
                  PCN.lastState[i] = PCN.currState[i].clone().detach()
                  for j in range(len(PCN.layers_d[i].lastState)):
                        PCN.layers_d[i].lastState[j] = PCN.layers_d[i].currState[j].clone().detach()

      """ Plotting """
      sequence_video(preds_t, title="trans_predictions", plt_title="Dynamical prediction", scale=1)
      sequence_video(preds_h, title="hier_predictions", plt_title="Hierarchical prediction", scale=1)
      sequence_video(err_t, title="trans_errors", plt_title="Dynamical prediction error", scale=1)
      sequence_video(err_h, title="hier_errors", plt_title="Hierarchical prediction error", scale=1)
      sequence_video(inputs, title="inputs", plt_title="Input", scale=1)

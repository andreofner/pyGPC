"""
Differentiable Generalized Predictive Coding
André Ofner 2021
"""

import gym
import torch
from tools import *
from torch.optim import SGD
import time, math, numpy as np
import matplotlib.pyplot as plt
from torch.nn import Linear, Sequential

class Model(torch.nn.Module):
      """ PCN with hierarchical and dynamical weights """
      def __init__(self, sizes, bias, act, sizes_d=[], bias_d=True, act_d=[]):
            super(Model, self).__init__()
            self.layers = [] # hierarchical layers
            self.layers_d = [] # dynamical layers for each hierarchical layer
            self.precisions = [] # hierarchical precision
            self.currState = [] # states at time t
            self.lastState = [] # states at time t+dt

            for i in range(0, len(sizes)-1, 2):
                  self.layers.append(Sequential(Linear(sizes[i + 1], sizes[i], bias), act[int(i/2)]))
                  if len(sizes_d) > 0: # todo higher dynamical layers
                        self.layers_d.append(Model([sizes[i],sizes[i]], bias_d, act_d))
                  self.precisions.append(torch.stack([(torch.eye(sizes[i]) * 9.9 + 0.1) for _ in range(BATCH_SIZE)]).requires_grad_())

      def w_h(self, l): return self.layers[l] # Hierarchical weights

      def w_d(self, l, l_d): return self.layers_d[l].layers[l_d] # Dynamical weights

      def w_t(self, l): return self.w_d(l, 0) # Transition (lowest dynamical) weights

      def parameters(self, l): # Relevant parameters to update a hierarchical layer
            return list(self.w_h(l=l).parameters()) + list(self.w_d(l=l, l_d=0).parameters())+\
                   [self.currState[l], self.precisions[l], self.lastState[l], self.lastState[l+1]]

      def predict(self, target=None, states=None):
            """ Backward pass through provided layers """
            states = [target] if states is None else [states[-1]]
            for w in list(reversed(self.layers)): states.append(w(states[-1]).detach())
            return list(reversed(states))

def GPC(m, l, loss=torch.abs, dynamical=False):
      """ Generalized Predictive Coding optimizer """

      # define learning rate for each variable
      opt_list = [SGD([m.lastState[l]], lr=0), # states l_{t}
      SGD([m.currState[l]], lr=10), # states l_{t+dt_{l+1}}
      SGD([m.lastState[l+1]], lr=0), # higher layer states l+1_{t}
      SGD(list(m.w_h(l=l).parameters()), lr=.001), # hierarchical weights l
      SGD(list(m.w_d(l=l, l_d=0).parameters()), lr=10), # dynamical weights l
      SGD([m.precisions[l]], lr=.1)] # precision l
      SGD([p.detach().requires_grad_() for p in m.parameters(l)], lr=0).zero_grad() # reset all gradients

      # dynamical prediction (state transition)
      pred_t = m.w_t(l=l).forward(m.lastState[l])  # predicted hidden state change
      e_t = loss(pred_t - m.currState[l])
      e_total = torch.mean(e_t)

      if dynamical: # higher layer is dynamical
            pred_h = m.w_d(l=l, l_d=0).forward((m.lastState[l+1])) # dynamical prediction
            e_h = loss((m.currState[l]-m.lastState[l]) - pred_h) # dynamical PE (state change or grad)
            e_total += torch.mean(torch.matmul(m.precisions[l]**-1, torch.reshape(e_h, [BATCH_SIZE, -1, 1]) ))
      else: # higher layer is hierarchical
            pred_h = m.w_h(l=l).forward((m.lastState[l+1])) # hierarchical prediction
            e_h = loss((m.lastState[l] - pred_h)) # hierarchical PE at t
            e_total += torch.mean(torch.matmul(m.precisions[l]**-1, torch.reshape(e_h, [BATCH_SIZE, -1, 1])))

      e_total.backward() # compute gradients
      for i, opt in enumerate(opt_list): opt.step() # step variables
      return [p.detach().numpy() for p in [pred_h, pred_t]], e_h, e_t

UPDATES = 1 # model updates per input
actions = [1 for i in range(10)] # actions in Moving MNIST
BATCH_SIZE = 8 # batch of agents
NOISE_SCALE = 0.0 # add gaussian noise to images
IMAGE_SIZE = 32*32 # image size after preprocessing

if __name__ == '__main__':

      # set up Moving MNIST in OpenAI gym
      env = gym.make('Mnist-s1-v0')
      env.reset()

      # output activations
      l_sizes, l_sizes_t = [IMAGE_SIZE,256, 256,256], []
      for i, s in enumerate(l_sizes[::2]): l_sizes_t += [s,s]

      # output activations
      act_h = [torch.nn.Sigmoid()] + [torch.nn.Sigmoid() for l in l_sizes[1::2]]
      act_d = [torch.nn.Sigmoid()] + [torch.nn.Sigmoid() for l in l_sizes[1::2]]

      # weights
      PCN = Model(sizes=l_sizes, bias=True, act=act_h, sizes_d=l_sizes_t, bias_d=True, act_d=act_d)

      # precision & state priors
      PCN.target = torch.tensor(torch.tensor([1 for i in range(l_sizes[-1])]).unsqueeze(0).repeat(BATCH_SIZE,1,1)*.1)
      PCN.currState = PCN.predict(target=PCN.target.float(), states=None) # prior for state t+dt
      PCN.lastState = PCN.currState # prior for state t

      # visualization
      [err_h, err_t, preds_h, preds_t], inputs = [[[] for _ in PCN.layers] for _ in range(4)], [[]]

      # iterate over time steps
      for i, action in enumerate(actions):
            obs, rew, done, _ = env.step([action for b in range(BATCH_SIZE)])
            input = (torch.Tensor(obs['agent_image'])/255).reshape([BATCH_SIZE, -1, IMAGE_SIZE])

            if True: # reduce size
                  input = input.reshape([BATCH_SIZE, -1, 64, 64])
                  input = torch.nn.MaxPool2d((2,2), stride=(2,2))(input).reshape([BATCH_SIZE, -1, IMAGE_SIZE])

            # update model
            for update in range(UPDATES):
                  currState = PCN.predict(states=PCN.currState) # hierarchical prediction
                  for i in range(len(PCN.layers)-1): # update individual layers
                        PCN.currState[0] = torch.tensor(input.detach().float().squeeze()).unsqueeze(1) # feed data
                        preds, e_h, e_t = GPC(PCN, i) # update layer variables

            # visualization
            inputs[0].append(input[:1]), preds_h[i].append(preds[0][:1]), preds_t[i].append(preds[1][:1])
            err_h[i].append(e_h[:1].detach()), err_t[i].append(e_t[:1].detach())

            # memorize last state
            for i in range(len(PCN.layers)): PCN.lastState[i] = PCN.currState[i]

      """ Plotting """
      sequence_video(preds_t, title="trans_predictions", plt_title="Dynamical prediction")
      sequence_video(preds_h, title="hier_predictions", plt_title="Hierarchical prediction")
      sequence_video(err_t, title="trans_errors", plt_title="Dynamical prediction error")
      sequence_video(err_h, title="hier_errors", plt_title="Hierarchical prediction error")
      sequence_video(inputs, title="inputs", plt_title="Input")
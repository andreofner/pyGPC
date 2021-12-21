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

# todo pretrain decoder weights --> freeze! (for initial results..)
# generally: we need disentangled representations --> more layers
# --> as disentangling happens only with many layers in equilibrium
# or some KL divergence term?..
# whats wrong with large batches?
# ---> check precision, it should allow to learn disentangled even with batch size 1..
# --> no precision wont prevent overfitting. so rather use LR states low?..


class Model(torch.nn.Module):
      """ Hierarchical and dynamical weights """
      def __init__(self, sizes, bias, act, sizes_d=[], bias_d=True, act_d=[]):
            super(Model, self).__init__()
            self.layers = [] # hierarchical layers
            self.layers_d = [] # dynamical layers for each hierarchical layer
            self.precisions = [] # hierarchical precision
            self.states_curr = [] # states at time t
            self.states_last = [] # states at time t+dt

            for i in range(0, len(sizes)-1, 2):
                  self.layers.append(Sequential(Linear(sizes[i + 1], sizes[i], bias), act[int(i/2)]))
                  if len(sizes_d) > 0: # todo higher dynamical layers
                        self.layers_d.append(Model([sizes[i],sizes[i]], bias_d, act_d))
                  self.precisions.append(torch.stack([(torch.eye(sizes[i]) * 9.9 + 0.1) for _ in range(BATCH_SIZE)]).requires_grad_())

      def w_h(self, l): # Hierarchical prediction weights
            return self.layers[l]

      def w_d(self, l, l_d): # Dynamical weights
            return self.layers_d[l].layers[l_d]

      def w_t(self, l): # Transition weights (lowest dynamical layer)
            return self.w_d(l, 0)

def predict(w_list, target=None, inp_list=None):
      """ Backward pass through provided layers """
      if inp_list is None:
            inp_list = [target]
      else:
            inp_list = [inp_list[-1]]
      for weights in list(reversed(w_list)):
            inp_list.append(weights(inp_list[-1]).detach())
      return list(reversed(inp_list))


def GPC(m, l, loss=torch.abs, dynamical=False):
      """ Generalized Predictive Coding optimizer """

      # optimizers for state inference in current layer
      opt_last_low = SGD([m.states_last[l]], lr=.01) # states l_{t}
      opt_low = SGD([m.states_curr[l]], lr=.01) # states l_{t+dt_{l+1}}

      # optimizers for state inference in higher layer
      opt_last_high = SGD([m.states_last[l+1]], lr=.01) # higher layer states l+1_{t}
      opt_high = SGD([m.states_curr[l+1]], lr=.01) # states l+1_{t+dt_{l+1}}

      # optimizers for weights
      opt_weights_h = SGD(list(m.w_h(l=l).parameters()), lr=.001) # hierarchical weights l
      opt_weights_d = SGD(list(m.w_d(l=l, l_d=0).parameters()), lr=.001) # dynamical weights l
      opt_weights_t = SGD(list(m.w_t(l=l).parameters()), lr=.001)#1) # dynamical weights l
      opt_weights_t_high = SGD(list(m.w_t(l=l+1).parameters()), lr=.001)#1) # transition weights l+1

      # optimizers for precision in current layer
      opt_var_h = SGD([m.precisions[l]], lr=.1)  # precision l

      # collect variables and optimizers
      p_list = list(m.w_h(l=l).parameters())+list(m.w_d(l=l, l_d=0).parameters())+\
               list(m.w_t(l=l).parameters())+list(m.w_t(l=l+1).parameters())+\
               [m.states_curr[l], m.precisions[l], m.states_curr[l+1], m.states_last[l], m.states_last[l+1]]
      opt_list = [opt_weights_h, opt_weights_d, opt_low, opt_var_h, opt_high,
                  opt_weights_t, opt_weights_t_high, opt_last_low, opt_last_high]

      # optimize all variables
      for _, _ in enumerate([None]): # optionally iterate over individual optimizers here

            # detach variables and reset gradients
            [p.detach() for p in p_list]
            [p.requires_grad_() for p in p_list]
            SGD(p_list, lr=0).zero_grad() # reset grads

            # 1) dynamical hidden state prediction
            change = (m.states_curr[l] - m.states_last[l])  # actual change of hidden state
            # e_t.backward(create_graph=True) for true gradient instead of state difference
            pred_change = m.w_t(l=l).forward(m.states_last[l])  # predicted change of hidden state
            e_t_ = loss(pred_change - m.states_curr[l])  #  change todo precision weighting
            e_total = torch.mean(e_t_)  # todo precision weighting
            #m.states_curr[l]_transitioned = m.w_t(l=l).forward(m.states_last[l])  # transition lower state

            if dynamical: # Higher layer is dynamical
                  # 2) dynamical top-down prediction of transition from t -> t+dt_{l}
                  TD_prediction_d = m.w_d(l=l, l_d=0).forward((m.states_curr[l+1])) # dynamical prediction
                  e_d_ = loss(change - TD_prediction_d) # dynamical PE
                  e_d = torch.mean(torch.matmul(m.precisions[l]**-1, torch.reshape(e_d_, [BATCH_SIZE, -1, 1]) )) # weighted PE
                  e_total += e_d
            else: # Higher layer is hierarchical
                  # 2) hierarchical top-down prediction
                  TD_prediction_h = m.w_h(l=l).forward((m.states_curr[l+1]))  # hierarchical prediction
                  e_h_ = loss((m.states_last[l] - TD_prediction_h))  # hierarchical PE at t
                  e_h = torch.mean(torch.matmul(m.precisions[l] ** -1, torch.reshape(e_h_, [BATCH_SIZE, -1, 1])))  # weighted PE
                  e_h2_ = loss((m.states_curr[l] - TD_prediction_h))  # hierarchical PE at t+dt_{l+1}
                  e_h2 = torch.mean(torch.matmul(m.precisions[l] ** -1, torch.reshape(e_h2_, [BATCH_SIZE, -1, 1])))  # weighted PE
                  e_total += e_h# + e_h2

            # compute total error and update variables
            e_total.backward()
            for i, opt in enumerate(opt_list): opt.step() # step variable # todo skip optimizers with LR=0

      # return errors and updated variables
      params = [m.w_h(l=l), m.w_d(l=l, l_d=0), m.w_t(l=l), m.states_curr[l], m.precisions[l], m.states_curr[l+1]]
      if dynamical: # higher layer is a dynamical layer
            predictions = [p.detach().numpy() for p in [torch.zeros_like(TD_prediction_d), TD_prediction_d, TD_prediction_d]]
            return params, None, predictions, torch.zeros_like(e_d_), e_d_, e_t_
      else: # higher layer is a hierarchical layer TODO return first hierarchical prediction
            predictions = [p.detach().numpy() for p in [TD_prediction_h, TD_prediction_h, TD_prediction_h]]
            return predictions, e_h_, e_h2_, e_t_

BATCH_SIZE = 4 # batch of agents TODO fix batch size = 1
NOISE_SCALE = 0.0 # add gaussian noise to images
IMAGE_SIZE = 32*32 # image size after preprocessing

if __name__ == '__main__':

      """ Moving MNIST in OpenAI gym"""
      env = gym.make('Mnist-s1-v0')
      obs = env.reset()

      """ Model setup"""
      l_sizes, l_sizes_t = [IMAGE_SIZE,32, 32,32, 32,32, 32,32], []
      for i, s in enumerate(l_sizes[::2]): l_sizes_t += [s,s]

      # output activation for each layer
      activations_h = [torch.nn.Sigmoid()] + [torch.nn.Sigmoid() for l in l_sizes[1::2]]
      activations_d = [torch.nn.Sigmoid()] + [torch.nn.Sigmoid() for l in l_sizes[1::2]]

      # weights
      model_h = Model(sizes=l_sizes, bias=True, act=activations_h,
                      sizes_d=l_sizes_t, bias_d=True, act_d=activations_d)

      # precision & state priors
      model_h.target = torch.tensor(torch.tensor([1 for i in range(l_sizes[-1])]).unsqueeze(0).repeat(BATCH_SIZE,1,1)*.1)
      model_h.states_curr = predict(model_h.layers, target=model_h.target.float(), inp_list=None) # prior for state t+dt
      model_h.states_last = model_h.states_curr # prior for state t

      # logging
      [err_h, err_d, err_t, preds_h, preds_d, preds_t], inputs = [[[] for _ in model_h.layers] for _ in range(6)], [[]]

      UPDATES = 15
      actions = [1 for i in range(10)]

      # iterate over time steps
      for i, action in enumerate(actions):

            # get observation from gym and preprocess
            obs, rew, done, _ = env.step([action for b in range(BATCH_SIZE)])
            input = torch.Tensor(obs['agent_image'])/255
            if True: # reduce input size
                  input = input.reshape([BATCH_SIZE, -1, 64, 64])
                  input = torch.nn.MaxPool2d((2,2), stride=(2,2))(input).reshape([BATCH_SIZE, -1, IMAGE_SIZE])

            # update model
            for i in list(reversed(list(range(len(model_h.layers) - 1)))):

                  # 1) predict
                  # todo order of inputs, order of propagation, prediction needed?
                  #states_curr = predict(model_h.layers, inp_list=model_h.states_curr) # hierarchical prediction

                  # 2) update
                  for update in range(UPDATES):

                        # input at lowest layer
                        model_h.states_curr[0] = torch.tensor(input.clone().detach().float().squeeze()).unsqueeze(1)

                        # update all trainable variables
                        preds, e_h, e_d, e_t = GPC(model_h, i)

                        # collect variables for visualization
                        if update >= UPDATES - 1:
                              inputs[0].append(input[:1])
                              preds_h[i].append(preds[0][:1])
                              preds_d[i].append(preds[1][:1])
                              preds_t[i].append(preds[2][:1])
                              err_h[i].append(e_h[:1].detach())
                              err_d[i].append(e_d[:1].detach())
                              err_t[i].append(e_t[:1].detach())

            # memorize last state
            for i in range(len(model_h.layers)):
                  model_h.states_last[i] = model_h.states_curr[i]

      """ Plotting """
      #sequence_video(preds_t, title="transition_predictions", plt_title="Transition prediction")
      #sequence_video(preds_d, title="dynamical_predictions", plt_title="Dynamical prediction")
      sequence_video(preds_h, title="hierarchical_predictions", plt_title="Hierarchical prediction")
      #sequence_video(err_t, title="transition_errors", plt_title="Transition prediction error")
      #sequence_video(err_d, title="dynamical_errors", plt_title="Dynamical prediction error")
      #sequence_video(err_h, title="hierarchical_errors", plt_title="Hierarchical prediction error")
      #sequence_video(inputs, title="model_inputs", plt_title="Input")
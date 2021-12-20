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
      """ Hierarchical and dynamical weights """
      def __init__(self, sizes, bias, act, sizes_d=[], bias_d=True, act_d=[]):
            super(Model, self).__init__()
            self.layers = [] # hierarchical layers
            self.layers_d = [] # dynamical layers for each hierarchical layer
            self.precisions = [] # v_h_list # hierarchical precision
            self.inp_list_z = []
            self.inp_list_save = []
            for i in range(0, len(sizes)-1, 2):
                  self.layers.append(Sequential(Linear(sizes[i + 1], sizes[i], bias), act[int(i/2)]))
                  if len(sizes_d) > 0: self.layers_d.append(Model(sizes_d, bias_d, act_d))
                  self.precisions.append(torch.stack([(torch.eye(sizes[i]) * 0.9 + 0.1) for _ in range(BATCH_SIZE)]).requires_grad_())

      def get_weights_h(self, layer_h):
            """ Hierarchical prediction weights """
            return [l for l in self.layers]

      def get_weights_t(self, layer_h):
            """ Transition weights (lowest dynamical layer) """
            return [l[0] for l in self.layers_d[layer_h]]

      def get_weights_d(self, layer_h, layer_d):
            """ Dynamical weights """
            return [l for l in self.layers_d[layer_h]]


def predict(w_list, target=None, inp_list=None):
      """ Backward pass through provided layers """
      if inp_list is None:
            inp_list = [target]
      else:
            inp_list = [inp_list[-1]]
      for weights in list(reversed(w_list)):
            inp_list.append(weights(inp_list[-1]).detach())
      return list(reversed(inp_list))


def GPC(model_h, model_d, model_t, model_t_high,
        z_low, z_var, z_high,
        last_state, last_state_high,
        loss=torch.abs, dynamical=False):
      """ Generalized Predictive Coding optimizer

      Perception:
      Cause states v (hierarchy describing the state of the model):
            - Predict lower hierarchical layer activity

      Hidden states x (dynamical hierarchy describing the state of a single layer):
            - Predict lower hierarchical layer activity (jointly with the inferred causal state)
            - Lowest layer predicts transitioned state
            - Higher dynamical layers predict the change in lower dynamical layers (higher order derivatives)

      Prediction:
            - Cause predictions (prior generalised coordinates) can be generated using higher hierarchical layers.
            - Dynamical predictions (change in generalised coordinates) for the current cause can be generated using dynamical layers.

      """

      # optimizers for state inference in current layer
      opt_last_low = SGD([last_state], lr=0) # states l_{t}
      opt_low = SGD([z_low], lr=0) # states l_{t+dt_{l+1}}

      # optimizers for state inference in higher layer
      opt_last_high = SGD([last_state_high], lr=.1) # higher layer states l+1_{t}
      opt_high = SGD([z_high], lr=.1) # states l+1_{t+dt_{l+1}}

      # optimizers for weights
      opt_weights_h = SGD(list(model_h.parameters()), lr=.01) # hierarchical weights l
      opt_weights_d = SGD(list(model_d.parameters()), lr=.01) # dynamical weights l
      opt_weights_t = SGD(list(model_t.parameters()), lr=.01)#1) # dynamical weights l
      opt_weights_t_high = SGD(list(model_t_high.parameters()), lr=.01)#1) # transition weights l+1

      # optimizers for precision in current layer
      opt_var_h = SGD([z_var], lr=0.1)  # precision l

      # collect variables and optimizers
      p_list = list(model_h.parameters())+list(model_d.parameters())+\
               list(model_t.parameters())+list(model_t_high.parameters())+\
               [z_low, z_var, z_high, last_state, last_state_high]
      opt_list = [opt_weights_h, opt_weights_d, opt_low, opt_var_h, opt_high,
                  opt_weights_t, opt_weights_t_high, opt_last_low, opt_last_high]

      # optimize all variables
      for _, _ in enumerate([None]): # optionally iterate over individual optimizers here

            # detach variables and reset gradients
            [p.detach() for p in p_list]
            [p.requires_grad_() for p in p_list]
            SGD(p_list, lr=0).zero_grad() # reset grads

            # 1) dynamical hidden state prediction
            change = (z_low - last_state)  # actual change of hidden state
            # e_t.backward(create_graph=True) for true gradient instead of state difference
            pred_change = model_t.forward(last_state)  # predicted change of hidden state
            e_t_ = loss(pred_change - z_low)  #  change todo precision weighting
            e_total = torch.mean(e_t_)  # todo precision weighting

            #z_low_transitioned = model_t.forward(last_state)  # transition lower state

            if dynamical:
                  """ Dynamical update: Higher layer is a dynamical layer. 
                  Compute dynamical top-down prediction. """

                  # 2) dynamical top-down prediction of transition from t -> t+dt_{l}
                  TD_prediction_d = model_d.forward((z_high)) # dynamical prediction
                  e_d_ = loss(change - TD_prediction_d) # dynamical PE
                  e_d = torch.mean(torch.matmul(z_var**-1, torch.reshape(e_d_, [BATCH_SIZE, -1, 1]) )) # weighted PE
                  e_total += e_d
            else:
                  """ Hierarchical update: Higher layer is a hierarchical layer. 
                  Compute higher layer transition & hierarchical top-down prediction. 
                  HIGHER LAYER: state_{t} -> state_{t+dt_{l+1}}
                  LOWER  LAYER: state_{t} -> skipped transitions -> state_{t+dt_{l+1}} """

                  # 2) hierarchical top-down prediction
                  TD_prediction_h = model_h.forward((z_high))  # hierarchical prediction
                  e_h_ = loss((last_state - TD_prediction_h))  # hierarchical PE at t
                  e_h = torch.mean(torch.matmul(z_var ** -1, torch.reshape(e_h_, [BATCH_SIZE, -1, 1])))  # weighted PE
                  e_h2_ = loss((z_low - TD_prediction_h))  # hierarchical PE at t+dt_{l+1}
                  e_h2 = torch.mean(torch.matmul(z_var ** -1, torch.reshape(e_h2_, [BATCH_SIZE, -1, 1])))  # weighted PE
                  e_total += e_h# + e_h2

            # compute total error and update variables
            e_total.backward()
            for i, opt in enumerate(opt_list): opt.step() # step variable # todo skip optimizers with LR=0

      # return errors and updated variables
      params = [model_h, model_d, model_t, z_low, z_var, z_high]
      if dynamical: # higher layer is a dynamical layer
            predictions = [p.detach().numpy() for p in [torch.zeros_like(TD_prediction_d), TD_prediction_d, TD_prediction_d]]
            return params, None, predictions, torch.zeros_like(e_d_), e_d_, e_t_
      else: # higher layer is a hierarchical layer TODO return first hierarchical prediction
            predictions = [p.detach().numpy() for p in [TD_prediction_h, TD_prediction_h, TD_prediction_h]]
            return params, None, predictions, e_h_, e_h2_, e_t_

BATCH_SIZE = 4 # batch of agents TODO fix batch size = 1
NOISE_SCALE = 0.0 # add gaussian noise to images
IMAGE_SIZE = 32*32 # image size after preprocessing

if __name__ == '__main__':

      """ Moving MNIST in OpenAI gym"""
      env = gym.make('Mnist-s1-v0')
      obs = env.reset()

      """ Model setup"""
      l_sizes = [IMAGE_SIZE,256, 256,256]
      hidden_sizes = [0, 0]

      # output activation for each PC layer
      activations_h = [torch.nn.Sigmoid()] + [torch.nn.Identity() for l in l_sizes[1::2]]
      activations_t = [torch.nn.Sigmoid()] + [torch.nn.Identity() for l in l_sizes[1::2]]
      activations_d = [torch.nn.Identity()] + [torch.nn.Identity() for l in l_sizes[1::2]]

      # weights
      l_sizes_t = []
      for i, s in enumerate(l_sizes[::2]): l_sizes_t += [s,s]

      if False:
            model_h = Model(sizes=l_sizes, bias=True, act=activations_h) # hierarchical
            model_d = Model(sizes=l_sizes, bias=True, act=activations_d) # dynamical
            model_t = Model(sizes=l_sizes_t, bias=True, act=activations_t) # transition
      else:
            model_h = Model(sizes=l_sizes, bias=True, act=activations_h) # hierarchical
            model_d = Model(sizes=l_sizes, bias=True, act=activations_d) # dynamical
            model_t = Model(sizes=l_sizes_t, bias=True, act=activations_t) # transition


      wl_h, wl_d, wl_t = [l for l in model_h.layers], [l for l in model_d.layers], [l for l in model_t.layers]

      # precision
      if False:
            v_h_list = [torch.stack([(torch.eye(l_sizes[::2][i])*0.9 + 0.1) for b in range(BATCH_SIZE)]).requires_grad_()
                              for i in range(len(l_sizes[::2]))] # prior hierarchical precision
      else:
            v_h_list = model_h.precisions

      # state priors
      target = torch.tensor(torch.tensor([1 for i in range(l_sizes[-1])]).unsqueeze(0).repeat(BATCH_SIZE,1,1)*.1)
      inp_list_z = predict(wl_h, target=target.float(), inp_list=None) # prior for state t+dt
      inp_list_save = inp_list_z # prior for state t

      # logging
      err_h, err_d, err_t, derivs, preds_h, preds_d, preds_t = [[[] for _ in wl_h] for _ in range(7)]
      inputs = [[]]

      UPDATES = 1
      actions = [0 for i in range(10)]

      # iterate over time steps
      for i, action in enumerate(actions):

            # get observation from gym and preprocess
            obs, rew, done, _ = env.step([action for b in range(BATCH_SIZE)])
            input = torch.Tensor(obs['agent_image'])/255
            if True: # reduce size
                  input = input.reshape([BATCH_SIZE, -1, 64, 64])
                  input = torch.nn.MaxPool2d((2,2), stride=(2,2))(input).reshape([BATCH_SIZE, -1, IMAGE_SIZE])

            # update model
            for update in range(UPDATES):

                  # 1) predict
                  inp_list_z = predict(wl_h, inp_list=inp_list_z) # hierarchical prediction

                  # 2) update
                  for i in range(len(hidden_sizes)-1):

                        # input at lowest layer
                        inp_list_z[0] = torch.tensor(input.clone().detach().float().squeeze()).unsqueeze(1)

                        # update all trainable variables
                        params, deriv, preds, e_h, e_d, e_t = GPC(wl_h[i], wl_d[i], wl_t[i], wl_t[i+1],
                                                                  inp_list_z[i], v_h_list[i], inp_list_z[i+1],
                                                                  inp_list_save[i], inp_list_save[i+1])

                        # collect variables for visualization
                        wl_h[i], wl_d[i], wl_t[i], inp_list_z[i], v_h_list[i], inp_list_z[i+1] = params
                        if update >= UPDATES - 1:
                              inputs[0].append(input[:1])
                              preds_h[i].append(preds[0][:1])
                              preds_d[i].append(preds[1][:1])
                              preds_t[i].append(preds[2][:1])
                              err_h[i].append(e_h[:1].detach())
                              err_d[i].append(e_d[:1].detach())
                              err_t[i].append(e_t[:1].detach())

            # memorize last state
            for i in range(len(hidden_sizes)): inp_list_save[i] = inp_list_z[i] # memorize last state

      """ Plotting """
      sequence_video(preds_t, title="transition_predictions", plt_title="Transition prediction")
      sequence_video(preds_d, title="dynamical_predictions", plt_title="Dynamical prediction")
      sequence_video(preds_h, title="hierarchical_predictions", plt_title="Hierarchical prediction")
      sequence_video(err_t, title="transition_errors", plt_title="Transition prediction error")
      sequence_video(err_d, title="dynamical_errors", plt_title="Dynamical prediction error")
      sequence_video(err_h, title="hierarchical_errors", plt_title="Hierarchical prediction error")
      sequence_video(inputs, title="model_inputs", plt_title="Input")
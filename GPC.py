"""
Differentiable Generalized Predictive Coding
André Ofner 2021
"""

import math
import torch
import gym, numpy as np
from torch.optim import SGD
import matplotlib.pyplot as plt
from tools import *

class Model(torch.nn.Module):
      """ Layer hierarchies with weights sharing """
      def __init__(self, sizes=[], hidden_sizes=[8], bias=False, activations=[torch.nn.Identity()]):
            super(Model, self).__init__()
            self.layers = []
            for i in range(0, len(sizes)-1, 2):
                  self.layers.append(
                        torch.nn.Sequential(
                              torch.nn.Linear(sizes[i+1], hidden_sizes[int(i/2)], bias=bias), torch.nn.ReLU(),
                              torch.nn.Linear(hidden_sizes[int(i/2)], sizes[i], bias=bias), activations[int(i/2)]
                        ))

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
      """ Generalized Predictive Coding optimizer """

      # optimizers for state inference in current layer
      opt_last_low = SGD([last_state], lr=0.1) # states l_{t}
      opt_low = SGD([z_low], lr=0.1) # states l_{t+dt_{l+1}}

      # optimizers for state inference in higher layer
      opt_last_high = SGD([last_state_high], lr=1) # higher layer states l+1_{t}
      opt_high = SGD([z_high], lr=1) # states l+1_{t+dt_{l+1}}

      # optimizers for learning of weights between current and higher layer
      opt_weights_h = SGD(list(model_h.parameters()), lr=0.01) # hierarchical weights l
      opt_weights_d = SGD(list(model_d.parameters()), lr=0.01) # dynamical weights l
      opt_weights_t = SGD(list(model_t.parameters()), lr=10) # dynamical weights l # todo LR
      opt_weights_t_high = SGD(list(model_t_high.parameters()), lr=0) # transition weights l+1 # todo LR

      # optimizers for precision of prediction error in current layer
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

            # 1) lower state transition
            z_low_transitioned = model_t.forward(last_state)  # transition lower state
            e_t_ = loss(z_low - z_low_transitioned)  # todo precision weighting
            e_t = torch.mean(e_t_)  # todo precision weighting
            # e_t.backward(create_graph=True) if we want the true gradient instead of state change
            grad_t = (z_low_transitioned - last_state)  # state change from transition
            e_total = e_t

            if dynamical:
                  """ Dynamical update: Higher layer is a dynamical layer. 
                  Compute dynamical top-down prediction. """

                  # 2) dynamical top-down prediction of transition from t -> t+dt_{l}
                  TD_prediction_d = model_d.forward((z_high)) # dynamical prediction
                  e_d_ = loss(grad_t - TD_prediction_d) # dynamical PE
                  e_d = torch.mean(torch.matmul(z_var**-1, torch.reshape(e_d_, [BATCH_SIZE, -1, 1]) )) # weighted PE
                  e_total += e_d
            else:
                  """ Hierarchical update: Higher layer is a hierarchical layer. 
                  Compute higher layer transition & hierarchical top-down prediction. 
                  HIGHER LAYER: state_{t} -> state_{t+dt_{l+1}}
                  LOWER  LAYER: state_{t} -> skipped transitions -> state_{t+dt_{l+1}} """

                  # 2) hierarchical top-down prediction at t
                  TD_prediction_h = model_h.forward((z_high))  # hierarchical prediction
                  e_h_ = loss((last_state - TD_prediction_h))  # hierarchical PE
                  e_h = torch.mean(torch.matmul(z_var ** -1, torch.reshape(e_h_, [BATCH_SIZE, -1, 1])))  # weighted PE

                  # 3) hierarchical top-down prediction at t+dt_{l+1} # todo sampling over >1 steps
                  z_high_transitioned = model_t_high.forward(last_state_high) # transition higher state
                  TD_prediction_h2 = model_h.forward((z_high_transitioned))  # hierarchical prediction
                  e_h2_ = loss((z_low - TD_prediction_h2))  # hierarchical PE
                  e_h2 = torch.mean(torch.matmul(z_var ** -1, torch.reshape(e_h2_, [BATCH_SIZE, -1, 1])))  # weighted PE
                  e_total += e_h + e_h2

            # compute total error and update variables
            e_total.backward()
            for i, opt in enumerate(opt_list): opt.step() # step variable # todo skip optimizers with LR=0

      # return errors and updated variables
      params = [model_h, model_d, model_t, z_low, z_var, z_high]
      if dynamical: # higher layer is a dynamical layer
            predictions = [p.detach().numpy() for p in [torch.zeros_like(TD_prediction_d), TD_prediction_d, z_low_transitioned]]
            return params, None, predictions, torch.zeros_like(e_d_), e_d_, e_t_
      else: # higher layer is a hierarchical layer TODO return first hierarchical prediction
            predictions = [p.detach().numpy() for p in [TD_prediction_h2, torch.zeros_like(TD_prediction_h2), z_low_transitioned]]
            return params, None, predictions, e_h_, e_h2_, e_t_

BATCH_SIZE = 16 # batch of agents TODO fix batch size = 1
NOISE_SCALE = 0.0 # add gaussian noise to images
IMAGE_SIZE = 16*16 # image size after preprocessing

if __name__ == '__main__':

      """ Moving MNIST in OpenAI gym"""
      env = gym.make('Mnist-s1-v0')
      obs = env.reset()

      """ Model setup"""
      l_sizes = [IMAGE_SIZE,128, 128,64]

      # output activation for each PC layer
      activations_h = [torch.nn.Sigmoid()] + [torch.nn.ReLU() for l in l_sizes[1::2]]
      activations_d = [torch.nn.Sigmoid()] + [torch.nn.ReLU() for l in l_sizes[1::2]]
      activations_t = [torch.nn.Sigmoid()] + [torch.nn.ReLU() for l in l_sizes[1::2]]

      # hidden layers within each PC layer. Activation is ReLU
      hidden_sizes = [64 for l in l_sizes[1::2]]

      # shared weights
      l_sizes_t = []
      for i, s in enumerate(l_sizes[::2]): l_sizes_t += [s,s]
      model_h = Model(sizes=l_sizes, bias=True, activations=activations_h, hidden_sizes=hidden_sizes) # hierarchical
      model_d = Model(sizes=l_sizes, bias=True, activations=activations_d, hidden_sizes=hidden_sizes) # dynamical
      model_t = Model(sizes=l_sizes_t, bias=True, activations=activations_t, hidden_sizes=hidden_sizes) # transition
      wl_h = [l for l in model_h.layers]
      wl_d = [l for l in model_d.layers]
      wl_t = [l for l in model_t.layers]

      # precision TODO transition error precision?
      v_h_list = [torch.stack([(torch.eye(l_sizes[::2][i])*0.9 + 0.1) for b in range(BATCH_SIZE)]).requires_grad_()
                        for i in range(len(l_sizes[::2]))] # prior hierarchical precision

      # state priors
      target = torch.tensor(torch.tensor([1 for i in range(l_sizes[-1])]).unsqueeze(0).repeat(BATCH_SIZE,1,1)*.1)
      inp_list_z = predict(wl_h, target=target.float(), inp_list=None) # prior for state t+dt
      inp_list_save = inp_list_z # prior for state t

      # logging
      errors_h = [[] for _ in wl_h] # hierarchical PE
      errors_d = [[] for _ in wl_d] # dynamical PE
      errors_t = [[] for _ in wl_d] # transition PE
      derivs = [[] for _ in wl_d] # first derivative of transition function
      preds_h, preds_d, preds_t = [[] for _ in wl_d], [[] for _ in wl_d], [[] for _ in wl_d]
      inputs = [[]]

      UPDATES = 5
      actions = [1 for i in range(10)]# + [0 for i in range(10)] + [1 for i in range(10)] + [0 for i in range(10)]
      for i, action in enumerate(actions):   # iterate over time steps

            # get observation from gym
            obs, rew, done, _ = env.step([action for b in range(BATCH_SIZE)])
            input = torch.Tensor(obs['agent_image'])

            # preprocess input
            if True:
                  input = input.reshape([BATCH_SIZE, -1, 64, 64])
                  input = torch.nn.MaxPool2d((2, 2), stride=(4, 4), padding=(0,0))(input)
                  input = input.reshape([BATCH_SIZE, -1, IMAGE_SIZE])
                  input = torch.nn.Sigmoid()(input)

            # update model
            inputs[0].append(input[:1])
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
                              preds_h[i].append(preds[0][:1])
                              preds_d[i].append(preds[1][:1])
                              preds_t[i].append(preds[2][:1])
                              errors_h[i].append(e_h[:1].detach())
                              errors_d[i].append(e_d[:1].detach())
                              errors_t[i].append(e_t[:1].detach())

            # memorize last state
            for i in range(len(hidden_sizes)): inp_list_save[i] = inp_list_z[i] # memorize last state

      """ Plotting """
      sequence_video(preds_t, title="transition_predictions", plt_title="Transition prediction")
      sequence_video(preds_d, title="dynamical_predictions", plt_title="Dynamical prediction")
      sequence_video(preds_h, title="hierarchical_predictions", plt_title="Hierarchical prediction")
      sequence_video(errors_t, title="transition_errors", plt_title="Transition prediction error")
      sequence_video(errors_d, title="dynamical_errors", plt_title="Dynamical prediction error")
      sequence_video(errors_h, title="hierarchical_errors", plt_title="Hierarchical prediction error")
      sequence_video(inputs, title="model_inputs", plt_title="Input")
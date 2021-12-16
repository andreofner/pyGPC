"""
Differentiable Generalized Predictive Coding in Python & Torch
André Ofner 2021
"""

import math
import torch
from torch.optim import SGD
import matplotlib.pyplot as plt
import gym, numpy as np
import tools

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

class ParallelModel(torch.nn.Module):
      """ Parallel hierarchical models without weights sharing """
      def __init__(self, parallel_models=1, sizes=[], hidden_sizes=[8], bias=False,
                          activations=[torch.nn.Identity()]):
            super(ParallelModel, self).__init__()
            self.models = []
            for i in range(parallel_models):
                  self.models.append(Model(sizes=sizes, hidden_sizes=hidden_sizes,
                                           bias=bias, activations=activations))

      def forward(self, x, l): # forward through layers without weights sharing
            return torch.tensor([m.layers[l].forward(x[b]) for b, m in enumerate(self.models)])

      def get_layer_weights(self, l): # weights for all models at layer l
            return [m.layers[l] for b, m in enumerate(self.models)]

def predict(w_list, target=None, inp_list=None):
      """ Backward pass through provided layers """
      if inp_list is None:
            inp_list = [target]
      else:
            inp_list = [inp_list[-1]]
      for weights in list(reversed(w_list)):
            inp_list.append(weights(inp_list[-1]).detach())
      return list(reversed(inp_list))


def GPC(model_h, model_d, model_t,
        h_low, d_low,
        h_var, d_var,
        h_high, d_high, last_state,
        loss=torch.abs, dynamical=True):
      """ Generalized Predictive Coding optimizer """

      # set hierarchical state as prior for dynamical state
      d_low = h_low
      d_high = h_high

      # define optimizer and LR for each variable
      opt_low_h, opt_high_h = SGD([h_low], lr=0.0), SGD([h_high], lr=0.1) # hierarchical states
      opt_low_d, opt_high_d = SGD([d_low], lr=0.0), SGD([d_high], lr=0.1) # dynamical states
      opt_weights_h = SGD(list(model_h.parameters()), lr=0.01) # hierarchical weights
      opt_weights_d = SGD(list(model_d.parameters()), lr=0.01) # dynamical weights
      opt_var_h, opt_var_d = SGD([h_var], lr=0.01), SGD([d_var], lr=0.01)  # precision

      # collect variables and optimizers
      p_list = list(model_h.parameters())+list(model_d.parameters())+[h_low, h_var, h_high] + [d_low, d_var, d_high]
      opt_list = [opt_weights_h, opt_weights_d, opt_low_h, opt_var_h, opt_high_h, opt_low_d, opt_var_d, opt_high_d]

      # collect variables and optimizers for transition model
      for b, m in enumerate(model_t):
            p_list += list(model_t[b].parameters())
            opt_list.append(SGD(list(model_t[b].parameters()), lr=0.001))

      # optimize all variables
      for _, _ in enumerate([None]): # optionally iterate over individual optimizers here

            # detach variables and reset gradients
            [p.detach() for p in p_list]
            [p.requires_grad_() for p in p_list]
            SGD(p_list, lr=0).zero_grad() # reset grads

            # 1) hierarchical top-down prediction
            TD_prediction_h = model_h.forward((h_high)) # hierarchical prediction
            e_h_ = loss((h_low - TD_prediction_h))   # hierarchical PE
            e_h = torch.mean(torch.matmul(h_var**-1, torch.reshape(e_h_, [batch_size, -1, 1]))) # weighted PE
            e_total = e_h

            # 2) transition and dynamical top-down prediction
            if dynamical:
                  # state transition
                  d_low_transitioned = []
                  for b, ls_b in enumerate(last_state):
                        if len(ls_b.shape) <= 2: ls_b.unsqueeze(0) # todo fix
                        d_low_transitioned.append(model_t[b].forward(ls_b))
                  d_low_transitioned = torch.stack(d_low_transitioned)
                  e_t_ = loss(d_low - d_low_transitioned) # todo precision weighting
                  e_t = torch.mean(e_t_)  # todo precision weighting
                  e_t.backward(create_graph=True)
                  grad_t = (d_low_transitioned-last_state) # state change from transition

                  # dynamical top-down prediction
                  TD_prediction_d = model_d.forward((d_high)) # dynamical prediction
                  e_d_ = loss(grad_t - TD_prediction_d) # dynamical PE
                  e_d = torch.mean(torch.matmul(d_var**-1, torch.reshape(e_d_, [batch_size, -1, 1]) )) # weighted PE
                  e_total += e_d + e_t
            else:
                  e_d_, e_t_ = torch.zeros_like(e_h_), torch.zeros_like(e_h_)
                  TD_prediction_d = torch.zeros_like(TD_prediction_h)
                  d_low_transitioned = torch.zeros_like(d_low)

            # compute total error and update variables
            e_total.backward()
            for i, opt in enumerate(opt_list): opt.step() # step variable

      # collect hierarchical and temporal predictions
      predictions = [p.detach().numpy() for p in [TD_prediction_h, TD_prediction_d, d_low_transitioned]]

      # collect updated variables
      params = [model_h, model_d, model_t, h_low, d_low, h_var, d_var, h_high, d_high]

      # return variables, predictions and PE
      return params, None, predictions, e_h_, e_d_, e_t_

if __name__ == '__main__':

      """ Moving MNIST in OpenAI gym"""
      env = gym.make('Mnist-s1-v0')
      obs = env.reset()

      """ Model setup"""
      input = torch.Tensor([obs['agent_image'][0].reshape([-1])])
      input = input.detach().unsqueeze(dim=1)
      batch = input

      IMAGE_SIZE = 16*16
      batch_size, l_sizes = 1, [IMAGE_SIZE,32, 32,16]
      hidden_sizes = [64,32]
      activations_h = [torch.nn.Sigmoid(), torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.Sigmoid()]
      activations_d = [torch.nn.Sigmoid(), torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.Sigmoid()]
      activations_t = [torch.nn.Sigmoid(), torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.Sigmoid()]

      # shared weights
      model_h = Model(sizes=l_sizes, bias=True, activations=activations_h, hidden_sizes=hidden_sizes) # hierarchical
      model_d = Model(sizes=l_sizes, bias=True, activations=activations_d, hidden_sizes=hidden_sizes) # dynamical
      wl_h = [l for l in model_h.layers]
      wl_d = [l for l in model_d.layers]

      # non shared weights
      l_sizes_t = []
      for i, s in enumerate(l_sizes[::2]):
            l_sizes_t.append(s)
            l_sizes_t.append(s)

      parallel_model_t = ParallelModel(parallel_models=batch_size, hidden_sizes=hidden_sizes,
                                   sizes=l_sizes_t, bias=True, activations=activations_t) # prior transition weights
      wl_t = [parallel_model_t.get_layer_weights(l=l) for l in range(len(l_sizes[::2]))]

      # precision
      v_h_list = [torch.stack([(torch.eye(l_sizes[i])*0.9 + 0.1) for b in range(batch_size)]).requires_grad_()
                        for i in range(len(l_sizes))] # prior hierarchical precision
      v_d_list = [torch.stack([(torch.eye(l_sizes[i])*0.9 + 0.1) for b in range(batch_size)]).requires_grad_()
                        for i in range(len(l_sizes))] # prior dynamical precision

      # states
      inp_list_h = predict(wl_h, target=torch.tensor(torch.ones_like(
            torch.tensor([i for i in range(l_sizes[-1])]).unsqueeze(0)) * 0.1).float(), inp_list=None) # state priors
      inp_list_d = predict(wl_d, target=torch.tensor(torch.ones_like(
            torch.tensor([i for i in range(l_sizes[-1])]).unsqueeze(0)) * 0.1).float(),inp_list=None) # dynamical priors
      inp_list_d_save = inp_list_d  # previous state for state transition

      # logging
      errors_h = [[] for _ in wl_h] # hierarchical PE
      errors_d = [[] for _ in wl_d] # dynamical PE
      errors_t = [[] for _ in wl_d] # transition PE
      derivs = [[] for _ in wl_d] # first derivative of transition function
      preds_h, preds_d, preds_t = [[] for _ in wl_d], [[] for _ in wl_d], [[] for _ in wl_d]
      states = [[] for _ in wl_d]
      inputs = []

      UPDATES = 5
      actions = [1 for i in range(10)]# + [1 for i in range(50)]
      for i, action in enumerate(actions):   # iterate over time steps

            """ Get observation from gym"""
            obs, rew, done, _ = env.step([action]) # feed your agent's action here
            input = torch.Tensor([obs['agent_image'][0].reshape([-1])])

            """ Process observation in model"""
            if True: # preprocess input size
                  input = input.reshape([-1, 1, 64, 64])
                  input = torch.nn.MaxPool2d((2, 2), stride=(4, 4), padding=(0,0))(input)
                  input = input.reshape([-1, IMAGE_SIZE])
                  input = torch.nn.Sigmoid()(input)

            input = input.detach().unsqueeze(dim=1)
            inputs.append(input[:])
            for update in range(UPDATES):
                  # predict
                  inp_list_h = predict(wl_h, inp_list=inp_list_h) # hierarchical prediction
                  inp_list_d = predict(wl_d, inp_list=inp_list_d) # dynamical prediction

                  # update
                  for i in range(len(hidden_sizes)-1):
                        inp_list_h[0] = torch.tensor(input.clone().detach().float())
                        inp_list_d[0] = torch.tensor(input.clone().detach().float())
                        params, deriv, preds, e_h, e_d, e_t = GPC(wl_h[i], wl_d[i], wl_t[i],
                                                                  inp_list_h[i], inp_list_d[i],
                                                                  v_h_list[i], v_d_list[i],
                                                                  inp_list_h[i+1], inp_list_d[i+1],
                                                                  last_state=inp_list_d_save[i])
                        wl_h[i], wl_d[i], wl_t[i], inp_list_h[i], \
                        inp_list_d[i], v_h_list[i], v_d_list[i], inp_list_h[i+1], inp_list_d[i + 1] = params
                        if update >= UPDATES - 1:
                              preds_h[i].append(preds[0][:])
                              preds_d[i].append(preds[1][:])
                              preds_t[i].append(preds[2][:])
                              #derivs[i].append(deriv[:])
                              errors_h[i].append(e_h.detach())
                              errors_d[i].append(e_d.detach())
                              errors_t[i].append(e_t.detach())
                              states[i + 1].append(inp_list_d[i + 1][0][0].detach().clone().numpy())
            for i in range(len(hidden_sizes)-1):
                  inp_list_d_save[i] = inp_list_d[i]   # memorize last state

      """ Plotting """

      def sequence_video(data, title="", plt_title="", scale=255, plot=False, plot_video=True):
            try:
                  predicts_plot = np.asarray([pred.squeeze() for pred in data[0]])
                  predicts_plot = predicts_plot.reshape([-1, int(math.sqrt(IMAGE_SIZE)), int(math.sqrt(IMAGE_SIZE)), 1])
            except:
                  predicts_plot = np.asarray([pred.detach().numpy().squeeze() for pred in data[0]])
                  predicts_plot = predicts_plot.reshape([-1, int(math.sqrt(IMAGE_SIZE)), int(math.sqrt(IMAGE_SIZE)), 1])

            if plot_video:
                  tools.plot_episode(predicts_plot*scale, title=str(title))
            if plot:
                  plt.imshow(predicts_plot[-1])
                  plt.title(str(plt_title))
                  plt.colorbar()
                  plt.show()

      sequence_video(preds_t, title="transition_predictions", plt_title="Transition prediction")
      sequence_video(preds_d, title="dynamical_predictions", plt_title="Dynamical prediction")
      sequence_video(preds_h, title="hierarchical_predictions", plt_title="Hierarchical prediction")

      sequence_video(errors_t, title="transition_errors", plt_title="Transition prediction error")
      sequence_video(errors_d, title="dynamical_errors", plt_title="Dynamical prediction error")
      sequence_video(errors_h, title="hierarchical_errors", plt_title="Hierarchical prediction error")

      #sequence_video(inp_list_d, title="model_inputs", plt_title="Input")

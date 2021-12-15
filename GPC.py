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
        h_high, d_high,
        last_state, transitioned_state,
        loss=torch.abs, dynamical=True):
      """ Generalized Predictive Coding optimizer """

      # todo state transition inference (not only weights learning):
      # d_low_transitioned right now is the predicted output from the transition weights
      # it should be also inferred. problem: mismatch with data --> that's fine it gets decoded

      # define optimizer and LR for each variable
      opt_low_h, opt_high_h = SGD([h_low], lr=0.001), SGD([h_high], lr=0.1) # hierarchical states
      opt_low_d, opt_high_d = SGD([d_low], lr=0.001), SGD([d_high], lr=0.1) # dynamical states
      opt_weights_h = SGD(list(model_h.parameters()), lr=0.01) # hierarchical weights
      opt_weights_d = SGD(list(model_d.parameters()), lr=0.01) # dynamical weights
      opt_var_h, opt_var_d = SGD([h_var], lr=0.1), SGD([d_var], lr=0.1)  # precision

      # collect variables and optimizers
      p_list = list(model_h.parameters())+list(model_d.parameters())+[h_low, h_var, h_high] + [d_low, d_var, d_high]
      opt_list = [opt_weights_h, opt_weights_d, opt_low_h, opt_var_h, opt_high_h, opt_low_d, opt_var_d, opt_high_d]

      # collect variables and optimizers for transition model
      for b, m in enumerate(model_t):
            p_list += list(model_t[b].parameters())
            opt_list.append(SGD(list(model_t[b].parameters()), lr=0.001))

      # optimize all variables
      for i, opt in enumerate([None]): # optionally iterate over individual optimizers here

            # detach variables and reset gradients
            [p.detach() for p in p_list]
            [p.requires_grad_() for p in p_list]
            SGD(p_list, lr=0).zero_grad() # reset grads

            # 1) hierarchical top-down prediction
            TD_prediction_h = model_h.forward((h_high)) # hierarchical prediction
            e_h_ = loss((h_low - TD_prediction_h))   # hierarchical PE
            e_h = torch.mean(torch.matmul(h_var**-1, torch.reshape(e_h_, [batch_size, -1, 1]))) # weighted PE
            e_total = e_h

            # 2) dynamical transition and top-down prediction
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

                  # top-down transition gradient prediction
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

if __name__ == '__main2__':

      def f_sin_2(x): # data generating function
            func = math.cos(x)*math.sin(x/2)
            deriv1 = math.cos(x)*math.sin(x/2) + (math.cos(x/2)*math.sin(x))/2 # for visualization
            deriv2 = math.cos(x/2)*math.cos(x) - (5*math.sin(x/2)*math.sin(x))/4 # for visualization
            return func, deriv1, deriv2, r'$ f(x) = sin(x) sin(0.5 x)$'

      def f_sin_pi(x): # data generating function
            func = math.sin((x/10)*math.pi)
            deriv1 = math.cos(x)*math.sin(x/2) + (math.cos(x/2)*math.sin(x))/2 # for visualization
            deriv2 = math.cos(x/2)*math.cos(x) - (5*math.sin(x/2)*math.sin(x))/4 # for visualization
            return func, deriv1, deriv2, r'$ f(x) = sin(x \pi )$'

      def f_sin_mod(x): # data generating function
            func = math.sin((x/10)*math.pi) * math.sin(0.7*(x/10)*math.pi)
            deriv1 = math.cos(x)*math.sin(x/2) + (math.cos(x/2)*math.sin(x))/2 # for visualization
            deriv2 = math.cos(x/2)*math.cos(x) - (5*math.sin(x/2)*math.sin(x))/4 # for visualization
            return func, deriv1, deriv2, r'$f(x) = sin( \pi x) sin(0.7 \pi x)$'

      def conv(input, stride=2): # strided conv
            conv1d = torch.nn.Conv1d(1, 1, 1, stride=stride, bias=False)
            torch.nn.init.constant_(conv1d.weight,1)
            return conv1d(input)

      for f, FUNC in enumerate([f_sin_mod]):
            for stride in [1,10]:
                  # 1D data batch where each batch element has a different stride
                  batch_size, l_sizes = 1, [1,1,1,1,1]
                  #strides = [(s+1) for s in range(batch_size)]
                  strides = [stride]
                  data = torch.tensor([FUNC(x)[0] for x in range(200)])
                  batch = torch.cat([conv(data.repeat(1,1,1), stride=s)[:,:,:] for s in strides])
                  batch = torch.reshape(batch, [batch_size,1,-1])

                  # shared weights
                  model_h = Model(sizes=l_sizes, bias=True) # prior hierarchical weights
                  model_d = Model(sizes=l_sizes, bias=True) # prior dynamical weights
                  wl_h = [l for l in model_h.layers]
                  wl_d = [l for l in model_d.layers]

                  # non shared weights
                  parallel_model_t = ParallelModel(parallel_models=batch_size, sizes=[1,1,1,1,1], bias=True)   # prior transition weights
                  wl_t = [parallel_model_t.get_layer_weights(l=l) for l in range(4)]

                  # precision
                  v_h_list = [torch.tensor([(torch.eye(l_sizes[i])*1) for b in range(batch_size)]).requires_grad_()
                                    for i in range(len(l_sizes))]   # prior hierarchical precision
                  v_d_list = [torch.tensor([(torch.eye(l_sizes[i])*1) for b in range(batch_size)]).requires_grad_()
                                    for i in range(len(l_sizes))]   # prior dynamical precision

                  # states
                  inp_list_h = predict(wl_h, target=torch.tensor(torch.ones_like(batch[:,:,0])*0.1).float(), inp_list=None) # state priors
                  inp_list_d = predict(wl_d, target=torch.tensor(torch.ones_like(batch[:,:,0])*0.1).float(), inp_list=None) # state priors
                  inp_list_d_save = inp_list_d # previous state for state transition

                  # logging
                  errors_h = [[] for _ in wl_h] # hierarchical PE
                  errors_d = [[] for _ in wl_d] # dynamical PE
                  errors_t = [[] for _ in wl_d] # transition PE
                  derivs = [[] for _ in wl_d] # first derivative
                  predicts = [[] for _ in wl_d] # predictions
                  states = [[] for _ in wl_d] # inferred states
                  inputs = [] # model inputs

                  UPDATES = 15 # todo stop when converged
                  batch = batch.squeeze()
                  if len(batch.shape) <= 1: batch = batch.unsqueeze(0) # make sure channel dim exists
                  for i, input in enumerate(batch.T): # iterate over time steps
                        input = input.detach().unsqueeze(dim=1)
                        inputs.append(input[:])
                        for update in range(UPDATES):
                              # 1) predict
                              inp_list_h = predict(wl_h, inp_list=inp_list_h)   # state priors
                              inp_list_d = predict(wl_d, inp_list=inp_list_d)   # state priors

                              # 2) update
                              for i in [0,1,2]:
                                    inp_list_h[0] = torch.tensor(input.clone().detach().float())
                                    inp_list_d[0] = torch.tensor(input.clone().detach().float())
                                    params, deriv, pred, e_h, e_d, e_t = GPC(wl_h[i], wl_d[i], wl_t[i],
                                                                  inp_list_h[i], inp_list_d[i], v_h_list[i], v_d_list[i],
                                                                  inp_list_h[i+1], inp_list_d[i+1], last_state=inp_list_d_save[i])
                                    wl_h[i], wl_d[i], wl_t[i], inp_list_h[i], inp_list_d[i], v_h_list[i], v_d_list[i], inp_list_h[i+1], inp_list_d[i+1] = params
                                    if update >= UPDATES-1:
                                          predicts[i].append(pred[:])
                                          derivs[i].append(deriv[:])
                                          errors_h[i].append(e_h.detach())
                                          errors_d[i].append(e_d.detach())
                                          errors_t[i].append(e_t.detach())
                                          states[i+1].append(inp_list_d[i+1][0][0].detach().clone().numpy())
                        for i in [0, 1]:
                              inp_list_d_save[i] = inp_list_d[i]   # memorize last state

                  # plot observed and inferred function
                  for b in range(batch_size):
                        fig = plt.figure(figsize=(10,5))
                        ax = plt.subplot(111)
                        plt.plot([d[b] for i, d in enumerate(inputs[:50])], label="Observed f(x)", color="black", linestyle='--')
                        plt.plot([d[b] for i, d in enumerate(predicts[0][:50])], label="Learned g(x)", color="black")
                        plt.plot([d[b] for i, d in enumerate(derivs[0][:50])], label="Learned g'(x)", color="green")
                        #plt.plot([d[b] for i, d in enumerate(derivs[1][:50])], label="Learned g''(x)", color="blue")
                        plt.plot([d[0] for i, d in enumerate(errors_t[0][:50])], label="Transition PE f(x)", color="red")
                        plt.plot([d[b] for i, d in enumerate(errors_d[0][:50])], label="Dynamical PE f(x)", color="red", linestyle='--')
                        plt.plot([d[b] for i, d in enumerate(errors_t[1][:50])], label="Transition PE f'(x)", color="orange")
                        plt.plot([d[b] for i, d in enumerate(errors_d[1][:50])], label="Dynamical PE f'(x)", color="orange", linestyle='--')
                        plt.grid()
                        plt.ylim(-2,2)
                        plt.ylabel(r'Observed f(x) and learned g(x)')
                        plt.xlabel(r'Update x')
                        plt.title(str(FUNC(0)[3])+r' with stride '+str(strides[b]))
                        box = ax.get_position()
                        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
                        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                        plt.savefig("Stride"+str(f)+str(strides[b])+".pdf")
                        plt.show()
                        plt.close()

if __name__ == '__main__':

      """ Moving MNIST in OpenAI gym"""

      """ Gym setup """
      env = gym.make('Mnist-s1-v0')
      obs = env.reset()
      frames = []
      agent_frames = []
      env_states = []

      """ Model setup"""
      input = torch.Tensor([obs['agent_image'][0].reshape([-1])])
      input = input.detach().unsqueeze(dim=1)
      batch = input

      IMAGE_SIZE = 16*16
      batch_size, l_sizes = 1, [IMAGE_SIZE,64, 64,32, 32,16]
      hidden_sizes = [64,32,16]
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

      UPDATES = 10
      actions = [1 for i in range(50)]# + [1 for i in range(5)]
      for i, action in enumerate(actions):   # iterate over time steps

            """ Get observation from gym"""
            obs, rew, done, _ = env.step([action]) # feed your agent's action here
            frames.append(obs['image']) # use this to visualize the episode
            agent_frames.append(obs['agent_image'])  # use this observation for your agent
            env_states.append(obs['state']) # the agent's positions, size and action sizes could also be fed to your agent
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

      PLOT_VIDEO = True
      PLOT_IMG = False
      agent_frames = np.asarray(agent_frames)
      env_states = np.asarray(env_states)

      # visualize transition prediction
      scale = 255
      predicts_plot = np.asarray([pred.squeeze() for pred in preds_t[0]]).reshape([-1, int(math.sqrt(IMAGE_SIZE)), int(math.sqrt(IMAGE_SIZE)), 1])#*255
      if PLOT_VIDEO: tools.plot_episode(predicts_plot*scale, agent_frames, env_states, title="transition_predictions")
      if True:
            plt.imshow(predicts_plot[-1])
            plt.title("dynamical prediction (transition)")
            plt.colorbar()
            plt.show()

      # visualize dynamical prediction
      predicts_plot = np.asarray([pred.squeeze() for pred in preds_d[0]]).reshape([-1, int(math.sqrt(IMAGE_SIZE)), int(math.sqrt(IMAGE_SIZE)), 1])#*255
      if PLOT_VIDEO: tools.plot_episode(predicts_plot*scale, agent_frames, env_states, title="dynamical_predictions")
      if PLOT_IMG:
            plt.imshow(predicts_plot[-1])
            plt.title("dynamical prediction (top-down)")
            plt.colorbar()
            plt.show()

      # visualize hierarchical prediction
      predicts_plot = np.asarray([pred.squeeze() for pred in preds_h[0]]).reshape([-1, int(math.sqrt(IMAGE_SIZE)), int(math.sqrt(IMAGE_SIZE)), 1])#*255
      if PLOT_VIDEO: tools.plot_episode(predicts_plot*scale, agent_frames, env_states, title="hierarchical_predictions")
      if PLOT_IMG:
            plt.imshow(predicts_plot[-1])
            plt.title("hierarchical prediction")
            plt.colorbar()
            plt.show()

      # visualize hierarchical model errors
      errors_h_plot = np.asarray([e.detach().numpy().squeeze() for e in errors_h[0]]).reshape([-1, int(math.sqrt(IMAGE_SIZE)), int(math.sqrt(IMAGE_SIZE)), 1])
      if PLOT_VIDEO: tools.plot_episode(errors_h_plot*scale, np.zeros_like(agent_frames), env_states, title="hierarchical_errors")
      if PLOT_IMG:
            plt.imshow(errors_h_plot[-1])
            plt.title("hierarchical prediction error")
            plt.colorbar()
            plt.show()

      # visualize dynamical model errors
      errors_d_plot = np.asarray([e.detach().numpy().squeeze() for e in errors_d[0]]).reshape([-1, int(math.sqrt(IMAGE_SIZE)), int(math.sqrt(IMAGE_SIZE)), 1])
      if PLOT_VIDEO: tools.plot_episode(errors_h_plot*scale, np.zeros_like(agent_frames), env_states, title="dynamical_errors")
      if PLOT_IMG:
            plt.imshow(errors_d_plot[-1])
            plt.title("dynamical prediction error")
            plt.colorbar()
            plt.show()

      # visualize model inputs
      inp_list_d_plot = np.asarray([e.detach().numpy().squeeze() for e in inp_list_d[0]]).reshape([-1, int(math.sqrt(IMAGE_SIZE)), int(math.sqrt(IMAGE_SIZE)), 1])
      if PLOT_VIDEO: tools.plot_episode(errors_h_plot*scale, np.zeros_like(agent_frames), env_states, title="model_inputs")
      if PLOT_IMG:
            plt.imshow(inp_list_d_plot[-1])
            plt.title("input")
            plt.colorbar()
            plt.show()
"""
Differentiable Generalized Predictive Coding
André Ofner 2021
"""

import torch
from tools import *
from torch.optim import SGD
from torch.nn import Linear, Sequential

class Model(torch.nn.Module):
      """ PCN model with hierarchical and dynamical weights """
      def __init__(self, sizes, bias, act, bias_d=True, act_d=[], dynamical=False, var=1., covar=.1, prior=.1):
            super(Model, self).__init__()
            self.layers = [] # hierarchical layers
            self.layers_d = [] # dynamical layers for each hierarchical layer
            self.prec = [] # hierarchical precision

            for i in range(0, len(sizes)-1, 2): # hierarchical layers
                  self.layers.append(Sequential(Linear(sizes[i+1], sizes[i], bias), act[int(i/2)]))
                  if not dynamical: # each hierarchical layer gets a dynamical model
                        self.layers_d.append(Model([sizes[i], sizes[i]]+sizes[i:], bias_d, act_d, dynamical=True))
                  self.prec.append((torch.eye(sizes[i])*(var-covar)+covar).repeat([B_SIZE,1,1])) # prior precision

            self.target = torch.ones([B_SIZE,1,sizes[-1]])*prior # prior target state
            self.lastState = self.predict(target=self.target.float(), states=None) # prior states at time t
            self.currState = self.predict(target=self.target.float(), states=None) # prior states at time t+dt
            if dynamical: self.lastState, self.currState = self.lastState[1:], self.currState[1:]

      def w_h(self, l):
            """ Weights of a hierarchical or dynamical model """
            return self.layers[l]

      def parameters(self, l):
            """ Parameters and learning rates to update a layer """
            return [{'params': [self.lastState[l]], 'lr': 100}, # state l, t
            {'params': [self.currState[l]], 'lr': 100}, # state l+1, t+dt
            {'params': [self.currState[l+1]], 'lr': 100}, # state l+1, t
            {'params': list(self.w_h(l=l).parameters()), 'lr': 100}, # top-down weights l
            {'params': [self.prec[l]], 'lr': 0.1}]  # precision l

      def predict(self, target=None, states=None):
            """ Backward pass through layers """
            states = [target] if states is None else [states[-1]]
            for w in list(reversed(self.layers)): states.append(w(states[-1]).detach())
            return list(reversed(states))

def GPC(m, l, loss=torch.square, dynamical=False):
      """ Generalized Predictive Coding optimizer """

      if dynamical: # dynamical prediction (state transition) & top-down prediction
            m_d = m.layers_d[l] # dynamical layers of this hierarchical layer
            opt = SGD(m_d.parameters(l), lr=0., momentum=0.) # dynamical optimizer
            opt.zero_grad() # reset gradients
            pred_t = m_d.w_h(l=0).forward(m.lastState[l])  # state transition in lowest layer # todo separate state_h/d
            pred_h = m_d.w_h(l=1).forward(m_d.currState[l+1]) # top-down prediction # todo higher order optimizers
            e_t = loss(pred_t - m.currState[l])
            e_h = loss((m.currState[l] - m.lastState[l]) - pred_h) # dynamical PE (state change or grad)
            if LEARN_PRECISION: torch.matmul(m.prec[l]**-1, torch.reshape(e_h, [B_SIZE, -1, 1]))
            e_total = torch.mean(e_t) + torch.mean(e_h)
      else: # hierarchical top-down prediction
            opt = SGD(m.parameters(l), lr=0., momentum=0.) # hierarchical optimizer
            opt.zero_grad()  # reset gradients
            pred_h = m.w_h(l=l).forward((m.currState[l+1])) # hierarchical prediction
            e_h = loss((m.currState[l] - pred_h)) # hierarchical PE at t
            e_t, pred_t = torch.zeros_like(e_h), torch.zeros_like(pred_h) # no transition
            if LEARN_PRECISION: torch.matmul(m.prec[l]**-1, torch.reshape(e_h, [B_SIZE, -1, 1]))
            e_total = torch.mean(e_h)

      e_total.backward() # compute gradients
      opt.step() # step variables
      return pred_h.detach().numpy(), pred_t.detach().numpy(), e_h, e_t

UPDATES = 5 # model updates per input
ACTIONS = [1 for i in range(100)] # actions in Moving MNIST
B_SIZE = 8 # batch of agents
IMAGE_SIZE = 32*32 # image size after preprocessing
LEARN_PRECISION = True

if __name__ == '__main__':

      # Moving MNIST in OpenAI gym
      env = gym.make('Mnist-s1-v0')
      env.reset()

      # hyperparameters
      l_sizes = [IMAGE_SIZE,64, 64,64, 64,64] # hierarchical layer sizes
      act_h = [torch.nn.Sigmoid()] + [torch.nn.Sigmoid() for l in l_sizes[1::2]] # hierarchical activations
      act_d = [torch.nn.Sigmoid()] + [torch.nn.Sigmoid() for l in l_sizes[1::2]] # dynamical activations

      # create hierarchical dynamical model
      PCN = Model(sizes=l_sizes, bias=True, act=act_h, bias_d=True, act_d=act_d)

      # visualization
      [err_h, err_t, preds_h, preds_t], inputs = [[[] for _ in PCN.layers] for _ in range(4)], [[]]

      for i, action in enumerate(ACTIONS): # iterate over time steps
            obs, rew, done, _ = env.step([action for b in range(B_SIZE)])
            input = (torch.Tensor(obs['agent_image'])/255).reshape([B_SIZE, -1, IMAGE_SIZE])

            if True: # reduce size
                  input = input.reshape([B_SIZE, -1, 64, 64])
                  input = torch.nn.MaxPool2d((2,2), stride=(2,2))(input).reshape([B_SIZE, -1, IMAGE_SIZE])

            # update model
            for update in range(UPDATES):
                  currState = PCN.predict(states=PCN.currState) # hierarchical prediction
                  for i in range(len(PCN.layers)-1): # update hierarchical layers
                        PCN.currState[0] = torch.tensor(input.detach().float().squeeze()).unsqueeze(1) # feed data
                        p_h, _, e_h, _ = GPC(PCN, i) # update hierarchical variables
                        _, p_d, _, e_t = GPC(PCN, i, dynamical=True) # update dynamical variables
                        if update == UPDATES-1 and i == 0: # visualization
                              inputs[0].append(input[:1])
                              preds_h[i].append(p_h[:1]), preds_t[i].append(p_d[:1])
                              err_h[i].append(e_h[:1].detach()), err_t[i].append(e_t[:1].detach())

            for i in range(len(PCN.layers)): PCN.lastState[i] = PCN.currState[i].clone().detach() # keep last state

      """ Plotting """
      sequence_video(preds_t, title="trans_predictions", plt_title="Dynamical prediction")
      sequence_video(preds_h, title="hier_predictions", plt_title="Hierarchical prediction")
      sequence_video(err_t, title="trans_errors", plt_title="Dynamical prediction error")
      sequence_video(err_h, title="hier_errors", plt_title="Hierarchical prediction error")
      sequence_video(inputs, title="inputs", plt_title="Input")

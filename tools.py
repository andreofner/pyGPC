"""
Tools for plotting and creating OpenAI gym environments from datasets
André Ofner 2021
"""

import gym
import random
import numpy as np
import sys, os, math
from gym.utils import seeding
from tensorflow import keras
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
from gym.envs.registration import register as gym_register
from GPC import B_SIZE, IMAGE_SIZE

""" Plotting helpers"""
def print_layer_variances(PCN, l=0, title="Prior"):
      print(str(title)+" Variance (batch mean): ", np.diag(np.array(PCN.covar[l].mean(dim=0).detach())).mean())
      print(str(title)+" Precision (batch mean): ", (np.diag(np.array(PCN.covar[l].mean(dim=0).detach())).mean() ** -1).round(3))
      print(str(title)+" Co-precision (batch mean): ", (np.array(PCN.covar[l].mean(dim=0).detach()).mean() ** -1).round(3))

def plot_variance_updates(variances, errors=None, title=""):
      plt.plot(variances, label="Variance (batch mean)", color="blue")
      if errors is not None:plt.plot(errors, label="Prediction error (batch mean)", color="red")
      plt.title(str(title)+"\nLayer 1")
      plt.ylabel("Magnitude (batch mean)")
      plt.xlabel("Update")
      plt.legend()
      plt.grid()
      plt.show()

def plot_thumbnails(variances, errors=None, inputs=None, datapoints=[], img_s=2, threshold = 0.2):
  inputs = np.asarray([p.detach().numpy() for p in inputs[0]]).squeeze()
  fig, ax = plt.subplots(1, 1, figsize=(10,10))
  plt.plot(variances, label="Inferred Variance", color="blue")
  for i in range(len(datapoints[:-1])):
      ax.vlines(datapoints[i], -5, 10, lw=1, color="grey", ls='dashed', alpha=0.5)
      if np.abs(errors[datapoints[i]]) > threshold:
          ax.imshow(inputs[i], extent=[datapoints[i]-(img_s//2), datapoints[i]+(img_s//2), -img_s, -0], aspect=1)
          ax.vlines(datapoints[i], errors[datapoints[i]], -img_s, lw=1, color="black")
  ax.plot(errors, label="Prediction error", color="red")
  ax.set_title("Prediction error precision (batch mean): Layer 1")
  ax.set_ylabel("Magnitude (batch mean)")
  ax.set_xlabel("Update")
  ax.set_ylim(-5,10)
  ax.legend(loc='upper right')
  plt.yticks([t for t in plt.yticks()[0] if t >= 0])
  plt.grid(axis='y')
  plt.show()

def model_sizes_summary(PCN):
      print("\nHierarchical weights: "), [print("Layer " + str(l) + ": " + str(list(s))) for l, s in
                                          enumerate(PCN.layers)];
      print("\nDynamical weights: "), [print("Layer " + str(l) + ": " + str(list(s.layers))) for l, s in
                                         enumerate(PCN.layers_d)];
      print("\nCause states: "), [print("Layer " + str(l) + ": " + str(list(s.shape))) for l, s in
                                         enumerate(PCN.curr_cause)];
      print("\nHidden states: "), [print("Layer " + str(l) + ": " + str(list(s.curr_cause[0].shape))) for l, s in
                                         enumerate(PCN.layers_d)];
      print("\nHierarchical covariances: "), [print("Layer " + str(l) + ": " + str(list(s.shape))) for l, s in
                                              enumerate(PCN.covar)];

def generate_videos(preds_h, inputs, preds_g, preds_gt, err_h, env_name, nr_videos=3, scale=1):
      preds_h = [[p.clip(min=0) for p in preds_h[0]]]
      preds_g = [[p.clip(min=0) for p in preds_g[0]]]
      preds_gt = [[p.clip(min=0) for p in preds_gt[0]]]
      for s, t in zip([preds_h, inputs, preds_g, preds_gt, err_h][:nr_videos], ['p_h', 'ins', 'p_g', 'p_gt', 'e_h'][:nr_videos]):
            sequence_video(s, t, scale=scale, env_name=str(env_name))

def visualize_covariance_matrix(PCN, skip_l=2, title=""):
      for covar, covar_type in zip([PCN.covar], [""]):
            fig, axs = plt.subplots(len(PCN.layers[::skip_l]), figsize=(5, 10))
            for i, ax in enumerate(axs):
                  l = ax.imshow(covar[i].detach().squeeze() ** -1)
                  ax.set_xticks([]);
                  ax.set_yticks([])
                  ax.set_xlabel("Layer " + str(i * skip_l + 1))
                  plt.colorbar(l, ax=ax)
            plt.suptitle(str(title)+"\nEstimated precision: " + str(covar_type))
            plt.tight_layout()
            plt.show()
            plt.close()

def plot_batch(batch, p=0, show=False, title=""):
      fig, axs = plt.subplots(4,4)
      for i in range(4):
        for j in range(4):
            try:
                  axs[i,j].imshow(batch[p].reshape([16,16]))
            except:
                  pass
            p += 1
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
      plt.tight_layout()
      if show:
          plt.show()
      else:
          plt.savefig(str(PLOT_PATH)+str(title)+".png", dpi=40)
      plt.close()

def sequence_video(data, title="", plt_title="", scale=255, plot=False, plot_video=True, env_name=""):

      try:
            predicts_plot = np.asarray([pred.squeeze() for pred in data[0]])
            predicts_plot = predicts_plot.reshape([-1, int(math.sqrt(IMAGE_SIZE)), int(math.sqrt(IMAGE_SIZE)), 1])
      except:
            predicts_plot = np.asarray([pred.detach().numpy().squeeze() for pred in data[0]])
            predicts_plot = predicts_plot.reshape([-1, int(math.sqrt(IMAGE_SIZE)), int(math.sqrt(IMAGE_SIZE)), 1])

      if plot_video: # save episode as gif
            clip = ImageSequenceClip(list(predicts_plot*scale), fps=20)
            clip.write_gif(str(PLOT_PATH) + str(title) + str(env_name) +'.gif', fps=20, verbose=False)

      if plot: # plot last frame
            plt.imshow(predicts_plot[-1])
            plt.title(str(plt_title))
            plt.colorbar()
            plt.show()


""" Moving MNIST in OpenAI Gym"""

# environment settings
ACTION_NAMES = {0:"Previous frame",1:"Next frame"} #2:"Up",3:"Down",4:"Left",5:"Right"}
ACTION_SPACE = list(ACTION_NAMES)

# optional: spatial actions
AGENT_SIZE = 32 # make smaller to move spatially
AGENT_STEP_SIZE = 1 # only for spatial actions
VIDEO_MODE = False # disables actions controlling time
OBSERVATION_SIZE = (AGENT_SIZE*2) * (AGENT_SIZE*2)  # size of observed patch

# logging and visualization settings
VERBOSE = False
PLOT_PATH = "./figures/"
try:
      os.makedirs(PLOT_PATH) # create folder
except:
      pass

def load_moving_mnist(nr_sequences=1000, test=False):
      """ Loads moving MNIST and saves to disk"""
      fpath = keras.utils.get_file("moving_mnist.npy","http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy")
      dataset = np.load(fpath)
      if not test:
            dataset = np.swapaxes(dataset, 0, 1) # Swap the axes representing the number of frames and number of data samples.
            dataset = dataset[:nr_sequences, ...] # select train data todo improve dataset split
            dataset = np.expand_dims(dataset, axis=-1) # Add a channel dimension since the images are grayscale.
      else:
            dataset = np.swapaxes(dataset, 0, 1) # Swap the axes representing the number of frames and number of data samples.
            dataset = dataset[nr_sequences:2*nr_sequences, ...] # select test data todo improve dataset split
            dataset = np.expand_dims(dataset, axis=-1) # Add a channel dimension since the images are grayscale.
      return dataset

class MnistEnv(gym.Env):
      """ See https://github.com/jbinas/gym-mnist  for static version"""
      def __init__(self, num_digits=2, dataset="moving_mnist", max_steps=100, noise_scale=0., seed=1337, test=False):
            self.shape = 28, num_digits * 28
            self.num_sym = num_digits
            self.max_steps = max_steps
            self.noise_scale = noise_scale # weighted gaussian noise on observation
            self.seed(seed=seed)
            self.dataset = dataset
            if self.dataset == "moving_mnist": # moving MNIST dataset
                  class MovingAgent:
                        """A single agent seeing a (partial) frame and moves across time (and space)"""
                        def __init__(self, dataset_size):
                              self.agent_size = AGENT_SIZE   # size of area currently covered by the agent
                              self.pos_x = 32   # horizontal position of agent
                              self.pos_y = 32   # vertical position of agent
                              self.sequence_state = random.randint(0,dataset_size-1)  # ID of currently observed sequence
                              self.time_state = 0   # position in current sequence
                              self.step_size_xy = AGENT_STEP_SIZE # how large steps within frames are

                  self.number_of_agents = B_SIZE
                  self.data = load_moving_mnist(test=test)
                  self.shape = 64, 64
                  self.nr_sequences = 1000 # sequences in dataset
                  self.moving_agents = [MovingAgent(self.data.shape[0]) for _ in range(self.number_of_agents)]
            else:
                  print("Please select a valid dataset!")
                  sys.exit()

            # the first action is the null action
            self.action_space = gym.spaces.Discrete(1 + 2 * self.num_sym)

            self.observation_space = gym.spaces.Box(
                  low=0,
                  high=1,
                  shape=self.shape + (1,),
                  dtype='uint8'
            )
            self.observation_space = gym.spaces.Dict({
                  'image': self.observation_space
            })

            self.reward_range = (0, 1)

      def step(self, actions):
            self.step_count += 1

            if VIDEO_MODE and self.step_count % 20 == 0:
                  for a in self.moving_agents:
                        a.time_state += 1
                        if a.time_state + 1 >  20:   # check if current sequence is done
                              if a.sequence_state + 1 >= self.nr_sequences:   # reset if all sequences have passed
                                    a.sequence_state = 0
                              a.sequence_state += 1
                              a.time_state = 0

            if self.dataset == "moving_mnist":
                  for curr_agent, action in zip(self.moving_agents, actions):
                        if action == 0:   # Go to previous Frame
                              curr_agent.time_state -= 1
                              if curr_agent.time_state < 0:   # check if reached beginning of sequence
                                    curr_agent.time_state = 0
                        elif action == 1:   # Go to next Frame
                              curr_agent.time_state += 1
                              if curr_agent.time_state + 1 >  20:   # check if current sequence is done
                                    if curr_agent.sequence_state + 1 >= self.nr_sequences:   # reset if all sequences have passed
                                          curr_agent.sequence_state = 0
                                    curr_agent.sequence_state += 1
                                    curr_agent.time_state = 0
                        elif action == 2:   # Move up
                              curr_agent.pos_y += curr_agent.step_size_xy
                        elif action == 3:   # Move down
                              curr_agent.pos_y -= curr_agent.step_size_xy
                        elif action == 4:   # Move left
                              curr_agent.pos_x -= curr_agent.step_size_xy
                        elif action == 5:   # Move right
                              curr_agent.pos_x += curr_agent.step_size_xy
                        # make sure agent position is within spatial bounds of frame
                        curr_agent.pos_x = min(curr_agent.pos_x, self.shape[0] - curr_agent.agent_size - 1)
                        curr_agent.pos_x = max(curr_agent.pos_x, curr_agent.agent_size)
                        curr_agent.pos_y = min(curr_agent.pos_y, self.shape[1] - curr_agent.agent_size - 1)
                        curr_agent.pos_y = max(curr_agent.pos_y, curr_agent.agent_size)

            done = self.step_count >= self.max_steps
            obs = self.gen_obs()
            return obs, 0, done, {}

      @property
      def steps_remaining(self):
            return self.max_steps - self.step_count

      def reset(self):
            ''' generate new world '''
            self.step_count = 0
            self.state = np.random.randint(10, size=(self.num_sym))
            return self.gen_obs()

      @property
      def observed_state_sequential(self):
            # create an image of the env for visualization
            img = self.data[self.moving_agents[0].sequence_state, self.moving_agents[0].time_state]
            img = img.astype(np.float64) + self.noise_scale * np.random.rand(img.shape[0], img.shape[1], 1) * 255 - 127
            img = np.clip(img, 0, 255).astype(int)
            img_RGB = np.repeat(img, 3, 2)
            raw_frame = img

            # process the observations for each agent
            ca_obs = []
            for ca in self.moving_agents:
                  # get frame from sequence
                  img = self.data[ca.sequence_state, ca.time_state]
                  agentobs = np.zeros_like(img)
                  state = [ca.time_state, ca.pos_x, ca.pos_y, ca.agent_size, ca.step_size_xy]
                  ca_obs.append([img, np.zeros_like(img_RGB), np.zeros_like(raw_frame), state, agentobs])

            return ca_obs

      def _reward(self):
            ''' Compute the reward to be given upon success '''
            return 0. # no reward function

      def gen_obs(self):
            if self.dataset == "mnist":
                  return {'image': self.observed_state.reshape(*self.shape, 1)}
            elif self.dataset == "moving_mnist":
                  agentobs_scaled_list = []
                  agentobs_list = []
                  states_list = []
                  for ca, ca_obs in zip(self.moving_agents, self.observed_state_sequential):
                        agentobs, img_RGB, raw_frame, state, agentobs_scaled = ca_obs
                        img_RGB = img_RGB.reshape(*self.shape, 3)
                        raw_frame = raw_frame.reshape(*self.shape, 1)
                        agentobs_scaled_list.append(agentobs_scaled.reshape(*self.shape, 1))
                        agentobs_list.append(agentobs.reshape(ca.agent_size*2,ca.agent_size*2,1))
                        states_list.append(state)
                  return {'raw_frame': raw_frame, # for environment visualization
                              'image': img_RGB, # for environment visualization
                              'agent_image': agentobs_list, # what each agent actually sees
                              'agent_image_scaled': agentobs_scaled_list, # for agent visualization
                              'state': states_list} # low-dimensional state for each agent with its position and other info
            else:
                  return None

      def seed(self, seed=1337):
            # Seed the random number generator
            self.np_random, _ = seeding.np_random(seed)
            return [seed]


class MnistEnv1(MnistEnv):
      def __init__(self):
            super().__init__(test=False)

class MnistEnv2(MnistEnv):
      def __init__(self):
            super().__init__(test=True)


""" Register as gym environment"""
gym_register(id='Mnist-Train-v0', entry_point='tools:MnistEnv1',  reward_threshold=900)
gym_register(id='Mnist-Test-v0', entry_point='tools:MnistEnv2',  reward_threshold=900)

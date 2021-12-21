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
from GPC import BATCH_SIZE, IMAGE_SIZE

""" Plotting helpers"""

def sequence_video(data, title="", plt_title="", scale=255, plot=False, plot_video=True):

      try:
            predicts_plot = np.asarray([pred.squeeze() for pred in data[0]])
            predicts_plot = predicts_plot.reshape([-1, int(math.sqrt(IMAGE_SIZE)), int(math.sqrt(IMAGE_SIZE)), 1])
      except:
            predicts_plot = np.asarray([pred.detach().numpy().squeeze() for pred in data[0]])
            predicts_plot = predicts_plot.reshape([-1, int(math.sqrt(IMAGE_SIZE)), int(math.sqrt(IMAGE_SIZE)), 1])

      if plot_video: # save episode as gif
            clip = ImageSequenceClip(list(predicts_plot*scale), fps=20)
            clip.write_gif(str(PLOT_PATH) + str(title) + '.gif', fps=20, verbose=False)

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

def load_moving_mnist(nr_sequences=1000):
      """ Loads moving MNIST and saves to disk"""
      fpath = keras.utils.get_file("moving_mnist.npy","http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy")
      dataset = np.load(fpath)
      dataset = np.swapaxes(dataset, 0, 1) # Swap the axes representing the number of frames and number of data samples.
      dataset = dataset[:nr_sequences, ...] # We'll pick out 1000 of the 10000 total examples and use those.
      dataset = np.expand_dims(dataset, axis=-1) # Add a channel dimension since the images are grayscale.
      return dataset

class MnistEnv(gym.Env):
      """ See https://github.com/jbinas/gym-mnist  for static version"""
      def __init__(self, num_digits=2, dataset="moving_mnist", max_steps=100, noise_scale=0., seed=1337):
            self.shape = 28, num_digits * 28
            self.num_sym = num_digits
            self.max_steps = max_steps
            self.noise_scale = noise_scale # weighted gaussian noise on observation
            self.seed(seed=seed)
            self.dataset = dataset
            if self.dataset == "moving_mnist": # moving MNIST dataset
                  class MovingAgent:
                        """A single agent seeing a (partial) frame and moves across time (and space)"""
                        def __init__(self):
                              self.agent_size = AGENT_SIZE   # size of area currently covered by the agent
                              self.pos_x = 32   # horizontal position of agent
                              self.pos_y = 32   # vertical position of agent
                              self.sequence_state = random.randint(0,BATCH_SIZE-1)  # ID of currently observed sequence
                              self.time_state = 0   # position in current sequence
                              self.step_size_xy = AGENT_STEP_SIZE # how large steps within frames are

                  self.number_of_agents = BATCH_SIZE
                  self.data = load_moving_mnist()
                  self.shape = 64, 64
                  self.nr_sequences = 1000 # sequences in dataset
                  self.moving_agents = [MovingAgent() for _ in range(self.number_of_agents)]
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
      """ Single moving digit """
      def __init__(self):
            super().__init__(num_digits=1)


""" Register as gym environment"""
gym_register(id='Mnist-s1-v0', entry_point='tools:MnistEnv1',  reward_threshold=900)

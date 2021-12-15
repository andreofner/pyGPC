""" Plotting helpers"""
import random
import torch
import gym
import sys
import numpy as np
import matplotlib.cm as cm
from gym.utils import seeding
from gym.envs.registration import register as gym_register
from tensorflow import keras
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip

def plot_episode(frames, agent_frames, states_batch, max_visualized_agents=8, title=""):
      """ save episode video as gif """
      states_batch = states_batch.swapaxes(0,1)
      clip = ImageSequenceClip(list(frames), fps=20)
      clip.write_gif(str(PLOT_PATH)+str(title)+'.gif', fps=20, verbose=False)

def plot_predictions(state_model, input, state, reconstruction, action, agent_size, layer2_pred=None, title=""):
      if layer2_pred is None: # single layer model
            predicted_state = state_model(torch.cat([torch.vstack([state]), torch.vstack([action])], dim=1))
            fig, axs = plt.subplots(1, 4)
            axs[0].imshow(input)
            axs[0].set_title("Input")
            axs[1].imshow(reconstruction.detach().numpy().reshape([agent_size * 2, agent_size * 2, 1]))
            axs[1].set_title("Input reconstruction")
            axs[2].imshow(state.detach().reshape([-1,1]))
            axs[2].set_title("Encoded state")
            for (j, i), label in np.ndenumerate(state.detach().reshape([-1,1])):
                  axs[2].text(i, j, label, ha='center', va='center')
            axs[3].imshow(predicted_state.detach().reshape([-1,1]))
            axs[3].set_title("Predicted state")
            for (j, i), label in np.ndenumerate(predicted_state.detach().reshape([-1,1])):
                  axs[3].text(i, j, label, ha='center', va='center')
            plt.tight_layout()
            plt.savefig(str(PLOT_PATH)+'predictions'+str(title)+'.png')
            plt.close()
      else:
            predicted_state = state_model(torch.cat([torch.vstack([state]), torch.vstack([action])], dim=1))
            fig, axs = plt.subplots(1, 5)
            axs[0].imshow(input)
            axs[0].set_title("Input")
            axs[1].imshow(reconstruction.detach().numpy().reshape([agent_size * 2, agent_size * 2, 1]))
            axs[1].set_title("Input\nreconstruction")
            axs[2].imshow(state.detach().reshape([-1,1]))
            axs[2].set_title("Encoded\nstate")
            axs[3].imshow(predicted_state.detach().reshape([-1,1]))
            axs[3].set_title("Predicted\nstate")
            axs[4].imshow(layer2_pred.detach().reshape([-1,1]))
            axs[4].set_title("Top-down\nprediction")
            plt.tight_layout()
            plt.savefig(str(PLOT_PATH)+'predictions'+str(title)+'.png')
            plt.close()

def plot_predictions_list(examples_to_plot, obs_, predicted_stimulus, title_):
      fig, axs = plt.subplots(max(2, examples_to_plot), 2)
      for batch_pos in range(examples_to_plot):
            axs[batch_pos, 0].imshow(obs_['STIM'][batch_pos], interpolation="nearest")
            axs[batch_pos, 0].set_title("Target")
            axs[batch_pos, 1].imshow(predicted_stimulus[batch_pos], interpolation="nearest")
            axs[batch_pos, 1].set_title("Prediction")
      plt.tight_layout()
      plt.savefig(str(PLOT_PATH)+'pred' + str(0) + '_stimulus' + str(title_) + '.png')
      plt.close()

def visualize_episode_video(frames, agent_frames, states):
      # visualize episode as video
      frames = np.asarray(frames)
      agent_frames = np.asarray(agent_frames)
      states = np.asarray(states)
      plot_episode(frames, agent_frames, states)

def visualize_losses(autoencoder_loss, state_loss, state_loss_l2=None):
      # plot losses
      if state_loss_l2 is None: # single layer DAI model
            fig, axs = plt.subplots(1, 2)
            axs[0].plot(autoencoder_loss)
            axs[0].set_title("Reconstruction loss")
            axs[1].plot(state_loss)
            axs[1].set_title("State prediction loss")
            plt.tight_layout()
            plt.savefig(str(PLOT_PATH)+'losses.png')
            plt.close()
      else: # 2 layer DAI model
            fig, axs = plt.subplots(1, 3)
            axs[0].plot(autoencoder_loss)
            axs[0].set_title("Reconstruction loss L1")
            axs[1].plot(state_loss)
            axs[1].set_title("State prediction loss L1")
            axs[2].plot(state_loss_l2)
            axs[2].set_title("State prediction loss L2")
            plt.tight_layout()
            plt.savefig(str(PLOT_PATH)+'losses.png')
            plt.close()

def plot_action_probabilities(action_probs):
      if len(action_probs) >  1000:
            plt.imshow(np.asarray(action_probs[0::int(len(action_probs)/100)]).T, aspect='50') # subsampling
      else:
            plt.imshow(np.asarray(action_probs).T, aspect='50')
      plt.title("Action selection probability:")
      plt.yticks(range(len(ACTION_SPACE)), ACTION_NAMES.values())
      plt.xlabel("Update")
      plt.xticks([])
      plt.tight_layout()
      plt.savefig(str(PLOT_PATH)+'action_probabilities.png')
      plt.close()

def video(frames, title="", plot_path=None):
      # save data seen by agent in this episode as gif
      clip = ImageSequenceClip(list(frames), fps=20)
      if plot_path is None:
            clip.write_gif(str(PLOT_PATH)+''+str(title)+'.gif', fps=20, verbose=False)
      else:
            clip.write_gif(str(plot_path)+''+str(title)+'.gif', fps=20, verbose=False)





""" Moving MNIST in OpenAI Gym"""

# environment settings
AGENT_SIZE = 32
NOISE_SCALE = 0.0
AGENT_STEP_SIZE = 1
# ACTION_NAMES = {0:"Previous frame",1:"Next frame",2:"Up",3:"Down",4:"Left",5:"Right"}
ACTION_NAMES = {2: "Up", 3: "Down", 4: "Left", 5: "Right"}
ACTION_SPACE = list(ACTION_NAMES)
VIDEO_MODE = True  # disable the time actions and the same moving MNIST frames to all DAI agents

# training settings
OBSERVATION_SIZE = (AGENT_SIZE * 2) * (AGENT_SIZE * 2)  # size of observed patch, the input for the autoencoder
STATE_SIZE = int(OBSERVATION_SIZE / 4)  # Size of observed states -->  amount of "compression"
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
LEARNING_RATE_AUTOENCODER = 1e-4
LEARNING_RATE_LAYER2 = 1e-4
DROPOUT_AE = 0.0
DROPOUT_STATE = 0.0

# logging and visualization settings
VERBOSE = False
PLOT_PATH = "./figures/"

def load_moving_mnist(plot=False, nr_sequences=1000):
      """ Loads moving MNIST and saves to disk"""
      fpath = keras.utils.get_file("moving_mnist.npy","http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy")
      dataset = np.load(fpath)
      dataset = np.swapaxes(dataset, 0, 1) # Swap the axes representing the number of frames and number of data samples.
      dataset = dataset[:nr_sequences, ...] # We'll pick out 1000 of the 10000 total examples and use those.
      dataset = np.expand_dims(dataset, axis=-1) # Add a channel dimension since the images are grayscale.
      if plot:
            print("Moving MNIST shape: " + str(dataset.shape))
            fig, axes = plt.subplots(4, 5, figsize=(10, 8))
            data_choice = 0 # select sequence to plot
            for idx, ax in enumerate(axes.flat):
                  ax.imshow(np.squeeze(dataset[data_choice][idx]), cmap="gray")
                  ax.set_title(f"Frame {idx + 1}")
                  ax.axis("off")
            plt.show()
      return dataset

class MnistEnv(gym.Env):
      """ See https://github.com/jbinas/gym-mnist  for static version"""
      def __init__(self, num_digits=2, dataset="moving_mnist", max_steps=100, noise_scale=NOISE_SCALE, seed=1337):
            self.shape = 28, num_digits * 28
            self.num_sym = num_digits
            self.max_steps = max_steps
            self.noise_scale = noise_scale # weighted gaussian noise on observation
            self.seed(seed=seed)
            self.dataset = dataset
            if self.dataset == "moving_mnist": # moving MNIST dataset
                  class MovingAgent:
                        """A single agent seeing a small input area and moves across time and space."""
                        def __init__(self):
                              self.agent_size = AGENT_SIZE   # size of area currently covered by the agent
                              self.pos_x = 32   # horizontal position of agent
                              self.pos_y = 32   # vertical position of agent
                              self.sequence_state = 0   # ID of currently observed sequence
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
                        # make sure agent position is within bounds of environment
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
            # create an image of the env for visualization purposes
            # todo when moving through time this visualization works only with a single agent
            img = self.data[self.moving_agents[0].sequence_state, self.moving_agents[0].time_state]
            img = img.astype(np.float64) + self.noise_scale * np.random.rand(img.shape[0], img.shape[1], 1) * 255 - 127
            img = np.clip(img, 0, 255).astype(int)
            img_RGB = np.repeat(img, 3, 2)
            raw_frame = img

            # now process the observations for each agent
            ca_obs = []

            for ca in self.moving_agents:
                  # get frame from sequence
                  img = self.data[ca.sequence_state, ca.time_state]
                  # add noise
                  img = img.astype(np.float64) + self.noise_scale * np.random.rand(img.shape[0], img.shape[1], 1) * 255 - 127
                  img = np.clip(img, 0, 255).astype(int)
                  # mask the area seen by the agent currently
                  agentobs = np.zeros_like(img)

                  if False:
                        observed_patch = img[ca.pos_x - ca.agent_size:ca.pos_x + ca.agent_size,ca.pos_y - ca.agent_size:ca.pos_y + ca.agent_size]
                        agentobs[ca.pos_x-ca.agent_size:ca.pos_x+ca.agent_size, ca.pos_y-ca.agent_size:ca.pos_y+ca.agent_size] = observed_patch
                        # mark agent position
                        img_RGB[ca.pos_x-ca.agent_size:ca.pos_x+ca.agent_size:,ca.pos_y-ca.agent_size, :] = 255
                        img_RGB[ca.pos_x-ca.agent_size:ca.pos_x+ca.agent_size:,ca.pos_y+ca.agent_size, :] = 255
                        img_RGB[ca.pos_x-ca.agent_size,ca.pos_y-ca.agent_size:ca.pos_y+ca.agent_size, :] = 255
                        img_RGB[ca.pos_x+ca.agent_size,ca.pos_y-ca.agent_size:ca.pos_y+ca.agent_size+1, :] = 255
                        # also return the spatial position, temporal position and agent characteristics
                        state = [ca.time_state, ca.pos_x, ca.pos_y, ca.agent_size, ca.step_size_xy]
                        ca_obs.append([observed_patch, img_RGB, raw_frame, state, agentobs])
                  else:
                        state = [ca.time_state, ca.pos_x, ca.pos_y, ca.agent_size, ca.step_size_xy]
                        ca_obs.append([img, img_RGB, raw_frame, state, agentobs])

            return ca_obs

      def _reward(self):
            ''' Compute the reward to be given upon success '''
            return 0. #1 - 0.9 * (self.step_count / self.max_steps)

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
            super().__init__(num_digits=1)

class MnistEnv2(MnistEnv):
      def __init__(self):
            super().__init__(num_digits=2)

class MnistEnv3(MnistEnv):
      def __init__(self):
            super().__init__(num_digits=3)

""" Register MNIST as fym environment"""

env_list = []
def register(id, entry_point, reward_threshold=900):
      assert id.startswith("Mnist-")
      assert id not in env_list
      # Register the environment with OpenAI gym
      gym_register(id=id, entry_point=entry_point,  reward_threshold=reward_threshold)
      # Add the environment to the set
      env_list.append(id)


register(
      id='Mnist-s1-v0',
      entry_point='gym_mnist.mnist:MnistEnv1'
)

register(
      id='Mnist-s2-v0',
      entry_point='gym_mnist.mnist:MnistEnv2'
)

register(
      id='Mnist-s3-v0',
      entry_point='gym_mnist.mnist:MnistEnv3'
)

"""
Differentiable Generalized Predictive Coding
André Ofner 2021

Hierarchical-dynamical prediction of local and global trajectories on Moving MNIST
"""

import torch
import numpy as np
from PIL import Image
from torch.nn import Parameter as P
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from tools import *
from MovingMNIST import *

plt.style.use(['seaborn'])


def plot_graph():
    lines, lstyles = [], ['-', '--', '-.', ':']

    # hierarchical prediction
    if True:
        lines += plt.plot(np.asarray(errors)[:, 0], color="red", linestyle=lstyles[0])
        lines += plt.plot(np.asarray(errors)[:, 1], color="red", linestyle=lstyles[1])
        lines += plt.plot(np.asarray(errors)[:, 2], color="red", linestyle=lstyles[2])
        lines += plt.plot(np.asarray(cov_h)[:, 0], color="green", linestyle=lstyles[0])
        lines += plt.plot(np.asarray(cov_h)[:, 1], color="green", linestyle=lstyles[1])
        lines += plt.plot(np.asarray(cov_h)[:, 2], color="green", linestyle=lstyles[2])

    # dynamical prediction
    if True:
        lines += plt.plot(np.asarray(errors_d1), color="black", linestyle=lstyles[0])
        lines += plt.plot(np.asarray(errors_d2), color="black", linestyle=lstyles[1])
        lines += plt.plot(np.asarray(errors_d3), color="black", linestyle=lstyles[2])
        lines += plt.plot(np.asarray(cov_d1), color="blue", linestyle=lstyles[0])
        lines += plt.plot(np.asarray(cov_d2), color="blue", linestyle=lstyles[1])
        lines += plt.plot(np.asarray(cov_d3), color="blue", linestyle=lstyles[2])

    # generalized coordinates
    if True:
        lines += plt.plot(np.asarray(cov_g1).mean(-1), color="yellow", linestyle=lstyles[0])
        lines += plt.plot(np.asarray(cov_g1).mean(-1), color="yellow", linestyle=lstyles[1])
        lines += plt.plot(np.asarray(cov_g1).mean(-1), color="yellow", linestyle=lstyles[2])
        lines += plt.plot(np.asarray(err_g1).mean(-1), color="black", linestyle=lstyles[0])
        lines += plt.plot(np.asarray(err_g2).mean(-1), color="black", linestyle=lstyles[1])
        lines += plt.plot(np.asarray(err_g3).mean(-1), color="black", linestyle=lstyles[2])

    plt.title("Hierarchical and dynamical prediction errors")
    plt.legend(lines,
               [f"Prediction error L{l + 1}" for l in range(3)] +
               [f"Error variance L{l + 1}" for l in range(3)] +
               [f"Dynamical prediction error L{l + 1}" for l in range(3)] +
               [f"Dynamical error variance L{l + 1}" for l in range(3)] +
               [f"Generalized motion error variance L{l + 1}" for l in range(3)]+
               [f"Generalized motion prediction error L{l + 1}"for l in range(3)])
    plt.xlabel("Update")
    plt.ylabel("Magnitude")
    #plt.yscale("log", base=10)
    plt.show()

def plot_2D(img_size=64, title="", plot=True, examples=1):
    if plot: fig, axs = plt.subplots(examples, 4)
    preds = []
    for example in range(examples):
        for ax, start_layer in enumerate(reversed(range(2, 5))):
            pred = net.predict_from(start_layer=start_layer)
            if plot:
                axs[example, ax].imshow(pred[example].reshape([img_size, img_size]))
                axs[example, ax].set_title("Layer " + str(3 - ax))
                axs[example, ax].set_xticks([]);
                axs[example, ax].set_yticks([])
            preds.append(pred)
        input = net.g_observation.states.cause_state[example].detach()
        if plot:
            error = pred[example] - input[example].detach().numpy()

            """
            # covar
            covar_h = net.layers[0].states.cause_covar.detach()
            projectedvar = torch.matmul(covar_h[example] ** -1, torch.from_numpy(error.reshape([img_size * img_size, 1])))
            axs[example, -3].imshow(projectedvar.reshape([img_size, img_size]))
            axs[example, -3].set_title("Precision\nLearning")
            axs[example, -3].set_xticks([]);
            axs[example, -2].set_yticks([])
            """

            if False:
                # slow covar
                covar_h_slow = net.layers[0].states.cause_covar_slow.detach()
                projectedvar_slow = covar_h_slow ** -1 * torch.ones([img_size * img_size, 1])
                axs[example, -2].imshow(projectedvar_slow.reshape([img_size, img_size]))
                axs[example, -2].set_title("Precision\nInference")
                axs[example, -2].set_xticks([]);
                axs[example, -2].set_yticks([])

            axs[example, -1].imshow(input.reshape([img_size, img_size]))
            axs[example, -1].set_title("Observation")
            axs[example, -1].set_xticks([]);
            axs[example, -1].set_yticks([])
    if plot:
        plt.suptitle(str(title))
        plt.tight_layout()
        plt.show()

    return input, preds


class GPC_state(torch.nn.Module):
    def __init__(self, b_size, n_states, g_coords, full_covar=False, var_prior=1., covar_prior=100000.):
        """
        Cause and hidden states of a layer
        """
        super().__init__()

        # cause states and hidden states
        self.register_parameter(name="cause_state",
                                param=P(torch.ones([b_size, n_states * g_coords]).requires_grad_()))
        self.register_parameter(name="hidd_state",
                                param=P(torch.ones([b_size, n_states * g_coords]).requires_grad_()))
        torch.nn.init.xavier_uniform_(self.cause_state)  # initialize with random prior
        torch.nn.init.xavier_uniform_(self.hidd_state)  # initialize with random prior

        # cause & hidden state error covariance: full covariance matrix for each batch element (~inference)
        if full_covar:
            self.register_parameter(name="cause_covar",
                                    param=P(torch.eye(self.cause_state.shape[-1]).repeat([b_size, 1, 1])
                                            * (var_prior - covar_prior) + covar_prior))
            self.register_parameter(name="hidd_covar",
                                    param=P(torch.eye(self.hidd_state.shape[-1]).repeat([b_size, 1, 1])
                                            * (var_prior - covar_prior) + covar_prior))

        # cause & hidden state error covariance: scalar for entire batch (~learning)
        self.register_parameter(name="cause_covar_slow", param=P(torch.tensor([var_prior])))
        self.register_parameter(name="hidd_covar_slow", param=P(torch.tensor([var_prior])))

        # state prediction error
        self.cause_error = torch.zeros_like(self.cause_state)
        self.hidd_error = torch.zeros_like(self.hidd_state)

        # temporary variables for state motion
        self.cause_previous = self.cause_state.detach()
        self.hidd_previous = self.hidd_state.detach()
        self.cause_change = self.cause_state.detach()*0
        self.hidd_change = self.hidd_state.detach()*0

        # parameter groups
        self.params_state = [self.cause_state, self.hidd_state]
        if full_covar:
            self.params_covar = [self.cause_covar, self.hidd_covar]
        self.params_covar = [self.cause_covar_slow, self.hidd_covar_slow]

    def forward(self):
        # Return the full state [causes, hiddens]
        return self.state

    def change(self):
        # difference between current and previous states
        self.cause_change = self.cause_state - self.cause_previous.detach()
        self.hidd_change = self.hidd_state - self.hidd_previous.detach()

        self.cause_previous = self.cause_state.clone().detach()
        self.hidd_previous = self.hidd_state.clone().detach()

        return self.cause_change, self.hidd_change

    def predstate(self, n_states_h=None, g_coords_h=None, n_states_d=None, g_coords_d=None):
        # View on subset of states used for outgoing prediction
        return [self.cause_state[:, :(n_states_h * g_coords_h)], self.hidd_state[:, :(n_states_d * g_coords_d)]]

    def precision(self):
        # Precision is the inverse of the estimated error covariance
        return self.covar**-1


class GPC_layer(torch.nn.Module):
    def __init__(self, h_states=1, d_states=0, g_coords=1, lower=None, b_size=1, protect_states=True):
        """
        Generalized predictive coding layer
        """
        super().__init__()

        # lower GPC layer
        self.lower = lower

        # layer settings
        self.protect_states = protect_states  # don't overwrite lower cause states when predicting
        self.g_coords = g_coords  # highest order of encoded state motion
        self.n_cause_states = h_states  # observable part of states
        self.n_hidd_states = d_states  # unobservable part of states
        self.n_predstates_h = self.n_cause_states   # amount of cause states used for outgoing prediction
        self.n_predstates_d = self.n_hidd_states   # amount of hidden states used for outgoing prediction
        self.n_states = self.n_hidd_states + self.n_cause_states

        # cause and hidden states x, v
        self.states = GPC_state(b_size, self.n_cause_states, self.g_coords)

        # helpful views for hierarchical predictions
        self.pred = None  # outgoing prediction

        # hierarchical weights connect hierarchical layers
        self.net_h = None
        if self.lower is not None:
            n_predstates = self.n_predstates_h * self.g_coords + self.n_predstates_d * self.g_coords
            self.net_h = torch.nn.Linear(n_predstates, # todo use only x, not its higher coords
                                         self.lower.n_cause_states * self.lower.g_coords, False)

        # dynamical weights couple state motion
        self.net_d = torch.nn.Linear(self.n_cause_states * self.g_coords,
                                     self.n_cause_states * min(self.g_coords, 1), False)

        # generalized coordinates encode orders of state motion
        self.net_g = None
        #if self.g_coords > 1: # todo remove
        #    self.net_g = [GPC_layer(h_states=3, d_states=0, p_states=3, g_coords=1, lower=None, b_size=b_size)
        #                  for _ in range(self.n_states)]  # one network per state

        # parameter groups and optimizers
        self.opts = []
        self.params_weight = torch.nn.ModuleList([self.net_d, self.net_g, self.net_h])
        self.opts.append(torch.optim.SGD(self.params_weight.parameters(), lr=LR_WEIGHTS))  # hierarchical weights
        self.opts.append(torch.optim.SGD(self.states.params_state, lr=LR_STATES))  # states used for prediction
        self.opts.append(torch.optim.SGD(self.states.params_covar, lr=LR_PRECISION))  # states used for prediction
        if self.lower is not None: # states receiving the prediction
            self.opts.append(torch.optim.SGD(self.lower.states.params_state, lr=LR_STATES))
            self.opts.append(torch.optim.SGD(self.lower.states.params_covar, lr=LR_PRECISION))


    def state_parameters(self, recurse=False):
        return self.parameters(recurse=recurse)

    def forward(self, prior=None):
        """
        Hierarchical prediction
        """

        # [causes, units] used for predictions
        predstate = self.states.predstate(n_states_h=self.n_predstates_h, g_coords_h=self.g_coords,
                                       n_states_d=self.n_predstates_d, g_coords_d=self.g_coords)

        # optionally overwrite cause prior used for prediction
        if prior is not None:
            predstate[0] = prior.detach()

        self.pred = self.net_h(torch.cat(predstate,1))

        # optionally overwrite cause state in lower layer with is prediction
        if not self.protect_states:
            self.lower.states.cause_state = P(self.pred.detach())

        return self.pred.detach()


    def infer(self, learn_covar=False, predict_change=True):
        """
        Update states and weights
        """

        learn_covar = True # todo fix

        for opt in self.opts:
            opt.zero_grad()

        if self.lower is not None:

            # select hierarchical prediction target
            target = self.lower.states.cause_state  # inferred cause state
            if predict_change:
                target = self.lower.states.cause_change  # difference between inferred cause states

            # hierarchical prediction error
            err = (self.pred - target).abs().unsqueeze(-1)+1
            #error_fast = 0. #(err * torch.matmul(self.lower.states.cause_covar**-1, err)).mean([1,2]) # todo
            error = (err * self.lower.states.cause_covar_slow**-1 * err).squeeze()
            error.backward(gradient=torch.ones_like(error))

            # freeze learned precision during inference
            if not learn_covar:
                self.lower.states.cause_covar_slow.grad.zero_()

            for opt in self.opts:
                opt.step()

            # precision decay
            ch = P((self.lower.states.cause_covar_slow - LR_PRECISION * self.lower.states.cause_covar_slow).detach())
            #cd = P((self.lower.states.hidd_covar - LR_PRECISION * self.lower.states.hidd_covar).detach())
            covars = [ch.detach(), ch.detach()]  # todo cd

            if learn_covar:
                if ch.min() > 1: self.lower.states.cause_covar_slow = ch
                #if cd.min() > 1: self.lower.states.hidd_covar = cd

            return err.mean(), [c.mean() for c in covars] # c[0].diagonal() # todo return weighted error

    def to_discrete(self, prior):
        """
        Convert a generalized state to a discrete sequence
        """
        return prior


class GPC_net(torch.nn.Module):
    def __init__(self, b_size=1, dynamical_net=True, var_prior=1.,
                 cause_sizes=[32,32,32,32], hidden_sizes=[32,32,32,32], obs_layer=None):
        """
        Generalized predictive coding network
        """
        super().__init__()

        # specify network type: hierarchical or dynamical
        self.dynamical_net = dynamical_net

        # input layer
        self.g_observation = GPC_layer(lower=None, b_size=b_size,
                                       h_states=cause_sizes[0], d_states=hidden_sizes[0])
        if obs_layer is None:  # sensory observation
            self.output_layer = GPC_layer(lower=self.g_observation, b_size=b_size,
                                          h_states=cause_sizes[1], d_states=hidden_sizes[1])
        else:  # state observation
            self.output_layer = GPC_layer(lower=obs_layer, b_size=b_size,
                                          h_states=cause_sizes[1], d_states=hidden_sizes[1])

        # layers
        self.hidden_layer = GPC_layer(lower=self.output_layer, b_size=b_size,
                                      h_states=cause_sizes[2], d_states=hidden_sizes[2])
        self.cause_layer = GPC_layer(lower=self.hidden_layer, b_size=b_size,
                                     h_states=cause_sizes[3], d_states=hidden_sizes[3])
        self.layers = [self.g_observation, self.output_layer, self.hidden_layer, self.cause_layer]

        # summary of cause and hidden states in all layers
        self.cause_state, self.hidd_state = [], []
        for l, layer in enumerate(self.layers):
            self.cause_state.append(layer.states.cause_state)
            self.hidd_state.append(layer.states.hidd_state)

        # dynamical weights
        if self.dynamical_net:
            c_size = torch.cat(self.cause_state, 1).shape[1]
            h_size = torch.cat(self.hidd_state, 1).shape[1]
            h_motion_size = torch.cat(self.hidd_state[1:], 1).shape[1]
            self.net_d = torch.nn.Linear(c_size+h_size, h_motion_size, False) # dynamical weights
            self.register_parameter(name="covar_d", param=P(torch.tensor([var_prior])))  # dynamical error covariance
            self.opt_d = torch.optim.SGD(self.net_d.parameters(), lr=LR_WEIGHTS)
            self.opts = [self.opt_d]
            for layer in self.layers[1:]:
                self.opts + layer.opts  # state optimizers from all layers # todo delete weights opts?

    def forward(self):
        """
        Hierarchical prediction through all layers
        """
        self.cause_layer.forward()  # prediction in generalized coordinates
        self.hidden_layer.forward()  # prediction in generalized coordinates
        self.output_layer.forward()  # sensory prediction in generalized coordinates
        prior = self.g_observation.to_discrete(self.output_layer.pred)  # sensory prediction in discrete samples
        return prior

    def predict_dynamical(self):
        """
        Dynamical prediction through all layers
        """
        cause = torch.cat(self.cause_state, 1)  # cause states from all layers
        hidden = torch.cat(self.hidd_state, 1)  # hidden states from all layers
        pred = self.net_d(torch.cat([cause, hidden], 1))  # predicted hidden state motion
        return pred.detach()

    def forward_dynamical(self):
        """
        Dynamical prediction and optimization through all layers
        """

        for opt in self.opts: opt.zero_grad()

        cause = torch.cat(self.cause_state, 1)
        hidden = torch.cat(self.hidd_state, 1)
        hidden_motion = torch.cat(self.hidd_state[1:], 1)

        # input and target of dynamical weights
        input = torch.cat([cause, hidden], 1)
        target = hidden_motion

        # predict
        pred = self.net_d(input)

        # dynamical prediction error
        err = (pred - target).abs().unsqueeze(-1) + 1
        error = (err * self.covar_d**-1 * err).squeeze()
        error.backward(gradient=torch.ones_like(error))

        for opt in self.opts: opt.step()

        # precision decay
        if True: # todo inference/learning
            cd = P((self.covar_d - LR_PRECISION * self.covar_d).detach())
            if cd.min() > 1: self.covar_d = cd

        return err.mean().item(), self.covar_d.item()

    def infer(self, new_datapoint=False, ):
        """
        Update states and weights in all layers
        """
        error1, [covar_h1, covar_d1] = self.output_layer.infer(learn_covar=new_datapoint,
                                                               predict_change=self.dynamical_net)
        error2, [covar_h2, covar_d2] = self.hidden_layer.infer(learn_covar=new_datapoint,
                                                               predict_change=self.dynamical_net)
        error3, [covar_h3, covar_d3] = self.cause_layer.infer(learn_covar=new_datapoint,
                                                              predict_change=self.dynamical_net)

        return [error1.item(), error2.item(), error3.item()], \
               [covar_h1.item(), covar_h2.item(), covar_h3.item()],\
               [covar_d1.item(), covar_d2.item(), covar_d3.item()],

    def feed(self, obs=None, prior=None):
        """
        Feed observation and target to network
        """
        if obs is not None:
            self.g_observation.states.cause_state = P(obs)
        if prior is not None:
            self.cause_layer.states.cause_state = P(prior)

    def transition(self):
        """
        Compute difference between current and previous state
        """
        for i, l in enumerate(self.layers):
            l.states.change()

    def print_states(self):
        for l, layer in enumerate(self.layers):
            print("Layer", l, "cause state", layer.states.cause_state.shape)
            print("Layer", l, "hidden state", layer.states.hidd_state.shape, "\n")

    def predict_from(self, prior=None, start_layer=-1):
        """
        Hierarchical prediction through selected layers
        """
        keeps = [l.protect_states for l in self.layers] # save setting
        for l in self.layers[2:]:
            l.protect_states = True  # don't change the network
        for layer in reversed(self.layers[1:start_layer]):
            prior = layer.forward(prior)
        pred = self.g_observation.to_discrete(prior)
        for i, l in enumerate(self.layers):
            l.protect_states = keeps[i]  # reset setting
        return pred.detach().numpy()

    def iterative_inference(self, data=None, target=None, updates=30):
        """
        Iteratively refine the prediction for observed data
        1) feed cause prior and input state
        2) predict causes in all layers (if dynamical: cause state motion)
        3) dynamical prediction (only if dynamical)
        """
        errors, errors_d, cov_h, cov_d = [], [], [], []
        for i in range(updates):
            self.feed(data, target)  # feed data and targets
            pred = self.forward()  # hierarchical prediction
            if self.dynamical_net:
                err_d, cd = self.forward_dynamical()  # dynamical prediction
                errors_d.append(err_d); cov_d.append(cd)
            err, ch, _ = self.infer(new_datapoint=(i == 0))  # todo remove dynamical covar
            errors.append(err);
            cov_h.append(ch);
        return errors, errors_d, cov_h, cov_d


if __name__ == '__main__':

    VIDEO = True
    BATCH_SIZE, IMG_SIZE = 16, 16
    LR_STATES, LR_WEIGHTS, LR_PRECISION, UPDATES = .1, 0.0001, 0.0001, 50
    train_set = MovingMNIST(root='.data/mnist', train=True, download=True,
                            transform=transforms.Compose([transforms.Scale(IMG_SIZE), transforms.ToTensor(), ]))
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)

    # hierarchical net with three layers
    cause_sizes = [IMG_SIZE * IMG_SIZE, 128, 64, 32]
    net = GPC_net(b_size=BATCH_SIZE, dynamical_net=False,
                  cause_sizes=cause_sizes)

    # the state motion of each hierarchical layer is predicted by a dynamical network
    net_d1 = GPC_net(b_size=BATCH_SIZE, dynamical_net=True, obs_layer=net.layers[0],
                     cause_sizes=[cause_sizes[1] for _ in range(4)],
                     hidden_sizes=[cause_sizes[1] for _ in range(4)]) # todo dynamical layers don't have cause states!
    net_d2 = GPC_net(b_size=BATCH_SIZE, dynamical_net=True, obs_layer=net.layers[1],
                     cause_sizes=[cause_sizes[2] for _ in range(4)],
                     hidden_sizes=[cause_sizes[2] for _ in range(4)]) # todo dynamical layers don't have cause states!
    net_d3 = GPC_net(b_size=BATCH_SIZE, dynamical_net=True, obs_layer=net.layers[2],
                     cause_sizes=[cause_sizes[3] for _ in range(4)],
                     hidden_sizes=[cause_sizes[3] for _ in range(4)]) # todo dynamical layers don't have cause states!
    net.print_states()

    # logging
    errors, cov_h, errors_d1, errors_d2, errors_d3, cov_d1, cov_d2, cov_d3 = [[] for _ in range(8)]
    err_g1, err_g2, err_g3, cov_g1, cov_g2, cov_g3 = [[] for _ in range(6)]
    vid_in, vid_p1, vid_p2, vid_p3 = [], [], [], []

    for seq_id, (seq, _) in enumerate(train_loader):
        seq = torch.transpose(seq, 0,1)
        for id, data in enumerate(seq):
            data = data.reshape([-1,IMG_SIZE*IMG_SIZE]).float()

            # step hierarchical net
            e, _, ch, _ = net.iterative_inference(data, updates=UPDATES)

            # step dynamical nets  # todo include hidden states
            eg1, ed1, cg1, cd1 = net_d1.iterative_inference(updates=UPDATES)
            eg2, ed2, cg2, cd2 = net_d2.iterative_inference(updates=UPDATES)
            eg3, ed3, cg3, cd3 = net_d3.iterative_inference(updates=UPDATES)

            # logging
            errors, cov_h = errors + e, cov_h + ch
            errors_d1, cov_d1 = errors_d1 + ed1, cov_d1 + cd1
            errors_d2, cov_d2 = errors_d2 + ed2, cov_d2 + cd2
            errors_d3, cov_d3 = errors_d3 + ed3, cov_d3 + cd3
            err_g1, cov_g1 = err_g1 + eg1, cov_g1 + cg1
            err_g2, cov_g2 = err_g2 + eg2, cov_g2 + cg2
            err_g3, cov_g3 = err_g3 + eg3, cov_g3 + cg3

            # track state motion
            [n.transition() for n in [net, net_d1, net_d2, net_d3]]

            # create video
            if VIDEO:
                input, preds = plot_2D(img_size=IMG_SIZE, plot=False)
                vid_in.append(input.detach().numpy().reshape([IMG_SIZE, IMG_SIZE]))
                vid_p1.append(preds[0][0].reshape([IMG_SIZE, IMG_SIZE]))
                vid_p2.append(preds[1][0].reshape([IMG_SIZE, IMG_SIZE]))
                vid_p3.append(preds[2][0].reshape([IMG_SIZE, IMG_SIZE]))

            if id % 1 == 0:
                print(id, np.asarray(errors)[-1].mean())
                break
        if seq_id == 0: break

    # Overview plots
    plot_graph()
    input, preds = plot_2D(img_size=IMG_SIZE, title="State prediction", examples=4, plot=True)

    # Create video
    if VIDEO:
        for data, name in zip([vid_in, vid_p1], ["input","predl1"]):
            images = np.asarray([v*100 for v in data])
            images *= (255.0 / images.max())
            imgs = [Image.fromarray(img) for img in images]
            imgs[0].save(f"./{name}.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)


"""
Differentiable Generalized Predictive Coding
André Ofner 2021

Hierarchical-dynamical prediction of local and global trajectories on Moving MNIST
"""

import torch
from torch.nn import Parameter as P
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
plt.style.use(['seaborn'])

from tools import *
from MovingMNIST import *


def plot_2D(img_size=64, title=""):
    examples = 4
    fig, axs = plt.subplots(examples, 6)
    for example in range(examples):
        for ax, start_layer in enumerate(reversed(range(2, 5))):
            pred = net.predict_from(start_layer=start_layer)
            axs[example, ax].imshow(pred[example].reshape([img_size, img_size]))
            axs[example, ax].set_title("Layer " + str(3 - ax))
            axs[example, ax].set_xticks([]);
            axs[example, ax].set_yticks([])
        input = net.g_observation.states.cause_state[example].detach()
        error = pred[example] - input[example].detach().numpy()

        # covar
        """
        covar_h = net.layers[0].states.cause_covar.detach()
        projected_var = torch.matmul(covar_h[example] ** -1, torch.from_numpy(error.reshape([img_size * img_size, 1])))
        axs[example, -3].imshow(projected_var.reshape([img_size, img_size]))
        axs[example, -3].set_title("Precision\nLearning")
        axs[example, -3].set_xticks([]);
        axs[example, -2].set_yticks([])
        """

        # slow covar
        covar_h_slow = net.layers[0].states.cause_covar_slow.detach()
        projected_var_slow = covar_h_slow ** -1 * torch.ones([img_size * img_size, 1])
        axs[example, -2].imshow(projected_var_slow.reshape([img_size, img_size]))
        axs[example, -2].set_title("Precision\nInference")
        axs[example, -2].set_xticks([]);
        axs[example, -2].set_yticks([])

        axs[example, -1].imshow(input.reshape([img_size, img_size]))
        axs[example, -1].set_title("Observation")
        axs[example, -1].set_xticks([]);
        axs[example, -1].set_yticks([])
    plt.suptitle(str(title))
    plt.tight_layout()
    plt.show()


class GPC_layer_state(torch.nn.Module):
    def __init__(self, b_size, n_states, g_coords, full_covar=False, var_prior=2., covar_prior=100000.):
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

        # cause & hidden state error covariance: full covariance matrix for each batch element (inference)
        if full_covar:
            self.register_parameter(name="cause_covar",
                                    param=P(torch.eye(self.cause_state.shape[-1]).repeat([b_size, 1, 1])
                                            * (var_prior - covar_prior) + covar_prior))
            self.register_parameter(name="hidd_covar",
                                    param=P(torch.eye(self.hidd_state.shape[-1]).repeat([b_size, 1, 1])
                                            * (var_prior - covar_prior) + covar_prior))

        # cause & hidden state error covariance: scalar for entire batch (learning)
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
        #self.params_covar = [self.cause_covar, self.hidd_covar]
        self.params_covar = [self.cause_covar_slow, self.hidd_covar_slow]

    def forward(self):
        # Return the full state [causes, hiddens]
        return self.state

    def change(self):
        # difference between current and previous states
        self.cause_change =  self.cause_state - self.cause_previous.detach()
        self.hidd_change = self.hidd_state - self.hidd_previous.detach()

        self.cause_previous = self.cause_state.clone().detach()
        self.hidd_previous = self.hidd_state.clone().detach()

        return self.cause_change, self.hidd_change

    def pred_state(self, n_states_h=None, g_coords_h=None, n_states_d=None, g_coords_d=None):
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
        self.n_hidd_states = d_states # d_states  # unobservable part of states todo
        self.n_pred_states_h = self.n_cause_states   # amount of cause states used for outgoing prediction
        self.n_pred_states_d = self.n_cause_states   # amount of hidden states used for outgoing prediction
        self.n_states = self.n_hidd_states + self.n_cause_states

        # cause and hidden states x, v
        self.states = GPC_layer_state(b_size, self.n_cause_states, self.g_coords)

        # helpful views for hierarchical predictions
        self.pred = None  # outgoing prediction

        # hierarchical weights connect hierarchical layers
        self.net_h = None
        if self.lower is not None:
            n_pred_states = self.n_pred_states_h * self.g_coords + self.n_pred_states_d * self.g_coords
            self.net_h = torch.nn.Linear(n_pred_states, # todo use only x, not its higher coords
                                         self.lower.n_cause_states * self.lower.g_coords, False)

        # dynamical weights connect moving states
        self.net_d = torch.nn.Linear(self.n_cause_states * self.g_coords,
                                     self.n_cause_states * min(self.g_coords, 1), False)

        # generalized coordinates encode orders of state motion
        self.net_g = None
        if self.g_coords > 1:
            self.net_g = [GPC_layer(h_states=3, d_states=0, p_states=3, g_coords=1, lower=None, b_size=b_size)
                          for _ in range(self.n_states)]  # one network per state

        # optimizer per parameter group
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

    def forward(self, prior=None, predict_change=False):
        """
        Hierarchical prediction
        """

        global ADD_CHANGE # todo remove
        predict_change = ADD_CHANGE # todo remove

        # [causes, units] used for predictions
        pred_state = self.states.pred_state(n_states_h=self.n_pred_states_h, g_coords_h=self.g_coords,
                                       n_states_d=self.n_pred_states_d, g_coords_d=self.g_coords)

        # optionally overwrite cause prior used for prediction
        if prior is not None:
            pred_state[0] = prior.detach()

        self.pred = self.net_h(torch.cat(pred_state,1))

        # optionally add prediction to lower cause state (state motion prediction)
        if predict_change:
            self.pred += self.lower.states.cause_state.detach()

        # optionally overwrite cause state in lower layer with is prediction
        if not self.protect_states:
            self.lower.states.cause_state = P(self.pred.detach())

        return self.pred.detach()


    def infer(self, learn_covar=False, predict_change=True):
        """
        Update states and weights
        """

        for opt in self.opts:
            opt.zero_grad()

        error = 0.
        if self.lower is not None:

            # select hierarchical prediction target
            target = self.lower.states.cause_state  # inferred cause state
            if predict_change: target = self.lower.states.cause_change  # difference between inferred cause states

            # hierarchical prediction error
            err = (self.pred - target).abs().unsqueeze(-1)+1
            #error_fast = 0. #(err * torch.matmul(self.lower.states.cause_covar**-1, err)).mean([1,2])
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

        return err.mean(), [c.mean() for c in covars] # c[0].diagonal() # todo return weighted error too

    def to_discrete(self, prior):
        """
        Convert a generalized state to a discrete sequence
        """
        return prior


class GPC_net(torch.nn.Module):
    def __init__(self, b_size=1, dynamical_net=True, var_prior=2.):
        """
        Generalized predictive coding network
        """
        super().__init__()

        # layers
        self.g_observation = GPC_layer(lower=None, b_size=b_size, h_states=IMG_SIZE*IMG_SIZE) # data in gen. coordinates
        self.output_layer = GPC_layer(lower=self.g_observation, b_size=b_size, h_states=256)
        self.hidden_layer = GPC_layer(lower=self.output_layer, b_size=b_size, h_states=128)
        self.cause_layer = GPC_layer(lower=self.hidden_layer, b_size=b_size, h_states=64)
        self.layers = [self.g_observation, self.output_layer, self.hidden_layer, self.cause_layer]

        # treat layers as hierarchical or dynamical-hierarchical
        self.dynamical_net = dynamical_net

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
        self.cause_layer.forward(predict_change=self.dynamical_net)  # prediction in generalized coordinates
        self.hidden_layer.forward(predict_change=self.dynamical_net)  # prediction in generalized coordinates
        self.output_layer.forward(predict_change=self.dynamical_net)  # sensory prediction in generalized coordinates
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
        return err.mean().item(), self.covar_d.item()

    def infer(self, new_datapoint=False):
        """
        Update states and weights in all layers
        """
        error1, [covar_h1, covar_d1] = self.output_layer.infer(learn_covar=new_datapoint)
        error2, [covar_h2, covar_d2] = self.hidden_layer.infer(learn_covar=new_datapoint)
        error3, [covar_h3, covar_d3] = self.cause_layer.infer(learn_covar=new_datapoint)

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


if __name__ == '__main__':

    ADD_CHANGE = False
    BATCH_SIZE, IMG_SIZE = 32, 32
    LR_STATES, LR_WEIGHTS, LR_PRECISION = .1, 0.001, 0.001
    train_set = MovingMNIST(root='.data/mnist', train=True, download=True,
                            transform=transforms.Compose([transforms.Scale(IMG_SIZE), transforms.ToTensor(), ]))
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    print('Train batches: {}'.format(len(train_loader)))

    net = GPC_net(b_size=BATCH_SIZE)
    net.print_states()
    errors, errors_d, cov_h, cov_d = [], [], [], []

    for seq_id, (seq, _) in enumerate(train_loader):
        seq = torch.transpose(seq, 0,1)
        for id, data in enumerate(seq):
            data = data.reshape([-1,IMG_SIZE*IMG_SIZE]).float()

            # infer current datapoint
            for i in range(10):
                net.feed(data, None)  # feed data and targets
                pred = net.forward()  # hierarchical prediction
                err_d, cd = net.forward_dynamical()  # dynamical prediction
                err, ch, _ = net.infer(new_datapoint=(i==0))  # todo remove dynamical covar?
                errors_d.append(err_d); errors.append(err);
                cov_h.append(ch); cov_d.append(cd)
            net.transition()  # track state motion
            if id % 1 == 0: print(id, np.asarray(errors)[-1].mean())
            if id == 9: break  # todo remove
        if seq_id == 9: break  # todo remove

    lines = []
    lines += plt.plot(np.asarray(errors), color="red")
    lines += plt.plot(np.asarray(cov_h), color="green")
    lines += plt.plot(np.asarray(errors_d), color="black")
    lines += plt.plot(np.asarray(cov_d), color="blue")
    plt.title("Hierarchical and dynamical prediction errors")
    plt.legend(lines,[f"Hier. PE Layer {l}" for l in range(3)] + [f"Hier. Precision Layer {l}" for l in range(3)] +
               [f"Dynamical PE"] + [f"Dynamical Precision"])
    plt.xlabel("Update")
    plt.ylabel("Magnitude")
    plt.show()

    ADD_CHANGE = False
    plot_2D(img_size=IMG_SIZE, title="Hierarchical state motion prediction")

    ADD_CHANGE = True
    plot_2D(img_size=IMG_SIZE, title="State + state motion prediction")

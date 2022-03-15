"""
Differentiable Generalized Predictive Coding
André Ofner 2021
"""

import matplotlib.pyplot as plt
from MovingMNIST import *
plt.style.use(['seaborn'])

# circles dataset:
BATCH_SIZE, IMG_SIZE = 1, 2
LR_STATES, LR_WEIGHTS, LR_PRECISION, UPDATES = 0.5, .0001, 0., 2
LR_SCALE = 0.1  # learning rate of states receiving prediction (regularization)
LR_SCALE_DYN = 0
ONE_TO_ONE = True  # generalized coordinates per unit or across units
VERBOSE = False
PRED_SPLIT = 64

# moving mnist dataset:
#BATCH_SIZE, IMG_SIZE = 64, 32
#LR_STATES, LR_WEIGHTS, LR_PRECISION, UPDATES = .1, .001, 0., 10
#LR_SCALE = 1  # learning rate of states receiving prediction (regularization)
#LR_SCALE_DYN = 0.1
#ONE_TO_ONE = True  # generalized coordinates per unit or across units
#VERBOSE = False
#PRED_SPLIT = 32

def plot_graph(errors, errors_d1, errors_d2, errors_d3, cov_d1, cov_d2, cov_d3,cov_g1,
           cov_g2, cov_g3, err_g1, err_g2, err_g3, err_h1, err_h2, err_h3,cov_h,
           hierarchical=True, g_coords=True, dynamical=True):

    lines, lstyles, labels = [], ['-', '--', '-.', ':'], []

    # hierarchical prediction
    if hierarchical:
        lines += plt.plot(np.asarray(errors)[:, 0], color="red", linestyle=lstyles[0])
        #lines += plt.plot(np.asarray(errors)[:, 1], color="red", linestyle=lstyles[1])
        #lines += plt.plot(np.asarray(errors)[:, 2], color="red", linestyle=lstyles[2])
        lines += plt.plot(np.asarray(cov_h)[:, 0], color="green", linestyle=lstyles[0])
        #lines += plt.plot(np.asarray(cov_h)[:, 1], color="green", linestyle=lstyles[1])
        #lines += plt.plot(np.asarray(cov_h)[:, 2], color="green", linestyle=lstyles[2])
        labels += [f"Prediction error L{l + 1}" for l in range(1)]
        labels += [f"Error variance L{l + 1}" for l in range(1)]

    # dynamical prediction
    if dynamical:
        lines += plt.plot(np.asarray(errors_d1), color="black", linestyle=lstyles[0])
        #lines += plt.plot(np.asarray(errors_d2), color="black", linestyle=lstyles[1])
        #lines += plt.plot(np.asarray(errors_d3), color="black", linestyle=lstyles[2])
        lines += plt.plot(np.asarray(cov_d1), color="blue", linestyle=lstyles[0])
        #lines += plt.plot(np.asarray(cov_d2), color="blue", linestyle=lstyles[1])
        #lines += plt.plot(np.asarray(cov_d3), color="blue", linestyle=lstyles[2])
        labels += [f"Dynamical prediction error L{l + 1}" for l in range(1)]
        labels += [f"Dynamical error variance L{l + 1}" for l in range(1)]

    # generalized coordinates
    if g_coords:
        # cause states # todo plot separately for each coordinate
        lines += plt.plot(np.asarray(cov_g1).mean(-1), color="grey", linestyle=lstyles[0])
        #lines += plt.plot(np.asarray(cov_g1).mean(-1), color="grey", linestyle=lstyles[1])
        #lines += plt.plot(np.asarray(cov_g1).mean(-1), color="grey", linestyle=lstyles[2])
        lines += plt.plot(np.asarray(err_g1).mean(-1), color="orange", linestyle=lstyles[0])
        #lines += plt.plot(np.asarray(err_g2).mean(-1), color="orange", linestyle=lstyles[1])
        #lines += plt.plot(np.asarray(err_g3).mean(-1), color="orange", linestyle=lstyles[2])
        labels += [f"Cause motion error variance L{l + 1}" for l in range(1)]
        labels += [f"Cause motion prediction error L{l + 1}" for l in range(1)]

        # hidden states # todo plot separately for each coordinate
        #lines += plt.plot(np.asarray(cov_g1).mean(-1), color="brown", linestyle=lstyles[0])
        #lines += plt.plot(np.asarray(cov_g1).mean(-1), color="brown", linestyle=lstyles[1])
        #lines += plt.plot(np.asarray(cov_g1).mean(-1), color="brown", linestyle=lstyles[2])
        # todo include covariance of hidden state motion
        lines += plt.plot(np.asarray(err_h1).mean(-1), color="olive", linestyle=lstyles[0])
        #lines += plt.plot(np.asarray(err_h2).mean(-1), color="olive", linestyle=lstyles[1])
        #lines += plt.plot(np.asarray(err_h3).mean(-1), color="olive", linestyle=lstyles[2])
        #labels += [f"Generalized hidden motion error variance L{l + 1}" for l in range(3)]
        labels += [f"Hidden motion prediction error L{l + 1}" for l in range(1)]


    plt.title("Hierarchical and dynamical prediction errors")
    plt.legend(lines, labels, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.xlabel("Update"); plt.ylabel("Magnitude")
    plt.savefig("./errors_graph.png", bbox_inches="tight")
    plt.tight_layout()
    plt.yscale("log")
    plt.show()


def plot_2D(net, img_size=64, title="", plot=True, examples=1):
    if plot: fig, axs = plt.subplots(examples, 4)
    preds = []
    for example in range(examples):
        for ax, start_layer in enumerate(reversed(range(2, 3))): # todo fixme 5
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
        plt.savefig("./batch.png", bbox_inches="tight")
        plt.show()

    return input, preds


class GPC_state(torch.nn.Module):
    def __init__(self, b_size, n_states_h, n_states_d, full_covar=False, var_prior=1., covar_prior=100000.):
        """
        Cause and hidden states of a layer
        """
        super().__init__()

        # cause states and hidden states
        self.cause_state = torch.ones([b_size, n_states_h]).requires_grad_()
        self.hidd_state = torch.ones([b_size, n_states_d]).requires_grad_()
        torch.nn.init.xavier_uniform_(self.cause_state)  # initialize with random prior
        torch.nn.init.xavier_uniform_(self.hidd_state)  # initialize with random prior

        # cause & hidden state error covariance: full covariance matrix for each batch element (~inference)
        if full_covar:
            self.cause_covar = torch.eye(self.cause_state.shape[-1]).repeat([b_size, 1, 1]) \
                               * (var_prior - covar_prior) + covar_prior
            self.hidd_covar = torch.eye(self.hidd_state.shape[-1]).repeat([b_size, 1, 1]) \
                              * (var_prior - covar_prior) + covar_prior

        # cause & hidden state error covariance: scalar for entire batch (~learning)
        self.cause_covar_slow = torch.tensor([var_prior])
        self.hidd_covar_slow = torch.tensor([var_prior])

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

    def change(self, step=False):
        """
        Difference between current and previous states
        """
        self.cause_change = self.cause_state - self.cause_previous.detach()
        self.hidd_change = self.hidd_state - self.hidd_previous.detach()
        if step:  # new datapoint: current state becomes previous state
            self.cause_previous = self.cause_state.clone().detach()
            self.hidd_previous = self.hidd_state.clone().detach()
        return self.cause_change, self.hidd_change

    def predstate(self, n_states_h=None, n_states_d=None, priors=None, dynamical=False):
        """
        View on subset of states used for outgoing prediction
        """

        if priors is not None:
            if dynamical:  # use priors[0] instead of cause state & priors[1] instead of hidden state
                return [priors[0].detach()[:, :n_states_h], priors[1].detach()[:, :n_states_d]]
            else:  # use priors instead of cause state
                return [priors.detach()[:, :n_states_h], self.hidd_state[:, :n_states_d]]

        return [self.cause_state[:, :n_states_h], self.hidd_state[:, :n_states_d]]


class GPC_layer(torch.nn.Module):
    def __init__(self, h_states=1, d_states=0, lower=None, b_size=1, protect_states=True, dynamical=True,
                 n_predstates_h=None, n_predstates_d=None):
        """
        Generalized predictive coding layer
        """
        super().__init__()

        self.gc = 4  # how many generalized coordinates to encode
        self.dynamical = dynamical
        self.lower = lower # lower GPC layer
        self.protect_states = protect_states  # don't overwrite lower cause states when predicting
        self.n_cause_states = h_states  # observable part of states
        self.n_hidd_states = d_states  # unobservable part of states
        self.n_states = self.n_hidd_states + self.n_cause_states
        self.states = GPC_state(b_size, self.n_cause_states, self.n_hidd_states) # cause and hidden states x, v
        self.pred = None  # outgoing prediction
        self.net_h = None

        # select amount of cause & hidden units used for prediction
        if not self.dynamical:
            self.n_predstates_h = n_predstates_h  # amount of cause states used for outgoing prediction
            self.n_predstates_d = min(n_predstates_d, PRED_SPLIT)  # amount of hidden states used for outgoing prediction
            if n_predstates_h is None: # default: all cause & hidden units are used for prediction
                self.n_predstates_h = self.n_cause_states
                self.n_predstates_d = min(self.n_hidd_states, PRED_SPLIT)
        else: # is dynamical layer
            if not ONE_TO_ONE:
                self.n_predstates_h = self.n_cause_states
                self.n_predstates_d = self.n_hidd_states
            else:
                if self.lower is not None:
                    self.n_predstates_h = self.lower.n_cause_states
                    self.n_predstates_d = self.lower.n_hidd_states
                else:
                    self.n_predstates_h = 0
                    self.n_predstates_d = 0

        # hierarchical weights
        if self.lower is not None:
            # hierarchical weights connect hierarchical layers
            self.net_h = torch.nn.Linear(self.n_predstates_h + self.n_predstates_d,
                                         self.lower.n_cause_states, False)
            if self.dynamical:
                # weights for generalized coordinates of cause motion
                if not ONE_TO_ONE:
                    self.net_h_cause = torch.nn.Linear(self.n_predstates_h, self.lower.n_cause_states, False) # full connectivity
                else:
                    self.net_h_cause = torch.ones_like(self.lower.states.cause_state).requires_grad_()

                # weights for generalized coordinates of hidden motion
                if not ONE_TO_ONE:
                    self.net_h_hidden = torch.nn.Linear(self.n_predstates_d, self.lower.n_hidd_states, False) # full connectivity
                else:
                    self.net_h_hidden = torch.ones_like(self.lower.states.hidd_state).requires_grad_()
                if VERBOSE: print("\tCause state gen. coord. net:", self.net_h.weight.shape)
                if VERBOSE: print("\tHidden state gen. coord. net:", self.net_h.weight.shape)
            else:
                if VERBOSE: print("\tCause states for outgoing prediction:", self.n_predstates_h)
                if VERBOSE: print("\tHidden states for outgoing prediction:", self.n_predstates_d)
                if VERBOSE: print("\tHierarchical net:", self.net_h.weight.shape)
            if VERBOSE: print("\tLayer cause states:", self.n_cause_states)
            if VERBOSE: print("\tLayer hidden states:", self.n_hidd_states)
            if VERBOSE: print("\tLower layer cause states:", self.lower.n_cause_states)
            if VERBOSE: print("\tLower layer hidden states:", self.lower.n_hidd_states)

        # parameter groups and optimizers
        self.opts = []

        # weight learning rate
        if self.dynamical:
            if self.lower is not None:
                if not ONE_TO_ONE:
                    self.opts.append(torch.optim.SGD([self.net_h_cause.weight, self.net_h_hidden.weight], lr=LR_WEIGHTS))  # hierarchical weights
                else:
                    self.opts.append(torch.optim.SGD([self.net_h_cause, self.net_h_hidden], lr=0.))  # don't optimize weights for 1:1
        else:
            if self.lower is not None:
                self.opts.append(torch.optim.SGD(self.net_h.parameters(), lr=LR_WEIGHTS))  # hierarchical weights

        # states & precision learning rate
        if self.dynamical:
            self.opts.append(torch.optim.SGD(self.states.params_state, lr=LR_STATES))  # states used for prediction
            self.opts.append(torch.optim.SGD(self.states.params_covar, lr=LR_PRECISION))  # states used for prediction
        else:
            self.opts.append(torch.optim.SGD(self.states.params_state, lr=LR_STATES))  # states used for prediction
            self.opts.append(torch.optim.SGD(self.states.params_covar, lr=LR_PRECISION))  # states used for prediction

        # lower layer states & precision learning rate
        if self.lower is not None:  # states receiving the prediction
            if self.dynamical:
                if self.lower.dynamical:
                    if not ONE_TO_ONE:
                        self.opts.append(torch.optim.SGD(self.lower.states.params_state, lr=LR_STATES*LR_SCALE_DYN))  # todo
                    else:
                        self.opts.append(torch.optim.SGD(self.lower.states.params_state, lr=LR_STATES*LR_SCALE_DYN))  # todo
                else:
                    self.opts.append(torch.optim.SGD(self.lower.states.params_state, lr=LR_STATES*LR_SCALE_DYN))  # todo
                self.opts.append(torch.optim.SGD(self.lower.states.params_covar, lr=LR_PRECISION))
            else:
                self.opts.append(torch.optim.SGD(self.lower.states.params_state, lr=LR_STATES*LR_SCALE))
                self.opts.append(torch.optim.SGD(self.lower.states.params_covar, lr=LR_PRECISION))

        # each hierarchical layer has a dynamical network that
        # - encodes states in generalized coordinates (state x, state change x', change of state change x'', ...)
        # - couples orders of generalized motion via dynamical weights (x->x', x'->x'', ...)
        if not self.dynamical:
            self.dyn_model = GPC_net(b_size=BATCH_SIZE, dynamical_net=True, obs_layer=self,
                                     cause_sizes=[self.n_cause_states for _ in range(self.gc)],
                                     hidden_sizes=[self.n_hidd_states for _ in range(self.gc)])

    def state_parameters(self, recurse=False):
        return self.parameters(recurse=recurse)




    def forward(self, prior=None, use_gen_coords=False):
        """
        Hierarchical prediction
        """

        # [causes, hiddens] used for prediction
        predstate = self.states.predstate(n_states_h=self.n_predstates_h,
                                          n_states_d=self.n_predstates_d,
                                          priors=prior, dynamical=self.dynamical)

        if self.dynamical:
            if not ONE_TO_ONE:
                self.pred_h = self.net_h_cause(predstate[0]) # full connectivity (#weights=input size * output size )
                self.pred_d = self.net_h_hidden(predstate[1])
            else:
                # 1:1 connectivity (#weights = output size)
                self.pred_h = self.net_h_cause * predstate[0]
                self.pred_d = self.net_h_hidden * predstate[1]
        else:
            # predict state
            self.pred = self.net_h(torch.cat(predstate, 1))

            # predict higher order gen. coordinates
            self.pred_gc = []
            if use_gen_coords:
                for gen_coord in range(1, self.gc, 1):
                    # predict from the current hierarchical layer's dynamical layers
                    predstate = self.dyn_model.layers[gen_coord].states.predstate(n_states_h=self.n_predstates_h,
                                                      n_states_d=self.n_predstates_d, priors=None, dynamical=False)
                    self.pred_gc.append(self.net_h(torch.cat(predstate, 1)))  # todo remove weights sharing?

        # optionally overwrite cause state in lower layer with is prediction
        if not self.protect_states:
            if self.dynamical:
                self.lower.states.cause_state = self.pred_h.detach()
                self.lower.states.hidd_state = self.pred_d.detach()
            else:
                # predict only state
                self.lower.states.cause_state = self.pred.detach()

                # predict higher order generalized coordinates
                if use_gen_coords:
                    self.lower.dyn_model.layers[0].cause_state = self.pred.detach()
                    for gen_coord in range(1, self.gc, 1):
                        self.lower.dyn_model.layers[gen_coord].cause_state = self.pred_gc[gen_coord-1].detach()

        if self.dynamical:
            return self.pred_h.detach(), self.pred_d.detach()  # todo
        else:
            return self.pred.detach()


    def infer(self):
        """
        Update states and weights
        """

        learn_covar = True  # todo select

        for opt in self.opts:
            opt.zero_grad()

        if self.lower is not None:

            # select hierarchical prediction target
            target = self.lower.states.cause_state  # inferred cause state
            if self.dynamical:
                target_h = self.lower.states.cause_change  # difference between inferred cause states
                target_d = self.lower.states.hidd_change  # difference between inferred cause states

            # hierarchical prediction error
            if self.dynamical:
                # hierarchical cause and hidden state motion prediction error
                err_d = (self.pred_d - target_d).abs().unsqueeze(-1)#+1 # todo +1 ?
                err_h = (self.pred_h - target_h).abs().unsqueeze(-1)#+1 # todo +1 ?
                error_h = (err_h * self.lower.states.cause_covar_slow**-1 * err_h).squeeze()
                error_d = (err_d * self.lower.states.hidd_covar_slow**-1 * err_d).squeeze()
                error_h.backward(gradient=torch.ones_like(error_h))
                error_d.backward(gradient=torch.ones_like(error_d))
                error = error_h.mean() + error_d.mean() # todo remove?
            else:
                # hierarchical cause state prediction error
                err = (self.pred - target).abs().unsqueeze(-1)#+1 # todo +1 ?
                #error_fast = 0. #(err * torch.matmul(self.lower.states.cause_covar**-1, err)).mean([1,2]) # todo
                error = (err * self.lower.states.cause_covar_slow**-1 * err).squeeze()
                error.backward(gradient=torch.ones_like(error))

            # freeze learned precision during inference
            if not learn_covar:
                self.lower.states.cause_covar_slow.grad.zero_()
                self.lower.states.hidd_covar_slow.grad.zero_()

            for opt in self.opts:
                opt.step()

            # precision decay
            ch = (self.lower.states.cause_covar_slow - LR_PRECISION * self.lower.states.cause_covar_slow).detach()
            cd = (self.lower.states.hidd_covar_slow - LR_PRECISION * self.lower.states.hidd_covar_slow).detach()
            covars = [ch.detach(), cd.detach()]

            if learn_covar:
                if ch.min() > 1: self.lower.states.cause_covar_slow = ch
                if cd.min() > 1: self.lower.states.hidd_covar_slow = cd

            if self.dynamical:
                return [err_h.mean(), err_d.mean()], [c.mean() for c in covars]  # c[0].diagonal()
            return [err.mean(), err.mean()*0], [c.mean() for c in covars] # c[0].diagonal() # todo return weighted error


class GPC_net(torch.nn.Module):
    def __init__(self, b_size=1, dynamical_net=True, var_prior=1.,
                 cause_sizes=[32,32,32,32], hidden_sizes=[0,0,0,0], obs_layer=None):
        """
        Generalized predictive coding network
        """
        super().__init__()

        # specify network type: hierarchical or dynamical
        self.dynamical_net = dynamical_net

        if self.dynamical_net:
            if VERBOSE: print("DYNAMICAL NETWORK:")
        else:
            if VERBOSE: print("HIERARCHICAL NETWORK:")

        # input layer (ignored by dynamical models since they observe hierarchical states)
        if not self.dynamical_net:
            if VERBOSE: print("LAYER: Input to generalized coordinates\n________\n")
        self.g_observation = GPC_layer(lower=None, b_size=b_size, dynamical=self.dynamical_net,
                                       h_states=cause_sizes[0], d_states=0,
                                       n_predstates_h=cause_sizes[0], n_predstates_d=0)

        # output layer
        if obs_layer is None:
            if VERBOSE: print("LAYER: Sensory observation")
            lower_layer = self.g_observation
        else:
            if VERBOSE: print("LAYER: Hierarchical State observation")
            lower_layer = obs_layer

        self.output_layer = GPC_layer(lower=lower_layer, b_size=b_size, dynamical=self.dynamical_net,
                                      h_states=cause_sizes[1], d_states=hidden_sizes[1],
                                      n_predstates_h=cause_sizes[1], n_predstates_d=hidden_sizes[1])

        self.layers = [self.g_observation, self.output_layer]

        # hidden layers
        for l in range(len(cause_sizes)-2):
            if VERBOSE: print("LAYER: Hidden")
            hidden_layer = GPC_layer(lower=self.layers[-1], b_size=b_size, dynamical=self.dynamical_net,
                                      h_states=cause_sizes[l+2], d_states=hidden_sizes[l+2],
                                      n_predstates_h=cause_sizes[l+2], n_predstates_d=hidden_sizes[l+2])
            self.layers.append(hidden_layer)
        if VERBOSE: print("________________________")

        # summary of cause and hidden states in all layers
        self.cause_state, self.hidd_state = self.hierarchical_states()

        # dynamical weights
        if self.dynamical_net:
            self.layers[0] = self.layers[1].lower
            self.nets_d = []
            self.opts_d = []
            self.covars_d = []
            for l, layer in enumerate(self.layers[:-1]):
                net_d = torch.nn.Linear(layer.n_cause_states+layer.n_hidd_states, self.layers[l+1].n_hidd_states) # f(x,v) = x'
                self.nets_d.append(net_d) # dynamical weights
                self.opts_d.append(torch.optim.SGD(net_d.parameters(), lr=LR_WEIGHTS))
                self.covars_d.append(torch.tensor([var_prior]).requires_grad_()) # dynamical error covariance

    def forward(self, use_gen_coords=False):
        """
        Hierarchical prediction through all layers
        """
        for l in reversed(range(len(self.layers))):
            if l > 0:  # input layer has no lower layer to predict
                self.layers[l].forward(use_gen_coords=use_gen_coords)
        return self.output_layer.pred

    def infer(self):
        """
        Update states and weights in all layers
        """
        errors_h, errors_d, covars_h, covars_d = [], [], [], []
        for l in range(len(self.layers)):
            if l > 0:  # input layer has no lower layer to predict
                [error_h, error_d], [covar_h, covar_d] = self.layers[l].infer()
                errors_h.append(error_h); errors_d.append(error_d)
                covars_h.append(covar_h); covars_d.append(covar_d)

        return [r.item() for r in errors_h], [r.item() for r in errors_d], \
               [r.item() for r in covars_h], [r.item() for r in covars_d]

    def feed(self, obs=None, prior=None):
        """
        Feed observation and target to network
        """
        if obs is not None:
            self.layers[0].states.cause_state = obs
        if prior is not None:
            self.layers[-1].states.cause_state = prior

    def state_diff(self, step=False):
        """
        Compute difference between current and previous state
        """
        for i, l in enumerate(self.layers):
            l.states.change(step=step)

    def predict_from(self, prior=None, protect_states=True, start_layer=-1, use_gen_coords=False):
        """
        Hierarchical prediction through selected layers
        """
        keeps = [l.protect_states for l in self.layers] # save setting
        for l in self.layers[2:]:
            l.protect_states = protect_states  # default: don't change the network
        for layer in reversed(self.layers[1:start_layer]):
            prior = layer.forward(prior, use_gen_coords=use_gen_coords)
        for i, l in enumerate(self.layers):
            l.protect_states = keeps[i]  # reset setting
        if self.dynamical_net:
            return [p.detach().numpy() for p in prior]
        return prior.detach().numpy()

    def iterative_inference(self, data=None, target=None, updates=30):
        """
        Iteratively refines the prediction for observed data
        1) feed cause state and input state to first and last layer
        2) predict lower cause states in all layers (dynamical: lower state motion)
        3) only in dynamical: prediction of next higher dynamical states (= higher order gen. coord.)
        """
        errors, errors_hidden, errors_d, cov_h, cov_d = [], [], [], [], []
        for i in range(updates):
            if self.dynamical_net and data is None:
                err_d, cd = self.forward_dynamical()  # dynamical prediction
                errors_d.append(err_d); cov_d.append(cd)
            else:
                self.feed(data, target)  # feed data and targets
            self.forward()  # hierarchical prediction
            err, err_hidd, ch, _ = self.infer()  # todo remove dynamical covar
            errors.append(err); errors_hidden.append(err_hidd); cov_h.append(ch);
        return errors, errors_hidden, errors_d, cov_h, cov_d

    def hierarchical_states(self, previous=False):
        """
        Summarizes cause and hidden states from all layers in the network
        """
        self.cause_state, self.hidd_state = [], []
        if previous:
            self.cause_state.append(self.layers[1].lower.states.cause_previous)
            self.hidd_state.append(self.layers[1].lower.states.hidd_previous)
            for l, layer in enumerate(self.layers[1:]):
                self.cause_state.append(layer.states.cause_previous)
                self.hidd_state.append(layer.states.hidd_previous)
        else:
            self.cause_state.append(self.layers[1].lower.states.cause_state)
            self.hidd_state.append(self.layers[1].lower.states.hidd_state)
            for l, layer in enumerate(self.layers[1:]):
                self.cause_state.append(layer.states.cause_state)
                self.hidd_state.append(layer.states.hidd_state)
        return self.cause_state, self.hidd_state

    def decode_dynamical_hidden(self, t=0):
        """
        Decodes the dynamical states of a hierarchical layer into
        a discrete sequence of the corresponding hierarchical state

        t : position in decoded discrete sequence
        """

        # apply encoded state motion to the lowest t dynamical layers, from highest to lowest coordinate
        for coord in reversed(range(min(t,len(self.layers)-1))):
            state = self.layers[coord].states.hidd_state.detach()
            higher_state = self.layers[coord+1].states.hidd_state.detach()
            if ONE_TO_ONE:
                net_d = self.layers[coord+1].net_h_hidden.detach()
            else:
                net_d = self.layers[coord + 1].net_h_hidden
                net_d.weight = torch.nn.Parameter(net_d.weight.detach())

            if not ONE_TO_ONE:
                self.layers[coord].states.hidd_state = state + net_d(higher_state)
            else:
                self.layers[coord].states.hidd_state = state + net_d * higher_state

        return [layer.states.hidd_state.detach()[0].numpy() for layer in self.layers]


    def decode_dynamical_cause(self, t=0):
        """
        Decodes the dynamical states of a hierarchical layer into
        a discrete sequence of the corresponding hierarchical state

        t : position in decoded discrete sequence
        """

        # apply encoded state motion to the lowest t dynamical layers, from highest to lowest coordinate
        for coord in reversed(range(min(t,len(self.layers)-1))):
            state = self.layers[coord].states.cause_state.detach()
            higher_state = self.layers[coord+1].states.cause_state.detach()

            if ONE_TO_ONE:
                net_d = self.layers[coord+1].net_h_cause.detach()
            else:
                net_d = self.layers[coord + 1].net_h_cause
                net_d.weight = torch.nn.Parameter(net_d.weight.detach())

            if not ONE_TO_ONE:
                self.layers[coord].states.cause_state = state + net_d(higher_state)
            else:
                self.layers[coord].states.cause_state = state + net_d * higher_state

        return [layer.states.cause_state.detach()[0].numpy() for layer in self.layers]

    def decode_dynamical_prediction(self):

        # get predicted generalized coordinates of the hidden state
        pred_coords = self.forward_dynamical(predict_only=True)  # x', x'', x''', etc.

        # overwrite hidden state coordinates with the prediction
        for l in range(len(self.layers[:-1])):
            self.layers[l+1].states.hidd_state = pred_coords[l]


    def forward_dynamical(self, predict_only=False):
        """
        Dynamical prediction and optimization between all dynamical layers in a hierarchical layer
        This couples orders of state motion: x' = f(x,v), x'' = f'(x',v'), ...
        """

        pred_coords = []
        for l in range(len(self.layers[:-1])):
            opt_states = torch.optim.SGD([self.layers[l].states.cause_state, self.layers[l].states.hidd_state,
                                          self.layers[l+1].states.hidd_state], lr=LR_STATES) # todo fix LR
            opt_weights = torch.optim.SGD(self.nets_d[l].parameters(), lr=LR_WEIGHTS) # todo fix LR
            opt_states.zero_grad()
            opt_weights.zero_grad()

            # predict higher dynamical layer state (= next higher generalized coordinate)
            input = torch.cat([self.layers[l].states.cause_state, self.layers[l].states.hidd_state], 1)
            pred = self.nets_d[l](input)

            if not predict_only:
                target = self.layers[l+1].states.hidd_state

                # dynamical prediction error
                err = (pred - target).abs().unsqueeze(-1)
                error = (err * self.covars_d[l]**-1 * err).squeeze()
                error.backward(gradient=torch.ones_like(error))

                opt_states.step()
                opt_weights.step()

                # precision decay
                cd = (self.covars_d[l] - LR_PRECISION * self.covars_d[l].detach())
                if cd.min() > 1:
                    self.covars_d[l] = cd

                if l == 0:
                    err_out = error.mean().detach().numpy()
                    covar_out = self.covars_d[l].mean().detach().numpy()
            else:
                pred_coords.append(pred.detach())

        if predict_only:
            return pred_coords

        return err_out, covar_out # todo return for all layers

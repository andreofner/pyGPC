"""
Differentiable Generalized Predictive Coding
André Ofner 2021
"""

import matplotlib.pyplot as plt
from MovingMNIST import *
plt.style.use(['seaborn'])

# hyper parameters for circles dataset
BATCH_SIZE, IMG_SIZE = 1, 1
WEIGHTS_SHARING = True  # share weights between generalized coords in hierarchical prediction
PROTECT_STATES = True  # overwrite of cause states with their prediction
ONE_TO_ONE = True  # encode generalized coordinates independently per unit
PRED_SPLIT = 32  # hidden units to use for outgoing prediction. remaining units are memory

# hierarchical LR (for hidden layers operating on generalized state coordinates)
LR_STATES, LR_WEIGHTS, LR_PRECISION, UPDATES = .1, .00, 0., 1
LR_SCALE = 0  # learning rate of states receiving prediction (regularization)

# dynamical LR (for hidden layers operating on generalized state coordinates)
LR_WEIGHTS_DYN = LR_WEIGHTS  # optimize dynamics towards observed state motion
LR_STATES_DYN = LR_STATES  # optimize state motion towards learned dynamics

# generalized coordinates LR (for sensory layer operating on discrete sequential data)
LR_SCALE_DYN = 0  # learning rate of states receiving prediction (regularization)


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
        for ax, start_layer in enumerate(reversed(range(2, 5))): # todo fixme 5
            pred = net.predict_from(start_layer=start_layer)
            if plot:
                axs[example, ax].imshow(pred[example].reshape([img_size, img_size]))
                axs[example, ax].set_title("Layer " + str(3 - ax))
                axs[example, ax].set_xticks([]);
                axs[example, ax].set_yticks([])
            preds.append(pred)
        input = net.g_observation.states.cause_state[example].detach()
        if plot:
            #error = pred[example] - input[example].detach().numpy()

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

    def change(self, step=False, dt=1.):
        """
        Compute difference between current and previous states
        step: new datapoint
        dt: temporal difference between samples
        """
        self.cause_change = (self.cause_state - self.cause_previous.detach()).detach()/dt
        self.hidd_change = (self.hidd_state - self.hidd_previous.detach()).detach()/dt
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
    def __init__(self, h_states=1, d_states=0, lower=None, b_size=1, dynamical=True,
                 n_predstates_h=None, n_predstates_d=None, gen_coords=5):
        """
        Generalized predictive coding layer
        """
        super().__init__()

        global PROTECT_STATES
        protect_states = PROTECT_STATES

        self.gc = gen_coords  # how many generalized coordinates to encode
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
            self.n_predstates_h = 0 # amount of cause states used for outgoing prediction
            self.n_predstates_d = min(n_predstates_d, PRED_SPLIT)  # amount of hidden states used for outgoing prediction
            if n_predstates_h is None: # default: all cause & hidden units are used for prediction
                self.n_predstates_h = 0
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

            # hierarchical weights connect hierarchical layers: one network per generalized coordinate
            if WEIGHTS_SHARING:
                net_h = torch.nn.Linear(self.n_predstates_h + self.n_predstates_d, self.lower.n_cause_states, False) # todo weights sharing?
                self.net_h = [net_h for _ in range(self.gc)]
            else:
                self.net_h = [torch.nn.Linear(self.n_predstates_h + self.n_predstates_d, self.lower.n_cause_states, False) for _ in range(self.gc)]

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

        # each hierarchical layer has a dynamical network
        # in sensory layer: encodes states in generalized coordinates (state x, state change x', change of state change x'', ...)
        # in hidden layers: couples orders of generalized motion via dynamical weights (x->x', x'->x'', ...)
        if not self.dynamical:
            self.dyn_model = GPC_net(b_size=BATCH_SIZE, dynamical_net=True, obs_layer=self,
                                     cause_sizes=[self.n_cause_states for _ in range(self.gc)],
                                     hidden_sizes=[self.n_hidd_states for _ in range(self.gc)])

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
                for nh in self.net_h:
                    self.opts.append(torch.optim.SGD(nh.parameters(), lr=LR_WEIGHTS))  # hierarchical weights

        # states & precision learning rate
        if self.dynamical:
            self.opts.append(torch.optim.SGD(self.states.params_state, lr=0.5))  # states used for prediction
            self.opts.append(torch.optim.SGD(self.states.params_covar, lr=LR_PRECISION))  # states used for prediction
        else:
            for layer in self.dyn_model.layers: # hierarchical states for every generalized coordinate
                self.opts.append(torch.optim.SGD(layer.states.params_state, lr=LR_STATES))  # states used for prediction
                self.opts.append(torch.optim.SGD(layer.states.params_covar, lr=LR_PRECISION))  # states used for prediction

        # lower layer states & precision learning rate
        if self.lower is not None:  # states receiving the prediction
            if self.dynamical:
                if self.lower.dynamical:
                    if not ONE_TO_ONE:
                        self.opts.append(torch.optim.SGD(self.lower.states.params_state, lr=0))  # todo
                    else:
                        self.opts.append(torch.optim.SGD(self.lower.states.params_state, lr=0))  # todo
                else:
                    self.opts.append(torch.optim.SGD(self.lower.states.params_state, lr=0))  # todo
                self.opts.append(torch.optim.SGD(self.lower.states.params_covar, lr=LR_PRECISION))
            else:
                for layer in self.lower.dyn_model.layers:  # hierarchical states for every generalized coordinate
                    self.opts.append(
                        torch.optim.SGD(layer.states.params_state, lr=LR_STATES*LR_SCALE))  # states used for prediction
                    self.opts.append(
                        torch.optim.SGD(layer.states.params_covar, lr=LR_PRECISION))  # states used for prediction

    def state_parameters(self, recurse=False):
        return self.parameters(recurse=recurse)


    def forward(self, prior=None, use_gen_coords=False):
        """
        Hierarchical prediction
        """

        if self.dynamical:
            # [causes, hiddens] used for prediction
            predstate = self.states.predstate(n_states_h=self.n_predstates_h, n_states_d=self.n_predstates_d,
                                              priors=prior, dynamical=self.dynamical)
            if not ONE_TO_ONE:
                self.pred_h = self.net_h_cause(predstate[0]) # full connectivity (#weights=input size * output size )
                self.pred_d = self.net_h_hidden(predstate[1])
            else:
                # 1:1 connectivity (#weights = output size)
                self.pred_h = self.net_h_cause * predstate[0]
                self.pred_d = self.net_h_hidden * predstate[1]
        else:
            # predict lower layer state in generalized coordinates
            self.pred = []
            for gen_coord in range(self.gc):
                dyn_layer = self.dyn_model.layers[gen_coord]
                predstate = dyn_layer.states.predstate(n_states_h=self.n_predstates_h,
                                                  n_states_d=self.n_predstates_d,
                                                  priors=None, dynamical=False)  # input state coordinate
                self.pred.append(self.net_h[gen_coord](torch.cat(predstate, 1)))  # output state coordinate

        if self.dynamical:
            return self.pred_h.detach(), self.pred_d.detach()  # todo
        else:
            return [p.detach() for p in self.pred]


    def infer(self):
        """
        Update states and weights
        """

        learn_covar = True  # todo select

        for opt in self.opts:
            opt.zero_grad()

        if self.lower is not None:
            if self.dynamical:  # hierarchical cause and hidden state motion prediction error
                target_h = self.lower.states.cause_change  # difference between inferred cause states
                target_d = self.lower.states.hidd_change  # difference between inferred cause states
                err_d = (self.pred_d - target_d).abs().unsqueeze(-1)
                err_h = (self.pred_h - target_h).abs().unsqueeze(-1)
                error_h = (err_h * self.lower.states.cause_covar_slow**-1 * err_h).squeeze()
                error_d = (err_d * self.lower.states.hidd_covar_slow**-1 * err_d).squeeze()
                error_h.backward(gradient=torch.ones_like(error_h))
                error_d.backward(gradient=torch.ones_like(error_d))
            else: # hierarchical cause state prediction error
                err = 0.
                for gen_coord in range(self.gc):
                    target = self.lower.dyn_model.layers[gen_coord].states.cause_state
                    if gen_coord > 0: target = target*10 # todo selectable dt scaling
                    err = err + (self.pred[gen_coord] - target).abs().unsqueeze(-1)
                error = (err * self.lower.states.cause_covar_slow**-1 * err).squeeze() # todo precision weighting for each dynamical state
                error.backward(gradient=torch.ones_like(error))

            # freeze learned precision during inference
            if not learn_covar:
                self.lower.states.cause_covar_slow.grad.zero_()
                self.lower.states.hidd_covar_slow.grad.zero_()

            # step dynamical variables
            if not self.dynamical:

                for opt in self.dyn_model.opts_d:
                    opt.step()  # todo merge with hierarchical opts
                    opt.zero_grad()  # todo fix

            # step hierarchical variables
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
                 cause_sizes=[32,32,32,32], hidden_sizes=[0,0,0,0], obs_layer=None, gen_coords=5):
        """
        Generalized predictive coding network
        """
        super().__init__()

        # specify network type: hierarchical or dynamical
        self.dynamical_net = dynamical_net
        self.gc = gen_coords

        # input layer (ignored by dynamical models since they observe hierarchical states)
        self.g_observation = GPC_layer(lower=None, b_size=b_size, dynamical=self.dynamical_net,
                                       h_states=cause_sizes[0], d_states=0,
                                       n_predstates_h=cause_sizes[0], n_predstates_d=0, gen_coords=self.gc)

        # output layer
        if obs_layer is None:
            lower_layer = self.g_observation
        else:
            lower_layer = obs_layer

        self.output_layer = GPC_layer(lower=lower_layer, b_size=b_size, dynamical=self.dynamical_net,
                                      h_states=cause_sizes[1], d_states=hidden_sizes[1],
                                      n_predstates_h=cause_sizes[1], n_predstates_d=hidden_sizes[1], gen_coords=self.gc)

        self.layers = [self.g_observation, self.output_layer]

        # hidden layers
        for l in range(len(cause_sizes)-2):
            hidden_layer = GPC_layer(lower=self.layers[-1], b_size=b_size, dynamical=self.dynamical_net,
                                      h_states=cause_sizes[l+2], d_states=hidden_sizes[l+2],
                                      n_predstates_h=cause_sizes[l+2], n_predstates_d=hidden_sizes[l+2], gen_coords=self.gc)
            self.layers.append(hidden_layer)

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
                self.opts_d.append(torch.optim.SGD(net_d.parameters(), lr=LR_WEIGHTS))
                self.nets_d.append(net_d) # dynamical weights
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

    def state_diff(self, step=False, dt=1.):
        """
        Compute difference between current and previous state
        """
        for i, l in enumerate(self.layers):
            l.states.change(step=step, dt=dt)

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
        return [p.detach().numpy() for p in prior]

    def iterative_inference(self, data=None, target=None, updates=1, predict_hierarchical=True):
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
            if predict_hierarchical:
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

    def decode_dynamical_hidden(self, t=0, dt=1.):
        """
        Decodes the dynamical states of a hierarchical layer into
        a discrete sequence of the corresponding hierarchical state

        t : position in decoded discrete sequence
        """

        if t==0: # return encoded state
            return [layer.states.hidd_state.detach()[0].numpy() for layer in self.layers]

        # apply encoded state motion to the lowest dynamical layers, from highest to lowest coordinate
        for coord in reversed(range(min(t,len(self.layers)-1))):
            state = self.layers[coord].states.hidd_state.detach()
            higher_state = self.layers[coord+1].states.hidd_state.detach()
            if ONE_TO_ONE:
                net_d = self.layers[coord+1].net_h_hidden.detach()
            else:
                net_d = self.layers[coord + 1].net_h_hidden
                net_d.weight = torch.nn.Parameter(net_d.weight.detach())

            if not ONE_TO_ONE:
                self.layers[coord].states.hidd_state = state + net_d(higher_state) * dt
            else:
                self.layers[coord].states.hidd_state = state + net_d * higher_state * dt

        return [layer.states.hidd_state.detach()[0].numpy() for layer in self.layers]


    def apply_encoded_motion(self, cause=True, hidden=True, dt=1.):
        """
        Compute the derivative of (causes or hidden) states in generalized coordinates
        and apply it to each respective lower state: x -> x + x', x' -> x' + x'', ...
        """
        for coord in range(len(self.layers) - 1):
            if cause:
                self.layers[coord].states.cause_state = self.layers[coord].states.cause_state.detach() \
                                                        + self.layers[coord + 1].states.cause_state.detach() * dt
            if hidden:
                self.layers[coord].states.hidd_state = self.layers[coord].states.hidd_state.detach() \
                                                       + self.layers[coord + 1].states.hidd_state.detach() * dt

    def decode_dynamical_cause(self, t=0, dt=1., autoregressive=True):
        """
        Decodes the dynamical states of a hierarchical layer into
        a discrete sequence of the corresponding hierarchical state

        t : position in decoded discrete sequence
        """

        if t==0: # return encoded state
            return [layer.states.cause_state.detach()[0].numpy() for layer in self.layers]

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
                self.layers[coord].states.cause_state = state + net_d(higher_state) * dt
            else:
                self.layers[coord].states.cause_state = state + net_d * higher_state * dt

        return [layer.states.cause_state.detach()[0].numpy() for layer in self.layers]

    def forward_dynamical(self, apply=False, dt=1.):
        """
        Dynamical prediction and optimization between all dynamical layers in a hierarchical layer
        This couples orders of state motion: x' = f(x,v), x'' = f'(x',v'), ...
        # todo condition f(x) on t: f(x,t)
        """

        for l in range(len(self.layers[:-1])):
            opt_states = torch.optim.SGD([self.layers[l].states.cause_state,
                                          self.layers[l].states.hidd_state,
                                          #self.layers[l+1].states.hidd_state,
                                          ], lr=LR_STATES_DYN) # todo fix LR
            opt_weights = torch.optim.SGD(self.nets_d[l].parameters(), lr=LR_WEIGHTS_DYN) # todo fix LR

            opt_states.zero_grad()
            opt_weights.zero_grad()

            # predict higher dynamical layer state (higher generalized coordinate)
            input_hidden = self.layers[l].states.hidd_state  # x
            input = torch.cat([self.layers[l].states.cause_state, input_hidden], 1)  # v,x
            target = self.layers[l + 1].states.hidd_state  # x'

            #pred = self.nets_d[l](input)  # f(v,x) # todo use this
            pred = torch.matmul(self.nets_d[0].weight, input.T).T  # matmul prediction # todo fix

            if not apply:
                # Dynamical prediction error
                err = ((pred - target)**2).mean(-1).unsqueeze(-1).unsqueeze(-1) # todo MSE?
                error = err # (err * self.covars_d[l]**-1 * err).squeeze()
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
            else:  # overwrite with prediction
                #self.layers[l].states.hidd_state = (self.layers[l].states.hidd_state + pred*dt).detach() # x_(t+1) = x_t + f(x)
                #if l == 0: print("x'", self.layers[l+1].states.hidd_state[0])
                self.layers[l+1].states.hidd_state = pred.detach() # x_(t+1) = x_t + f(x)
                #if l == 0:print("Predicted x'", self.layers[l+1].states.hidd_state[0])
                #if l == 0:print("---")
        if not apply:
            return err_out, covar_out

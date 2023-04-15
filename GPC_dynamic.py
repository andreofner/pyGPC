#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.distributions import MultivariateNormal
from torchvision.utils import make_grid
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time, random, logging

logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL)
plt.rcParams['figure.dpi'] = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[ ]:


""" Load dSprites dataset """
#!git clone https://github.com/deepmind/dsprites-dataset.git
#dataset_zip = np.load('./dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', allow_pickle=True, encoding="latin1")

dataset_zip = np.load('../../datasets/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', allow_pickle=True, encoding="latin1")
imgs = dataset_zip['imgs']
latents_values = dataset_zip['latents_values']
latents_classes = dataset_zip['latents_classes']
metadata = dataset_zip['metadata'][()]
latents_sizes = metadata['latents_sizes']
latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:], np.array([1,])))


# In[ ]:


# Helper functions
def latent_to_index(latents):
    return np.dot(latents, latents_bases).astype(int)

def sample_latent(size=1):
    samples = np.zeros((size, latents_sizes.size))
    for lat_i, lat_size in enumerate(latents_sizes):
        samples[:, lat_i] = np.random.randint(lat_size, size=size)
    return samples


# In[ ]:


GEN_COORDS = 3
LOCATIONS = 8
BATCH_SIZE= 32//LOCATIONS
DT = 1. 


# In[ ]:


""" Taylor series expansion kernels"""

def taylor_operator(n=3, dt=.1, t=0, plot=False):
    """ Finite difference coefficients for discrete Taylor series expansion"""
    s = np.round((t)/dt)
    k = np.asarray(range(1,n+1,1)) + np.trunc(s - (n + 1)/2)
    x = s - np.min(k) + 1

    # inverse embedding operator T: sequence = T*embedding
    T = np.zeros([n,n]) 
    for i in range(1, n+1, 1):
        for j in range(1, n+1, 1):
            T[i-1,j-1] = ((i-x)*dt)**(j-1) / np.prod(np.asarray(range(1,j,1)))

    # embedding operator E: embedding = E*sequence
    E = np.linalg.inv(np.matrix(T)) 
    
    if plot:
        print("E", E.round(2))
        print("T", T.round(2))
        plt.imshow(T, cmap="Greys_r"); plt.title("Inverse embedding operator"); plt.colorbar(); plt.show()
        plt.imshow(E, cmap="Greys_r"); plt.title("Forward embedding operator"); plt.colorbar(); plt.show()
        
    return torch.from_numpy(T).unsqueeze(-1).float(), torch.from_numpy(E).unsqueeze(1).float()

# temporal embedding operator (1D convolution): sequence -> embedding
T, E = taylor_operator(n=GEN_COORDS, dt=DT, t=0)
conv_E = torch.nn.Conv1d(1, GEN_COORDS, GEN_COORDS, stride=1, padding="valid", bias=False)  # same padding creates inaccuracies
conv_E.weight = torch.nn.Parameter(E)

# inverse temporal embedding operator (1D convolution): embedding -> sequence
conv_T = torch.nn.Conv1d(1, GEN_COORDS, GEN_COORDS, stride=1, padding="valid", bias=False)  # same padding creates inaccuracies
conv_T.weight = torch.nn.Parameter(T)

def embed_batch(seq_tensor_flat):
    out, labels_out = [], []
    for b in range(seq_tensor_flat.shape[0]):
        res = conv_E(seq_tensor_flat[b].transpose(0,1).unsqueeze(1).float())
        time_step = random.randint(0, res.shape[-1]-1)
        out.append(res[...,time_step].unsqueeze(0))
    return torch.cat(out)

def inverse_embed_batch(generalized):
    out = []
    for b in range(generalized.shape[0]):
        out.append(conv_T(generalized[b].unsqueeze(-1)).unsqueeze(0).squeeze(-1))
    return torch.cat(out)

T, E = taylor_operator(n=GEN_COORDS+2, dt=1, plot=True)


# In[ ]:


""" dSprites rotation dataset """
blurrer = torchvision.transforms.GaussianBlur(kernel_size=(3, 3), sigma=(10,10))
rot = torchvision.transforms.functional.rotate
interpol = torchvision.transforms.InterpolationMode.BILINEAR

def get_batch(steps=3, stride=20, scale=1, generalized=True, width=32, blur=True):
    """ dSprites rotation dataset """
    if generalized:
        latents_sampled = sample_latent(size=BATCH_SIZE)
        seq_out = []
        timesteps = GEN_COORDS*2+1+(steps*stride)
        random_direction = [random.randint(0,1)*2-1 for _ in range(BATCH_SIZE)]
        for t in range(timesteps):
            if t ==0:
                latents_sampled[:, 4] = random.randint(13,16) # Position X
                latents_sampled[:, 5] = random.randint(13,16) # Position Y
            indices_sampled = latent_to_index(latents_sampled)
            imgs_sampled = imgs[indices_sampled] 
            if blur:
                if t == 0:
                    img = blurrer(torch.tensor(imgs_sampled).float().unsqueeze(1))
                    imgs_sampled = rot(img, angle=0, interpolation=interpol)
                else:
                    imgs_sampled = torch.cat([rot(img_, angle=t*random_direction[i], interpolation=interpol) for i, img_ in enumerate(img)])
                imgs_sampled = imgs_sampled.squeeze(1).detach().numpy()
            seq_out.append(imgs_sampled)

        sequence, sequence_label, latents = [], [], []
        for step in range(0, steps*stride, stride):
            seq_tensor = torch.tensor([seq_out[time][:] for time in range(step, GEN_COORDS*2+1+step, 1)]).transpose(0,1).contiguous() # [batch, time, x, y]
            seq_tensor = seq_tensor[:, :, (32-width//2):(32+width//2), (32-width//2):(32+width//2)].contiguous()
            seq_tensor_flat = seq_tensor.view([BATCH_SIZE, -1, width*width]).float() * scale
            generalized = embed_batch(seq_tensor_flat) # [batchsize, 64*64, GEN_COORDS]
            sequence.append(generalized.unsqueeze(0))
            sequence_label.append(torch.Tensor([random_direction]))
            latents.append(torch.Tensor([latents_sampled]))

        return torch.cat(sequence), torch.cat(sequence_label), torch.cat(latents)
    else:
        latents_sampled = sample_latent(size=BATCH_SIZE)
        latents_sampled[:, 4] = random.randint(12,17) # Position X: 32 values in [0, 1]
        latents_sampled[:, 5] = random.randint(12,17) # Position Y: 32 values in [0, 1]
        indices_sampled = latent_to_index(latents_sampled)
        imgs_sampled = imgs[indices_sampled]    
        return torch.tensor(imgs[indices_sampled]).view(BATCH_SIZE, 1, 64, 64).float().contiguous()[:,:,(32-width//2):(32+width//2), (32-width//2):(32+width//2)] * scale


# In[ ]:


""" Create rotating dSprites dataset in generalized coordinates """
if False:
    dataset = []
    dataset_labels = []
    dataset_latents = []
    for batch in range(1000):
        if batch % 100 == 0: print("batch", batch)
        gen, gen_label, gen_latents = get_batch(steps=LOCATIONS, blur=True)
        dataset.append(gen.unsqueeze(0))
        dataset_labels.append(gen_label.unsqueeze(0))
        dataset_latents.append(gen_latents.unsqueeze(0))
    dsprites_generalized = torch.cat(dataset)
    dsprites_generalized_labels = torch.cat(dataset_labels).unsqueeze(-1)
    dsprites_generalized_latents = torch.cat(dataset_latents)
    del dataset; del dataset_labels

    torch.save(dsprites_generalized, f"./dsprites_generalized_blur")
    torch.save(dsprites_generalized_labels, f"./dsprites_generalized_blur_labels")
    torch.save(dsprites_generalized_latents, f"./dsprites_generalized_blur_latents")
    
# load dataset 
dsprites_generalized = torch.load(f"./dsprites_generalized_blur").to(device).detach()
dsprites_generalized_labels = torch.load(f"./dsprites_generalized_blur_labels").to(device).detach()
dsprites_generalized_latents = torch.load(f"./dsprites_generalized_blur_latents").to(device).detach()
print(dsprites_generalized.shape) 
print(dsprites_generalized_labels.shape) 
print(dsprites_generalized_latents.shape) 


# In[ ]:


# treat direction as latent factor
dsprites_generalized_labels = torch.concat([dsprites_generalized_labels, dsprites_generalized_latents],-1)


# In[ ]:


""" Full covariance and diagonal gaussian distributions """

class Gaussian():
    def __init__(self, mu, precision):
        self.mu = mu 
        self.precision = precision 
        self.dim = self.mu.shape[1]
        
    def compute_L(self):
        if self.L is None:
            try:
                self.L = torch.linalg.cholesky(torch.inverse(precision))
            except:
                print("Cholesky failed. Using standard normal.")
                self.L = torch.linalg.cholesky(torch.inverse(precision.cuda()*0.+torch.eye(mu.shape[-1]).cuda()).cuda()).cuda()

    def log_probability(self, x):
        self.compute_L()
        indices = np.arange(self.L.shape[-1])
        return -0.5 * (self.dim * np.log(2.0*np.pi)
                       + 2.0 * torch.log(self.L[:, indices, indices]).sum(1)
                       + torch.matmul(torch.matmul((x - self.mu).unsqueeze(1), self.precision),
                                      (x - self.mu).unsqueeze(-1)).sum([1, 2]))

    def sample(self):
        self.compute_L()
        eps = torch.randn_like(self.mu)
        return self.mu + torch.matmul(self.L, eps.unsqueeze(-1)).squeeze(-1)

    def repeat(self, n):
        mu = self.mu.unsqueeze(1).repeat(1, n, 1).view(-1, self.mu.shape[-1])
        precision = self.precision.unsqueeze(1).repeat(1, n, 1, 1).view(-1, *self.precision.shape[1:])
        return Gaussian(mu, precision)
    
    @staticmethod
    def weighted_error(p, q):
        error = (q.mu - p.mu).unsqueeze(-1)
        weighted_error = torch.matmul(torch.matmul(error.transpose(dim0=-2, dim1=-1), q.precision), error)
        return 0.5 * weighted_error
    
    @staticmethod
    def kl_div(p, q):
        n = p.mu.shape[1]
        weighted_precision = torch.matmul(q.precision, torch.inverse(p.precision))
        trace = torch.diagonal(weighted_precision, dim1=-2, dim2=-1).sum(-1) 
        det_q = torch.linalg.det(torch.inverse(q.precision))
        det_p = torch.linalg.det(torch.inverse(p.precision))
        det = torch.log(det_q/det_p)
        error = (q.mu - p.mu).unsqueeze(-1)
        weighted_error = torch.matmul(torch.matmul(error.transpose(dim0=-2, dim1=-1), q.precision), error)
        return 0.5 * (det + trace + weighted_error - n)
    
class DiagonalGaussian():
    def __init__(self, mu, logvar):
        self.mu = mu
        self.logvar = logvar

    def log_probability(self, x):
        return -0.5 * torch.sum(np.log(2.0*np.pi) + self.logvar + ((x - self.mu)**2)
                                    / torch.exp(self.logvar), dim=1)

    def sample(self):
        eps = torch.randn_like(self.mu)
        return self.mu + torch.exp(0.5 * self.logvar) * eps

    def repeat(self, n):
        mu = self.mu.unsqueeze(1).repeat(1, n, 1).view(-1, self.mu.shape[-1])
        logvar = self.logvar.unsqueeze(1).repeat(1, n, 1).view(-1, self.logvar.shape[-1])
        return DiagonalGaussian(mu, logvar)

    @staticmethod
    def kl_div(p, q):
        return 0.5 * torch.sum(q.logvar - p.logvar - 1.0 + (torch.exp(p.logvar) + (p.mu - q.mu)**2)/(torch.exp(q.logvar)), dim=1)


# In[ ]:


class DenseMLP(nn.Module):
    """ Hierarchical or dynamical prediction weights """
    def __init__(self, input_dim, output_dim, hidden_dim=512, activation=F.relu, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.linear_hidden0 = nn.Linear(self.input_dim, self.hidden_dim, bias=bias)
        self.linear_hidden1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)
        self.linear_mu = nn.Linear(self.hidden_dim, self.output_dim, bias=bias)
        self.activation = activation 
        self.logvar = torch.Tensor([0.0]).cuda()
        self.W_out = None

    def forward(self, z, order=0):
        if order == 0: # forward through ReLU and compute masks
            # input layer
            h = self.activation(self.linear_hidden0(z))
            activation_mask = torch.unsqueeze(h > 0.0, dim=-1).to(torch.float) 
            self.mask1 = activation_mask.detach()
            W = self.linear_hidden0.weight 
            W = activation_mask * W 
            
            # hidden layer
            h = self.activation(self.linear_hidden1(h))
            activation_mask = torch.unsqueeze(h > 0.0, dim=-1).to(torch.float)
            self.mask2 = activation_mask.detach()
            W = torch.matmul(self.linear_hidden1.weight, W)
            W = activation_mask * W
            
            # output layer
            self.W_out = torch.matmul(self.linear_mu.weight, W).detach()
            mu = self.linear_mu(h)
            return DiagonalGaussian(mu, self.logvar.detach()), self.W_out.detach()
        else: # forward through masks
            h = self.mask1.squeeze(-1)*self.linear_hidden0(z)
            h = self.mask2.squeeze(-1)*self.linear_hidden1(h)
            mu = self.linear_mu(h)
            return DiagonalGaussian(mu, self.logvar.detach()), self.W_out.detach()


# In[ ]:


class GPC(nn.Module):
    """ Hierarchical dynamical predictive coding layer """
    def __init__(self, obs_size=784, beta_cause=1, beta_hidden=1, activation=F.relu, skip_connections=False):
        super().__init__()
        
        # state sizes
        self.obs_size = obs_size # size of observation
        self.cause_states = latent_dim//2 # number of cause states
        self.hidden_states = latent_dim-self.cause_states  # number of hidden states
        self.z_dim = latent_dim # total number of states
        self.I = torch.eye(self.z_dim).unsqueeze(0).cuda()
        
        # inference hyperparameters
        self.lr = inference_lr # inference learning rate
        self.inference_steps = inference_steps # number of inference steps before each weight update
        self.beta_cause = beta_cause # weight cause states divergence from N(0,I) prior
        self.beta_hidden = beta_hidden # weight cause states divergence from N(0,I) prior
        self.skip_connections = skip_connections # additional connections from cause to response
        
        # hierarchical and dynamical weights
        self.decoder = DenseMLP(self.z_dim, self.obs_size, 1024, activation, bias=False).cuda() 
        self.transition = DenseMLP(self.z_dim, self.z_dim, 512, activation=activation, bias=False).cuda()
       
        #logging
        self.e_y, self.e_t, self.e_z = [[[] for gc in range(GEN_COORDS)] for _ in range(3)]  

    def forward(self, example, labels=None, init=True):
        example = example.unsqueeze(0)
        if labels is not None: 
            self.labels = labels.unsqueeze(0)

        if init:
            # initialise states
            self.z_inferred = torch.zeros([GEN_COORDS, example[0].shape[0], self.z_dim]).cuda()+0.000001
            
            # logging
            self.observation = example[0] # observed data  
            self.losses = [None for gc in range(GEN_COORDS)] # errors
            self.pred_h = [None for gc in range(GEN_COORDS)] # hierarchical prediction
            self.precision = [None for _ in range(self.inference_steps+1)] 
            self.precision_t = [None for _ in range(self.inference_steps+1)] 

        for step in range(self.inference_steps+1):
            # use labels if available
            if labels is not None:
                self.z_inferred[:,self.hidden_states:] = self.z_inferred[:,self.hidden_states:]*0
                self.z_inferred[0][:,self.hidden_states:] += labels[0]
            
            # compute jacobians
            states = self.z_inferred[0].clone().detach()
            if not self.skip_connections:
                    states[:,self.hidden_states:] *= 0 # exclude causes from hierarchical prediction
            _, W_dec = self.decoder.forward(states, 0) # hierarchical prediction
            _, W_dec_t = self.transition.forward(self.z_inferred[0], 0) # dynamical prediction

            # hierarchical precision
            self.W_dec = W_dec
            precision = torch.matmul(W_dec.transpose(1, 2), W_dec).detach() + self.I 
            self.precision[step] = precision.clone().detach()
            damped_precision = torch.inverse(precision)

            for gc in reversed(range(GEN_COORDS)):
                # predict
                x = example[0,...,gc].reshape([-1,width*width]).detach()
                states = self.z_inferred[gc].clone().detach()
                if not self.skip_connections:
                    states[:,self.hidden_states:] *= 0 # exclude causes from hierarchical prediction
                p_x_z, W_dec = self.decoder.forward(states, gc) # hierarchical prediction
                p_x_z_t, W_dec_t = self.transition.forward(self.z_inferred[gc], gc) # dynamical prediction
                
                # hierarchical prediction error (response)
                error_ = (p_x_z.mu - x.view(p_x_z.mu.shape)).unsqueeze(-1) # error
                error = torch.matmul(W_dec.transpose(1, 2), error_) # gradient
                error = torch.matmul(damped_precision, error).squeeze(-1) # precision weighted error
  
                # complexity (cause states)
                if (self.beta_cause+self.beta_hidden) > 0 and gc == 0:
                    error_prior_ = (self.z_inferred[gc] - 0) # standard normal prior
                    error_prior_[:,:self.hidden_states] *= self.beta_hidden # complexity on hidden states
                    error_prior_[:,self.hidden_states:] *= self.beta_cause # complexity on cause states
                    error_prior = torch.matmul(damped_precision, error_prior_.unsqueeze(-1)).squeeze(-1) # precision weighted error
                else:
                    error_prior = torch.zeros_like(self.z_inferred[gc])
                            
                # dynamical prediction error (hidden states)
                if gc < GEN_COORDS-1:
                    error_t_ = (p_x_z_t.mu - self.z_inferred[gc+1].detach().view(p_x_z_t.mu.shape)).unsqueeze(-1)
                    error_t_[:,self.hidden_states:] *= 0 # exclude causes from dynamical prediction # todo zero out input causes
                    error_t = torch.matmul(W_dec_t.transpose(1, 2), error_t_) # gradient
                    error_t = torch.matmul(damped_precision, error_t).squeeze(-1) # precision weighted error todo precision type
                else:
                    error_t_ = error.unsqueeze(-1)*0
                    error_t = error_t_.squeeze(-1)

                if step == self.inference_steps: # learning
                    self.losses[gc] = (error_**2).squeeze(-1).sum(-1).mean() + (error_t_**2).squeeze(-1).sum(-1).mean()

                    # logging
                    self.pred_h[gc] = p_x_z.mu # final hierarchical prediction
                    self.e_y[gc] = (error_**2).mean().item() # hierarchical accuracy
                    self.e_z[gc] = (error_prior_**2).mean().item() # complexity
                    self.e_t[gc] = (error_t_**2).mean().item() # dynamical accuracy
                else: # inference
                    # update hidden state wrt. each individual step in sequence
                    grad = (error+error_prior+error_t)[:,:self.hidden_states]
                    self.z_inferred[gc].data[:,:self.hidden_states] = self.z_inferred[gc].data[:,:self.hidden_states] - self.lr*grad

                    # update cause state wrt. mean of all steps in sequence
                    grad_cause = (error_prior+error_t)[:,self.hidden_states:]
                    if self.skip_connections: # skip connections: hierarchical error for causes
                        grad_cause += error[:,self.hidden_states:]
                    chunks = torch.stack(torch.chunk(grad_cause, BATCH_SIZE, dim=0))
                    chunks = chunks.mean(1, keepdim=True).repeat([1,LOCATIONS,1]).view(grad_cause.shape)
                    self.z_inferred[gc].data[:,self.hidden_states:] = self.z_inferred[gc].data[:,self.hidden_states:] - self.lr*chunks

        return self.losses
    
    def get_prediction(self, plot=True, batch_id=0):
        """ Decode hierarchical prediction and project to discrete sequence """
        self.gen_response = torch.cat([p.detach().unsqueeze(-1) for p in self.pred_h],-1)
        self.true_response = self.observation.view(self.gen_response.shape)
        self.state =  torch.cat([p.detach().unsqueeze(-1) for p in self.z_inferred],-1)
        self.inv = inverse_embed_batch(self.gen_response.float().cpu()) 
        self.inv_true = inverse_embed_batch(self.true_response.float().cpu())
        self.inv_state = inverse_embed_batch(self.state.float().cpu())

        if plot:
            plt.rcParams['figure.dpi'] = 100
            fig, ax = plt.subplots(nrows=GEN_COORDS, ncols=4)
            fig.subplots_adjust(top=0.8)
            for t in range(GEN_COORDS):
                ax[t][0].imshow(self.inv_true[batch_id][...,t].reshape([width,width]).cpu().detach().numpy(), cmap="Greys_r");
                ax[t][1].imshow(self.true_response[batch_id][...,t].reshape([width,width]).cpu().detach().numpy(), cmap="Greys_r");
                ax[t][2].imshow(self.gen_response[batch_id][...,t].reshape([width,width]).cpu().detach().numpy(), cmap="Greys_r"); 
                ax[t][3].imshow(self.inv[batch_id][...,t].reshape([width,width]).cpu().detach().numpy(), cmap="Greys_r");   
                [ax[t][i].axis('off') for i in range(4)]
            for i, name in enumerate(["Observation", "Generalized\nobservation", "Generalized\nprediction", "Projected\nprediction"]):
                ax[0][i].set_title(name, size=9)
            plt.show()
            
        return self.state
            
    def predict_sequence(self, plot=True, timesteps=5, stride=1, direction=1, title="", plot_orders=GEN_COORDS, save=False):
        """ Predict latent state sequence by transitioning the generalized latent state """
        state = self.get_prediction(plot=False)
        seq = [state.float().cpu()]

        # transition hidden state sequence in generalised coordinates
        for t in range(timesteps):
            for gc in range(1, GEN_COORDS,1): # predict [state->velocity, velocity->acceleration, ...]
                state[:,:self.hidden_states,gc] = self.transition.forward(state[...,gc-1].cuda().detach(), 0)[0].mu[:,:self.hidden_states] 
            for gc in reversed(range(GEN_COORDS-1)): # apply [..., acceleration -> velocity, velocity -> state]
                state[:,:self.hidden_states,gc] += direction*state[:,:self.hidden_states, gc+1]*DT
            seq.append(state.clone())

        if plot:
            # decode hidden state sequence into sensory predictions
            fig, ax = plt.subplots(nrows=plot_orders, ncols=(timesteps//stride)+1, figsize=(timesteps//stride, plot_orders))
            if plot_orders <= 1: ax=[ax]
            for i, curr in enumerate(seq[::stride]):
                for gc in range(plot_orders):
                    p_x_z, _ = self.decoder.forward(curr[...,gc].cuda(), gc)
                    ax[plot_orders-gc-1][i].imshow(p_x_z.mu[0].reshape([width,width]).cpu().detach(), cmap="Greys_r")
                    ax[plot_orders-gc-1][i].set_xticks([])
                    ax[plot_orders-gc-1][i].set_yticks([])
            del state ; del seq; torch.cuda.empty_cache()
            state_name = f"dt={direction}" if plot_orders ==1 else f"State"
            for gc, gc_name in enumerate([state_name, "Velocity", "Acceleration"][:plot_orders]):
                ax[plot_orders-gc-1][0].set_ylabel(gc_name, rotation='horizontal', ha="right", va="center")
            if save: 
                plt.savefig("extrapolation_"+title+".pdf")
            plt.show()
    


# In[ ]:


inference_lr = .9 
latent_dim = 20
width = 32
inference_steps = 5

# load dataset
dataset = TensorDataset(dsprites_generalized.transpose(2,1).contiguous(), 
                        dsprites_generalized_labels.transpose(2,1).contiguous())
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# create model
pcn = GPC(obs_size=width**2, beta_cause=0.1, beta_hidden=0.1).to(device) 
opt = torch.optim.Adam(pcn.parameters(), lr=0.0001)

layer_losses, acc, compl, acc_t = [[[] for gc in range(GEN_COORDS)] for _ in range(4)]
for epoch in range(15): 
    start = time.time()
    for update, (batch, labels) in enumerate(loader):
        example = batch[0].view([-1, width,width, GEN_COORDS])
        labels = labels[0][...,:1].view([-1, 1])
        opt.zero_grad()
        losses = pcn(example, None)
        for gc in range(GEN_COORDS):
            losses[gc].backward(retain_graph=True)
            layer_losses[gc].append(losses[gc].item())
            acc[gc].append(pcn.e_y[gc])
            compl[gc].append(pcn.e_z[gc])
            acc_t[gc].append(pcn.e_t[gc])
        opt.step()
    if epoch % 1 == 0:
        print("Epoch", epoch, " runtime ", str(time.time()-start))
        pcn.get_prediction(plot=True)
        pcn.predict_sequence(plot=True, timesteps=50, stride=5, title=f"100", plot_orders=1, direction=1)
        pcn.predict_sequence(plot=True, timesteps=50, stride=5, title=f"100", plot_orders=1, direction=-1)

for i, stats in enumerate([layer_losses, acc, compl, acc_t]):
    np.save(f"./stats_{i}_{inference_steps}", np.asarray(stats))


# In[ ]:


""" 2D projection of causes and hidden states"""
import seaborn as sns
import pandas as pd  
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

gc = 0 
causes, hiddens, label = [], [], []
for update, (batch, labels) in enumerate(loader):
    example = batch[0].view([-1, width,width, GEN_COORDS])
    labels = labels[0].view([-1, 7])
    losses = pcn(example, None)
    causes.append(pcn.z_inferred[gc,:,pcn.hidden_states:])
    hiddens.append(pcn.z_inferred[gc,:,:pcn.hidden_states])
    label.append(labels)
    if update == 200: 
        break

# keep only first location in each sequence
causes = torch.concat(causes).cpu().detach().numpy()[::BATCH_SIZE] 
hiddens = torch.concat(hiddens).cpu().detach().numpy()[::BATCH_SIZE] 
label = torch.concat(label).cpu().detach().numpy()[::BATCH_SIZE]

# tSNE on causes
if causes.shape[-1] == 2:
    z = causes # use two dimensional cause (no tSNE)
else:
    z = TSNE(n_components=2).fit_transform(causes)  

# tSNE on hiddens
if hiddens.shape[-1] == 2:
    z_hidden = hiddens # use two dimensional cause (no tSNE)
else:
    z_hidden = TSNE(n_components=2).fit_transform(hiddens)  

df = pd.DataFrame()
df["y"] = label[:,0].astype(np.int) # direction
df["y2"] = label[:,2] # shape
df["y3"] = label[:,3] # scale
df["Cause state t-SNE 1"] = z[:,0]
df["Cause state t-SNE 2"] = z[:,1]
df["Hidden state t-SNE 1"] = z_hidden[:,0]
df["Hidden state t-SNE 2"] = z_hidden[:,1]

plt.rcParams.update({'font.size': 18})

# plot cause states
fig, axs = plt.subplots(1,2, figsize=(12,5))
palette = sns.color_palette("colorblind", 5) 
res = sns.scatterplot(x="Cause state t-SNE 1", y="Cause state t-SNE 2", ax=axs[0],
                      hue=df.y.tolist(), size=df.y3.tolist(),
                palette=palette[:len(df.y.unique())], data=df).set(title="Cause states")
legend_elements = [Line2D([0], [0], color="w", markersize=10, marker='o', label='CW', markerfacecolor=palette[0]),
                   Line2D([0], [0], color="w", markersize=10, marker='o', label='CCW',  markerfacecolor=palette[1])]
axs[0].legend(handles=legend_elements, title="Direction", loc='best', fontsize=12);  #axs[0].grid()
axs[0].axis("off")

# plot hidden states
res = sns.scatterplot(x="Hidden state t-SNE 1", y="Hidden state t-SNE 2", ax=axs[1], 
                      hue=df.y2.tolist(), size=df.y3.tolist(),
                palette=palette[2:len(df.y2.unique())+2], data=df).set(title="Hidden states")
legend_elements = [Line2D([0], [0], color="w", marker='o', markersize=10, label='Square', markerfacecolor=palette[2]),
                   Line2D([0], [0], color="w", marker='o', markersize=10, label='Ellipse',  markerfacecolor=palette[3]),
                   Line2D([0], [0], color="w", marker='o', markersize=10, label='Heart', markerfacecolor=palette[4]),]
axs[1].legend(handles=legend_elements, title="Shape", loc="best", fontsize=12); #axs[1].grid()
axs[1].axis("off")
plt.tight_layout()
plt.savefig(f"clustering_{gc}.pdf")
plt.title(f"Order {gc}")
plt.show()
plt.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.distributions import MultivariateNormal
import torchvision
import datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

plt.rcParams['figure.dpi'] = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BaseEncoder(nn.Module):
    def __init__(self, z_dim, x_dim, h_dim):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.linear_hidden0 = nn.Linear(self.x_dim, self.h_dim)
        self.linear_hidden1 = nn.Linear(self.h_dim, self.h_dim)
        self.linear_mu = nn.Linear(self.h_dim, self.z_dim)
        self.linear_logvar = nn.Linear(self.h_dim, self.z_dim)
        self.activation = F.relu

    def forward(self, x):
        h = self.activation(self.linear_hidden0(x))
        h = self.activation(self.linear_hidden1(h))
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)

        return DiagonalGaussian(mu, logvar), h

class BaseDecoder(nn.Module):
    def __init__(self, z_dim, x_dim, h_dim):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.linear_hidden0 = nn.Linear(self.z_dim, self.h_dim)
        self.linear_hidden1 = nn.Linear(self.h_dim, self.h_dim)
        self.linear_mu = nn.Linear(self.h_dim, self.x_dim)
        self.activation = F.relu
        self.logvar = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, z, compute_jacobian=False):
        if compute_jacobian:
            h = self.activation(self.linear_hidden0(z))
            activation_mask = torch.unsqueeze(h > 0.0, dim=-1).to(torch.float)
            W = self.linear_hidden0.weight
            W = activation_mask * W

            h = self.activation(self.linear_hidden1(h))
            activation_mask = torch.unsqueeze(h > 0.0, dim=-1).to(torch.float)
            W = torch.matmul(self.linear_hidden1.weight, W)
            W = activation_mask * W

            W = torch.matmul(self.linear_mu.weight, W)

            mu = self.linear_mu(h)
            W_out = W

            return DiagonalGaussian(mu, self.logvar), W_out

        else:
            h = self.activation(self.linear_hidden0(z))
            h = self.activation(self.linear_hidden1(h))
            mu = self.linear_mu(h)
            return DiagonalGaussian(mu, self.logvar)

        
class Gaussian():
    def __init__(self, mu, precision):
        self.mu = mu #[batch_size, z_dim]
        self.precision = precision # [batch_size, z_dim, z_dim]
        self.L = None
        self.dim = self.mu.shape[1]
        
    def compute_L(self):
        if self.L is None:
            try:
                self.L = torch.linalg.cholesky(torch.inverse(self.precision))
            except:
                print("Cholesky failed. Using standard normal.")
                self.L = torch.linalg.cholesky(torch.inverse(self.precision.cuda()*0.+torch.eye(mu.shape[-1]).cuda()).cuda()).cuda()

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

def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        try:
            z = autoencoder.encoder(x.to(device).view([-1,784])).mu
        except:
            z = autoencoder.encoder(x.to(device).view([-1,784]))[0].mu
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
    plt.show()

def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.mu.reshape(28, 28).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])
    plt.show()
    


# In[ ]:


class VLAE(nn.Module):
    def __init__(self, image_size=784, activation=F.relu):
        super().__init__()
        self.test = False
        self.update_lr = 0.5
        self.z_dim = latent_dim
        self.image_size = image_size
        self.encoder = BaseEncoder(z_dim=latent_dim, x_dim=784, h_dim=hidden_dim)
        self.decoder = BaseDecoder(z_dim=latent_dim, x_dim=784, h_dim=hidden_dim)
        self.prior = DiagonalGaussian(mu=torch.zeros(1,1).cuda(), logvar=torch.zeros(1,1).cuda())
        
    def solve_mu(self, x, mu_prev, p_x_z, W_dec):
        var_inv = torch.exp(-self.decoder.logvar).unsqueeze(1)
        precision = torch.matmul(W_dec.transpose(1, 2) * var_inv, W_dec)
        precision += torch.eye(precision.shape[1]).unsqueeze(0).cuda()
        bias = p_x_z.mu.unsqueeze(-1) - torch.matmul(W_dec, mu_prev.unsqueeze(-1))
        mu = torch.matmul(W_dec.transpose(1, 2) * var_inv, x.view(-1, self.image_size, 1) - bias)
        mu = torch.matmul(torch.inverse(precision), mu)
        mu = mu.squeeze(-1)
        return mu, precision
    
    def update_rate(self, t):
        return self.update_lr / (t+1)

    def forward(self, x, sample):
        q_z_x, _ = self.encoder.forward(x)
        mu = q_z_x.mu

        for i in range(inference_steps):
            p_x_z, W_dec = self.decoder.forward(mu, compute_jacobian=True)
            mu_new, precision = self.solve_mu(x, mu, p_x_z, W_dec)

            lr = self.update_rate(i)
            mu = (1 - lr) * mu + lr * mu_new

        p_x_z, W_dec = self.decoder.forward(mu, compute_jacobian=True)
        var_inv = torch.exp(-self.decoder.logvar).unsqueeze(1)      

        precision = torch.matmul(W_dec.transpose(1, 2) * var_inv, W_dec)
        precision += torch.eye(precision.shape[1]).unsqueeze(0).cuda()

        q_z_x = Gaussian(mu, precision)
        z = q_z_x.sample() # reparam trick
        self.p_x_z = self.decoder.forward(z)

        if self.test:
            return self.loss_torch(x, z, self.p_x_z, self.prior, q_z_x), self.p_x_z 
        else:
            return self.loss(x, z, self.p_x_z, self.prior, q_z_x), self.p_x_z 

    def loss_torch(self, x, z, p_x_z, p_z, q_z_x):
        self.prior_full = Gaussian(mu=torch.zeros(z.shape[0], self.z_dim).cuda(), 
                           precision=torch.eye(self.z_dim,self.z_dim).repeat(z.shape[0],1,1).cuda())
        self.prior_full_torch = MultivariateNormal(self.prior_full.mu, 
                                                   precision_matrix=self.prior_full.precision)
        
        
        q_z_x_torch = MultivariateNormal(q_z_x.mu, precision_matrix=q_z_x.precision) # posterior 
        self.kl_inferred = torch.distributions.kl.kl_divergence(q_z_x_torch, self.prior_full_torch) # complexity
        self.error_p_x_z = p_x_z.log_probability(x.view(-1, self.image_size)) # accuracy 
        self.ELBO = -torch.mean(self.error_p_x_z - self.kl_inferred)
        return self.ELBO
    
    def loss(self, x, z, p_x_z, p_z, q_z_x):
        return -torch.mean(p_x_z.log_probability(x) + p_z.log_probability(z) - q_z_x.log_probability(z))
        


# In[ ]:


class PC(nn.Module):
    def __init__(self, image_size=784, activation=F.relu):
        super().__init__()
        self.test = False
        self.z_dim = latent_dim
        self.image_size = image_size
        self.encoder = BaseDecoder(z_dim=784, x_dim=latent_dim, h_dim=hidden_dim)
        self.decoder = BaseDecoder(z_dim=latent_dim, x_dim=784, h_dim=hidden_dim)
        self.prior = DiagonalGaussian(mu=torch.zeros(1,1).cuda(), logvar=torch.zeros(1,1).cuda())

    def forward(self, x, sample=False):
        # top-down prediction
        q_z_x = self.encoder.forward(x)
        p_x_z, W_dec = self.decoder.forward(q_z_x.mu.detach(), compute_jacobian=True)
        precision = torch.matmul(W_dec.transpose(1, 2), W_dec)
        precision += torch.eye(precision.shape[1]).unsqueeze(0).cuda()
        self.q_z_x_TD = Gaussian(q_z_x.mu, precision.detach()) 
        mu = self.q_z_x_TD.mu
        
        # iterative inference
        mu = mu.detach() 
        self.I = torch.eye(precision.shape[1]).unsqueeze(0).cuda()
        for step in range(inference_steps):
            p_x_z, W_dec = self.decoder.forward(mu, compute_jacobian=True)
            
            pred_no_bias = torch.matmul(W_dec, mu.unsqueeze(-1)).squeeze(-1)
            bias = p_x_z.mu - pred_no_bias

            error = (pred_no_bias - x.view(p_x_z.mu.shape) - bias).unsqueeze(-1)
            error = torch.matmul(W_dec.transpose(1, 2), error) 
            error_prior = (mu - 0)

            self.lr = inference_lr
            if False: # precision weighted inference
                error_prior = error_prior.unsqueeze(-1)

                precision = torch.matmul(W_dec.transpose(1, 2), W_dec) 
                precision += torch.eye(precision.shape[1]).unsqueeze(0).cuda() 

                mu = mu - self.lr * torch.matmul(precision, error).squeeze(-1) \
                    - self.lr *  torch.matmul(precision, error_prior).squeeze(-1)
            else:
                mu = mu - self.lr * error.squeeze(-1) - self.lr * error_prior

        p_x_z, W_dec = self.decoder.forward(mu, compute_jacobian=True)
        var_inv = torch.exp(-self.decoder.logvar).unsqueeze(1)      

        # precision
        precision = torch.matmul(W_dec.transpose(1, 2) * var_inv, W_dec)
        precision += torch.eye(precision.shape[1]).unsqueeze(0).cuda()

        # decode posterior
        q_z_x = Gaussian(mu, precision.detach())
        if sample:
            z = q_z_x.sample()
        else:
            z = q_z_x.mu
        self.p_x_z = self.decoder.forward(z.detach()) 
        
        if self.test:
            # decode posterior sample
            z = q_z_x.sample()
            self.p_x_z = self.decoder.forward(z.detach()) 
            return self.loss_torch(x, z, self.p_x_z, self.prior, q_z_x), self.p_x_z 
        else:
            return self.loss_separate(x, z, self.p_x_z, self.prior, q_z_x, self.q_z_x_TD), self.p_x_z 

    def loss_torch(self, x, z, p_x_z, p_z, q_z_x):
        self.prior_full = Gaussian(mu=torch.zeros(z.shape[0], self.z_dim).cuda(), 
                           precision=torch.eye(self.z_dim,self.z_dim).repeat(z.shape[0],1,1).cuda())
        self.prior_full_torch = MultivariateNormal(self.prior_full.mu, 
                                                   precision_matrix=self.prior_full.precision)
        
        q_z_x_torch = MultivariateNormal(q_z_x.mu, precision_matrix=q_z_x.precision) # posterior 
        self.kl_inferred = torch.distributions.kl.kl_divergence(q_z_x_torch, self.prior_full_torch) # complexity
        self.error_p_x_z = p_x_z.log_probability(x.view(-1, self.image_size)) # accuracy 
        self.ELBO = -torch.mean(self.error_p_x_z - self.kl_inferred)
        return self.ELBO
    
    def loss(self, x, z, p_x_z, p_z, q_z_x):
        self.kl_inferred = -(p_z.log_probability(z) - q_z_x.log_probability(z)) # todo
        self.error_p_x_z = p_x_z.log_probability(x) # todo
        self.ELBO = -torch.mean(self.error_p_x_z - self.kl_inferred) # todo
        return -torch.mean(p_x_z.log_probability(x) + p_z.log_probability(z) - q_z_x.log_probability(z))
    
    def loss_separate(self, x, z, p_x_z, p_z, q_z_x, q_z_x_TD, use_KL=False):
        if use_KL:
            q_z_x_TD_torch = MultivariateNormal(q_z_x_TD.mu, precision_matrix=q_z_x_TD.precision) # top-down prediction 
            q_z_x_torch = MultivariateNormal(q_z_x.mu, precision_matrix=q_z_x.precision) # inferred posterior 
            
            # full KL
            self.kl_TD = torch.distributions.kl.kl_divergence(q_z_x_torch, q_z_x_TD_torch) # top-down PE
            self.kl_inferred = 0. #torch.distributions.kl.kl_divergence(q_z_x_TD_torch, self.prior_full_torch) # complexity
        else:
            # weighted prediction errors
            self.kl_TD = Gaussian.weighted_error(q_z_x, q_z_x_TD).mean() # top-down PE
            self.kl_inferred = 0. #Gaussian.weighted_error(q_z_x_TD, self.prior_full).mean() # complexity
        
        self.error_p_x_z = p_x_z.log_probability(x.view(-1, self.image_size)) # accuracy (bottom-up PE)
        
        # total error
        self.ELBO = -torch.mean(self.error_p_x_z - self.kl_inferred - self.kl_TD)
        return self.ELBO
    


# In[ ]:


class VAE(nn.Module):
    def __init__(self, image_size=784, activation=F.relu):
        super().__init__()
        self.test = False
        self.z_dim = latent_dim
        self.encoder = BaseEncoder(z_dim=latent_dim, x_dim=784, h_dim=hidden_dim)
        self.decoder = BaseDecoder(z_dim=latent_dim, x_dim=784, h_dim=hidden_dim)
        self.prior = DiagonalGaussian(mu=torch.zeros(1,1).cuda(), logvar=torch.zeros(1,1).cuda())
        self.image_size = image_size

    def forward(self, x, sample):
        q_z_x, _ = self.encoder(x)
        z = q_z_x.sample()
        p_x_z = self.decoder(z)
        return self.loss(x, z, p_x_z, self.prior, q_z_x), p_x_z

    def loss(self, x, z, p_x_z, p_z, q_z_x):
        self.kl_inferred = DiagonalGaussian.kl_div(q_z_x, p_z) 
        self.error_p_x_z = p_x_z.log_probability(x) 
        self.ELBO = -torch.mean(self.error_p_x_z - self.kl_inferred) 
        return self.ELBO


# In[ ]:


def train(model, data, dataset, max_updates):
    model.test = False
    start = time.time()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    updates = 0
    for epoch in range(100):
        #print("Epoch", epoch)
        for batch, (x, y) in enumerate(data):
            x = x.to(device)
            x = dataset.preprocess(x).view([-1,784])
            
            opt.zero_grad()
            loss, pred = model.forward(x, sample=SAMPLE)
            loss.backward()
            opt.step()
            updates += 1
            if updates > max_updates:
                #print("Runtime: ", time.time() - start)
                return model

def visualize(data, dataset):
    for batch, (x, y) in enumerate(data):
        #x = x.to(device).view([-1,784])

        x = x.to(device)
        x = dataset.preprocess(x).view([-1,784])

        plt.imshow((dataset.unpreprocess(x).view([-1,784])[0]).detach().cpu().reshape([28,28]));
        plt.colorbar(); plt.title("Target"); plt.show()

        loss, pred = vae.forward(x, sample=False)
        plt.imshow((dataset.unpreprocess(pred.mu[0]).view([-1,784])).detach().cpu().reshape([28,28]));
        plt.colorbar(); plt.title("Mean"); plt.show()   

        loss, pred = vae.forward(x, sample=True)
        plt.imshow((dataset.unpreprocess(pred.mu[0]).view([-1,784])).detach().cpu().reshape([28,28]));
        plt.colorbar(); plt.title("Sample"); plt.show()   

        if batch > 1: 
            break

    #plot_latent(vae, test_data, num_batches=100)
    #plot_reconstructed(vae, r0=(-3, 3), r1=(-3, 3))


# In[ ]:


SAMPLE = False
hidden_dim = 256 
latent_dim = 16
inference_steps = 1
inference_lr = 0.001
updates = 10000
BATCH_SIZE = 64

for inference_steps in range(1,11,1):
    for DS in [datasets.MNIST, datasets.OMNIGLOT, datasets.FashionMNIST][1:2]:
        for SAMPLE in [False]:
            dataset = DS(batch_size=BATCH_SIZE, logit_transform=True)
            train_data = dataset.train_loader
            test_data = dataset.test_loader

            for run in range(1):
                vae = PC().to(device) # create model
                vae = train(vae, train_data, dataset, updates) # train model
                E, C, A = test(vae, test_data, dataset) # test model
                print(inference_steps, E,C,A)

inference_lr = 0.01
Es_high, Cs_high, As_high = [], [], []
for inference_steps in range(1,11,1):
    for DS in [datasets.MNIST, datasets.OMNIGLOT, datasets.FashionMNIST]:
        for SAMPLE in [False]:
            dataset = DS(batch_size=BATCH_SIZE, logit_transform=True)
            train_data = dataset.train_loader
            test_data = dataset.test_loader

            for run in range(1):
                vae = PC().to(device) # create model
                vae = train(vae, train_data, dataset, updates) # train model
                E, C, A = test(vae, test_data, dataset) # test model
                print(inference_steps, E,C,A)
                


# In[ ]:





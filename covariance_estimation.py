import torch
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn'])

# variance and noise settings
SAMPLES = 300  # number of observations
MOD = 3  # scales variance of observation mean
random_scale = .3  # scales additive gaussian noise on observation mean

# initialise variables
ins = torch.zeros((1,3))
sigma, sigma_weighted = torch.zeros((1,3,3)), torch.zeros((1,3,3))
means = torch.zeros((1,3))
observations, bbs, est_covars, est_means = [], [], [], []

def ML_covar(sigma, x, x_mean, N):
    """ Updates ML covariance estimate given new observation and estimated mean"""
    for j in range(sigma.shape[-1]):
        for k in range(sigma.shape[-2]):
            sigma[0, j, k] = sigma[0, j, k] + (x[0, j]-x_mean[0, j]) * (x[0, k]-x_mean[0, k])
    return sigma, sigma/N

def ML_mean(x_mean, x, N):
    """ Updates ML mean estimate given new observation"""
    x_mean = x_mean + x
    return x_mean, x_mean/N

def plot_covariance(observations, est_means, est_covars):
    """ Plots variance estimation updates and the resulting covariance matrix"""
    fig, axes = plt.subplots(ins.shape[-1]+1)
    for pos, ax in enumerate(axes[:-1]):
        ax.plot(np.asarray(observations)[:,pos], color="lightblue", label="Observation")
        ax.plot(np.asarray(est_means)[:,pos], color="blue", label="Estimated mean", linestyle="--")
        ax.plot(np.asarray(est_covars)[:, pos, pos], color="green", label="Estimated variance", linestyle="-.")
        ax.grid("both")
    ax.set_xlabel("Update");
    plt.suptitle("Covariance estimation: Maximum Likelihood")
    clb = axes[-1].imshow(sigma_weighted[0].detach())
    axes[-1].set_xlabel("Estimated\nCovariance")
    axes[-1].set_yticks([]); axes[-1].set_xticks([])
    plt.tight_layout()
    fig.colorbar(clb)
    ax.legend(loc="lower left", bbox_to_anchor=(0, -2))
    plt.show()

for i in range(SAMPLES):
    # generative some data with constant variance and add random noise
    ins[0, 0] = (i % MOD) + 0.5 + np.random.normal(0,random_scale)
    ins[0, 1] = -ins[0, 0] + np.random.normal(0,random_scale)
    ins[0, 2] = -ins[0, 0]*0.5 - 2 + np.random.normal(0,random_scale)

    # update mean estimate
    means, mean_weighted = ML_mean(means, ins, i)

    # update covariance estimate
    if i > 20: # wait until mean is estimated
        sigma, sigma_weighted = ML_covar(sigma, ins, mean_weighted, i)

    # log results
    observations.append(ins[0].clone().detach().numpy())
    est_means.append(mean_weighted.clone().detach().numpy()[0])
    est_covars.append(sigma_weighted[0].clone().detach().numpy())

plot_covariance(observations, est_means, est_covars)

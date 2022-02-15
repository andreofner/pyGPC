import torch
import matplotlib.pyplot as plt
plt.style.use(['seaborn'])
import numpy as np

""" 
Minimal example for dynamical weights optimisation: 
- estimating the dynamics of a Lorenz attractor
- top-down parameters (= cause states = Prandtl number, Rayleigh number, etc.) are known, 
- weights parameterizing x_dot, y_dot, z_dot (hidden states motion) have to be estimated
- single layer PCN without incoming or outgoing prediction, hidden state are set equal to the sensory observation
- precision is estimated on the dynamical prediction error
See https://matplotlib.org/stable/gallery/mplot3d/lorenz_attractor.html for the ground truth attractor
"""

dt = 0.01
scale = 1  # changes start position
s,r,b = 10, 28, 2.667
LR_PRECISION = 0
BATCH_SIZE = 256
plt_range = 0
num_steps = BATCH_SIZE*2
start_interpolation = num_steps-plt_range

def lorenz(x, y, z, s=10, r=28, b=2.667):
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

def optimize_PCN(x, x_past,  s=10, r=28, b=2.667, optimize=True):
    """ Optimize dynamical prediction """
    global datapoint; global sigma
    error = torch.tensor([100.])
    x = x.float()  # current hidden state
    x_past = x_past.float().detach()  # past hidden state
    v = torch.tensor([[s, r, b]]).detach()  # cause state
    opt = torch.optim.SGD([net_dynamical_x.weight, net_dynamical_y.weight, net_dynamical_z.weight], lr=.00001)
    opt_sigma = torch.optim.SGD([sigma], lr=LR_PRECISION)
    opt.zero_grad()
    opt_sigma.zero_grad()
    updates = 0

    for i in range(1000):
        updates += 1
        nets = [net_dynamical_x, net_dynamical_y, net_dynamical_z]
        preds = []
        for pos in range(3):
            if True:
                x_past_ = x_past.clone().detach()*0.
                if pos == 0:
                    x_past_[:, 1:] = x_past[:, 1:]
                elif pos == 1:
                    x_past_[:, 2] = x_past[:, 2]
                    x_past_[:, 0] = x_past[:, 0]
                else:
                    x_past_[:, :2] = x_past[:, :2]
            else:
                x_past_ = x_past
            preds.append((nets[pos](torch.cat([v.detach().repeat([BATCH_SIZE-1,1]),x_past_], -1)) * x_past[:,pos:pos+1]))
        pred = torch.cat(preds, -1)
        if optimize:
            opt.zero_grad()
            opt_sigma.zero_grad()
            error = (x.float() - pred).abs().mean()
            error_weighted = sigma**-1 * error
            error_weighted.mean().backward() # todo sum or mean batch aggreagtion?
            opt.step()
            if updates == 1: # tod learn not infer precision
                opt_sigma.step()
                if (sigma - LR_PRECISION * sigma ** -1).detach() >= 1:  # define maximum allowed precision
                    sigma = (sigma - LR_PRECISION * sigma ** -1).detach()  # cause state variance decay
        if error.abs().sum() < 1:
            break
    print(updates, "updates", error.sum())
    return pred[:,0].detach(), pred[:,1].detach(), pred[:,2].detach(), sigma.detach().numpy()

# set up model logging
m_xs = [0.*scale]
m_ys = [1.*scale]
m_zs = [1.05*scale]
sigmas = [0.]

# set up true attractor
xs = [0.*scale]
ys = [1.*scale]
zs = [1.05*scale]

# set up weights
net_dynamical_x = torch.nn.Linear(6, 1, bias=False)
net_dynamical_y = torch.nn.Linear(6, 1, bias=False)
net_dynamical_z = torch.nn.Linear(6, 1, bias=False)
sigma = (torch.ones([1])*1).requires_grad_()

for i in range(int(num_steps/BATCH_SIZE)):
    """ step true attractor and turn sequence into batch of shape [[steps], [x,y,z]] """
    for j in range(BATCH_SIZE):
        x_dot, y_dot, z_dot = lorenz(xs[-1], ys[-1], zs[-1])
        xs.append(xs[-1] + (x_dot * dt))
        ys.append(ys[-1] + (y_dot * dt))
        zs.append(zs[-1] + (z_dot * dt))
        if j == 0 : # add first in batched sequence (not predicted) to model output # todo improve
            m_xs.append(xs[-1])
            m_ys.append(ys[-1])
            m_zs.append(zs[-1])
    seq = torch.cat([torch.tensor([d[-BATCH_SIZE:]]) for d in [xs, ys, zs]]).T

    """ step model """
    if i < start_interpolation:
        m_x_dot, m_y_dot, m_z_dot, sima = optimize_PCN(seq[1:], seq[:-1],optimize=True)
    else:
        m_x_dot, m_y_dot, m_z_dot, sima = optimize_PCN(seq[1:], seq[:-1], optimize=False) # todo use own prediction

    for j in range(BATCH_SIZE-1):
        m_xs.append(m_x_dot[j])
        m_ys.append(m_y_dot[j])
        m_zs.append(m_z_dot[j])
        sigmas.append(sigma)


if False:
    hist = 3
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(xs[-plt_range*hist:], ys[-plt_range*hist:], zs[-plt_range*hist:], lw=0.5, color='black', label="True trajectory")
    ax1.legend()
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot(m_xs[-plt_range*hist:-plt_range], m_ys[-plt_range*hist:-plt_range], m_zs[-plt_range*hist:-plt_range], lw=0.5, color='blue', label="Filtering & optimization")
    ax2.plot(m_xs[-plt_range-1:], m_ys[-plt_range-1:], m_zs[-plt_range-1:], lw=0.5, color='green', label="Filtering with fixed weights")
    ax2.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

if True:
    print("True x:\t", xs[-1])
    print("Pred x:\t", m_xs[-1].item())
    print("\nTrue y:\t", ys[-1])
    print("Pred y:\t", m_ys[-1].item())
    print("\nTrue z:\t", zs[-1])
    print("Pred z:\t", m_zs[-1].item())

fig, axs = plt.subplots(2, 2)
axs[0,0].plot(m_xs, color="blue")
axs[0,0].plot(xs, color="black")
axs[0,0].set_title("x1")
axs[0,1].plot(m_ys, color="blue")
axs[0,1].plot(ys, color="black")
axs[0,1].set_title("x2")
axs[1,0].plot(m_zs, color="blue")
axs[1,0].plot(zs, color="black")
axs[1,1].plot(sigmas, color="orange")
axs[1,1].set_title("Variance")
plt.suptitle("Dynamical prediction\nLorenz attractor")
plt.show()
plt.show()

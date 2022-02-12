import torch
import matplotlib.pyplot as plt
import numpy as np
plt.style.use(['seaborn'])

""" 
Minimal example for dynamical weights optimisation
"""

def true_derivative(x, s=2.):
    return s*x

def optimize_PCN(x, true_x, s=2., optimize=True):
    """ Optimize dynamical prediction """
    global sigma; global pred; global error
    error = torch.tensor([100.])
    x = torch.tensor(x).float()
    v = torch.tensor([[s]]).detach()  # cause state (not optimised)
    x_past = torch.tensor([[x]]).detach()  # past hidden state (not optimised)
    x = torch.tensor([[true_x]]).detach().float()  # current hidden state (optimised)
    opt = torch.optim.SGD([net_dynamical.weight], lr=.001)
    opt_sigma = torch.optim.SGD([sigma], lr=LR_PRECISION)
    updates = 0
    while error.abs() > 0.001:
        updates += 1
        pred = net_dynamical(torch.cat([v.detach(), x_past.detach()*0], -1)) * x_past.detach()
        if optimize:
            opt.zero_grad()
            opt_sigma.zero_grad()
            error = (x - pred)
            error_weighted = torch.matmul(error.T, torch.matmul(sigma**-1, error))
            error_weighted.backward()
            opt.step()
            opt_sigma.step()
            if (sigma - LR_PRECISION * sigma**-1).detach() >= .0001: # define maximum allowed precision
                sigma = (sigma - LR_PRECISION * sigma**-1).detach()  # cause state variance decay
        else:
            break
    print(updates, "updates", net_dynamical.weight)
    return pred[0][0].detach(), sigma.detach()

if __name__ == '__main__':

    # model
    dt = 0.01
    s,r,b = 10, 28, 2.667
    LR_PRECISION = .1
    net_dynamical = torch.nn.Linear(2, 1, bias=False)
    sigma = (torch.ones([1]) * 1).requires_grad_()

    # logging
    num_steps = 100
    plt_range = 60
    start_interpolation = num_steps-plt_range
    m_xs, xs = torch.empty(num_steps + 1), np.empty(num_steps + 1)
    m_xs[0], xs[0] = 2., 2.
    sigmas = np.empty(num_steps + 1)
    sigmas[0] = sigma.detach()

    for i in range(num_steps):
        xs[i+1] = xs[i] + (true_derivative(xs[i]) * dt) # get observation
        if i < start_interpolation:
            m_x_dot, sigma = optimize_PCN(xs[i], xs[i + 1], optimize=True)
        else:
            m_x_dot, sigma = optimize_PCN(m_xs[i], m_xs[i + 1], optimize=False) # model's predictions as inputs
        m_xs[i+1] = m_x_dot
        sigmas[i+1] = (sigma**-1).log()

    fig = plt.figure(figsize=(5,3))
    ax1 = fig.add_subplot(1, 2, 1); ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot(xs, lw=0.5, color='black', label="Observation")
    ax2.plot(m_xs[:-plt_range+1], lw=0.5, color='blue', label="Filtering")
    ax2.plot(list(range(len(m_xs[:-plt_range]),len(m_xs),1)), m_xs[-plt_range:], lw=0.5, color='green', label="Prediction")
    ax2.plot(sigmas, lw=0.5, color='orange', label="Log precision")
    ax1.legend(); ax2.legend()
    ax1.set_xlabel("Update"); ax2.set_xlabel("Update"); ax2.grid(True), ax2.set_yticklabels([]);
    plt.suptitle("Dynamical prediction \n $ \dot{x} = f(x,v) = v*x$ with known cause state $v=2$")
    plt.tight_layout()
    plt.show()

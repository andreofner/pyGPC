import torch
import math
import matplotlib.pyplot as plt
from lorenz_attractor import *

""" 
Hierarchical predictions g(x,v) in generalized coordinates by fitting n-th order polynomials to (noisy) datapoints 
- exploits the temporal locality of outgoing predictions in generalized predictive coding
- the number of generalized coordinates (coefficients of the polynomial) 
  is independent of the temporal resolution (= sequence length = number of fitted samples in btach dimension) 
- allows to treat predictions (and their precision) over time and space equally 

Converting generalised predictions to sensory samples would normally require 
autoregressive prediction using Taylor's theorem.

Hierarchical predictions of v are wrt. points in time (or space), i.e. y(t) = g(x,v,t)  
    -> n-th order (Taylor polynomial) expansion of observed datapoint 
Dynamical predictions of x_dot are wrt. x and v, i.e. x_dot_dt = f(x,v,dt)
    -> n_th order autoregressive neural ODE 
"""

# model settings
n_g_coords = 10  # how many generalized coordinates to use
output_dim = 1  # shape of observation
coeffs = [torch.randn((), requires_grad=True) for coord in range(n_g_coords)]

# optimisation settings
seqs = 5
seq_length = 16
updates = 1000
dt = 0.01

net = torch.nn.Linear(n_g_coords, output_dim, bias=True)
opt = torch.optim.SGD(net.parameters(), 0.01)

initial = (0., 1., 1.05)
for chunk in range(seqs):

    # create data: lorenz attractor (ODE)
    x, y, z = generate_data(num_steps=seq_length, dt=dt, plot=False, initial=initial)
    initial = (x[-1].item(), y[-1].item(), z[-1].item())
    x_ = torch.tensor(x) * .01
    y = torch.tensor(y) * .01
    z_ = torch.tensor(z) * .01
    x = torch.range(1, y.shape[0]) / seq_length  # time axis (ODE sampled at specific time points)

    # create n-th order inputs (x, x^2, x^3, ...) for generalized coordinates
    x_g = [x]
    [x_g.append(x**n) for n, coeff in enumerate(coeffs[1:])]
    x_g = torch.stack(x_g)

    y = y + torch.rand_like(y) * 0.01 # add noise to observations
    x_g, y = x_g.T, y.T.unsqueeze(1)  # view time as batch

    for t in range(updates):
        y_pred = net(x_g)
        loss = ((y_pred - y)**2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    plt.plot(x, y, label="Observation", color="black")
    plt.plot(torch.tensor(x), torch.tensor(y_pred.detach()), label="Prediction", color="green")
plt.title("Noisy observations of a Lorenz attractor")
plt.show()

import torch
import math
import matplotlib.pyplot as plt
from lorenz_attractor import *

""" 
Hierarchical predictions g(x,v) in generalized coordinates by fitting n-th order polynomials to (noisy) datapoints 
- exploits the temporal locality of outgoing predictions in generalized predictive coding
- the number of generalized coordinates (coefficients of the polynomial) 
  is independent of the temporal resolution (= sequence length = number of fitted samples in batch dimension) 
- allows to treat predictions (and their precision) over time and space equally 

Converting generalised predictions to sensory samples would normally require 
autoregressive prediction using Taylor's theorem and finite differences for their estimation.

Hierarchical predictions of v are wrt. points in time (or space), i.e. y(t) = g(x,v,t)  
    -> n-th order (Taylor polynomial) expansion of observed datapoint 
Dynamical predictions of x_dot are wrt. x and v, i.e. x_dot_dt = f(x,v,dt)
    -> n_th order autoregressive neural ODE 
"""

# model settings
n_g_coords = 8  # how many generalized coordinates to use
input_dim = 2  # number of inputs in generalised coordinates

n_g_coords_out = 1  # how many generalized coordinates to use for output. use 1 for sensory observations
output_dim = 1  # shape of observation
coeffs = [torch.randn((), requires_grad=True) for coord in range(n_g_coords)]

# optimisation settings
seqs = 1
seq_length = 16
updates = 10
dt = 0.01

net = torch.nn.Linear(n_g_coords*input_dim, output_dim*n_g_coords_out, bias=True)
opt = torch.optim.SGD(net.parameters(), 0.01)

initial = (0., 1., 1.05)
for chunk in range(seqs):

    # create data: lorenz attractor (ODE)
    x, y, z = generate_data(num_steps=seq_length, dt=dt, plot=False, initial=initial)
    initial = (x[-1].item(), y[-1].item(), z[-1].item())

    # scale observations to useful range and add noise
    scale = 0.1
    noise_scale = scale*.1
    x = torch.tensor(x) * scale + torch.rand_like(torch.tensor(x)) * noise_scale
    y = torch.tensor(y) * scale + torch.rand_like(torch.tensor(y)) * noise_scale
    z = torch.tensor(z) * scale + torch.rand_like(torch.tensor(z)) * noise_scale
    t = (torch.range(1, y.shape[0]) / seq_length)#.unsqueeze(-1)  # time axis

    # create n-th order inputs (in generalized coordinates)
    # just (x, x^2, x^3, ...), (y, y^2, y^3, ...), etc.
    t_g = [t]
    [t_g.append(t**(n+1)) for n in range(1,n_g_coords,1)]
    t_g = torch.stack(t_g).float()

    x_g = [x]
    [x_g.append(x**(n+1)) for n in range(1,n_g_coords,1)]
    x_g = torch.stack(x_g).float()

    y_g = [y]
    [y_g.append(y**(n+1)) for n in range(1,n_g_coords,1)]
    y_g = torch.stack(y_g).float()

    z_g = [z]
    [z_g.append(z**(n+1)) for n in range(1,n_g_coords_out,1)]
    z_g = torch.stack(z_g).float()

    t_g, x_g, y_g, z_g = t_g.T, x_g.T, y_g.T, z_g.T

    # the original first order values
    t = t.unsqueeze(-1)
    x = x.unsqueeze(-1)
    y = y.unsqueeze(-1)
    z = z.unsqueeze(-1)

    input = torch.cat([t_g, y_g],-1)
    target = z_g
    for update in range(updates):
        prediction = net(input)
        loss = ((prediction - target)**2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    plt.plot(t, target, label="Observation", color="black")
    plt.plot(t, torch.tensor(prediction.detach()), label="Prediction", color="green")
plt.title("Noisy observations of a Lorenz attractor")
plt.show()


# After estimating the nth-order polynomial coefficients from data via regression
# we get their derivatives (-> generalized coordinates) simply via the derivative matrix D:

coeffs = net.weight[:,:n_g_coords].unsqueeze(-1)
print("Estimated polynomial coefficients: ", coeffs.detach()[0])
for deriv in range(n_g_coords-1):
    D = (torch.eye(coeffs.shape[1]) * torch.range(0, coeffs.shape[1]-1, 1))[1:].unsqueeze(0)
    d_coeff = torch.matmul(D, coeffs)
    print("Derivative operator ", D.detach()[0])
    print(f'Coefficients of {deriv} derivative:', d_coeff.detach()[0])
    coeffs = d_coeff

"""
Circles dataset from https://github.com/greydanus/piecewise_node
"""

import numpy as np

class ObjectView(object):  # make a dictionary look like an object
    def __init__(self, d): self.__dict__ = d

def get_dataset_args(as_dict=False):
    arg_dict = {'num_samples': 10000, 'train_split': 0.9, 'r_range': [1., 2], 'arcsize': 1.45 * 6.28,
                # makes a sequence of length 30
                'dt': 0.1, 'seed': 0}
    return arg_dict if as_dict else ObjectView(arg_dict)


def make_circle(radius=1, dt=0.1, domain=[0, 6.28]):
    start, stop = domain
    L = 2 * np.pi * radius * (stop - start) / (2 * np.pi)
    theta = np.linspace(start, stop, int(L / dt))
    x = radius * np.stack([np.sin(theta), np.cos(theta)])
    return x

def get_dataset(args):
    np.random.seed(args.seed)  # random seed for reproducibility
    smallest_circle = make_circle(radius=args.r_range[0], dt=args.dt, domain=[0, args.arcsize])
    min_seq_len = smallest_circle.shape[-1]  # this is the shortest possible trajectory
    trajectories = []

    for i in range(args.num_samples):  # this loop generates the synthetic 'circles' dataset
        radius = args.r_range[0] + (args.r_range[1] - args.r_range[0]) * np.random.rand()  # random radius
        start_angle = 2 * np.pi * np.random.rand()  # random angle
        x = make_circle(radius=radius, dt=args.dt, domain=[start_angle, start_angle + args.arcsize])
        trajectories.append(x[..., :min_seq_len])  # append trajectory to list

    xs = np.stack(trajectories).transpose(2, 0, 1)  # reshape tensor dimensions -> [time, batch, state]
    split_ix = int(args.num_samples * args.train_split)  # train / test split
    dataset = {'x': xs[:, :split_ix], 'x_test': xs[:, split_ix:]}
    return dataset

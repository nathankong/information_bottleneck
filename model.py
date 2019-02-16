from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, x_dim=1, out_dim=1):
        super(Model, self).__init__()
        self.x_dim = x_dim
        self.out_dim = out_dim
        self.lin_x_to_output = nn.Linear(x_dim, out_dim)

    def forward(self, x):
        h = self.lin_x_to_output(x)
        ps = torch.tanh(h)
        return ps


class NoiseModel(nn.Module):
    def __init__(self, beta=None, mu=None, x_dim=1, out_dim=1):
        # mu is column vector (i.e. 2x1 or 1x1)
        super(Model, self).__init__()
        if mu is None:
            self.mu = np.array([[0]])
        else:
            self.mu = mu

        self.beta = beta
        self.x_dim = x_dim
        self.out_dim = out_dim
        self.lin_x_to_output = nn.Linear(x_dim, out_dim)

    def forward(self, x):
        h = self.lin_x_to_output(x)
        ps = torch.tanh(h)
        # Add noise

        return ps + z



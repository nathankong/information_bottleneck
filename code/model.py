from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

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
    def __init__(self, beta=None, x_dim=1, out_dim=1):
        super(NoiseModel, self).__init__()
        self.beta = beta
        self.x_dim = x_dim
        self.out_dim = out_dim
        self.lin_x_to_output = nn.Linear(x_dim, out_dim)
        nn.init.zeros_(self.lin_x_to_output.weight)
        nn.init.zeros_(self.lin_x_to_output.bias)

    def forward(self, x):
        h = self.lin_x_to_output(x)
        ps = torch.tanh(h)
        # Add noise from N(0, beta^2)
        z = torch.randn(x.size(), device=x.device) * self.beta
        ps_noise = ps + z
        return ps_noise, ps


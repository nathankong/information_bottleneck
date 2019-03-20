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
        assert beta is not None
        self.beta = beta
        self.x_dim = x_dim
        self.out_dim = out_dim

        # Define layer
        self.lin_x_to_output = nn.Linear(x_dim, out_dim)

        # Initialize weights to 0
        nn.init.zeros_(self.lin_x_to_output.weight)
        nn.init.zeros_(self.lin_x_to_output.bias)

    def forward(self, x):
        h = self.lin_x_to_output(x)
        ps = torch.tanh(h)

        # Add noise from N(0, beta^2)
        z = torch.randn(x.size(), device=x.device) * self.beta
        ps_noise = ps + z

        activation_dict = dict()
        activation_dict["output"] = ps
        activation_dict["output_noise"] = ps_noise

        return ps_noise, activation_dict

class NoiseModelSingleNeuronReLU(nn.Module):
    def __init__(self, beta=None, x_dim=1, out_dim=1):
        super(NoiseModelSingleNeuronReLU, self).__init__()
        assert beta is not None
        self.beta = beta
        self.x_dim = x_dim
        self.out_dim = out_dim

        #self.non_lin = torch.nn.LeakyReLU(negative_slope=0.1)
        self.non_lin = torch.tanh

        # Define layer
        self.lin_x_to_output = nn.Linear(x_dim, out_dim)

        nn.init.zeros_(self.lin_x_to_output.weight)
        nn.init.zeros_(self.lin_x_to_output.bias)

    def forward(self, x):
        h = self.lin_x_to_output(x)
        ps = self.non_lin(h)

        # Add noise from N(0, beta^2)
        z = torch.randn(x.size(), device=x.device) * self.beta
        ps_noise = ps + z

        activation_dict = dict()
        activation_dict["output"] = ps
        activation_dict["output_noise"] = ps_noise

        return ps_noise, activation_dict


class NoiseModelReLU(nn.Module):
    # Assuming single dimension only
    def __init__(self, beta=None, x_dim=1, h1_dim=1, out_dim=1):
        super(NoiseModelReLU, self).__init__()
        assert beta is not None
        self.beta = beta
        self.x_dim = x_dim
        self.h1_dim = h1_dim
        self.out_dim = out_dim

        # Define layers (only one hidden layer)
        self.lin_x_to_h1 = nn.Linear(x_dim, h1_dim)
        self.lin_h1_to_output = nn.Linear(h1_dim, out_dim)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.1)

        # Initialize weights
        self.lin_x_to_h1.weight.data.fill_(0.0)
        self.lin_x_to_h1.bias.data.fill_(4.5)
        self.lin_h1_to_output.weight.data.fill_(-1.0)
        self.lin_h1_to_output.bias.data.fill_(0.0)

        #nn.init.zeros_(self.lin_x_to_h1.weight)
        #nn.init.zeros_(self.lin_x_to_h1.bias)
        #nn.init.zeros_(self.lin_h1_to_output.weight)
        #nn.init.zeros_(self.lin_h1_to_output.bias)

    def forward(self, x):
        h = self.leaky_relu(self.lin_x_to_h1(x))
        #h_lin = self.lin_x_to_h1(x)
        #h = torch.max(h_lin, h_lin/10.)

        h_noise = h + (torch.randn(h.size(), device=x.device) * self.beta)

        out = self.leaky_relu(self.lin_h1_to_output(h_noise))
        #h_noise_lin = self.lin_h1_to_output(h_noise)
        #out = torch.max(h_noise_lin, h_noise_lin/10.)

        out_noise = out + (torch.randn(out.size(), device=x.device) * self.beta)

        activation_dict = dict()
        activation_dict["hidden"] = h
        activation_dict["hidden_noise"] = h_noise
        activation_dict["output"] = out
        activation_dict["output_noise"] = out_noise

        return out_noise, activation_dict


class NoiseModelTwoNeuronTanh(nn.Module):
    # Assuming single dimension only
    def __init__(self, beta=None, x_dim=1, h1_dim=1, out_dim=1):
        super(NoiseModelTwoNeuronTanh, self).__init__()
        assert beta is not None
        self.beta = beta
        self.x_dim = x_dim
        self.h1_dim = h1_dim
        self.out_dim = out_dim

        # Define layers (only one hidden layer)
        self.lin_x_to_h1 = nn.Linear(x_dim, h1_dim)
        self.lin_h1_to_output = nn.Linear(h1_dim, out_dim)
        self.nonlin = torch.tanh

        # Initialize weights
        nn.init.zeros_(self.lin_x_to_h1.weight)
        nn.init.zeros_(self.lin_x_to_h1.bias)
        nn.init.zeros_(self.lin_h1_to_output.weight)
        nn.init.zeros_(self.lin_h1_to_output.bias)

    def forward(self, x):
        h = self.nonlin(self.lin_x_to_h1(x))
        h_noise = h + (torch.randn(h.size(), device=x.device) * self.beta)

        out = self.nonlin(self.lin_h1_to_output(h_noise))
        out_noise = out + (torch.randn(out.size(), device=x.device) * self.beta)

        activation_dict = dict()
        activation_dict["hidden"] = h
        activation_dict["hidden_noise"] = h_noise
        activation_dict["output"] = out
        activation_dict["output_noise"] = out_noise

        return out_noise, activation_dict



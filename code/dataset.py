from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np
import scipy.stats as sps


class ScalarDatasetEight(data.Dataset):
    def __init__(self, N):
        self.N = N
        self.X, self.Y = build_dataset_eight(N)
        print(self.X.shape, self.Y.shape)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        data_x, data_y = self.X[index], self.Y[index]

        data_x = data_x.T
        data_x = torch.from_numpy(data_x).float()
        data_y = data_y.T
        data_y = torch.from_numpy(data_y).float()

        return data_x, data_y


class ScalarDataset(data.Dataset):
    def __init__(self, N, test=False):
        self.N = N
        if test:
            self.X, self.Y = build_dataset_size_4()
        else:
            self.X, self.Y = build_dataset(N)
        print(self.X.shape, self.Y.shape)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        data_x, data_y = self.X[index], self.Y[index]

        data_x = data_x.T
        data_x = torch.from_numpy(data_x).float()
        data_y = data_y.T
        data_y = torch.from_numpy(data_y).float()

        return data_x, data_y


class GaussianMixtureDataset(data.Dataset):
    def __init__(self, N, n_components, means, variances, mixture_probs):
        # N: number of samples
        # n_components: scalar value
        # means: (n_components, dimension)
        # variances: (n_components, dimension, dimension)
        # mixture_probs: (n_components,)

        assert n_components == means.shape[0], "Number of components and means mismatch."
        assert n_components == variances.shape[0], "Number of components and variances mismatch."
        assert means.shape[1] == variances.shape[1], "Dimension between mean and variance mismatch."
        assert means.shape[1] == variances.shape[2], "Dimension between mean and variance mismatch."
        assert n_components == mixture_probs.shape[0], "Number of components and mixture probabilities mismatch."

        self.N = N
        self.n_components = n_components
        self.means = means
        self.variances = variances
        self.mixture_probs = mixture_probs

        self.X, self.Y = build_gaussian_mixture_dataset(N, n_components, means, variances, mixture_probs)
        print(self.X.shape, self.Y.shape)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        data_x, data_y = self.X[index], self.Y[index]

        data_x = data_x.T
        data_x = torch.from_numpy(data_x).float()
        data_y = data_y.T
        data_y = torch.from_numpy(data_y).float()

        return data_x, data_y


def build_gaussian_mixture_dataset(N, n_components, means, variances, mixture_probs):
    # N: number of samples
    # n_components: scalar value
    # means: (n_components, dimension)
    # variances: (n_components, dimension, dimension)
    # mixture_probs: (n_components,)
    # TODO: We are currently assuming one dimensional input
    dimension = means.shape[1]
    assert dimension == 1

    component_idx = np.random.choice(n_components, size=N, replace=True, p=mixture_probs)
    X = [sps.multivariate_normal.rvs(means[i,:], variances[i,:,:]) for i in component_idx]
    X = np.array(X).reshape(N,dimension)

    X = X.reshape((N,))

    #Y = component_idx.reshape(N,1)
    Y = np.zeros((N,))
    Y[X<=0] = -1
    Y[X>0] = 1

    X = X.reshape((N,1))
    Y = Y.reshape((N,1))

    return X, Y


def build_dataset_eight(N):
    X = np.array([1,2,3,4,5,6,7,8])
    ind = np.random.randint(8,size=N)
    X = X[ind] # (N,)

    Y = np.zeros((N,))
    Y[X<=4] = 0
    Y[X>4] = 0.25

    assert X.ndim == 1 and Y.ndim == 1
    X = X.reshape((N,1))
    Y = Y.reshape((N,1))

    return X, Y


def build_dataset(N):
    X = np.array([-3,-1,1,3])
    ind = np.random.randint(4,size=N)
    X = X[ind] # (N,)

    Y = np.zeros((N,))
    Y[X<3] = -1
    Y[X==3] = 1

    assert X.ndim == 1 and Y.ndim == 1
    X = X.reshape((N,1))
    Y = Y.reshape((N,1))

    return X, Y


def build_dataset_size_4():
    X = np.array([-3,-1,1,3])
    Y = np.zeros((4,))
    Y[X<3] = -1
    Y[X==3] = 1

    assert X.ndim == 1 and Y.ndim == 1
    X = X.reshape((4,1))
    Y = Y.reshape((4,1))

    return X, Y


if __name__ == "__main__":
    x,y = build_dataset(15)
    print(x)
    print(y)

    x,y = build_dataset_eight(15)
    print(x)
    print(y)

    n_components = 2
    means = np.array([-1,1]).reshape(n_components,1)
    variances = np.array([1,1]).reshape(n_components,1,1)
    mixture_probs = np.array([0.5,0.5])
    y = build_gaussian_mixture_dataset(15, n_components, means, variances, mixture_probs)
    print(y.shape)
    print(y)

    n_components = 2
    means = np.array([[-1,1],[1,-1]]).reshape(n_components,2)
    variances = np.array([[[1,0],[0,1]],[[1,0],[0,1]]]).reshape(n_components,2,2)
    mixture_probs = np.array([0.5,0.5])
    y = build_gaussian_mixture_dataset(200, n_components, means, variances, mixture_probs)



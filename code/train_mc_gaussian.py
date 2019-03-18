from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np

import torch
import torch.nn as nn
from torch import optim

from dataset import GaussianMixtureDataset, build_gaussian_mixture_dataset
from model import NoiseModel, NoiseModelSingleNeuronReLU
from mutual_information_estimator_continuous import MutualInformationEstimator
# TODO: Need to remove UniformDataDistribution and modify to Gaussian mixture PDF
from utils import UnivariateGaussian, UniformDataDistribution


def sample(m, device, n_components, means, variances, mixture_probs, num_samp=1000):
    X, _ = build_gaussian_mixture_dataset(
        num_samp,
        n_components,
        means,
        variances,
        mixture_probs
    )

    output_noise, output_dict = m(torch.from_numpy(X).float().to(device))
    return output_noise, output_dict


def test_model(m, test_loader, device):
    avg_acc = 0
    num_batch = 0
    for _, (test_x, test_y) in enumerate(test_loader):
        test_out_noise, _ = m(test_x.to(device))
        avg_acc += compute_acc(test_out_noise.detach().cpu().numpy(), test_y.detach().cpu().numpy())
        num_batch += 1

    avg_acc /= num_batch
    assert num_batch == 1

    return avg_acc, test_out_noise, test_y


def compute_acc(model_out, true_out):
    # model_out = tanh values
    assert model_out.shape[0] == true_out.shape[0]
    #model_out[model_out<=0.5] = 0
    #model_out[model_out>0.5] = 1
    model_out[model_out<0] = -1
    model_out[model_out>=0] = 1
    correct = np.sum((model_out==true_out).astype(int))
    return correct / true_out.shape[0]


if __name__ == "__main__":
    # Sample size
    N = 30

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--noise', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--results_dir', type=str, default="")
    parser.add_argument('--num_mc_samples', type=int, default=200)
    args = parser.parse_args()

    # MC samples
    print("Number of MC samples:", args.num_mc_samples)

    # Results dir
    if args.results_dir == "":
        assert 0, "Need results directory."
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    print("Results directory:", args.results_dir)

    # Use GPU or not
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print("Using cuda:", cuda)
    print("Device:", device)
    # Noise or no noise
    print("Using noise model:", args.noise)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Learning rate
    print("Learning rate:", args.lr)

    # Dataset
    #n_components = 2
    #means = np.array([-1,1]).reshape(n_components,1)
    #variances = np.array([0.1,0.1]).reshape(n_components,1,1)
    #mixture_probs = np.array([0.5,0.5])

    n_components = 3
    means = np.array([-2,0,2]).reshape(n_components,1)
    variances = np.array([0.1,0.1,0.1]).reshape(n_components,1,1)
    mixture_probs = np.ones((n_components,), dtype=np.float) / n_components

    train_dataset = GaussianMixtureDataset(N, n_components, means, variances, mixture_probs)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = GaussianMixtureDataset(N, n_components, means, variances, mixture_probs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=N, shuffle=False)

    # Model
    if not args.noise:
        assert 0, "Should only run noise model."
    else:
        #m = NoiseModel(beta=args.beta) # Add noise to layer
        m = NoiseModelSingleNeuronReLU(beta=args.beta)
        #m = NoiseModelReLU(beta=args.beta)
    m = m.to(device)

    # Optimizer
    optimizer = optim.SGD(m.parameters(), lr=args.lr)
    loss_func = nn.MSELoss()

    # Set up probability distributions
    noise_distr = UnivariateGaussian(0, np.square(args.beta))
    # TODO: ADD THE APPROPRIATE GAUSSIAN MIXTURE PDF
    data_distr = UniformDataDistribution(8, outcomes=np.array([1,2,3,4,5,6,7,8]).reshape(8,1))

    # Mutual information estimator
    mi = MutualInformationEstimator(
        m,
        noise_distr,
        data_distr
    )

    # Start the training
    accs = list()
    mutual_info = list()
    losses = list()
    for i in xrange(args.epochs):
        if (i+1) % 1 == 0:
            m.eval()
            m.to(torch.device("cpu"))
            acc, test_out_noise, test_y = test_model(m, test_loader, torch.device("cpu"))
            accs.append(acc)

            # Get noise samples
            gen_noise_outputs, output_dict = sample(
                m,
                torch.device("cpu"),
                n_components,
                means,
                variances,
                mixture_probs,
                num_samp=1000
            )
            np.save(args.results_dir + "/epoch_{}_outputs_noise.npy".format(i+1), gen_noise_outputs.detach().cpu().numpy())

            # Compute MI
            curr_mutual_info = 0.0 # TODO: FIX AFTER TRAINING SUCCESSFUL
#            curr_mutual_info = mi.compute_mutual_information(
#                gen_outputs["output"].detach().cpu().numpy(),
#                1000,
#                "output",
#                args.num_mc_samples
#            )
#            mutual_info[i] = curr_mutual_info

            print("Epoch {}; MI {}; Acc {}".format(i+1, curr_mutual_info, acc))
            if acc >= 0.99:
                for g in optimizer.param_groups:
                    g['lr'] = args.lr / 20.

        m.train()
        m.to(device)
        for batch_idx, (data_x, data_y) in enumerate(train_loader):
            data_x, data_y = data_x.to(device), data_y.to(device) 

            output_noise, _ = m(data_x)
            loss = loss_func(output_noise, data_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())


    # Diagnostics
    for param in m.parameters():
        print("Weight:", param.data)

    n_components = 3
    means = np.array([-2,0,2]).reshape(n_components,1)
    variances = np.array([0.1,0.1,0.1]).reshape(n_components,1,1)
    mixture_probs = np.ones((n_components,), dtype=np.float) / n_components
    new_set_x, new_set_y = build_gaussian_mixture_dataset(
        10,
        n_components,
        means,
        variances,
        mixture_probs
    )
    model_y = np.tanh(1.0556*new_set_x + 0.1573)
    print("True out:", new_set_y.T)
    print("Model out:",  model_y.T)
    new_set_x, new_set_y = build_gaussian_mixture_dataset(
        10,
        n_components,
        means,
        variances,
        mixture_probs
    )
    print("Random out:", new_set_x.T)
    print("Random out label:", new_set_y.T)


    # Save stuff
    np.save(args.results_dir + "/accuracies.npy", np.array(accs))
    np.save(args.results_dir + "/mutual_information.npy", np.array(mutual_info))
    np.save(args.results_dir + "/losses.npy", np.array(losses))




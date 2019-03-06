from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn
from torch import optim

from dataset import ScalarDatasetEight, build_dataset_eight
from model import NoiseModelReLU
from mutual_information_estimator import MutualInformationEstimator
from utils import UnivariateGaussian, UniformDataDistribution


def sample(m, num_samp=1000):
    X, _ = build_dataset_eight(num_samp)
    output_noise, output_dict = m(torch.from_numpy(X).float().to(device))
    return output_noise, output_dict


def test_model(m, test_loader):
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
    assert model_out.shape[0] == true_out.shape[0]
    model_out[model_out<=0.125] = 0
    model_out[model_out>0.125] = 0.25
    correct = np.sum((model_out==true_out).astype(int))
    return correct / true_out.shape[0]


if __name__ == "__main__":
    # python information_bottleneck/code/train_eight.py --epochs 300 --batch_size 100 --noise True --beta 0.05 --lr 0.0001

    # Sample size
    N = 100

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--noise', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

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

    # Datasets
    train_dataset = ScalarDatasetEight(N)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = ScalarDatasetEight(N)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=N, shuffle=False)

    # Model
    if not args.noise:
        assert 0, "Should only run noise model."
    else:
        m = NoiseModelReLU(beta=args.beta) # Add noise to layer
    m = m.to(device)

    # Optimizer
    optimizer = optim.SGD(m.parameters(), lr=args.lr)
    loss_func = nn.MSELoss()

    # Set up probability distributions
    noise_distr = UnivariateGaussian(0, np.square(args.beta))
    # The '8' is hard coded since the dataset is {1,2,3,4,5,6,7,8}
    data_distr = UniformDataDistribution(8) 

    # Mutual information estimator
    mi = MutualInformationEstimator(
        m,
        device,
        noise_distr.compute_probability,
        data_distr.compute_probability,
        np.array([1,2,3,4,5,6,7,8]).reshape(8,1)
    )

    # Start the training
    accs = list()
    mutual_info = list()
    losses = list()
    for i in xrange(args.epochs):
        if (i+1) % 1 == 0:
            m.eval()
            acc, test_out_noise, test_y = test_model(m, test_loader)
            accs.append(acc)

            # Get noise samples
            gen_noise_outputs, output_dict = sample(m, num_samp=1000)
            np.save("results/eight/epoch_{}_outputs_noise.npy".format(i+1), gen_noise_outputs.detach().cpu().numpy())
            np.save("results/eight/epoch_{}_h1_noise.npy".format(i+1), output_dict["hidden_noise"].detach().cpu().numpy())

            # TODO: Call compute_mutual_information() twice for each layer
            # Compute MI
            #curr_mutual_info = mi.compute_mutual_information(gen_outputs.detach().cpu().numpy(), 1000)
            #mutual_info[i] = curr_mutual_info
            curr_mutual_info = 0
            mutual_info.append(curr_mutual_info)

            print("Epoch {}; MI {}; Acc {}".format(i+1, curr_mutual_info, acc))
            if acc >= 0.99:
                for g in optimizer.param_groups:
                    g['lr'] = args.lr / 20.

        m.train()
        for batch_idx, (data_x, data_y) in enumerate(train_loader):
            data_x, data_y = data_x.to(device), data_y.to(device) 

            output_noise, _ = m(data_x)
            loss = loss_func(output_noise, data_y)
            loss.backward()
            optimizer.step()
            #print("Loss val {}".format(loss.item()))
            losses.append(loss.item())


    # Diagnostics
    for param in m.parameters():
        print("Weight:", param.data)

    new_set_x, new_set_y = build_dataset_eight(10)
    model_y = -0.88*new_set_x + 4.3
    model_y = np.maximum(model_y, model_y/10)
    model_y = -0.23*model_y + 0.24
    model_y = np.maximum(model_y, model_y/10)
    print("New set x:", new_set_x)
    print("True out:", new_set_y.T)
    print("Model out:",  model_y.T)
    new_set_x, new_set_y = build_dataset_eight(10)
    print("Random out:", new_set_x.T)
    print("Random out label:", new_set_y.T)

    # Save stuff
    np.save("results/eight/accuracies.npy", np.array(accs))
    np.save("results/eight/mutual_information.npy", np.array(mutual_info))
    np.save("results/eight/losses.npy", np.array(losses))




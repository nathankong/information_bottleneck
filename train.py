from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim

from dataset import ScalarDataset
from model import Model


def test_model(m, test_loader):
    avg_acc = 0
    num_batch = 0
    for _, (test_x, test_y) in enumerate(test_loader):
        test_out = m(test_x.to(device))
        avg_acc += compute_acc(test_out.detach().cpu().numpy(), test_y.detach().cpu().numpy())
        num_batch += 1
    avg_acc /= num_batch
    return avg_acc, test_out, test_y


def compute_acc(model_out, true_out):
    # model_out = tanh values
    assert model_out.shape[0] == true_out.shape[0]
    model_out[model_out<=0] = -1
    model_out[model_out>0] = 1
    correct = np.sum((model_out==true_out).astype(int))
    return correct / true_out.shape[0]


if __name__ == "__main__":
    N = 500

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # Use GPU or not
    cuda = torch.cuda.is_available()
    print("Using cuda:", cuda)
    device = torch.device('cuda' if cuda else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Datasets
    train_dataset = ScalarDataset(N)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=25, shuffle=True)

    test_dataset = ScalarDataset(N)
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=25, shuffle=True)

    # Model
    m = Model() # Default dim=1
    m = m.to(device)

    # Optimizer
    optimizer = optim.SGD(m.parameters(), lr=0.01)
    loss_func = nn.MSELoss()

    accs = np.zeros((args.epochs,))
    for i in xrange(args.epochs):
        print("Epoch", i+1)
        for batch_idx, (data_x, data_y) in enumerate(train_loader):
            data_x, data_y = data_x.to(device), data_y.to(device) 

            output = m(data_x)
            loss = loss_func(output, data_y)
            loss.backward()
            optimizer.step()

        if (i+1) % 1 == 0:
            acc, test_out, test_y = test_model(m, test_loader)
            accs[i] = acc

    for param in m.parameters():
        print("Weight:", param.data)

    from dataset import build_dataset
    new_set_x, new_set_y = build_dataset(10)
    model_y = np.tanh(156.4934*new_set_x - 329.3788)
    print("True out:", new_set_y.T)
    print("Model out:",  model_y.T)
    new_set_x, new_set_y = build_dataset(10)
    print("Random out:", new_set_x.T)
    print("Random out label:", new_set_y.T)

    # Plot test set accuracies
    plt.figure()
    plt.plot(accs)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.show()



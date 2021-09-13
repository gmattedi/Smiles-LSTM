import time

import torch
from torch import nn

import model
from utils import *

from typing import Sequence, List
from sklearn.model_selection import train_test_split


# Declaring the train method
def train(
        net: model.CharRNN, data: Sequence[int],
        epochs: int=1, batch_size: int=32,
        seq_length: int=100, lr: float=0.001, clip: int=5,
        val_frac: float=0.1, print_every: int=10000) -> List[List[float,float]]:
    """
    Train the network
    Args:
        net (model.CharRNN):
        data (Sequence[int]): Flat one-hot SMILES encoding
        epochs (int): Number of epochs
        batch_size (int)
        seq_length (int): Number of character steps per mini-batch
        lr (float)
        clip (float): Clip gradients
        val_frac (float)
        print_every (int): Print info every N samples

    Returns:
        loss_hist (List[List[float,float]]): Loss history for every `print_every`.
            First column is training loss, second validation loss

    """

    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # create training and validation data
    data, val_data = train_test_split(data, test_size=val_frac, shuffle=True)
    # Check if GPU is available
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        net.cuda()

    loss_hist = []

    counter = 0
    n_chars = len(net.chars)
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)

        timer_start = time.time()
        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1

            # One-hot encode our data and make them Torch tensors
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if train_on_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output, targets.view(batch_size * seq_length).long())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for xval, yval in get_batches(val_data, batch_size, seq_length):
                    # One-hot encode our data and make them Torch tensors
                    xval = one_hot_encode(xval, n_chars)
                    xval, yval = torch.from_numpy(xval), torch.from_numpy(yval)

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    inputs, targets = xval, xval
                    if train_on_gpu:
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size * seq_length).long())

                    val_losses.append(val_loss.item())

                net.train()  # reset to train mode after iterationg through validation data

                timer_end = time.time()
                loss_hist.append([loss.item(), np.mean(val_losses)])

                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {:8d}...".format(counter),
                      "Loss: {:8.4f}...".format(loss.item()),
                      "Val Loss: {:8.4f}".format(np.mean(val_losses)),
                      "Epoch Time {:8.1f}s".format(timer_end - timer_start))

    return loss_hist

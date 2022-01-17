import time
from typing import Sequence, List, Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

import model
import utils

logger = utils.logger


# Declaring the train method
def train(
        net: model.CharRNN, data: Sequence[int],
        epochs: int = 1, batch_size: int = 32,
        seq_length: int = 100, lr: float = 0.001, clip: int = 5,
        val_frac: float = 0.1, print_every: int = 10000) -> List[List[float]]:
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
        loss_hist (List[List[float,float,float,float]]): Loss history for every `print_every`.
            columns are epoch, step, training loss and validation loss

    """

    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # create training and validation data
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

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
        for x, y in utils.get_batches(data, batch_size, seq_length):
            counter += 1

            # One-hot encode our data and make them Torch tensors
            x = utils.one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if(train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output, targets.view(
                batch_size*seq_length).long())
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
                for x, y in utils.get_batches(val_data, batch_size, seq_length):
                    # One-hot encode our data and make them Torch tensors
                    x = utils.one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    inputs, targets = x, y
                    if(train_on_gpu):
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(
                        batch_size*seq_length).long())

                    val_losses.append(val_loss.item())

                net.train()  # reset to train mode after iterationg through validation data
                timer_end = time.time()

                train_loss, val_loss = loss.item(), np.mean(val_losses)
                loss_hist.append([e+1, counter, train_loss, val_loss])

                msg = \
                    f"""
Epoch: {e + 1}/{epochs}
Step:  {counter:8d}
Loss:  {train_loss:8.4f}
Val loss: {val_loss:8.4f}
Epoch time: {timer_end - timer_start:8.1f}s
"""
                logger.info(msg)

    return loss_hist

from typing import Sequence

import torch
from torch import nn


# Declaring the model
class CharRNN(nn.Module):
    """
    CharRNN
    """

    def __init__(self, tokens: Sequence[int], n_hidden: int = 10, n_layers: int = 2,
                 drop_prob: float = 0.2, lr: float = 0.001):
        """
        Init model

        Args:
            tokens (Sequence[int]): Flat one-hot encoding of concatenated training data
            n_hidden (int)
            n_layers (int)
            drop_prob (float): Dropout
            lr (float)
        """

        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        # creating character dictionaries
        self.chars = tokens
        self.int2char = {i: char for i, char in enumerate(self.chars)}
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        # define the LSTM
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)

        # define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # define the final, fully-connected output layer
        self.fc = nn.Linear(n_hidden, len(self.chars))

    def forward(self, x, hidden):
        """ Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. """

        # get the outputs and the new hidden state from the lstm
        r_output, hidden = self.lstm(x, hidden)

        # pass through a dropout layer
        out = self.dropout(r_output)

        # Stack up LSTM outputs using view
        out = out.contiguous().view(-1, self.n_hidden)

        # put x through the fully-connected layer
        out = self.fc(out)

        # return the final output and the hidden state
        return out, hidden

    def init_hidden(self, batch_size):
        """ Initializes hidden state """
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        # Check if GPU is available
        train_on_gpu = torch.cuda.is_available()

        if train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden

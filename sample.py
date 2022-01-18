from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import RDLogger, Chem

import model
import utils

RDLogger.DisableLog('rdApp.*')


def predict(
        net: model.CharRNN, char,
        h: Tuple[torch.Tensor, torch.Tensor],
        top_k: Optional[int] = None) -> Tuple[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Given a character, predict the next character.
    Returns the predicted character and the hidden state.

    Args:
        net (model.CharRNN)
        char (str): Character
        h (Tuple[torch.Tensor, torch.Tensor]): Hidden state
        top_k (Optional[int]): Pick from top K characters

    Returns:

    """

    # tensor inputs
    x = np.array([[net.char2int[char]]])
    x = utils.one_hot_encode(x, len(net.chars))
    inputs = torch.from_numpy(x)
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        inputs = inputs.cuda()

    # detach hidden state from history
    h = tuple([each.data for each in h])
    # get the output of the model
    out, h = net(inputs, h)

    # get the character probabilities
    p = F.softmax(out, dim=1).data
    # train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        p = p.cpu()  # move to cpu

    # get top characters
    if top_k is None:
        top_ch = np.arange(len(net.chars))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()

    # select the likely next character with some element of randomness
    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p / p.sum())

    # return the encoded value of the predicted char and the hidden state
    return net.int2char[char], h


def get_sample(net: model.CharRNN, size: int, prime: str = 'B', top_k: Optional[int] = None) -> str:
    """
    Get `size` characters from the network
    Args:
        net (model.CharRNN)
        size (int):  Number of characters
        prime (str): Prime net with string
        top_k (Optional[int]): Pick form top K characters

    Returns:
        chars (str): Characters

    """
    # Check if GPU is available
    train_on_gpu = torch.cuda.is_available()

    if train_on_gpu:
        net.cuda()
    else:
        net.cpu()

    net.eval()  # eval mode

    # First off, run through the prime characters
    chars = [ch for ch in prime]
    char = None
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)

    chars.append(char)

    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)


def get_sample_frame(net: model.CharRNN, size: int, prime: str = 'B', top_k: Optional[int] = None,
                     verbose: bool = True) -> pd.DataFrame:
    """
    Wrapper for sampling the net, splitting the output into SMILES string, converting to
    RDKit mols, checking validity, and computing descriptors

    Args:
        net (model.CharRNN)
        size (int): Sample this many characters
        prime (str): Prime net with string
        top_k (Optional[int]): Pick from top K characters
        verbose (bool)

    Returns:
        sample (pd.DataFrame)
    """

    net.eval()
    sample = get_sample(net, size=size, prime=prime, top_k=top_k).split('\n')
    sample = pd.DataFrame(sample, columns=['SMILES'])
    sample['ROMol'] = sample.SMILES.map(Chem.MolFromSmiles)

    num_valid = sample.ROMol.notna().sum()
    num_invalid = sample.shape[0] - num_valid
    sample = sample[sample.ROMol.notna()]

    if verbose:
        print(f'Valid molecules {num_valid}/{num_valid + num_invalid}')

    # Compute descriptors of samples
    for desc in utils.descriptors:
        sample[desc] = sample.ROMol.map(utils.descriptors[desc])

    return sample


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Sample trained SmilesLSTM model')
    parser.add_argument('-m', '--model', help='Trained model checkpoint (.pt file)', required=True)
    parser.add_argument('--hidden', help='Hidden units (default: %(default)d)', required=False, default=56, type=int)
    parser.add_argument('--layers', help='Layers (default: %(default)d)', required=False, default=2, type=int)
    parser.add_argument('-s', '--size', help='Sample this many characters (default: %(default)d)', required=False,
                        default=100000,
                        type=int)
    parser.add_argument('-o', '--output', help='Output CSV file', required=True, type=str)
    args = parser.parse_args()

    train_on_gpu = torch.cuda.is_available()
    device = 'cuda' if train_on_gpu else 'cpu'

    net = model.CharRNN(utils.chars, n_hidden=args.hidden, n_layers=args.layers)
    net.load_state_dict(torch.load(args.model, map_location=torch.device(device)))

    net_sample = get_sample_frame(net, size=args.size, verbose=False)
    net_sample.drop(columns=['ROMol']).to_csv(args.output, index=False)

import logging
from typing import Sequence

import numpy as np
from rdkit.Chem import Descriptors, rdMolDescriptors

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger('main')

# Hardcode chars for consistency
chars = (
    '9', '1', 'B', 'p', 'X', '3', 'i', '6', '/', 'l', '7', 'S', '(', 'N',
    'F', 'c', 'g', 'I', '5', '8', 'e', 't', 'Z', '#', 'r', '%', 'O', 'o',
    'n', 'C', '[', '4', 'M', '.', 'A', 'L', 'T', 's', '+', '0', ')', '\n',
    'H', '@', 'b', '\\', 'K', '=', 'a', 'R', 'P', '-', '2', ']'
)
int2char = {i: char for i, char in enumerate(chars)}
char2int = {ch: ii for ii, ch in int2char.items()}

descriptors = {
    'HBA': Descriptors.NumHAcceptors,
    'HBD': Descriptors.NumHDonors,
    'MW': Descriptors.MolWt,
    'rotatable': rdMolDescriptors.CalcNumRotatableBonds,
    'TPSA': Descriptors.TPSA,
    'cLogP': Descriptors.MolLogP,
    'fsp3': rdMolDescriptors.CalcFractionCSP3,
    'stereo': rdMolDescriptors.CalcNumAtomStereoCenters
}


def one_hot_encode(arr: np.ndarray, n_labels: int) -> np.ndarray:
    """
    Convert dense one-hot representation to matrix.
    Args:
        arr (np.ndarray): Array of dense one-hot embedding
        n_labels (int): Total number of possible labels

    Returns:
        one_hot (np.array)
    """

    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot


# Defining method to make mini-batches for training
def get_batches(arr: np.ndarray, batch_size: int, seq_length: int):
    """Create a generator that returns batches of size
       batch_size x seq_length from arr.

    Args:
        arr (np.array): Dense one-hot, continuous encoding
        batch_size (int): Batch size, the number of sequences per batch
        seq_length (int): Number of encoded chars in a sequence

    Returns:

    """

    batch_size_total = batch_size * seq_length
    # total number of batches we can make
    n_batches = len(arr) // batch_size_total

    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size_total]
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))

    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n + seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y

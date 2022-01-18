from typing import Sequence, Callable

import numpy as np
import yaml
from scipy.stats import norm

"""
Functions for MPO. Implemented as classes
The only requirement a __call__ method that accepts np.ndarray-like objects
and returns a 1D np.ndarray-like of floats of the same size
"""


class Gaussian:
    def __init__(self, loc: float = 0, scale: float = 1) -> None:
        self.loc = loc
        self.scale = scale
        self.f = norm(loc=loc, scale=scale)

    def __call__(self, x) -> np.ndarray:
        return self.f.pdf(x)

    def __repr__(self) -> str:
        return f'Gaussian(loc={self.loc}, scale={self.scale})'


class Linear:
    def __init__(self, m: float = 1, b: float = 0) -> None:
        self.m = m
        self.b = b

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.m * x + self.b

    def __repr__(self) -> str:
        return f'Linear(m={self.m}, b={self.b})'


class Sigmoid:
    def __init__(self, loc: float = 0, scale: float = 1) -> None:
        self.loc = loc
        self.scale = scale

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-self.scale * (x - self.loc)))

    def __repr__(self) -> str:
        return f'Sigmoid(loc={self.loc}, scale={self.scale})'


class ReLU:
    def __init__(self, m: float = 1, b: float = 0, p: float = 1, mode: str = 'upper') -> None:
        self.m = m
        self.b = b
        self.p = p
        self.mode = mode.lower()

        if not self.mode.lower() in ['upper', 'lower']:
            raise ValueError(f'mode must be "upper" or "lower"')

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = self.m * x + self.b
        plateau_array = np.full(shape=y.shape, fill_value=self.p)
        y = np.c_[y, plateau_array]

        if self.mode == 'upper':
            return y.min(axis=1)
        elif self.mode == 'lower':
            return y.max(axis=1)

    def __repr__(self) -> str:
        return f'ReLu(m={self.m}, b={self.b}, p={self.p}, mode="{self.mode}")'


class MPO:
    """
    Main MPO class
    """

    def __init__(self, funcs: Sequence[Callable], weights: Sequence[float], norm: bool = False) -> None:
        """
        Init the Multi Parameter Optimization

        Args:
            funcs (Sequence[Callable]): Sequence of instances of the MPO functions.
                Accepts also any callable that takes 1D np.ndarray and returns 1D np.array
            weights (Sequence[float]): Weights. Can be single float or sequence of floats
            norm (bool): Normalise all scores to [0,1] (i.e. divide by maximum score)
        """

        if len(funcs) != len(weights):
            raise ValueError(f'{len(funcs)} functions but {len(weights)} weights')

        self.funcs = funcs
        self.weights = weights
        self.norm = norm

        if self.norm:
            self.max_score = np.sum(weights)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Process input data
        Args:
            x (np.ndarray)

        Returns:
            scores (np.ndarray): scores
        """

        if x.ndim < 2:
            raise ValueError(f'Expected input x with > 1 dimensions, got {x.ndim}')
        elif x.shape[1] != len(self.funcs):
            raise ValueError(f'Dimension 1 has shape {x.shape[1]} but {len(self.funcs)} functions have been passed')

        scores = np.array([
            self.funcs[i](x[:, i]) * self.weights[i]
            for i in range(x.shape[1])
        ])
        scores = scores.sum(axis=0)

        if self.norm:
            scores = scores / self.max_score

        return scores

    @classmethod
    def from_yaml(cls, yaml_file: str):
        """
        Initialize MPO from yaml file. Refer to example provided

        Args:
            yaml_file (str)
        """

        with open(yaml_file) as handle:
            yaml_data = yaml.load(handle, Loader=yaml.FullLoader)
        config = yaml_data['functions']
        norm = yaml_data['normalize']

        allowed_entries = ['gaussian', 'linear', 'sigmoid', 'relu']

        functions, weights = [], []
        for entry in config:
            entry_name = list(entry.keys())[0].lower()
            entry_data = entry[list(entry.keys())[0]]

            if entry_name == 'gaussian':
                fn = Gaussian
            elif entry_name == 'linear':
                fn = Linear
            elif entry_name == 'sigmoid':
                fn = Sigmoid
            elif entry_name == 'relu':
                fn = ReLU
            else:
                raise ValueError(f'Function type "{entry_name}" not in allowed list {allowed_entries}')

            if 'weight' in entry_data:
                weights.append(entry_data['weight'])
                del entry_data['weight']
            else:
                weights.append(1)
            functions.append(fn(**entry_data))

        return cls(funcs=functions, weights=weights, norm=norm)

    def __repr__(self) -> str:
        msg = f'MPO(\n'
        msg += '\tfuncs=(%s),\n' % ', '.join(map(str, self.funcs))
        msg += '\tweights=(%s)\n' % ', '.join(map(str, self.weights))
        msg += ')'
        return msg


# if __name__ == '__main__':
#
#     x = np.random.uniform(low=(0, 10), high=(10, 20), size=(10, 2))
#
#     functions = [
#         Gaussian(loc=2, scale=1),
#         Linear(m=2, b=0)
#     ]
#     weights = [1, 0.5]
#     mpo = MPO(functions, weights, norm=True)
#     scores = mpo(x)
#
#     print('%6s %6s %6s' % ('x1', 'x2', 'score'))
#     for x1, x2, score in zip(x[:, 0], x[:, 1], scores):
#         print(f'{x1:6.2f} {x2:6.2f} {score:6.2f}')
#
#     mpo = MPO.from_yaml('exampleMPO.yaml')
#     scores = mpo(x)

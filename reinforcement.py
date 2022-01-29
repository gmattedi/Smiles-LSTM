from typing import Optional, List, Tuple, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from rdkit import Chem
from rdkit.Chem import Descriptors

import model
import sample
import utils


def sigmoid(x, x0: float = 0., b: float = 1.):
    return 1 / (1 + np.exp(-b * (x - x0)))


class Reinforcement:
    """
    Policy gradient.
    Draws from https://github.com/isayev/ReLeaSE
    """

    def __init__(self, net: model.CharRNN, scorer: Callable[[Chem.Mol], float], gamma: float = 0.97, lr: float = 0.001):
        """
        Init Policy Gradient instance
        
        Args:
            net (CharRNN) 
            scorer (Callable[[Chem.Mol], float]): function or method that takes
                a RDKit molecule and returns a real score 
            gamma (float): discount factor 
            lr (float): Optimizer learning rate 
        """

        self.net = net
        self.train_on_gpu = torch.cuda.is_available()
        self.device = 'cuda' if self.train_on_gpu else 'cpu'

        self.gamma = gamma
        self.scorer = scorer

        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

        # We use this to determine how many characters to sample
        # to obtain the required number of episodes.
        # Not the most elegant solution.
        self.avg_episode_length = 50  # 48.9

    def gradient(self, batch_size: int, clip: Optional[float] = None) -> Tuple[float, Tensor]:
        """
        Sample a batch of episodes (SMILES strings), score them and apply the policy gradient
        
        Args:
            batch_size (int): Number of episodes
            clip (Optiona[float]): Clip gradients 

        Returns:
            total_reward: float
            rl_loss: Tensor

        """

        rl_loss = Tensor([0], device=self.device)
        total_reward = 0

        # Sample N episodes from the net
        num_chars = int(1.25 * self.avg_episode_length * batch_size)
        batch = self._get_batch(size=num_chars)[:batch_size]

        self.opt.zero_grad()
        self.net.train()

        for episode in batch:

            # Assign reward of 0 if the molecule is invalid
            # Otherwise call the scorer
            mol = Chem.MolFromSmiles(episode[:-1])
            if mol is None:
                reward = 0
            else:
                reward = self.scorer(mol)

            # Convert string of characters back into sparse one hot matrix.
            # Can we make this more efficient? We're doing the work twice
            episode_onehot = np.array([[self.net.char2int[char] for char in episode]])
            episode_onehot = utils.one_hot_encode(episode_onehot, len(self.net.chars))

            discounted_reward = reward
            total_reward += reward

            h = self.net.init_hidden(1)
            # "Following" (replaying) the trajectory and accumulating the loss
            for i in range(len(episode) - 1):
                # log_p is a vector of log probs for the next action
                log_p, h = self._get_step_probs(episode[i], h, log=True)
                # Index of actual future token
                top_i = episode_onehot[0, i + 1].argmax()
                # Compute loss and discount the reward
                rl_loss -= log_p[0, top_i] * discounted_reward
                discounted_reward = discounted_reward * self.gamma

        rl_loss = rl_loss / batch_size
        total_reward = total_reward / batch_size
        rl_loss.backward()

        if clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), clip
            )

        self.opt.step()

        return total_reward, rl_loss.item()

    def _get_batch(self, size: int, prime: str = 'C', sep: str = '\n') -> List[str]:
        """
        Get batch of episodes from the net
        Args:
            size (int): Number of characters to sample 
            prime (str): Prime the net with a character 

        Returns:
            batch (List[str]): list of episodes
            
        """

        batch = sample.get_sample(self.net, size=size, prime=prime)
        # Split sampled chars by the separator and drop the last string,
        # which is likely incomplete
        batch = [episode + sep for episode in batch.split(sep)][:-1]
        return batch

    def _get_step_probs(
            self, char: str,
            h: Tuple[Tensor, Tensor],
            log: bool = True) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Given a character and hidden state, return the probabilities of each action
        and the new hidden state)
        
        Args:
            char (str) 
            h (Tuple[Tensor, Tensor]): hidden state 
            log (bool): Return log softmax values instead of softmax 

        Returns:
            p (Tensor): probabilities of the next h
            h (Tuple[Tensor, Tensor]): new hidden state 

        """

        # tensor inputs
        x = np.array([[self.net.char2int[char]]])
        x = utils.one_hot_encode(x, len(self.net.chars))
        inputs = torch.from_numpy(x)

        if train_on_gpu:
            inputs = inputs.cuda()

        # detach hidden state from history
        h = tuple([each.data for each in h])
        # get the output of the model
        out, h = self.net(inputs, h)

        # get the character probabilities
        if log:
            p = F.softmax(out, dim=1)
        else:
            p = F.log_softmax(out, dim=1)

        return p, h

    @staticmethod
    def logp_scorer(mol: Chem.Mol, scale: float = 1e-6, sigmoid_x0: float = 3., sigmoid_b: float = 1.) -> float:
        """
        Simple scorer that returns the sigmoid of the cLogP
        
        Args:
            mol (Chem.Mol): RDKit molecule 
            scale (float): Scale the score by a factor
            sigmoid_x0 (float): Sigmoid inflection  
            sigmoid_b (float): Sigmoid sharpness 

        Returns:
            val (float): Score

        """
        val = Descriptors.MolLogP(mol)
        val = scale * sigmoid(val, sigmoid_x0, sigmoid_b)
        return val

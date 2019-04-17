import torch
from torch import nn

from collections import deque


class Linear(nn.Module):
    """A simple Linear model for solving RL problems"""

    def __init__(self, input_dimension=128, n_actions=4):
        """
        Initialises the object.

        Args:
            input_dimension (int): The dimension of the input.
            n_actions (int): The number of possible actions.
        """

        super(Linear, self).__init__()

        self.name = 'Linear'

        self.layer = nn.Linear(input_dimension, n_actions)

    def get_config(self):

        config = {
            'input_dimension': self.layer.in_features,
            'n_actions': self.layer.out_features
        }

        return config

    def forward(self, x):
        """
        Performs the forward computation.

        Args:
            x (torch.Tensor): The input vector.

        Returns:
            actions (torch.Tensor): The estimated action-value function on the current state.
        """

        actions = self.layer(x)

        return actions

    def reset(self):
        pass

    def predict(self, x):
        pass

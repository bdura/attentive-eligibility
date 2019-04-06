import torch
from torch import nn

from collections import deque


class MLP(nn.Module):
    """A simple RNN for solving RL problems"""

    def __init__(self, input_dimension=128, hidden_dimension=50, n_hidden_layers=3, n_actions=4, dropout=.1):
        """
        Initialises the object.

        Args:
            input_dimension (int): The dimension of the input.
            hidden_dimension (int): The dimension of the hidden unit.
            n_actions (int): The number of possible actions.
            dropout (float): The dropout factor.
        """

        super(MLP, self).__init__()

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.input_layer = nn.Linear(input_dimension, hidden_dimension)

        self.hidden = nn.Sequential(*[
            nn.Sequential(
                self.dropout,
                nn.Linear(hidden_dimension, hidden_dimension),
                self.activation,
            )
            for _ in range(n_hidden_layers)
        ])

        self.action_layer = nn.Linear(hidden_dimension, n_actions)

    def forward(self, x):
        """
        Performs the forward computation.

        Args:
            x (torch.Tensor): The input vector.

        Returns:
            actions (torch.Tensor): The estimated action-value function on the current state.
        """

        x = self.dropout(x)

        x = self.input_layer(x)
        x = self.activation(x)

        x = self.hidden(x)

        x = self.dropout(x)

        actions = self.action_layer(x)

        return actions

    def reset(self):
        pass

    def predict(self, x):
        pass

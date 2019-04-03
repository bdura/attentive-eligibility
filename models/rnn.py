import torch
from torch import nn

from collections import deque


class RNN(nn.Module):

    def __init__(self, input_dimension=5, hidden_dimension=5, key_dimension=4,
                 n_actions=4, dropout=.1, horizon=-1):

        super(RNN, self).__init__()

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.input_layer = nn.Linear(input_dimension, hidden_dimension)

        self.context_layer = nn.Linear(2 * hidden_dimension, hidden_dimension)

        # Attention mechanism
        self.key = nn.Linear(hidden_dimension, key_dimension)
        self.query = nn.Linear(hidden_dimension, key_dimension)

        self.first_context = nn.Parameter(torch.zeros(2, 5))

        self.context = deque()
        self.keys = deque()
        self.horizon = horizon

        self.action_layer = nn.Linear(hidden_dimension, n_actions)

    def reset(self):

        self.context.clear()
        self.keys.clear()

        self.context.append(self.first_context.data)

        self.keys.append(self.key(self.first_context.data))

    def forward(self, x):
        """
        Performs the forward computation.

        Args:
            x (torch.Tensor): The input vector.
            # context (torch.Tensor): A torch tensor containing the past context.
            # keys (torch.Tensor): A torch tensor containing the past keys.

        Returns:
            actions (torch.Tensor): The estimated action-value function on the current state.
            context (torch.Tensor): The full context.
            keys (torch.Tensor): The full keys
        """

        x = self.input_layer(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.context_layer(torch.cat((self.context[-1], x), dim=1))
        x = self.activation(x)
        x = self.dropout(x)

        self.context.append(x)
        if self.horizon > -1 and len(self.context) > self.horizon + 1:
            self.context.popleft()

        context = torch.stack(tuple(self.context))

        query = self.query(x)
        key = self.key(x)

        self.keys.append(key)
        if self.horizon > -1 and len(self.keys) > self.horizon + 1:
            self.keys.popleft()

        keys = torch.stack(tuple(self.keys))

        attention = nn.functional.softmax((keys.unsqueeze(2) @ query.unsqueeze(0).unsqueeze(3)).squeeze(-1), dim=0)

        weighted_context = (attention * context).sum(dim=0)

        actions = self.action_layer(weighted_context)

        return actions


if __name__ == '__main__':

    rnn = RNN()

    tensor = torch.randn((2, 5))

    rnn.reset()
    rnn(tensor)

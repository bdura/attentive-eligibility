import torch
from torch import nn

from collections import deque


class RNN(nn.Module):
    """A simple RNN for solving RL problems"""

    def __init__(self, input_dimension=5, hidden_dimension=5, n_actions=4, dropout=0., truncate=20):
        """
        Initialises the object.

        Args:
            input_dimension (int): The dimension of the input.
            hidden_dimension (int): The dimension of the hidden unit.
            n_actions (int): The number of possible actions.
            dropout (float): The dropout factor.
            truncate (int): The number of hidden states to backpropagate into.
        """

        super(RNN, self).__init__()

        self.name = 'RNN'

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.input_layer = nn.Linear(input_dimension, hidden_dimension)

        self.context_layer = nn.Linear(2 * hidden_dimension, hidden_dimension)

        self.first_context = nn.Parameter(torch.zeros((1, hidden_dimension)))

        self.context = None

        self.contexts = deque()
        self.truncate = truncate

        self.action_layer = nn.Linear(hidden_dimension, n_actions)

    def get_config(self):

        config = {
            'input_dimension': self.input_layer.in_features,
            'hidden_dimension': self.input_layer.out_features,
            'n_actions': self.action_layer.out_features,
            'dropout': self.dropout.p,
            'truncate': self.truncate,
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

        x = self.input_layer(x)
        x = self.activation(x)
        x = self.dropout(x)

        if self.context is not None:
            x = self.context_layer(torch.cat((self.context, x), dim=1))
        else:
            n = x.size(0)
            context = torch.cat(tuple([self.first_context for _ in range(n)]))
            x = self.context_layer(torch.cat((context, x), dim=1))

        x = self.activation(x)

        self.context = x

        self.contexts.append(x)
        if len(self.contexts) > self.truncate:
            self.contexts.popleft().detach_()

        x = self.dropout(x)

        actions = self.action_layer(x)

        return actions

    def predict(self, x):
        """
        Performs an alternative forward computation, wherein gradients are
        not computed and the hiddens is left unchanged.

        Args:
            x (torch.Tensor): The input vector.

        Returns:
            actions (torch.Tensor): The estimated action-value function on the current state.
        """

        with torch.no_grad():

            x = self.input_layer(x)
            x = self.activation(x)

            if self.context is not None:
                x = self.context_layer(torch.cat((self.context, x), dim=1))
            else:
                n = x.size(0)
                context = torch.cat(tuple([self.first_context for _ in range(n)]))
                x = self.context_layer(torch.cat((context, x), dim=1))

            x = self.activation(x)

            actions = self.action_layer(x)

            return actions

    def reset(self):
        """Resets the time-dependency of the model"""
        self.context = None


class AttentiveRNN(nn.Module):

    def __init__(self, input_dimension=128, hidden_dimension=50, key_dimension=5,
                 n_actions=4, dropout=0., horizon=-1, truncate=20):
        """
        Initialises the object.

        Args:
            input_dimension (int): The dimension of the input.
            hidden_dimension (int): The dimension of the hidden unit.
            key_dimension (int): The dimension of the keys/queries.
            n_actions (int): The number of possible actions.
            dropout (float): The dropout factor.
            horizon (int): The number of contexts considered during the attention phase. -1 means consider all.
        """

        super(AttentiveRNN, self).__init__()

        self.name = 'AttentiveRNN'

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.tanh = nn.Tanh()

        self.input_layer = nn.Linear(input_dimension, hidden_dimension)

        self.hidden_layer = nn.Linear(2 * hidden_dimension, hidden_dimension)

        # Attention mechanism
        self.key_hidden = nn.Linear(hidden_dimension, key_dimension)
        self.key_context = nn.Linear(hidden_dimension, key_dimension)
        self.query = nn.Parameter(.001 * torch.randn(1, key_dimension))

        self.first_hidden = nn.Parameter(torch.zeros(1, hidden_dimension))
        self.context = None

        self.hiddens = deque()
        self.horizon = horizon

        self.contexts = deque()
        self.truncate = truncate

        self.action_layer = nn.Linear(hidden_dimension, n_actions)

    def get_config(self):

        config = {
            'input_dimension': self.input_layer.in_features,
            'hidden_dimension': self.input_layer.out_features,
            'n_actions': self.action_layer.out_features,
            'dropout': self.dropout.p,
            'truncate': self.truncate,
            'horizon': self.horizon,
            'key_dimension': self.key.out_features,
        }

        return config

    def reset(self):
        """Resets the time-dependency of the model"""

        self.hiddens.clear()
        self.context = None

    def forward(self, x):
        """
        Performs the forward computation.

        Args:
            x (torch.Tensor): The input vector.

        Returns:
            actions (torch.Tensor): The estimated action-value function on the current state.
        """

        n = x.size(0)

        query = torch.cat(tuple([self.query for _ in range(n)]))

        if len(self.hiddens) == 0:
            hidden = torch.cat(tuple([self.first_hidden for _ in range(n)]))
            self.hiddens.append(hidden)

            self.context = hidden

        x = self.input_layer(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.hidden_layer(torch.cat((self.hiddens[-1], x), dim=1))
        x = self.activation(x)

        self.hiddens.append(x)
        if self.horizon > -1 and len(self.hiddens) > self.horizon + 1:
            self.hiddens.popleft()

        hiddens = torch.stack(tuple(self.hiddens))

        key_context = self.key_context(self.context)

        keys = []
        for hidden in self.hiddens:
            keys.append(self.tanh(key_context + self.key_hidden(hidden)))

        keys = torch.stack(tuple(keys))

        pre_attention = (keys.unsqueeze(2) @ query.unsqueeze(0).unsqueeze(3)).squeeze(-1)
        attention = nn.functional.softmax(pre_attention, dim=0)

        self.context = (attention * hiddens).sum(dim=0)

        actions = self.action_layer(self.context)

        return actions

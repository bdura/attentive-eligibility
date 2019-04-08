import torch
from torch import nn

from collections import deque


class RNN(nn.Module):
    """A simple RNN for solving RL problems"""

    def __init__(self, input_dimension=5, hidden_dimension=5, n_actions=4, dropout=.1, truncate=20):
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
            self.contexts.popleft().detach()

        x = self.dropout(x)

        actions = self.action_layer(x)

        return actions

    def predict(self, x):
        """
        Performs an alternative forward computation, wherein gradients are
        not computed and the context is left unchanged.

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
                 n_actions=4, dropout=.1, horizon=-1, truncate=20):
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

        self.input_layer = nn.Linear(input_dimension, hidden_dimension)

        self.context_layer = nn.Linear(2 * hidden_dimension, hidden_dimension)

        # Attention mechanism
        self.key = nn.Linear(hidden_dimension, key_dimension)
        self.query = nn.Linear(hidden_dimension, key_dimension)

        self.first_context = nn.Parameter(torch.zeros(1, hidden_dimension))

        self.context = deque()
        self.keys = deque()
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

        self.context.clear()
        self.keys.clear()

    def forward(self, x):
        """
        Performs the forward computation.

        Args:
            x (torch.Tensor): The input vector.

        Returns:
            actions (torch.Tensor): The estimated action-value function on the current state.
        """

        if len(self.context) == 0:
            n = x.size(0)
            context = torch.cat(tuple([self.first_context for _ in range(n)]))
            self.context.append(context)
            self.keys.append(self.key(context))

        x = self.input_layer(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.context_layer(torch.cat((self.context[-1], x), dim=1))
        x = self.activation(x)
        x = self.dropout(x)

        self.context.append(x)
        if self.horizon > -1 and len(self.context) > self.horizon + 1:
            self.context.popleft()

        self.contexts.append(x)
        if len(self.contexts) > self.truncate:
            self.contexts.popleft().detach()

        context = torch.stack(tuple(self.context))

        query = self.query(x)
        key = self.key(x)

        self.keys.append(key)
        if self.horizon > -1 and len(self.keys) > self.horizon + 1:
            self.keys.popleft()

        keys = torch.stack(tuple(self.keys))

        pre_attention = (keys.unsqueeze(2) @ query.unsqueeze(0).unsqueeze(3)).squeeze(-1)
        attention = nn.functional.softmax(pre_attention, dim=0)

        weighted_context = (attention * context).sum(dim=0)

        actions = self.action_layer(weighted_context)

        return actions

    def predict(self, x):
        """
        Performs an alternative forward computation, wherein gradients are
        not computed and the context is left unchanged.

        Args:
            x (torch.Tensor): The input vector.

        Returns:
            actions (torch.Tensor): The estimated action-value function on the current state.
        """

        full_context = deque()
        full_context.extend(self.context)

        full_keys = deque()
        full_keys.extend(self.keys)

        with torch.no_grad():

            if len(self.context) == 0:
                n = x.size(0)
                context = torch.cat(tuple([self.first_context for _ in range(n)]))
                full_context.append(context)
                full_keys.append(self.key(context))

            x = self.input_layer(x)
            x = self.activation(x)

            x = self.context_layer(torch.cat((full_context[-1], x), dim=1))
            x = self.activation(x)

            full_context.append(x)
            if self.horizon > -1 and len(full_context) > self.horizon + 1:
                full_context.popleft()

            context = torch.stack(tuple(full_context))

            query = self.query(x)
            key = self.key(x)

            full_keys.append(key)
            if self.horizon > -1 and len(full_keys) > self.horizon + 1:
                full_keys.popleft()

            keys = torch.stack(tuple(full_keys))

            pre_attention = (keys.unsqueeze(2) @ query.unsqueeze(0).unsqueeze(3)).squeeze(-1)
            attention = nn.functional.softmax(pre_attention, dim=0)

            weighted_context = (attention * context).sum(dim=0)

            actions = self.action_layer(weighted_context)

            return actions

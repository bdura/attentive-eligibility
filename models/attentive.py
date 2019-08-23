from torch import nn

import math
import torch


class Attentive(nn.Module):

    def __init__(self, input_dim, hidden_dimensions=(64, 64, 64, 64), memory_size=256, memory_dimension=32, key_dimension=16, output_dim=2):

        super(Attentive, self).__init__()

        self.memory = nn.Parameter(torch.randn(memory_size, memory_dimension))

        dims = [input_dim] + list(hidden_dimensions)

        self.mlp = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(d1, d2),
                nn.ReLU()
            )
            for d1, d2 in zip(dims[:-1], dims[1:])
        ])

        self.memory2key = nn.Linear(memory_dimension, key_dimension)
        self.hidden2query = nn.Linear(hidden_dimensions[-1], key_dimension)
        self.softmax = nn.Softmax(dim=2)
        self.d = key_dimension

        self.hidden2value = nn.Linear(hidden_dimensions[-1], memory_dimension)

        self.output_layer = nn.Linear(memory_dimension, output_dim)

    def forward(self, x):

        h = self.mlp(x)
        x = self.hidden2value(h)

        q = self.hidden2query(h)
        k = self.memory2key(self.memory)

        if len(q.size()) == 2:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = self.memory.unsqueeze(0)
        else:
            v = self.memory

        z = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d)
        s = torch.bmm(self.softmax(z), v)

        s = s.squeeze(0)

        x = x + s.detach()

        x = self.output_layer(x)

        return x

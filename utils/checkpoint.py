import torch
import json

import os

from models import rnn, mlp
from control import agents, environments

import gym


models_dict = {
    'MLP': mlp.MLP,
    'RNN': rnn.RNN,
    'AttentiveRNN': rnn.AttentiveRNN,
}

agents_dict = {
    'DQNAgent': agents.DQNAgent,
}


def load_environment(directory):

    with open(os.path.join(directory, 'config.json'), 'r') as f:
        config = json.load(f)

    buffer = torch.load(os.path.join(directory, 'buffer.pth'))

    model = models_dict[config['types']['model']](**config['model'])
    model.load_state_dict(torch.load(os.path.join(directory, 'state_dict.pth')))

    optimiser = torch.optim.Adam(model.parameters(), lr=.001)

    agent = agents_dict[config['types']['agent']](
        model=model,
        optimiser=optimiser,
        **config['agent']
    )

    environment = environments.Environment(
        agent=agent,
        environment=gym.make('Breakout-ram-v0'),
        **config['environment']
    )
    environment.replay_memory = buffer

    return environment

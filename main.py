import torch

import gym

from models import rnn
from control import agents, environments


def main():

    pass


if __name__ == '__main__':

    rnn = rnn.RNN(input_dimension=128, hidden_dimension=50, truncate=1)  # , key_dimension=15)
    optimiser = torch.optim.Adam(rnn.parameters(), lr=.001)

    agent = agents.DQNAgent(rnn, optimiser)

    environment = environments.ExpectedSarsa(
        environment=gym.make('Breakout-ram-v0'),
        agent=agent,
        gamma=.9,
        verbose=True
    )

    environment.reset()

    for _ in range(10):
        print(environment.backup())

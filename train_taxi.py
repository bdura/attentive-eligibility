import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import models.rnn as rnns
import models.mlp as mlps
import models.linear as linears
import control.agents as agents
import control.environments as env
import copy


def experiment_train(environment, n_trainings):
    returns_train, returns_eval, q_estimation = [], [], []

    for _ in range(n_trainings):
        returns = environment.train(segments=10, episodes=100)

        plt.figure()
        plt.plot(returns.T[0], label='training')
        plt.plot(returns.T[1], label='evaluation')
        plt.legend()
        plt.show()

        # agent.temperature *= .8

        returns_train.append(np.mean(returns.T[0]))
        returns_eval.append(np.mean(returns.T[1]))

        q_estimation.append(environment.agent.q(environment.state_representation(environment.environment.reset())))

    return returns_train, returns_eval, q_estimation

if __name__ == '__main__':

    env_name = 'Taxi-v2'
    # env_name = 'Breakout-ram-v0'

    environment = env.Environment(
        environment=gym.make(env_name),
        agent=None,
        verbose=True,
        max_steps=300,
        capacity=5000,
        representation_method='one_hot_encoding'
    )

    model = mlps.MLP(input_dimension=environment.get_input_dimension(),
                     hidden_dimension=200,
                     n_hidden_layers=2,
                     n_actions=environment.n_actions,
                     dropout=.5)

    optimiser = torch.optim.Adam(model.parameters(), lr=.001)

    agent = agents.DQNAgent(model,
                            optimiser,
                            gamma=.99,
                            temperature=1,
                            algorithm='expsarsa',
                            n_actions=environment.n_actions)

    environment.agent = agent
    agent.commit()

    returns_train, returns_eval, q_estimation = experiment_train(environment, 10)

    plt.figure()
    plt.plot(returns_train, label='mean training')
    plt.plot(returns_eval, label='mean evaluation')
    plt.legend()
    plt.show()

    q_estimation = np.asarray(q_estimation)
    plt.figure()
    plt.plot(q_estimation[:, 0], label='initial q of wait')
    plt.plot(q_estimation[:, 1], label='initial q of left')
    plt.plot(q_estimation[:, 2], label='initial q of right')
    plt.legend()
    plt.show()

    environment.agent.save('../saved/mixed_mlp')

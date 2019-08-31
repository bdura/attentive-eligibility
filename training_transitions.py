import gym
from torch import nn

from control.agents import DQNAgent


class DQN(nn.Module):

    def __init__(self, input_dim=4, hidden_dim=50, n_layers=1, n_actions=2):
        super(DQN, self).__init__()

        self.model = nn.Sequential(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ),
            *[
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                for _ in range(n_layers)
            ],
            nn.Linear(hidden_dim, n_actions),
        )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        return self.model(x)


BATCH_SIZE = 128
BUFFER_SIZE = 1000
N_EPISODES = 1000

model_params = dict(input_dim=4, hidden_dim=50, n_layers=1, n_actions=2)

if __name__ == '__main__':
    target_net = DQN(**model_params)
    policy_net = DQN(**model_params)
    environment = gym.make('CartPole-v0')
    agent = DQNAgent(target_net=target_net, policy_net=policy_net, environment=environment)
    agent.train(10000)

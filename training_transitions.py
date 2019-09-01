import torch
from gym_minigrid.wrappers import *
from torch import nn
from torch import optim

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


class CustomImgObsWrapper(ImgObsWrapper):
    """
    Flatten the output.
    """

    def observation(self, obs):
        return torch.flatten(torch.tensor(super(CustomImgObsWrapper, self).observation(obs)))


class CustomFullyObsWrapper(FullyObsWrapper):
    """
    Flatten the output.
    """

    def observation(self, obs):
        return torch.flatten(torch.tensor(super(CustomFullyObsWrapper, self).observation(obs)))


class CustomTensorized(gym.core.ObservationWrapper):
    """
    Flatten the output.
    """

    def observation(self, observation):
        return torch.flatten(torch.tensor(observation))

    def _observation(self, obs):
        return torch.flatten(torch.tensor(obs))


# env = gym.make('MiniGrid-Empty-5x5-v0')
# env = CustomFullyObsWrapper(env)

env = gym.make('CartPole-v0')
env = CustomTensorized(env)

n_observations = np.prod(env.observation_space.shape)
n_actions = env.action_space.n

model_params = dict(input_dim=n_observations, hidden_dim=200, n_layers=2, n_actions=n_actions)

if __name__ == '__main__':
    target_net = DQN(**model_params).eval()
    policy_net = DQN(**model_params).train()

    agent = DQNAgent(
        target_net=target_net,
        policy_net=policy_net,
        environment=env,
        temperature=20,
        annealing=.999,
        tboard_path='tensorboard/cartpole-qlearning',
        optimizer=lambda x: optim.RMSprop(x, lr=1e-2),
        attention_k=10,
        use_memory_attention=True
    )
    agent.train(10000)

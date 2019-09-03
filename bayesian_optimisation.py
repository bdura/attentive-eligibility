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

# model_params = dict(input_dim=n_observations, hidden_dim=200, n_layers=2, n_actions=n_actions)


from skopt.space import Real, Integer, Categorical
from skopt import gp_minimize

dimensions = [
    Integer(30, 60),
    Integer(0, 3),
    Real(1e-4, 1e-1),
    Categorical([optim.RMSprop, optim.Adam]),
    Real(.1, 4),
    Real(.97, 1)
    # Categorical([True, False]),
    # Integer(5, 20),
    # Real(1e-2, 1),
]


def objective(hidden_dim, n_layers, lr,
              optimiser, temperature, annealing):  # , use_memory_attention, attention_k, attention_t):
    params = dict(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        lr=lr,
        optimiser=optimiser,
        temperature=temperature,
        annealing=annealing,
        # use_memory_attention=use_memory_attention,
        # attention_k=attention_k,
        # attention_t=attention_t,
    )

    print(*[f'{k}={v}' for k, v in params.items()], sep='\n')

    model_params = dict(input_dim=n_observations, hidden_dim=hidden_dim, n_layers=n_layers, n_actions=n_actions)

    target_net = DQN(**model_params).eval()
    policy_net = DQN(**model_params).train()

    agent = DQNAgent(
        target_net=target_net,
        policy_net=policy_net,
        environment=env,
        temperature=temperature,
        annealing=annealing,
        optimizer=lambda x: optimiser(x, lr=lr),
        # attention_k=attention_k.item(),
        # attention_t=attention_t.item(),
        # use_memory_attention=use_memory_attention,
        tboard_path=f'tensorboard/skopt2'
    )

    agent.train(500)

    results = np.array([agent.episode(greedy=True)[0] for _ in range(100)])

    print('>> Result :\t', results.mean(), '\n')

    # Negative mean since we are minimising
    return - results.mean()


if __name__ == '__main__':
    res = gp_minimize(
        lambda x: objective(*x),
        dimensions,
        n_calls=15,
        n_random_starts=5,
        random_state=123,
    )

    print(res)

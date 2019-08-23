import math
import random
from itertools import count

import gym
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim

from control.replay import ReplayMemory, Transition

from tqdm import tqdm
from control.agents import DQNAgent



# ENVIRONMENT



# MODEL (we may import a model)


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

model_params = dict(input_dim=4, hidden_dim=50, n_layers=1, n_actions=2)

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
# init_screen = get_screen()
# _, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space



# optimizer = optim.SGD(policy_net.parameters(), lr=.9)

# optimizer = optim.Adam(policy_net.parameters())
# optimizer = eligibility.EligibilitySGD(policy_net.parameters(), lr=.9, gamma=.9, lambd=.9)






def train():

    print('>> Beginning training')

    for i_episode in tqdm(range(N_EPISODES), ascii=True, ncols=100):
        # Initialize the environment and state
        state = torch.tensor(environment.reset()).float().unsqueeze(0)  # .unsqueeze(0)

        full_return = 0

        for t in count():
            # Select and perform an action
            action = select_action(state)
            s, reward, done, _ = environment.step(action.item())

            full_return += reward

            reward = torch.tensor([reward], device=device)

            # Observe new state
            if not done:
                next_state = torch.tensor(s).float().unsqueeze(0)  # .unsqueeze(0)
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model()

            if done:
                writer.add_scalar('duration', t, i_episode)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())


if __name__ == '__main__':
    target_net = DQN(**model_params)
    policy_net = DQN(**model_params)
    environment = gym.make('CartPole-v0')
    agent = DQNAgent(target_net=target_net, policy_net=policy_net, environment=environment)
    agent.train(10)

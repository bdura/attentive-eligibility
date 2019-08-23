import math
import random
from itertools import count

import gym
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from tqdm import tqdm

import numpy as np

from control.replay import Transition, EpisodeReplayMemory, Episode

# TENSORBOARD
writer = SummaryWriter('tensorboard/episode')

# ENVIRONMENT
environment = gym.make('CartPole-v0')


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
    def forward(self, x, mask=None):
        return self.model(x)

    def reset(self):
        pass


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
N_EPISODES = 500
BUFFER_SIZE = 1000

model_params = dict(input_dim=4, hidden_dim=50, n_layers=1, n_actions=2)

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
# init_screen = get_screen()
# _, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = environment.action_space.n

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

policy_net = DQN(**model_params).to(device)
target_net = DQN(**model_params).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# optimizer = optim.SGD(policy_net.parameters(), lr=.9)
optimizer = optim.Adam(policy_net.parameters())
# optimizer = optim.Adam(policy_net.parameters())
# optimizer = eligibility.EligibilitySGD(policy_net.parameters(), lr=.9, gamma=.9, lambd=.9)
memory = EpisodeReplayMemory(BUFFER_SIZE)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model_per_episode():
    if len(memory) < BATCH_SIZE:
        return

    policy_net.reset()
    target_net.reset()

    episodes = memory.sample(BATCH_SIZE)

    max_length = np.max([len(episode) for episode in episodes])
    episodes = np.array([episode.get_length(max_length) for episode in episodes])

    steps = episodes.swapaxes(0, 1)

    for i in range(len(steps)):

        batch = steps[i]

        episode_mask = np.array([b[0] is not None for b in batch], dtype=bool)
        episode_index = np.arange(len(episode_mask))[episode_mask]

        steps = steps[:, episode_index]

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*batch[episode_mask]))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=device, dtype=torch.uint8)

        try:
            non_final_next_states = torch.cat(tuple([s for s in batch.next_state
                                                     if s is not None]))
        except:
            non_final_next_states = None

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        mask = torch.tensor(episode_index)
        state_action_values = policy_net(state_batch, mask).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(len(batch.state), device=device)

        if non_final_next_states is not None:
            mask = torch.arange(len(batch.state))[non_final_mask == 1]
            next_state_values[non_final_mask] = target_net(non_final_next_states, mask).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch.float()

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        writer.add_scalar('loss', loss.item(), steps_done)
        # for param in policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)

        optimizer.step()


def train():
    print('>> Beginning training')

    for i_episode in tqdm(range(N_EPISODES), ascii=True, ncols=100):

        policy_net.reset()

        # Initialize the environment and state
        state = torch.tensor(environment.reset()).float().unsqueeze(0)  # .unsqueeze(0)
        episode = Episode()

        full_return = 0

        for t in count():
            # Select and perform an action
            action = select_action(state)
            s, reward, done, _ = environment.step(action.item())
            reward = torch.tensor([reward], device=device)

            full_return += reward

            # Observe new state
            if not done:
                next_state = torch.tensor(s).float().unsqueeze(0)  # .unsqueeze(0)
            else:
                next_state = None

            episode.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            if done:
                writer.add_scalar('duration', t, i_episode)
                break

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        memory.push(episode)

        # Perform one step of the optimization (on the target network)
        batch = optimize_model_per_episode()

        if batch is not None:
            break


if __name__ == '__main__':
    train()

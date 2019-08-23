import math

import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

from control.utils import BaseAgent
from control.replay import ReplayMemory
from torch import optim
from control.replay import Transition
import random
from tqdm import tqdm
from tensorboardX import SummaryWriter
from itertools import count
import copy


DEFAULT_BUFFER_SIZE = 1000
TARGET_UPDATE = 10
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

class DQNAgent(BaseAgent):
    """
    A general class for value approximation by a neural network
    """

    name = 'DQNAgent'

    def __init__(self, policy_net, target_net, environment, gamma=.99, temperature=1, algorithm='expsarsa',
                 use_eligibility=False, use_double_learning=True, terminal_state=None, buffer_size=DEFAULT_BUFFER_SIZE,
                 optimizer=optim.Adam, criterion=nn.SmoothL1Loss, batch_size=128, tboard_path='tensorboard/transitions'):
        """
        Initialises the object.

        Args:
            model (nn.Module): A Pytorch module.
            optimiser (torch.optim.self.optimizer): An self.optimizer.
        """

        super(DQNAgent, self).__init__(temperature=temperature, environment=environment, gamma=gamma, algorithm=algorithm,
                                       use_eligibility=use_eligibility)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.memory = ReplayMemory(buffer_size)
        self.criterion = criterion()

        self.batch_size = batch_size
        self.policy_net = policy_net.to(self.device)
        self.target_net = target_net.to(self.device)
        self.target_net.load_state_dict(policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optimizer(policy_net.parameters())
        self.criterion = nn.SmoothL1Loss()
        self.scheduler = MultiStepLR(self.optimizer, gamma=0.3, milestones=[10, 30, 60])

        self.use_double_learning = use_double_learning

        self.terminal_state = terminal_state

        # TENSORBOARD
        if tboard_path is not None:
            self.writer = SummaryWriter(tboard_path)

    def get_config(self):
        """
        Compute the configuration of the agent as a dictionary .

        Returns:
            config, dict: configuration of the agent.
        """

        config = {
            'gamma': self.gamma,
            'temperature': self.temperature,
            'algorithm': self.algorithm,
            'use_eligibility': self.use_eligibility
        }

        return config

    def commit(self):
        """Commits the changes made to the model, by moving them over to the fixed model."""

        self.fixed.load_state_dict(copy.deepcopy(self.model.state_dict()))

    def tensorise(self, array):
        """
        Returns the tensorised version of an array.

        Args:
            array (np.array): A numpy array.

        Returns:
            tensor (torch.Tensor): The tensorised version of the array.
        """

        if array.dtype == np.int:
            dtype = None
        else:
            dtype = torch.float

        tensor = torch.tensor(array, dtype=dtype).to(self.device)

        return tensor

    def eval(self):
        """Puts the model in evaluation mode."""
        self.model.eval()

    def train(self):
        """Puts the model in training mode."""
        self.model.train()

    def q(self, state):
        """
        Returns the current approximation for the action-value function.

        Args:
            state (np.array): The representation of a state.

        Returns:
            actions (np.array): The value for each possible action.
        """

        with torch.no_grad():

            state = self.tensorise(state)

            # Add a batch dimension
            squeezed = len(state.size()) == 1

            if squeezed:
                state = state.unsqueeze(0)

            if self.use_double_learning:
                actions = self.fixed(state)
            else:
                actions = self.model(state)

            if self.terminal_state is not None:
                for i in range(state.size()[0]):
                    if torch.argmax(state[i, :]) == self.terminal_state:
                        actions[i, :] = torch.zeros(actions[i, :].size())

            # Remove the batch dimension
            if squeezed:
                actions = actions.squeeze()

            return actions.detach().cpu().numpy()

    def update(self, state, action, target):
        """
        Performs a gradient descent step on the model.

        Args:
            state (np.array): The representation for the state.
            action (int): The action taken.
            target (float): The target (be it Sarsa, ExpSarsa or QLearning).
        """

        state = self.tensorise(state)
        self.optimizer.zero_grad()

        # Add a batch dimension
        state = state.unsqueeze(0)

        actions = self.model(state)

        q = actions.squeeze()[action]

        loss = self.criterion(q, self.tensorise(target))
        loss.backward(retain_graph=True)

        if self.use_eligibility:
            self.optimizer.step(loss)
        else:
            self.optimizer.step()

    def target(self, reward, next_state):
        """
        Compute a single target to perform an update.

        Args:
            reward: float, observed reward by the agent.
            next_state: np.array, following state.

        Returns:
            target: float, target for the update.
        """

        q = self.q(next_state)

        if self.algorithm == 'sarsa':
            probability = self.boltzmann(q)
            action = self.sample_action(probability)

            target = torch.Tensor(
                [reward[i] + self.gamma * q[i][action[i]] for i in range(len(next_state))]
            ).reshape(-1, 1)

        elif self.algorithm == 'expsarsa':
            probability = self.boltzmann(q)

            target = torch.Tensor(
                [reward[i] + self.gamma * (probability[i] @ q[i].T) for i in range(len(next_state))]
            ).reshape(-1, 1)

        elif self.algorithm == 'qlearning':
            target = torch.Tensor(
                [reward[i] + self.gamma * q[i].max() for i in range(len(next_state))]
            ).reshape(-1, 1)

        else:
            raise Exception("Wrong agent.algorithm.")

        return target

    def targets(self, rewards, next_states):
        r"""
        Computes the targets corresponding to a tuple (reward, next_state).

        ..math::
            \mathrm{target} = r + \gamma \times \hat{q}(s', a')

        Args:
            rewards (np.array): The rewards.
            next_states (np.array): Next states.

        Returns:
            targets (np.array): The targets.
        """

        targets = []

        for reward, next_state in zip(rewards, next_states):
            targets.append(self.target(reward, next_state))

        return np.stack(targets)

    def batch_update(self, states, actions, rewards, next_states):
        """
        Performs a gradient descent step on the model.

        Args:
            states (np.array): The representation for the state.
            actions (np.array): The action taken.
            rewards (np.array): The reward.
            next_states (np.array): The representation for the next state.
        """

        # Resetting the networks.
        self.reset()

        targets = self.targets(rewards, next_states)

        for state, action, target in zip(states, actions, targets):

            # Zeroing the gradients
            self.optimizer.zero_grad()

            state = self.tensorise(state)

            q = torch.gather(self.model(state), dim=1, index=self.tensorise(action).unsqueeze(1))
            loss = self.criterion(q, self.tensorise(target))
            loss.backward(retain_graph=True)

            if self.use_eligibility:
                self.optimizer.step(loss)

            else:
                self.optimizer.step()

        return loss.item()

    def optimize_model(self, steps_done):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat(tuple([s for s in batch.next_state
                                                 if s is not None]))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch.float()

        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        self.writer.add_scalar('loss', loss.item(), steps_done)

        # for param in policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss

    def select_action(self, state, steps_done):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def train(self, num_episodes):

        print('>> Beginning training')
        steps_done = 0
        for i_episode in tqdm(range(num_episodes), ascii=True, ncols=100):
            # Initialize the self.environment and state
            state = torch.tensor(self.environment.reset()).float().unsqueeze(0)

            full_return = 0

            for t in count():
                # Select and perform an action
                action = self.select_action(state, steps_done)
                steps_done+=1
                s, reward, done, _ = self.environment.step(action.item())

                full_return += reward

                reward = torch.tensor([reward], device=self.device)

                # Observe new state
                if not done:
                    next_state = torch.tensor(s).float().unsqueeze(0)
                else:
                    next_state = None

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                self.optimize_model(steps_done)

                if done:
                    self.writer.add_scalar('duration', t, i_episode)
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def reset(self):
        """Resets the model."""

        self.model.reset()
        self.fixed.reset()

        # self.commit()

    def save(self, directory):
        """Saves the model weights."""

        torch.save(self.model.cpu().state_dict(), '{}/state_dict.pth'.format(directory))
        self.model = self.model.to(self.device)

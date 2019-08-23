import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

from control.utils import BaseAgent

import copy


class DQNAgent(BaseAgent):
    """
    A general class for value approximation by a neural network
    """

    name = 'DQNAgent'

    def __init__(self, model, optimiser, n_actions, gamma=.9, temperature=1, algorithm='expsarsa',
                 use_eligibility=False, use_double_learning=True, terminal_state=None):
        """
        Initialises the object.

        Args:
            model (nn.Module): A Pytorch module.
            optimiser (torch.optim.Optimizer): An optimizer.
        """

        super(DQNAgent, self).__init__(temperature=temperature, n_actions=n_actions, gamma=gamma, algorithm=algorithm,
                                       use_eligibility=use_eligibility)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.fixed = copy.deepcopy(self.model).eval()
        # self.criterion = nn.SmoothL1Loss()
        self.criterion = nn.SmoothL1Loss()
        self.optimiser = optimiser
        self.scheduler = MultiStepLR(self.optimiser, gamma=0.3, milestones=[10, 30, 60])

        self.use_double_learning = use_double_learning

        self.terminal_state = terminal_state

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
        self.optimiser.zero_grad()

        # Add a batch dimension
        state = state.unsqueeze(0)

        actions = self.model(state)

        q = actions.squeeze()[action]

        loss = self.criterion(q, self.tensorise(target))
        loss.backward(retain_graph=True)

        if self.use_eligibility:
            self.optimiser.step(loss)
        else:
            self.optimiser.step()

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
            self.optimiser.zero_grad()

            state = self.tensorise(state)

            q = torch.gather(self.model(state), dim=1, index=self.tensorise(action).unsqueeze(1))
            loss = self.criterion(q, self.tensorise(target))
            loss.backward(retain_graph=True)

            if self.use_eligibility:
                self.optimiser.step(loss)

            else:
                self.optimiser.step()

        return loss.item()

    def reset(self):
        """Resets the model."""

        self.model.reset()
        self.fixed.reset()

        # self.commit()

    def save(self, directory):
        """Saves the model weights."""

        torch.save(self.model.cpu().state_dict(), '{}/state_dict.pth'.format(directory))
        self.model = self.model.to(self.device)

import torch
from torch import nn

import numpy as np

from control.utils import BaseAgent

import copy


class DQNAgent(BaseAgent):
    """
    A general class for value approximation by a neural network
    """

    name = 'DQNAgent'

    def __init__(self, model, optimiser, gamma=.9, temperature=1, algorithm='expsarsa', n_actions=4,
                 use_eligibility=False, use_double_learning=True):
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

        self.use_double_learning = use_double_learning

        if self.use_double_learning:
            self.fixed = copy.deepcopy(self.model).eval()

        self.criterion = nn.MSELoss(reduction='sum')
        self.optimiser = optimiser

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

        if self.use_double_learning:
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
        self.model.zero_grad()

        # Add a batch dimension
        state = state.unsqueeze(0)

        actions = self.model(state)

        q = actions.squeeze()[action]

        loss = self.criterion(q, self.tensorise(target))
        # loss.backward(retain_graph=True)
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        # torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.5)

        if self.use_eligibility:
            self.optimiser.step(loss)
        else:
            self.optimiser.step()

    def target(self, reward, next_state, next_action):
        """
        Compute a single target to perform an update.

        Args:
            reward: float, observed reward by the agent.
            next_state: np.array, following state.
            next_action: the following action.

        Returns:
            target: float, target for the update.
        """

        q = self.q(next_state)

        reward = np.asarray(reward)

        probability = self.boltzmann(q)

        if self.algorithm == 'sarsa':
            target = reward + self.gamma * q[next_action]

        elif self.algorithm == 'expsarsa':
            target = reward + self.gamma * probability @ q.T

        else:
            if len(q.shape) == 1:
                target = reward + self.gamma * q.max()
            else:
                target = reward + self.gamma * q.max(axis=1)

        return target

    def targets(self, rewards, next_states, next_actions):
        r"""
        Computes the targets corresponding to a tuple (reward, next_state).

        ..math::
            \mathrm{target} = r + \gamma \times \hat{q}(s', a')

        Args:
            rewards (np.array): The rewards.
            next_states (np.array): Next states.
            next_actions (np.array): Next actions

        Returns:
            targets (np.array): The targets.
        """

        targets = []

        for reward, next_state, next_action in zip(rewards, next_states, next_actions):
            targets.append(self.target(reward, next_state, next_action))

        return np.stack(targets)

    def batch_update(self, states, actions, rewards, next_states, next_actions):
        """
        Performs a gradient descent step on the model.

        Args:
            states (np.array): The representation for the state.
            actions (np.array): The action taken.
            rewards (np.array): The reward.
            next_states (np.array): The representation for the next state.
            next_actions (np.array): The next actions.
        """

        # Resetting the networks.
        self.reset()

        targets = self.targets(rewards, next_states, next_actions)

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

    def single_batch_update(self, states, actions, rewards, next_states, next_actions):
        """
        Performs a gradient descent step on the model.

        Args:
            states (np.array): The representation for the state.
            actions (np.array): The action taken.
            rewards (np.array): The reward.
            next_states (np.array): The representation for the next state.
            next_actions (np.array): The next actions.
        """

        # Resetting the networks.
        self.reset()

        targets = self.targets(rewards, next_states, next_actions)

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

    def reset(self):
        """Resets the model."""

        self.model.reset()
        if self.use_double_learning:
            self.fixed.reset()

        # self.commit()

    def save(self, directory):
        """Saves the model weights."""

        torch.save(self.model.cpu().state_dict(), '{}/state_dict.pth'.format(directory))
        self.model = self.model.to(self.device)

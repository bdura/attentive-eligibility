import torch
from torch import nn

from control.utils import BaseAgent

import copy


class TabularAgent(BaseAgent):
    """
    A simple agent that uses tabular methods to compute the optimal policy.

    Todo:
        Implement it.
    """
    pass


class DQNAgent(BaseAgent):
    """
    A general class for value approximation by a neural network
    """

    def __init__(self, model, optimiser):
        """
        Initialises the object.

        Args:
            model (nn.Module): A Pytorch module.
            optimiser (torch.optim.Optimizer): An optimizer.
        """

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.fixed = copy.deepcopy(self.model).eval()

        self.criterion = nn.MSELoss()
        self.optimiser = optimiser

    def commit(self):
        self.fixed.load_state_dict(copy.deepcopy(self.model.state_dict()))

    def state(self, state):
        """
        Returns the tensorised version of the state representation.

        Args:
            state (np.array): The representation of the state.

        Returns:
            state (torch.Tensor): The tensorised version of the state representation.
        """

        state = torch.tensor(state, dtype=torch.float).to(self.device)

        return state

    def eval(self):
        self.model.eval()

    def train(self):
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

            state = self.state(state)

            # Add a batch dimension
            state = state.unsqueeze(0)

            actions = self.fixed(state)

            # Remove the batch dimension
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

        state = self.state(state)

        # Add a batch dimension
        state = state.unsqueeze(0)

        actions = self.model(state)

        q = actions.squeeze()[action]

        loss = self.criterion(q, torch.tensor(target).to(self.device))
        loss.backward(retain_graph=True)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.5)

        self.optimiser.step()

    def reset(self):
        """Resets the model"""

        self.model.reset()
        self.fixed.reset()

        # self.commit()

    def save(self, directory):

        torch.save(self.model.cpu().state_dict(), '{}/model_weights.pth'.format(directory))

from time import time
import numpy as np

from tqdm import tqdm

import gc
import os
import json

from collections import namedtuple

from functools import wraps


def softmax(x):
    """
    Stable implementation of the softmax operation.

    Args:
        x (np.array): An array on which to perform the softmax operation.

    Returns:
        (np.array): The result of the softmax operation.
    """

    if len(x.shape) == 1:
        axis = 0
    else:
        axis = 1

    z = x - x.max(axis=axis, keepdims=True)

    return np.exp(z)/np.exp(z).sum(axis=axis, keepdims=True)


def description(desc):
    """
    Decorator function generator in order to print a description with ``self.print``.

    Args:
        desc (str): The text description to print.

    Returns:
        A decorator function.
    """

    def decorator(func):

        @wraps(func)
        def f(*args, **kwargs):

            self = args[0]
            self.print('>> ' + desc + '... ', end='')
            t = time()

            res = func(*args, **kwargs)

            self.print('Done.   (' + str(round(time() - t, 2)) + 's)')

            return res

        return f

    return decorator


def memory(func):

    @wraps(func)
    def f(*args, **kwargs):
        res = func(*args, **kwargs)
        gc.collect()
        return res

    return f


def delete(func):

    @wraps(func)
    def f(*args, **kwargs):
        self = args[0]

        res = func(*args, **kwargs)

        del self
        gc.collect()

        return res

    return f


class BaseEnvironment:

    def __init__(self, verbose):
        self.verbose = verbose

    def print(self, text, end='\n'):
        if self.verbose:
            print(text, end=end)

    def tqdm(self, iterator, ascii=True, ncols=100, *args, **kwargs):
        if self.verbose:
            return tqdm(iterator, ascii=ascii, ncols=ncols, *args, **kwargs)
        return iterator


class BaseAgent(object):

    def __init__(self, temperature, n_actions):
        self.temperature = temperature
        self.n_actions = n_actions

    def q(self, state):
        """
        Defines a method that returns the approximation for the action-value function.

        Args:
            state (np.array): An array defining the state.

        Returns:
            actions (np.array): The value of each action.
        """

        pass

    def update(self, state, action, target):
        """
        Performs an update of the value function.

        Args:
            state (np.array): An array defining the state.
            action (int): The action taken.
            target (float): The TD-target for the state-action pair.
        """
        pass

    def eval(self):
        """Puts the agent in evaluation mode"""
        pass

    def train(self):
        """Puts the agent in training mode"""
        pass

    def reset(self):
        """Resets the agent"""
        pass

    def greedy(self, q):

        best_as = np.arange(self.n_actions)[q == q.max()]

        return best_as

    def epsilon_greedy(self, state, epsilon=.1):

        assert 0 <= epsilon <= 1

        best_as = self.greedy(state)

        p = np.ones(self.n_actions) * epsilon / self.n_actions
        p[best_as] += (1 - epsilon) / len(best_as)

        return p

    def boltzmann(self, q):

        p = softmax(q / self.temperature)

        return p

    def sample_action(self, p):

        action = np.random.choice(self.n_actions, p=p)
        return action


def save_json(obj, directory, name):
    path = os.path.join(directory, name)

    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


Transition = namedtuple(
    'Transition',
    ('state', 'action', 'reward', 'next_states')
)


class Episode(object):

    def __init__(self):

        self.transitions = []

    def push(self, transition):

        self.transitions.append(transition)

    def __len__(self):
        return len(self.transitions)

    def output(self, length=None):

        if length is None:
            length = len(self) + 1

        states = np.array([t[0] for t in self.transitions[:length]])
        actions = np.array([t[1] for t in self.transitions[:length]])
        rewards = np.array([t[2] for t in self.transitions[:length]])
        next_states = np.array([t[3] for t in self.transitions[:length]])

        return states, actions, rewards, next_states


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, episode):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = episode
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return np.random.choice(self.memory, size=batch_size)

    def __len__(self):
        return len(self.memory)

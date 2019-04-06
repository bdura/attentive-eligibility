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

    z = x - x.max()

    return np.exp(z)/np.exp(z).sum()


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


def save_json(obj, directory, name):
    path = os.path.join(directory, name)

    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


Transition = namedtuple(
    'Transition',
    ('state', 'action', 'reward', 'next_state', 'next_action')
)


class Episode(object):

    def __init__(self):

        self.transitions = []

    def push(self, state, action, reward, next_state, next_action):

        self.transitions.append(Transition(state, action, reward, next_state, next_action))


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

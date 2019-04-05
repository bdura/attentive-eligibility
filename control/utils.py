from time import time
import numpy as np

from tqdm import tqdm

import gc

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

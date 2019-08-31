import random
from collections import namedtuple
import numpy as np
import torch

Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward')
)


class ReplayMemory(object):
    """
    Basic replay memory, keeping transitions.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.states = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.states.append(None)
        self.memory[self.position] = Transition(*args)
        self.states[self.position] = self.memory[self.position].state.squeeze(0)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def sample_states(self, num_samples, similarity_vec):
        index = sorted(np.argsort(similarity_vec)[-num_samples:].numpy())
        state_batch = torch.stack(tensors=[self.states[i] for i in index])
        return state_batch, index

    def __len__(self):
        return len(self.memory)


class Episode(object):
    """
    Class representing a full episode/trajectory.
    """

    def __init__(self):
        self.transitions = []

    def push(self, *args):
        self.transitions.append(Transition(*args))

    def __len__(self):
        return len(self.transitions)

    def get_length(self, length):

        n = len(self.transitions)

        if n >= length:
            # If there are enough transitions, just return the given length
            return self.transitions[:length]
        else:
            # If the length is greater than the number of transitions, append dummy transitions
            transitions = self.transitions + [Transition(None, 0, None, 0) for _ in range(length - n)]
            return transitions


class EpisodeReplayMemory(object):
    """
    Replay memory that stores episodes rather than transitions
    """

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
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

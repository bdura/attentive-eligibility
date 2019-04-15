import numpy as np
import os

import json

import torch

from control.utils import softmax, tiling, BaseEnvironment
from control.utils import Episode, ReplayMemory, Transition

import time


class Environment(BaseEnvironment):

    def __init__(self, environment, agent, tiling=False, seed=None, verbose=False,
                 max_steps=1000, slack=None, capacity=10000, n_tilings=1, n_bins=10, min_ram=0, max_ram=256):

        super(Environment, self).__init__(verbose=verbose)

        np.random.seed(seed)

        self.environment = environment
        self.agent = agent

        self.state = None
        self.action = None

        self.max_steps = max_steps

        self.replay_memory = ReplayMemory(capacity=capacity)

        self.slack = slack

        self.tiling = tiling

        self.n_tiling = n_tilings
        self.n_bins = n_bins

        self.min_ram = min_ram
        self.max_ram = max_ram

    def get_config(self):

        config = {
            'verbose': self.verbose,
            'max_steps': self.max_steps
        }

        return config

    def notify(self, text):
        if self.slack is not None:
            self.slack.send_message(text)

    def greedy(self, state):
        """
        Looks for all greedy actions in order to randomise ties.

        Args:
            state (int): The current state.

        Returns:
            best_as (np.array): Array (of possibly one element only) containing the greedy actions to perform.
        """

        q = self.agent.q(state)

        best_as = np.arange(self.agent.n_actions)[q == q.max()]

        return best_as

    def epsilon_greedy(self, state, epsilon=.1):
        """
        Returns the epsilon-greedy weighting of each actions.

        Args:
            state (int): The current state. Needed because we sample
            epsilon (float): The epsilon parameter.

        Returns:
            p (np.array): The epsilon-greedy weighting of the available actions.
        """

        assert 0 <= epsilon <= 1

        best_as = self.greedy(state)

        p = np.ones(self.agent.n_actions) * epsilon / self.agent.n_actions
        p[best_as] += (1 - epsilon) / len(best_as)

        return p

    def boltzmann(self, state, return_q=False):
        """
        Returns the softmax-weighting of the available actions.

        Args:
            state: The current state.
            return_q (bool): Whether to return the vector of qs.

        Returns:
            p (np.array): The Boltzmann (softmax) weighting of the available actions.

        """

        q = self.agent.q(state)

        p = softmax(q / self.agent.temperature)

        if return_q:
            return p, q

        return p

    def sample_action(self, p):
        """
        Samples an action according to weighting p.

        Args:
            p (np.array): Probability weighting (sums to one) of each actions.

        Returns:
            action (int): The next action to take
        """

        action = np.random.choice(self.agent.n_actions, p=p)

        return action

    def step(self, action):

        s, r, d, i = self.environment.step(action)

        s = self.state_representation(s)

        try:
            d = d or i['ale.lives'] < 5
        except KeyError:
            pass

        return s, r, d, i

    def evaluate(self):
        """
        Performs a single evaluation/greedy step (no training).

        Returns:
            d (bool): Whether we've reached the end of the episode.
        """

        s, r, d, i = self.step(self.action)

        # If there are ties, we might want to choose between actions at random
        a = np.random.choice(self.greedy(s))

        # We store the new state and action
        self.state, self.action = s, a

        return d, r

    def explore(self):
        """
        Performs a single exploration step and stores the results in the buffer.

        Returns:
            d (bool): Whether we've reached the end of the episode.
        """

        s, r, d, i = self.step(self.action)

        # If there are ties, we might want to choose between actions at random
        p, q = self.boltzmann(s, return_q=True)
        a = self.sample_action(p)

        transition = Transition(self.state, self.action, r, s)

        # We store the new state and action
        self.state, self.action = s, a

        return d, r, transition

    def reset(self):
        self.agent.reset()
        self.state = self.state_representation(self.environment.reset())
        self.action = 1

    def evaluation_episode(self, render=False):
        """
        Runs a full evaluation episode.

        Args:
            render (bool): Whether to display the render of the episode.

        Returns:
            full_return (float): The full return obtained during the experiment.
            counter (int): The number of timesteps.
        """

        self.reset()

        if render:
            self.environment.render()
            time.sleep(0.01)

        step = self.evaluate
        self.agent.eval()

        done = False
        full_return = 0.

        counter = 0
        while not done and counter < self.max_steps:
            done, reward = step()
            full_return = self.agent.gamma * full_return + reward
            counter += 1

            if render:
                self.environment.render()
                time.sleep(0.01)

        if render:
            self.environment.close()

        return full_return, counter

    def exploration_episode(self, render=False):
        """
        Runs a full exploration episode.

        Args:
            render (bool): Whether to display the render of the episode.

        Returns:
            full_return (float): The full return obtained during the experiment.
            counter (int): The number of timesteps.
        """

        episode = Episode()

        self.reset()

        if render:
            self.environment.render()
            time.sleep(0.01)

        done = False
        full_return = 0.

        counter = 0
        while not done and counter < self.max_steps:
            done, reward, transition = self.explore()
            episode.push(transition)

            full_return = self.agent.gamma * full_return + reward
            counter += 1

            if render:
                self.environment.render()
                time.sleep(0.01)

        self.replay_memory.push(episode)

        if render:
            self.environment.close()

        return full_return, counter

    def exploration_segment(self, episodes=100):
        """
        Runs a full segment, which consists of ten training episodes followed by
        one evaluation episode (following the greedy policy obtained so far).

        Args:
            episodes (int): The number of training episodes to run (and average).

        Returns:
            (float): The return obtained after the evaluation episode.
        """

        # self.agent.commit()

        training_return = np.mean([self.exploration_episode()[0] for _ in range(episodes)])
        testing_return = self.evaluation_episode()[0]

        return training_return, testing_return

    def batch(self, batch_size=20):
        """
        Performs a gradient descent on a batch of episodes.

        Args:
            batch_size (int): The number of episodes to train on.
        """

        assert len(self.replay_memory) > 0

        buffer = self.replay_memory

        episodes = buffer.sample(batch_size)

        length = np.min([len(episode) for episode in buffer.memory])

        states = []
        actions = []
        rewards = []
        next_states = []

        for episode in episodes:
            s, a, r, n = episode.output(length=length)

            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(n)

        states = np.stack(states).swapaxes(0, 1)
        actions = np.stack(actions).swapaxes(0, 1)
        rewards = np.stack(rewards).swapaxes(0, 1)
        next_states = np.stack(next_states).swapaxes(0, 1)

        self.agent.batch_update(states, actions, rewards, next_states)

    def train(self, segments=100, episodes=100):
        """
        Trains the agent. Alternates exploration and batch gradient descent.

        Args:
            segments (int): The number of segments of exploration to perform.
            episodes (int): The number of episodes for each segment.

        Returns:
            returns (np.array): The mean return for each segment.
        """

        iterator = self.tqdm(range(segments), ascii=True, ncols=100)

        returns = []

        with iterator as it:
            for _ in it:

                self.agent.commit()
                returns.append(self.exploration_segment(episodes))

                for _ in range(2 * len(self.replay_memory) // 100):
                    self.batch(100)

        return np.array(returns)

    def run(self, epochs=10, segments=10, episodes=50, wall_time=10, save_directory=None):

        self.notify('Beginning training')

        t0 = time.time()

        for i in range(epochs):

            returns = self.train(segments, episodes).mean(axis=0)[0]

            self.notify('>> Training return : {:.2f}'.format(returns))
            self.print('>> Training return : {:.2f}'.format(returns))

            mean_return, steps = np.array([self.evaluation_episode() for _ in range(200)]).mean(axis=0)

            self.notify('>> Evaluation return : {:.2f}, steps : {:.2f}'.format(mean_return, steps))
            self.print('>> Evaluation return : {:.2f}, steps : {:.2f}'.format(mean_return, steps))

            if save_directory is not None:
                self.save(save_directory)

            now = (time.time() - t0) / 3600

            if now / (i + 1) * (i + 2) > wall_time * .95:
                break

        self.notify('Training ended.')

    def save(self, directory):

        os.makedirs(directory, exist_ok=True)

        config = dict()

        config['types'] = {
            'model': self.agent.model.name,
            'agent': self.agent.name
        }

        config['agent'] = self.agent.get_config()
        config['model'] = self.agent.model.get_config()
        config['environment'] = self.get_config()

        with open(os.path.join(directory, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

        self.agent.save(directory)

        torch.save(self.replay_memory, os.path.join(directory, 'buffer.pth'))

    def state_representation(self, state):

        if self.tiling:

            tilings = np.zeros((len(state), self.n_tilings, self.n_bins))

            for i in range(len(state)):
                tilings[i, :, :] = tiling(value=state[i], min_value=self.min_ram, max_value=self.max_ram,
                                          n_tilings=self.n_tilings, n_bins=self.n_bins)

            return np.ravel(tilings, order='F')

        else:
            return state


class SimplifiedEnvironment(Environment):
    """A simplified environment with only 3 actions."""

    def step(self, action):
        if action > 0:
            action += 1

        s, r, d, i = self.environment.step(action)

        s = self.state_representation(s)

        try:
            d = d or i['ale.lives'] < 5
        except KeyError:
            pass

        return s, r, d, i

    def reset(self):
        super().reset()

        s, r, d, i = self.environment.step(1)

        self.state = self.state_representation(s)


class OverSimplifiedEnvironment(SimplifiedEnvironment):
    """An over simplified environment with only 3 actions and 15 features."""

    features = np.array([
        57, 70, 71, 72, 74, 75, 86, 90, 94, 95, 99, 101, 103, 105, 119
    ])

    def reset(self):
        super().reset()
        self.state = self.state_representation(self.state[self.features])

    def step(self, action):

        s, r, d, i = super().step(action)

        s = self.state_representation(s[self.features])

        return s, r, d, i

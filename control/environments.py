import numpy as np
import os

import json

import torch
import gym

from control.utils import softmax, tiling, one_hot_encoding, BaseEnvironment
from control.utils import Episode, ReplayMemory, Transition

import time


class Environment(BaseEnvironment):

    def __init__(self, environment, agent, seed=None, verbose=False, max_steps=1000, slack=None, capacity=10000,
                 representation_method='observation', n_tilings=1, n_bins=10):
        """ Initializes the environment. """

        super(Environment, self).__init__(verbose=verbose)

        np.random.seed(seed)

        assert type(environment.action_space) is gym.spaces.discrete.Discrete
        assert \
            type(environment.observation_space) is gym.spaces.discrete.Discrete or \
            type(environment.observation_space) is gym.spaces.box.Box

        self.environment = environment
        self.agent = agent

        self.state = None
        self.action = None

        self.max_steps = max_steps

        self.replay_memory = ReplayMemory(capacity=capacity)

        self.slack = slack

        self.representation_method = representation_method

        self.n_tilings = n_tilings
        self.n_bins = n_bins

        self.n_actions = environment.action_space.n

        if type(environment.observation_space) is gym.spaces.discrete.Discrete:
            self.obs_dim = 1
            self.min_obs = 0
            self.max_obs = environment.observation_space.n - 1

        elif type(environment.observation_space) is gym.spaces.box.Box:
            self.obs_dim = environment.observation_space.shape[0]
            self.min_obs = min(environment.observation_space.low)
            self.max_obs = max(environment.observation_space.high) + 1

    def get_config(self):
        """
        Compute the configuration of the environment as a dictionary.

        Returns:
            config: dict, configuration of the environment.
        """

        config = {
            'verbose': self.verbose,
            'max_steps': self.max_steps,
            'representation_method': self.representation_method,
            'n_tilings': self.n_tilings,
            'n_bins': self.n_bins
        }

        return config

    def notify(self, text):
        """ Slack notification function. """

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

        best_as = np.arange(self.n_actions)[q == q.max()]

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

        p = np.ones(self.n_actions) * epsilon / self.n_actions
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

        action = np.random.choice(self.n_actions, p=p)

        return action

    def step(self, action):
        """
        Performs a step of self.environment.

        Args:
            action: int, action to perform during the step.

        Returns:
            s: np.array, state following the step.
            r: float, reward observed.
            d: bool, whether the episode is done.
            i: dict, info of the environment after the step.
        """

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
        """ Reset the agent, the environment. """

        self.agent.reset()
        self.state = self.state_representation(self.environment.reset())

        if self.agent is not None:
            self.action = self.sample_action(self.boltzmann(self.state))

    def evaluation_episode(self, render=False, return_observations=False):
        """
        Runs a full evaluation episode.

        Args:
            render (bool): Whether to display the render of the episode.
            return_observations (bool): Whether to return the (full) observation of the states.

        Returns:
            full_return (float): The full return obtained during the experiment.
            counter (int): The number of timesteps.
            observations (list): full observations of the states.
        """

        observations = []

        self.reset()

        if return_observations:
            observations.append(self.environment.unwrapped._get_obs())

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
            #full_return = self.agent.gamma * full_return + reward
            full_return += reward
            counter += 1

            if render:
                self.environment.render()
                time.sleep(0.01)

            if return_observations:
                observations.append(self.environment.unwrapped._get_obs())

        if render:
            self.environment.close()

        if not return_observations:
            return full_return, counter

        else:
            return full_return, counter, observations

    def exploration_episode(self, render=False, return_observations=False):
        """
        Runs a full exploration episode.

        Args:
            render (bool): Whether to display the render of the episode.
            return_observations (bool): Whether to return the (full) observation of the states.

        Returns:
            full_return (float): The full return obtained during the experiment.
            counter (int): The number of timesteps.
            observations (list): full observations of the states.
        """

        observations = []
        episode = Episode()

        self.reset()

        if return_observations:
            observations.append(self.environment.unwrapped._get_obs())

        if render:
            self.environment.render()
            time.sleep(0.01)

        done = False
        full_return = 0.

        counter = 0
        while not done and counter < self.max_steps:
            done, reward, transition = self.explore()

            episode.push(transition)

            #full_return = self.agent.gamma * full_return + reward
            full_return += reward
            counter += 1

            if render:
                self.environment.render()
                time.sleep(0.01)

            if return_observations:
                observations.append(self.environment.unwrapped._get_obs())

        self.replay_memory.push(episode)

        if render:
            self.environment.close()

        if not return_observations:
            return full_return, counter

        else:
            return full_return, counter, observations

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

    def train(self, segments=100, episodes=100, batch_size=100):
        """
        Trains the agent. Alternates exploration and batch gradient descent.

        Args:
            segments (int): The number of segments of exploration to perform.
            episodes (int): The number of episodes for each segment.
            batch_size (int): The number of episodes per batch.

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
                    self.batch(batch_size)

        return np.array(returns)

    def run(self, epochs=10, segments=10, episodes=50, wall_time=10, num_evaluation=200, batch_size=100,
            save_directory=None):
        """
        Run a full training of the agent in the environment.

        Args:
            epochs: int, number of epochs.
            segments: int, number of segments.
            episodes: int, number of episodes.
            wall_time: float, time limit of the run.
            save_directory: str, directory where to save the environment.
        """

        self.notify('Beginning training')

        t0 = time.time()

        for i in range(epochs):

            returns = self.train(segments, episodes, batch_size).mean(axis=0)[0]

            self.notify('>> Training return : {:.2f}'.format(returns))
            self.print('>> Training return : {:.2f}'.format(returns))

            mean_return, steps = np.array([self.evaluation_episode() for _ in range(num_evaluation)]).mean(axis=0)

            self.notify('>> Evaluation return : {:.2f}, steps : {:.2f}'.format(mean_return, steps))
            self.print('>> Evaluation return : {:.2f}, steps : {:.2f}'.format(mean_return, steps))

            if save_directory is not None:
                self.save(save_directory)

            now = (time.time() - t0) / 3600

            if now / (i + 1) * (i + 2) > wall_time * .95:
                break

        self.notify('Training ended.')

    # TODO: debug
    def save(self, directory):
        """
        Save the environment.

        Args:
            directory: str, directory where to save.
        """

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
        """
        Compute the representation of a state; can be the full observation or a tiling.

        Args:
            state: np.array, full observation of the state.

        Returns:
            np.array, representation of the state as a vector.
        """

        if self.representation_method == 'observation':
            return state

        elif self.representation_method == 'tiling':
            assert type(self.environment.observation_space) is gym.spaces.box.Box

            tilings = np.zeros((len(state), self.n_tilings, self.n_bins))

            for i in range(len(state)):
                tilings[i, :, :] = tiling(value=state[i], min_value=self.min_obs, max_value=self.max_obs,
                                          n_tilings=self.n_tilings, n_bins=self.n_bins)

            return np.ravel(tilings, order='F')

        elif self.representation_method == 'one_hot_encoding':
            representation = []

            if type(self.environment.observation_space) is gym.spaces.discrete.Discrete:
                representation = one_hot_encoding(value=state,
                                                  min_value=self.min_obs,
                                                  max_value=self.max_obs)

            elif type(self.environment.observation_space) is gym.spaces.box.Box:
                for i in range(self.obs_dim):
                    representation.extend(list(one_hot_encoding(value=state[i],
                                                                min_value=self.min_obs,
                                                                max_value=self.max_obs)))

            return np.asarray(representation)

        else:
            raise Exception("No such method (must be vector, tiling or one-hot-encoding for the base environment).")

    def get_input_dimension(self):
        """
        Compute the dimension of the state representation.

        Returns:
            int, dimension of the input.
        """

        return self.state_representation(self.environment.reset()).shape[0]

    def bytes_evolution_range(self, n_episodes_exploration=100, n_episodes_evaluation=100):
        """
        Compute the range of evolution of each of the 128 bytes.

        Args:
            n_episodes_exploration: int, number of exploration episodes to test.
            n_episodes_evaluation: int, number of evaluation episodes to test.

        Returns:
            min_features: np.array, vector with the minimal value of each bytes
            max_features: np.array, vector with the maximal value of each bytes
        """

        min_features = int(self.max_obs) * np.ones(self.obs_dim, dtype=np.int_)
        max_features = int(self.min_obs) * np.ones(self.obs_dim, dtype=np.int_)

        for _ in range(n_episodes_exploration):
            full_return, counter, observations = self.exploration_episode(return_observations=True)

            observations = np.asarray(observations)
            min_obs, max_obs = np.amin(observations, axis=0), np.amax(observations, axis=0)

            for i in range(self.obs_dim):
                if min_obs[i] < min_features[i]:
                    min_features[i] = int(min_obs[i])
                if max_obs[i] > max_features[i]:
                    max_features[i] = int(max_obs[i])

        for _ in range(n_episodes_evaluation):
            full_return, counter, observations = self.evaluation_episode(return_observations=True)

            observations = np.asarray(observations)
            min_obs, max_obs = np.amin(observations, axis=0), np.amax(observations, axis=0)

            for i in range(self.obs_dim):
                if min_obs[i] < min_features[i]:
                    min_features[i] = int(min_obs[i])
                if max_obs[i] > max_features[i]:
                    max_features[i] = int(max_obs[i])

        return min_features, max_features


class SimplifiedEnvironment(Environment):
    """ A simplified environment with only 3 actions. """

    def __init__(self, environment, agent, seed=None, verbose=False, max_steps=1000, slack=None, capacity=10000,
                 representation_method='vector', n_tilings=1, n_bins=10):
        """ Initializes the simplified environment. """

        assert environment.unwrapped.spec.id == 'Breakout-ram-v0'

        super(SimplifiedEnvironment, self).__init__(environment, agent, seed, verbose, max_steps, slack, capacity,
                                                    representation_method, n_tilings, n_bins)

        self.n_actions = 3

    def step(self, action):
        """ Take a step in the environment; deal with the initial action and the final one (loss of first life). """

        if action == -1:
            action = 1
        elif action > 0:
            action += 1

        s, r, d, i = self.environment.step(action)

        s = self.state_representation(s)

        try:
            d = d or i['ale.lives'] < 5
        except KeyError:
            pass

        return s, r, d, i

    def reset(self):
        """ Reset the agent, the environment, take the first step (fire), and chose the next action. """

        self.agent.reset()
        self.environment.reset()

        state, _, _, _ = self.environment.step(1)
        self.state = self.state_representation(state)

        if self.agent is not None:
            self.action = self.sample_action(self.boltzmann(self.state))


class OverSimplifiedEnvironment(SimplifiedEnvironment):
    """An over simplified environment with only 3 actions and about 15 features."""

    def __init__(self, environment, agent, seed=None, verbose=False, max_steps=1000, slack=None, capacity=10000,
                 representation_method='vector', n_tilings=1, n_bins=10, mixed_threshold=15, max_tiling=8):
        """ Initializes the over simplified environment. """

        super(OverSimplifiedEnvironment, self).__init__(environment, agent, seed, verbose, max_steps, slack, capacity,
                                                        representation_method, n_tilings, n_bins)

        self.features = np.array([
            0, 6, 12, 18, 19, 24, 30, 31, 49, 57, 70, 71, 72, 74, 75, 77, 84, 86, 90, 91, 94, 95, 96, 99, 100, 101, 102,
            103, 104, 105, 106, 107, 109, 119, 121, 122
        ])

        self.min_range = np.array([
            15, 63, 63, 63, 252, 15, 0, 0, 225, 4, 0, 1, 55, 2, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 150
        ])
        self.max_range = np.array([
            63, 255, 255, 255, 255, 255, 192, 192, 240, 5, 184, 6, 191, 8, 13, 3, 15, 25, 255, 3, 226, 135, 28, 230,
            128, 208, 128, 255, 128, 255, 128, 131, 2, 255, 150, 246
        ])

        self.mixed_threshold = mixed_threshold
        self.max_tiling = max_tiling

        self.tilings = self.get_tilings()

    def get_tilings(self):
        """
        Compute the number of tilings for each feature as an affine function evolving between 1 and self.max_tiling.

        Returns:
            tilings: list, number of tiling for each feature.
        """

        tilings = []

        for i in range(len(self.features)):
            d = self.max_range[i] - self.min_range[i]

            tilings.append(
                int(1 + d * self.max_tiling // (self.max_obs - self.min_obs))
            )

        return tilings

    def state_representation(self, state):
        """
        Compute the representation of a state; can be the full observation, a tiling, a 1 hot encoding, or a mixed
        version of the last two.

        Args:
            state: np.array, full observation of the state.

        Returns:
            np.array, representation of the state as a vector.
        """

        state = state[self.features]

        if self.representation_method == 'observation':
            return state

        elif self.representation_method == 'tiling':
            representation = []

            for i in range(len(self.features)):
                unraveled_tiling = tiling(value=state[i],
                                          min_value=self.min_range[i],
                                          max_value=self.max_range[i],
                                          n_tilings=self.tilings[i],
                                          n_bins=self.mixed_threshold)

                representation.extend(list(np.ravel(unraveled_tiling, order='F')))

            return np.asarray(representation)

        elif self.representation_method == 'one_hot_encoding':
            representation = []

            for i in range(len(self.features)):
                representation.extend(list(one_hot_encoding(value=state[i],
                                                            min_value=self.min_range[i],
                                                            max_value=self.max_range[i])))

            return np.asarray(representation)

        elif self.representation_method == 'mixed':
            representation = []

            for i in range(len(self.features)):

                if self.max_range[i] - self.min_range[i] >= self.mixed_threshold:

                    unraveled_tiling = tiling(value=state[i],
                                              min_value=self.min_range[i],
                                              max_value=self.max_range[i],
                                              n_tilings=self.tilings[i],
                                              n_bins=self.mixed_threshold)

                    representation.extend(list(np.ravel(unraveled_tiling, order='F')))

                else:
                    representation.extend(one_hot_encoding(value=state[i],
                                                           min_value=self.min_range[i],
                                                           max_value=self.max_range[i]))

            return np.asarray(representation)

        else:
            raise Exception("No such method (must be vector, tiling, one_hot_encoding or mixed, for the " +
                            "OverSimplified environment)")

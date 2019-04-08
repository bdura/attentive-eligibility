import numpy as np
import os

from control.utils import softmax, BaseEnvironment, save_json
from control.utils import Episode, ReplayMemory


class Environment(BaseEnvironment):

    def __init__(self, environment, agent, temperature=1, gamma=1, alpha=.1,
                 decay=.9, seed=None, verbose=False, max_steps=1000):

        super(Environment, self).__init__(verbose=verbose)

        np.random.seed(seed)

        self.environment = environment
        self.agent = agent

        # There are 4 possible actions
        self.n_actions = 4

        self.temperature = temperature

        self.gamma = gamma
        self.alpha = alpha
        self.decay = decay

        self.state = None
        self.action = None

        self.max_steps = max_steps

        self.replay_memory = ReplayMemory(capacity=1000)

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

    def boltzmann(self, state):
        """
        Returns the softmax-weighting of the available actions.

        Args:
            state: The current state

        Returns:
            p (np.array): The Boltzmann (softmax) weighting of the available actions.

        """

        q = self.agent.q(state)

        p = softmax(q / self.temperature)

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

    def backup(self):
        pass

    def evaluate(self):
        """
        Performs a single evaluation/greedy step (no training).

        Returns:
            d (bool): Whether we've reached the end of the episode.
        """

        s, r, d, i = self.environment.step(self.action)

        # If there are ties, we might want to choose between actions at random
        a = np.random.choice(self.greedy(s))

        # We store the new state and action
        self.state, self.action = s, a

        return d, r

    def reset(self):
        self.agent.reset()
        self.state = self.environment.reset()
        self.action = np.random.choice(self.greedy(self.state))

    def episode(self, evaluation=False):
        """
        Runs a full episode.

        Args:
            evaluation (bool): Whether the agent is in evaluation or training mode.

        Returns:
            full_return (float): The full return obtained during the experiment.
        """

        self.reset()

        if evaluation:
            step = self.evaluate
            self.agent.eval()
        else:
            step = self.backup
            self.agent.train()

        done = False
        full_return = 0.

        if evaluation:
            self.action = np.random.choice(self.greedy(self.state))
        else:
            p = self.boltzmann(self.state)
            self.action = self.sample_action(p)

        counter = 0
        while not done and counter < self.max_steps:
            done, reward = step()
            full_return = self.gamma * full_return + reward
            counter += 1

        return full_return, counter

    def segment(self, episodes=10):
        """
        Runs a full segment, which consists of ten training episodes followed by
        one evaluation episode (following the greedy policy obtained so far).

        Args:
            episodes (int): The number of training episodes to run (and average).

        Returns:
            (float): The return obtained after the evaluation episode.
        """

        self.agent.commit()

        training_return = np.mean([self.episode()[0] for _ in range(episodes)])
        testing_return = self.episode(evaluation=True)[0]

        return training_return, testing_return

    def run(self, segments=100):
        """
        Perform a full run, which consists of 100 independent segments.

        Args:
            segments (int): The number of segments to run.

        Returns:
            returns (np.array): An array containing the independent returns obtained
                by each segment.
        """

        iterator = self.tqdm(range(segments), ascii=True, ncols=100)

        returns = np.array([self.segment() for _ in iterator])

        return returns

    def save(self, directory):

        os.makedirs(directory, exist_ok=True)

        config = {
            'temperature': self.temperature,
            'gamma': self.gamma,
            'alpha': self.alpha,
            'decay': self.decay,
            'max_steps': self.max_steps,
        }

        save_json(config, directory, 'env_config.json')

        self.agent.save(directory)


class Sarsa(Environment):

    def backup(self):
        """
        Performs a single Sarsa backup (state-action -> reward -> state-action).

        Returns:
            d (bool): Whether we've reached the end of the episode.
        """

        s, r, d, i = self.environment.step(self.action)

        p = self.epsilon_greedy(s, 1)
        a = self.sample_action(p)

        target = r + self.gamma * self.agent.q(s)[a]

        # Regular Sarsa is an on-policy method
        self.agent.update(state=self.state, action=self.action, target=target)

        # We store the new state and action
        self.state, self.action = s, a

        return d, r


class ExpectedSarsa(Environment):

    def backup(self):
        """
        Performs a single Sarsa backup (state-action -> reward -> state-action).

        Returns:
            d (bool): Whether we've reached the end of the episode.
        """

        s, r, d, i = self.environment.step(self.action)

        p = self.boltzmann(s)
        # p = self.epsilon_greedy(s, 1)
        a = self.sample_action(p)

        target = r + self.gamma * p @ self.agent.q(s)

        # Regular Sarsa is an on-policy method
        self.agent.update(state=self.state, action=self.action, target=target)

        # We store the new state and action
        self.state, self.action = s, a

        return d, r


class QLearning(Environment):

    def backup(self):
        """
        Performs a single Sarsa backup (state-action -> reward -> state-action).

        Returns:
            d (bool): Whether we've reached the end of the episode.
        """

        s, r, d, i = self.environment.step(self.action)

        p = self.boltzmann(s)
        a = self.sample_action(p)

        target = r + self.gamma * self.agent.q(s).max()

        # Regular Sarsa is an on-policy method
        self.agent.update(state=self.state, action=self.action, target=target)

        # We store the new state and action
        self.state, self.action = s, a

        return d, r

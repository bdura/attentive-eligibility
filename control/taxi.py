import gym
import numpy as np

from control.utils import softmax, BaseAgent


class Agent(BaseAgent):

    def __init__(self, environment, temperature=1, gamma=1, alpha=1, seed=None, verbose=False):

        super(Agent, self).__init__(verbose=verbose)

        np.random.seed(seed)

        self.environment = environment

        # Assert that the environment has discrete state-action space
        assert isinstance(environment.action_space, gym.spaces.Discrete)
        assert isinstance(environment.observation_space, gym.spaces.Discrete)

        # Building the action-value function
        self.n = int(environment.observation_space.n)
        self.m = int(environment.action_space.n)
        self.q = np.zeros((self.n, self.m))

        # Parameter of the agent
        self.temperature = temperature
        self.gamma = gamma
        self.alpha = alpha

        self.state = None
        self.action = None

    def initialise(self):
        """
        Initialises the agent by setting the action-value function to 0.
        """
        self.q = np.zeros((self.n, self.m))

    def greedy(self, state):
        """
        Looks for all greedy actions in order to randomise ties.

        Args:
            state (int): The current state.

        Returns:
            best_as (np.array): Array (of possibly one element only) containing the greedy actions to perform.
        """

        best_as = np.arange(self.m)[self.q[state] == self.q[state].max()]

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

        p = np.ones(self.m) * epsilon / self.m
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

        values = self.q[state]
        p = softmax(values / self.temperature)

        return p

    def sample_action(self, p):
        """
        Samples an action according to weighting p.

        Args:
            p (np.array): Probability weighting (sums to one) of each actions.

        Returns:
            action (int): The next action to take
        """

        action = np.random.choice(self.m, p=p)
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

    def episode(self, evaluation=False):
        """
        Runs a full episode.

        Args:
            evaluation (bool): Whether the agent is in evaluation or training mode.

        Returns:
            full_return (float): The full return obtained during the experiment.
        """

        if evaluation:
            step = self.evaluate
        else:
            step = self.backup

        done = False
        full_return = 0.

        self.state = self.environment.reset()
        p = self.boltzmann(self.state)

        self.action = np.random.choice(self.greedy(self.state)) if evaluation else self.sample_action(p)

        while not done:
            done, reward = step()
            full_return = self.gamma * full_return + reward

        return full_return

    def segment(self, episodes=10):
        """
        Runs a full segment, which consists of ten training episodes followed by
        one evaluation episode (following the greedy policy obtained so far).

        Args:
            episodes (int): The number of training episodes to run (and average).

        Returns:
            (float): The return obtained after the evaluation episode.
        """

        training_return = np.mean([self.episode() for _ in range(episodes)])
        testing_return = self.episode(evaluation=True)

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

        self.initialise()

        iterator = self.tqdm(range(segments), ascii=True, ncols=100)

        returns = np.array([self.segment() for _ in iterator])

        return returns


class Sarsa(Agent):

    def backup(self):
        """
        Performs a single Sarsa backup (state-action -> reward -> state-action).

        Returns:
            d (bool): Whether we've reached the end of the episode.
        """

        s, r, d, i = self.environment.step(self.action)

        p = self.boltzmann(s)
        a = self.sample_action(p)

        # Regular Sarsa is an on-policy method
        diff = r + self.gamma * self.q[s, a] - self.q[self.state, self.action]

        self.q[self.state, self.action] += self.alpha * diff

        # We store the new state and action
        self.state, self.action = s, a

        return d, r


class ExpectedSarsa(Agent):

    def backup(self):
        """
        Performs a single Expected-Sarsa step.

        Returns:
            d (bool): Whether we've reached the end of the episode.
        """

        s, r, d, i = self.environment.step(self.action)

        p = self.boltzmann(s)
        a = self.sample_action(p)

        # In Expected-Sarsa, we make a weighted sum
        diff = r + self.gamma * (p @ self.q[s]) - self.q[self.state, self.action]

        self.q[self.state, self.action] += self.alpha * diff

        # We store the new state and action
        self.state, self.action = s, a

        return d, r


class QLearning(Agent):

    def backup(self):
        """
        Performs a single Q-Learning step.

        Returns:
            d (bool): Whether we've reached the end of the episode.
            r (float): The reward obtained from this step.
        """

        s, r, d, i = self.environment.step(self.action)

        p = self.boltzmann(s)
        a = self.sample_action(p)

        # In Q-Learning, we make the optimistic choice
        diff = r + self.gamma * self.q[s].max() - self.q[self.state, self.action]

        self.q[self.state, self.action] += self.alpha * diff

        # We store the new state and action
        self.state, self.action = s, a

        return d, r

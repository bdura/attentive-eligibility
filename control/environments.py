import numpy as np
import os

from control.utils import softmax, BaseEnvironment, save_json
from control.utils import Episode, ReplayMemory, Transition


class Environment(BaseEnvironment):

    def __init__(self, environment, agent, temperature=1., gamma=1., alpha=.1,
                 decay=.9, seed=None, verbose=False, max_steps=1000, slack=None):

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

        self.replay_memory = ReplayMemory(capacity=10000)

        self.slack = slack

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

        p = softmax(q / self.temperature)

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

    def target(self, reward, next_state, next_action, probability):
        pass

    def backup(self):
        s, r, d, i = self.environment.step(self.action)

        p, q = self.boltzmann(s, return_q=True)
        # p = self.epsilon_greedy(s, 1)
        a = self.sample_action(p)

        target = self.target(r, s, a, p)

        # Regular Sarsa is an on-policy method
        self.agent.update(state=self.state, action=self.action, target=target)

        # We store the new state and action
        self.state, self.action = s, a

        return d, r

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

    def explore(self):
        """
        Performs a single exploration step and stores the results in the buffer.

        Returns:
            d (bool): Whether we've reached the end of the episode.
        """

        s, r, d, i = self.environment.step(self.action)

        # If there are ties, we might want to choose between actions at random
        p, q = self.boltzmann(s, return_q=True)
        a = self.sample_action(p)

        transition = Transition(self.state, self.action, self.target(r, s, a, p))

        # We store the new state and action
        self.state, self.action = s, a

        return d, r, transition

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

    def exploration_episode(self):

        episode = Episode()

        self.reset()

        done = False
        full_return = 0.

        p = self.boltzmann(self.state)
        self.action = self.sample_action(p)

        counter = 0
        while not done and counter < self.max_steps:

            done, reward, transition = self.explore()
            episode.push(transition)

            full_return = self.gamma * full_return + reward
            counter += 1

        self.replay_memory.push(episode)

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
        testing_return = self.episode(evaluation=True)[0]

        return training_return, testing_return

    def batch(self, batch_size=20):

        assert len(self.replay_memory) > 0

        buffer = self.replay_memory

        episodes = buffer.sample(batch_size)

        length = np.min([len(episode) for episode in buffer.memory])

        states = []
        actions = []
        targets = []

        for episode in episodes:
            s, a, t = episode.output(length=length)

            states.append(s)
            actions.append(a)
            targets.append(t)

        states = np.stack(states).swapaxes(0, 1)
        actions = np.stack(actions).swapaxes(0, 1)
        targets = np.stack(targets).swapaxes(0, 1)

        self.agent.batch_update(states, actions, targets)

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

    def train(self, segments=100, episodes=100):

        iterator = self.tqdm(range(segments), ascii=True, ncols=100)

        returns = []

        with iterator as it:
            for i in it:

                self.agent.commit()
                returns.append(self.exploration_segment(episodes))

                for _ in range((i + 1) ** 2):
                    self.batch(50)

        return np.array(returns)

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

    def target(self, reward, next_state, next_action, probability):
        target = reward + self.gamma * self.agent.q(next_state)[next_action]
        return target


class ExpectedSarsa(Environment):

    def target(self, reward, next_state, next_action, probability):
        target = reward + self.gamma * probability @ self.agent.q(next_state)
        return target


class QLearning(Environment):

    def target(self, reward, next_state, next_action, probability):
        target = reward + self.gamma * self.agent.q(next_state).max()
        return target


if __name__ == '__main__':
    import models.mlp as mlps
    import control.agents as agents

    import torch
    import gym

    model = mlps.MLP()
    optimiser = torch.optim.Adam(model.parameters(), lr=.001)

    agent = agents.DQNAgent(model, optimiser)

    environment = ExpectedSarsa(
        environment=gym.make('Breakout-ram-v0'),
        agent=agent,
        gamma=.999,
        temperature=10,
        verbose=True,
        max_steps=1000
    )

    environment.exploration_segment(1)

    environment.batch(batch_size=2)

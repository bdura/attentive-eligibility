# Attentive Eligibility

In this class project, we explore a novel mechanism for credit assignment, relying on the concept of attention.

We wish to discuss how this may extend the intuition of the "backwards view" of eligibility traces for the tabular setting.

## Control

The folder control contains the main files of this project:
- agents.py contains a class DQNAgent which defines the agent that evolves in the environment, with a PyTorch Model, the algorithm used,...
- environments.py define a class Environment which deals with the interaction between the agent and a gym environment. It contains a subclass designed to simply the environment Breakout-v0 by specifying the first action (launch the game), removing the corresponding action for the game, and limiting a party to a single life to simplify the number of encountered states
- utils.py contains various functions (Boltzmann of sparse coding for instance) and the classes ReplayMemory and Episode

## Models

The folder models contains the files defining the models of the agent, using PyTorch:
- linear.py defines a linear model (no bias and no activation) to simulate the tabular case when using 1 hot encoding
- mlp.py defines a classic Multi-Layer Perceptron model
- rnn.py defines a classic Recurrent Neural Network and the Attentive Neural Network that we implement

## Notebooks

The folder notebooks contains the notebooks we use to perform the training and the experiment:
- Train - Taxi is the generic file to launch the training of a model on the Taxi-v2 environment
- The other notebooks contain some of the actual experiments launched

## Utils

The folder util contains some functions:
- the implementation of checkpoints for the training
- the implementation of the eligibility traces optimizer
- the implementation of Slack notifications during training
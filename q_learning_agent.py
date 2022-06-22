import gym
import gym_examples
from gym import spaces
import numpy as np
import random

class QLearningAgent:

    def __init__(self, state_size, action_size, action_space):
        # hyper parameters
        self.state = None
        self.learning_rate = 0.9
        self.discount_rate = 0.8
        self.epsilon = 1.0
        self.decay_rate = 0.005

        self.state_size = state_size
        self.action_size = action_size
        self.action_space = action_space

        self.qtable = np.zeros((state_size, action_size))

    def __setstate__(self, state):
        self.state = state

    def learn(self, action, new_state, reward):
        # Q-learning algorithm
        self.qtable[self.state, action] = self.qtable[self.state, action] + self.learning_rate * (
                reward + self.discount_rate * np.max(self.qtable[new_state, :]) - self.qtable[self.state, action])

        self.state = new_state

    def decrease_epsilon(self, episode):
        self.epsilon = np.exp(-self.decay_rate * episode)

    def choose_action(self):

        # exploration-exploitation tradeoff
        if random.uniform(0, 1) < self.epsilon:
            # explore
            action = self.action_space.sample()
        else:
            # exploit
            action = np.argmax(self.qtable[self.state, :])

        return action



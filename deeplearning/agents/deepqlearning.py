# Project: Neural Network From Scratch
# Date: Mars 2019
# Author: Flavien LOISEAU

from collections import deque
import numpy as np
import random

class DeepQLearningAgent():

    def __init__(self, model, memory_size=2000, gamma=0.9, epsilon_min=0.001, epsilon_decay=0.999, learning_rate = 0.001, epochs=3):
        # Learning parameters
        self.gamma = gamma# Discount rate
        self.epsilon = 1.0 # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.memory = deque(maxlen=memory_size)
        # Model
        self.model = model

    def remember(self, state, action, reward, next_state, done):
        """Store state, action, reward, next_state, done in the memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self,state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(np.size(self.model.layers[-2].weights,1))
        else:
            return np.argmax(self.model.predict(state))

    def replay(self, batch_size=128):
        if len(self.memory) >= batch_size:
            # Choose experiences
            minibatch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not(done):
                    target += self.gamma * np.amax(self.model.predict(next_state))
                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.train([state], [target_f], self.epochs, self.learning_rate)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
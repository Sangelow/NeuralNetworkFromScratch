import gym
import gym.spaces
from deeplearning.agents.deepqlearning import DeepQLearningAgent
from deeplearning.neuralnetwork.model import NeuralNetwork
from deeplearning.neuralnetwork.layers import FCLayer, ActivationLayer
from deeplearning.neuralnetwork.activationfunctions import linear, linear_prime, leakyReLU, leakyReLU_prime
from deeplearning.neuralnetwork.lossfunctions import mse, mse_prime
import numpy as np

# Parameters
episodes = 5000

# Initialise space
env = gym.make('CartPole-v1')

model = NeuralNetwork()
model.add(FCLayer(4, 16))
model.add(ActivationLayer(leakyReLU, leakyReLU_prime))
model.add(FCLayer(16, 16))
model.add(ActivationLayer(leakyReLU, leakyReLU_prime))
model.add(FCLayer(16, 2))
model.add(ActivationLayer(linear, linear_prime))
model.set_loss(mse, mse_prime)

# Create DQL Agent
agent = DeepQLearningAgent(model)

for episode in range(episodes):
    # Reset state at beginning of each episode
    state = env.reset()
    state = np.reshape(state, [1, 4])
    for t in range(1000):
        # Render env
        # env.render()
        # Act
        action = agent.act(state)
        # Observe
        next_state, reward, done, info = env.step(action)
        next_state = next_state / np.linalg.norm(next_state, ord=2)
        next_state = np.reshape(next_state, [1, 4])
        # Remember
        agent.remember(state, action, reward, next_state, done)
        # Update state
        state = next_state

        if done:
            print(f"Episode: {episode+1}/{episodes} | Score: {t} | Epsilon: {agent.epsilon}")
            break

        # train agent
        agent.replay()
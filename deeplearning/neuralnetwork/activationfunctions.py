# Project: Neural Network From Scratch
# Date: Mars 2019
# Author: Flavien LOISEAU

import numpy as np

def linear(x):
    return x

def linear_prime(x):
    return 1

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def sigmoid(x):                                        
   return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex/ex.sum()

def softmax_prime(x):
    pass

def ReLU(x):
    return x * (x > 0)

def ReLU_prime(x):
    return 1 * (x > 0)

def leakyReLU(x):
    return np.where(x > 0, x, 0.01 * x)

def leakyReLU_prime(x):
    return np.where(x > 0, 1, 0.01)
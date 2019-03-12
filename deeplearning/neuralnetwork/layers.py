# Project: Neural Network From Scratch
# Date: Mars 2019
# Author: Flavien LOISEAU

import numpy as np

class Layer():
    """docstring for Layer"""
    def __init__(self):
        super(Layer, self).__init__()
        self.input = None
        self.output = None

    def __repr__(self):
        return f"Layer object: {type(self).__name__}."

    def forward_propagation(self, input):
        """Forward propagation throught layer."""
        raise NotImplementedError

    def backward_propagation(self, ouputError, learningRate):
        """Backward propagation throught layer.
        Note: About learngin rate: This parameter should be something 
        like an update policy, or an optimizer as they call it in Keras,
        but for the sake of simplicity we’re simply going to pass a
        learning rate and update our parameters using gradient descent."""
        raise NotImplementedError


class FCLayer(Layer):
    """docstring for FCLayer"""
    def __init__(self, input_size, output_size):
        super(FCLayer, self).__init__()
        # Initialise weights and bias with random small values
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias    = np.random.randn(         1, output_size) * 0.1
        
    def forward_propagation(self, input_data):
        """Return the output of the layer for a given input."""
        self.input  = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        """Backward propagation of output error."""
        # Calculate errors
        input_error   = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # bias_error = output_error
        # Update parameters
        self.weights -= learning_rate * weights_error
        self.bias    -= learning_rate * output_error
        return input_error


class ActivationLayer(Layer):
    """docstring for ActivationLayer"""
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        """Return output of theh layer for a given input."""
        self.input  = input_data
        self.output = self.activation(input_data)
        return self.output

    def backward_propagation(self, output_error, learning_rate = None):
        """Backward propgation of output error."""
        return self.activation_prime(self.input) * output_error
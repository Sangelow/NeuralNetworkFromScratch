# Project: Neural Network From Scratch
# Date: Mars 2019
# Author: Flavien LOISEAU

class NeuralNetwork():
    """docstring for NeuralNetwork"""
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self,layer):
        """Add layer."""
        self.layers.append(layer)

    def set_loss(self, loss, loss_prime):
        """Set loss/cost function (function to minimize)."""
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        """Predict output for given input."""
        output = input_data
        for layer in self.layers:
            output = layer.forward_propagation(output)
        # print(output)
        return output

    def predict_multiple(self, input_data):
        """Predict output for given list of input."""
        # Create list of results
        results = []
        # Loop throught inputs
        for data in input_data:
            output = data
            for layer in self.layers:
                output = layer.forward_propagation(output)
            results.append(output)
        return output

    def train(self, x_train, y_train, epochs, learning_rate):
        """Train neural network."""
        for i in range(epochs):
            err = 0
            for j in range(len(x_train)):
                # Forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display)
                err += self.loss(y_train[j], output)

                # Backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # Calculate average error for all samples
            err /= len(x_train)
            # print(f'Epoch {i+1}/{epochs} | Error: {err}')
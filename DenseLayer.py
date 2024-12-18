import numpy as np


class DenseLayer:
    def __init__(self, input_size, num_units):
        # input_size = number of features
        # num_units = number of unit
        # initialize the weights into a 2d vector using he initialization.
        self.weights = np.random.randn(input_size, num_units) * np.sqrt(2 / input_size)
        self.biases = np.zeros((1, num_units))

    def forward(self, a_in):
        self.a_in = a_in
        self.z = np.dot(self.a_in, self.weights) + self.biases
        return self.z

    def backwards(self, output_gradient, learning_rate):

        # derived using the chain rule. partial derivative of e with respect to Y times X transposed
        weights_gradient = np.dot(self.a_in.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weights.T)
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)
        return input_gradient
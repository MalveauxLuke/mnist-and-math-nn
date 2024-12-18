import numpy as np

# different layer types cant use lambda functions in mapping because of pickle

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def linear(x):
    return x

# Derivatives
def relu_prime(x):
    return np.where(x > 0, 1, 0)

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def linear_prime(x):
    return np.ones_like(x)
class ActivationLayer:
    def __init__(self, activation):
        activation_mapping = {
            "relu": relu,
            "sigmoid": sigmoid,
            "linear": linear,
        }

        # Derivative (prime) mapping
        activation_prime_mapping = {
            "relu": relu_prime,
            "sigmoid": sigmoid_prime,
            "linear": linear_prime,
        }
        # maps onto the correct activation function. Prime is used for backwards propogation.
        if activation in activation_mapping:
            self.activation = activation_mapping[activation]
            self.actiavtion_prime = activation_prime_mapping[activation]
        else:
            raise ValueError("Unsupported operation")

    def forward(self, a_in):
        # a_in is the inputs
        # calls activation function
        self.a_in = a_in
        self.a = self.activation(self.a_in)
        return self.a

    def backwards(self, output_gradient, learning_rate):
        # chain rule
        return np.multiply(output_gradient, self.actiavtion_prime(self.a_in))


class SoftMax:
    # used in predict statement
    def forward(self, a_in):
        a_in_stable = a_in - np.max(a_in, axis=-1, keepdims=True)  # Subtract max for stability
        tmp = np.exp(a_in_stable)
        self.output = tmp / np.sum(tmp, axis=-1, keepdims=True)
        return self.output

    def backwards(self, output_gradient, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)

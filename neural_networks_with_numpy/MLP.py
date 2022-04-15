import numpy as np
import random


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))


class MLP(object):
    def __init__(self, sizes: list):
        """
        The list ``sizes`` contains the numbers of neurons in each layers.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [
            np.random.randn(y, x)
            for x, y in zip(sizes[:-1], sizes[1:])
        ]
        self.biases = [
            np.zeros((y, 1))
            for y in sizes[1:]
        ]
    
    def load_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases
    
    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(w @ a + b)
        return a
    
    def forward(self, x) -> int:
        """
        Get the max component and return an integer.
        """
        y = self.feedforward(x)
        return int(np.argmax(y))
    
    def backprop(self, x: np.ndarray, y: np.ndarray):
        """
        Update parameters according to ONE point in dataset.
        """
        nabla_b = [np.zeros_like(b) for b in self.biases]
        nabla_w = [np.zeros_like(w) for w in self.weights]

        activation = x
        # store all the activations in network
        activations = [x]
        zs = []

        for w, b in zip(self.weights, self.biases):
            z = w @ activation + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        delta = self.cost_prime(activations[-1], y) + sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = delta @ activations[-2].T
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = (self.weights[-l+1].T @ delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = delta @ activations[-1-l].T
        return nabla_b, nabla_w
    
    def evaluate(self, test_data):
        test_results = [
            (np.argmax(self.feedforward(x)), y)
            for x, y in test_data
        ]
        return sum(int(x == y) for x, y in test_results)
    
    def SGD(self, mini_batch, eta):
        nabla_b = [np.zeros_like(b) for b in self.biases]
        nabla_w = [np.zeros_like(w) for w in self.weights]
        # For each single point in mini batch, calculate the gradient.
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        self.weights = [w - eta / len(mini_batch) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [b - eta / len(mini_batch) * nb
            for b, nb in zip(self.biases, nabla_b)
        ]
    
    def train(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.SGD(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(
                    j, self.evaluate(test_data), n_test
                ))
            else:
                print("Epoch {} complete".format(j))
    
    def cost_prime(z, y_hat, y):
        return y_hat - y
import numpy as np
import logging
from utilities import get_batches

class Model:

    def __init__(self, N, V, set_random_seed=None):

        logging.basicConfig(format="%(message)s", level=logging.INFO)

        if set_random_seed:
            np.random.seed(set_random_seed)

        self.W1 = np.random.rand(N, V)
        self.W2 = np.random.rand(V, N)
        self.b1 = np.random.rand(N, 1)
        self.b2 = np.random.rand(V, 1)
        self.N = N
        self.V = V

    def get_weights(self):
        return self.W1, self.W2

    def get_biases(self):
        return self.b1, self.b2

    def get_weights_and_biases(self):
        return self.W1, self.W2, self.b1, self.b2

    def update_params(self, W1, W2, b1, b2):
        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2

    def sigmoid(z):
    """
    Input:
        z: input neuron/s
    
    Output:
        sigmoid_z: sigmoid activation of z
    """
    sigmoid_z = 1.0 / (1.0 + np.exp(-z))
    return sigmoid_z

    def softmax(z):
        e = np.exp(z)
        y_hat = e / np.sum(e, axis=0)
        return y_hat

    def relu(z):
        h = np.maximum(0, z)
        return h

    def forward_pass(self, x, W1, W2, b1, b2):
        h = np.dot(W1, x) + b1
        h = self.relu(h)
        z = np.dot(W2, h) + b2
        return z, h

    def cost(y, y_hat, batch_size):
        log_p = np.multiply(np.log(y_hat), y) + np.multiply(np.log(1-y_hat), 1-y)
        c = -1 / batch_size * np.sum(log_p)
        c = np.squeeze(c)
        return c

    def backward_pass(self, x, y_hat, y, h, W1, W2, b1, b2, batch_size):
        l1 = np.dot(W2.T, (y_hat-y))
        l1 = self.relu(l1)
        grad_W1 = (1 / batch_size) * np.dot(l1, x.T)
        grad_W2 = (1 / batch_size) * np.dot(y_hat-y, h.T)
        grad_b1 = np.sum((1 / batch_size) * np.dot(l1, x.T), axis=1, keepdims=True)
        grad_b2 = np.sum((1 / batch_size) * np.(y_hat - y, h.T), axis=1, keepdims=True)
        return grad_W1, grad_W2, grad_b1, grad_b2

    def gradient_descent(self, data, word_to_index, num_iters, batch_size, alpha=0.03, C=2):
        W1, W2, b1, b2 = self.get_weights_and_biases()
        V = self.V
        iterations = 0
        for x, y in get_batches(data, word_to_index, V, C, batch_size):
            z, h = self.forward_pass(x, W1, W2, b1, b2)
            y_hat = self.softmax(z)
            cost = self.cost(y, y_hat, batch_size)
            if ((iterations + 1) % 10 == 0):
                iteration_log = f"iteration: {iters + 1} cost: {cost:.6f}"
                logging.info(iteration_log)

            # backprop gradients
            grad_W1, grad_W2, grad_b1, grad_b2 = self.backward_pass(x, y_hat, y, h, W1, W2, b1, b2, batch_size)

            # upadte weights and biases
            W1 -= alpha*grad_W1
            W2 -= alpha*grad_W2
            b1 -= alpha*grad_b1
            b2 -= alpha*grad_b2
            self.update_params(W1, W2, b1, b2)

            iterations += 1

            if iterations == num_iters:
                break

            if iterations % 100 == 0:
                alpha *= 0.66

    def get_word_embeddings(self):
        embeddings = (self.W1.T + self.W2) / 2.0
        return embeddings
        


    
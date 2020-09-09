"""
Model:
    1. Defines the Model class (shallow neural network) for training on dataset
"""

import numpy as np
import logging
from utilities import get_batches
import PlotCode as PC

class Model:

    """
    class to define the shalow neural network and training 'gsd' algorithm
    """

    def __init__(self, N, V, set_random_seed=None):
        """
        Input:
            N: number of dimensions for vocabulary words
            V: Length of vocablary
            set_random_seed: set a random seed for test consistency, defaults to None

        Actions:
            1. configures the logging format and level
            2. sets the random seed if available
            3. initializes the weights, biases, N, V and cost
        """

        logging.basicConfig(format="%(message)s", level=logging.INFO)

        if set_random_seed:
            np.random.seed(set_random_seed)

        self.W1 = np.random.rand(N, V)
        self.W2 = np.random.rand(V, N)
        self.b1 = np.random.rand(N, 1)
        self.b2 = np.random.rand(V, 1)
        self.N = N
        self.V = V
        self.cost = None

    def get_weights(self):
        """
        Returns the weights
        """
        return self.W1, self.W2

    def get_biases(self):
        """
        Returns the biases
        """
        return self.b1, self.b2

    def get_weights_and_biases(self):
        """
        Returns the weights and biases
        """
        return self.W1, self.W2, self.b1, self.b2

    def get_cost(self):
        """
        Returns the cost
        """
        return self.cost

    def update_params(self, W1, W2, b1, b2, cost):
        """
        Inputs:
            W1: weight matrix of first layer
            W2: weight matric of second layer
            b1: bias of first layer
            b2: bias of second layer
            cost: cost of current iteration
        
        Output:
            updates the input parameters to the class
        """
        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2
        self.cost = cost

    def sigmoid(self, z):
        """
        Input:
            z: input neuron/s
    
        Output:
            sigmoid_z: sigmoid activation of z
        """
        sigmoid_z = 1.0 / (1.0 + np.exp(-z))
        return sigmoid_z

    def softmax(self, z):
        """
        Input:
            z: input neuron/s

        Output:
            y_hat: softmax evaluation of z
        """
        e = np.exp(z)
        y_hat = e / np.sum(e, axis=0)
        return y_hat

    def relu(self, z):
        """
        Input:
            z: input neuron/s

        Output:
            h: relu activation of z
        """
        h = np.maximum(0, z)
        return h

    def forward_pass(self, x, W1, W2, b1, b2):
        """
        Inputs:
            x: input matrix 
            W1: weight matrix of first layer
            W2: weight matrix of second layer
            b1: bias of first layer
            b2: bias of second layer

        Outputs:
            z: activations ofter second layer
            h: inputs to second layer
        """
        h = np.dot(W1, x) + b1
        h = self.relu(h)
        z = np.dot(W2, h) + b2
        return z, h

    def cal_cost(self, y, y_hat, batch_size):
        """
        Inputs:
            y: real labels
            y_hat: calculated labels
            batch_size: batch size of the set

        Output:
            c: cost of the batch
        """
        log_p = np.multiply(np.log(y_hat), y) + np.multiply(np.log(1-y_hat), 1-y)
        c = -1 / batch_size * np.sum(log_p)
        c = np.squeeze(c)
        return c

    def backward_pass(self, x, y_hat, y, h, W1, W2, b1, b2, batch_size):
        """
        Inputs:
            x: input matrix
            y_hat: calculated labels
            y: real labels
            h: inputs to the second layer
            W1: weight matrix of first layer
            W2: weight matrix of second layer
            b1: bias of first layer
            b2: bias of second layer
            batch_size: batch size of the set

        Output:
            grad_W1: gradient of W1
            grad_W2: gradient of W2
            grad_b1: gradient of b1
            grad_b2: gradient of b2
        """
        l1 = np.dot(W2.T, (y_hat-y))
        l1 = self.relu(l1)
        grad_W1 = (1 / batch_size) * np.dot(l1, x.T)
        grad_W2 = (1 / batch_size) * np.dot(y_hat-y, h.T)
        grad_b1 = np.sum((1 / batch_size) * np.dot(l1, x.T), axis=1, keepdims=True)
        grad_b2 = np.sum((1 / batch_size) * np.dot(y_hat - y, h.T), axis=1, keepdims=True)
        return grad_W1, grad_W2, grad_b1, grad_b2

    def gradient_descent(self, data, word_to_index, num_iters, batch_size, alpha=0.03, C=2, verbose=False):
        """
        Inputs:
            data: processes dataset
            word_to_index: dictionary that maps word to index
            num_iters: number of desired iterations
            batch_size: custom batch size 
            alpha: training hyperparameter, defaults to 0.03
            C: number of context words on each side, defaults to 2
            verbose: boolean to print info logs

        Actions:
            Trains the model according to the given input parameters using gradient descent algorithm
        """
        W1, W2, b1, b2 = self.get_weights_and_biases()
        V = self.V
        iterations = 0
        for x, y in get_batches(data, word_to_index, V, C, batch_size):
            z, h = self.forward_pass(x, W1, W2, b1, b2)
            y_hat = self.softmax(z)
            cost = self.cal_cost(y, y_hat, batch_size)
            
            if verbose:
                if ((iterations + 1) % 10 == 0):
                    iteration_log = f"iteration: {iterations + 1} cost: {cost:.6f}"
                    logging.info(iteration_log)

            # backprop gradients
            grad_W1, grad_W2, grad_b1, grad_b2 = self.backward_pass(x, y_hat, y, h, W1, W2, b1, b2, batch_size)

            # upadte weights and biases
            W1 -= alpha*grad_W1
            W2 -= alpha*grad_W2
            b1 -= alpha*grad_b1
            b2 -= alpha*grad_b2

            # update the lerant parameters to the class
            self.update_params(W1, W2, b1, b2, cost)

            iterations += 1

            if iterations == num_iters:
                break

            if iterations % 100 == 0:
                alpha *= 0.66

    def get_word_embeddings(self):
        """
        Returns the word embeddings
        """
        W1 = self.W1
        W2 = self.W2
        embeddings = (W1.T + W2) / 2.0
        return embeddings
        


    
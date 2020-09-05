"""
utilities:
    1. bundle of utility functions to be called globally for this project
"""
import numpy as np
from scipy import linalg
from collections import defaultdict
import logging

logging.basicConfig(format="%(message)s", level=logging.INFO)
# helper utility functions

# 1. sigmoid activation function
def sigmoid(z):
    """
    Input:
        z: input neuron/s
    
    Output:
        sigmoid_z: sigmoid activation of z
    """
    sigmoid_z = 1.0 / (1.0 + np.exp(-z))
    return sigmoid_z 

# 2. get index fucntion
def get_indices(words, word_to_index):
    """
    Input:
        words: list of words
        word_to_index: dictionary mapping words to index
    
    Output:
        indices: list of indices corresponding to the words
    """
    indices = []
    for word in words:
        indices = indices + [word_to_index[word]]
    return indices

# 3. zip index with frequency function
def zip_index_and_feq(context_words, word_to_index):
    """
    Input:
        context_words: list of context words
        word_to_index: dictionary that maps words to index

    Output:
        zipped: list of tupples containing index of 
                context words and their frequency
    """
    frequency_dict = defaultdict(int)
    for word in context_words:
        frequency_dict[word] += 1
    indices = get_indices(context_words, word_to_index)
    zipped = []
    for i in range(len(indices)):
        index = indices[i]
        frequency = frequency_dict[context_words[i]]
        zipped.append((index, frequency))
    return zipped

# 4. get vectors function
def get_vectors(data, word_to_index, V, C):
    """
    Input:
        data: processed text data (list of sentences)
        word_to_index: dictionary that maps words to index
        V: number of unique words in the corpus
        C: number of context words on each side
    """
    i = C
    while True:
        y = np.zeros(V)
        x = np.zeros(V)
        center_word = data[i]
        context_words = data[(i - C):i] + data[(i+1):(i+C+1)]
        num_context_words = len(context_words)
        y[word_to_index[center_word]] = 1
        for index, frequency in zip_index_and_feq(context_words, word_to_index):
            x[index] = frequency / num_context_words
        yield x, y
        i += 1
        if i >= len(data):
            info_message = 'i is being set to 0'
            logging.info(info_message)
            i = 0

# 5. get batches fucntion
def get_batches(data, word_to_index, V, C, batch_size):
    """
    Input:
        data: processed dataset
        word_to_index: dictionary that maps words to index
        V: number of unique words in corpus
        C: number of context words on each side
        batch_size: custom batch size for training

    Output:
        1. yields x, y batch and label one at a time
    """
    x_batch = []
    y_batch = []
    for x, y in get_vectors(data, word_to_index, V, C):
        while len(x_batch) < batch_size:
            x_batch.append(x)
            y_batch.append(y)
        else:
            yield np.array(x_batch).T, np.array(y_batch).T
             # batch = [] # TODO: remove comment if bugs in batch formation

# 6. compute the pca (pricipal component analysis)
def pca(data, num_dims=2):
    """
    Input:
        data: data corpus of dimension (m, n) where each row corresponding to a words vector
        num_dims: number of dimensions/components you want to keep
    
    Output:
        collapsed_X = data transformed into num_dims dims/columns + regenerated original data 
        pass in: data as 2D numpy array
    """
    m, n = data.shape

    # mean centering the data
    data -= data.mean(axis=0)

    # covariance matrix
    COV = np.cov(data, rowvar=False)

    # calculating eigenvectors & eigenvalues of covariance matrix
    evals, evecs = linalg.eigh(COV)

    # sorting eigenvalues in decending order
    # to return the corresponding indices of evals and evecs
    indices = np.argsort(evals)[::-1]
    evecs = evecs[:, indices]

    # sorting eigenvectors according to same  indices
    evals = evals[indices]

    # selecting first n eigenvectors (n = num_dims od rescaled data array)
    evecs = evecs[:, :num_dims]

    return np.dot(evecs.T, data.T).T

# 7. get dictionaries function
def get_dictionaries(data):
    """
    Input:
        data: the data corpus
    
    Output:
        word_to_index: dictionary mapping word to index
        index_to_word: dictionary mapping index to word
    """
    words = sorted(list(set(data)))
    n = len(words)
    index = 0

    word_to_index = {}
    index_to_word = {}
    for word in words:
        word_to_index[word] = index
        index_to_word[index] = word
        index += 1
    return word_to_index, index_to_word




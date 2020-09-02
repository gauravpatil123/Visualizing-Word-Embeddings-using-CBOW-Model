import numpy as np
import DatasetProcessing as DP
import Model as M
import PlotCode as PC
import logging
from utilities import pca

logging.basicConfig(format="%(message)s", level=logging.INFO)

# 1. Loading and processing dataset and extarcting parameters

data = DP.ProcessData("data") # <-- TODO: input dataset dir path
# prints dataset sumamry
data.summary()
processed_data = data.get_processed_data()
freq_dist = data.get_freq_distribution()
word_to_index, index_to_word = data.get_dicts()
V = data.get_vocab_size()

# 2. Initializing and training the model
model = M.Model(N=50, V)
model.gradient_descent(processed_data, word_to_index, num_iters=100, batch_size=128)

# 3. Extracting word embeddings from trained model
embeddings = model.get_word_embeddings()

# 4. Getting most common words to plot embeddings
most_common_words, _ = data.most_common_words_and_counts(num=20)

indices = data.get_indices_of_words(most_common_words)

# Defining datapoints for plot
X = embeddings[indices, :]
info_log = "Shape of embeddings of most common words along with their index positions:"
shape_log = str(X.shape) + str(indices)
logging.info(info_log)
logging.info(shape_log)

# reducing the dimensions of embeddings to 2
X_2D = pca(X, 2)

# reducing the dimentions of embeddings to 3
X_3D = pca(X, 3)


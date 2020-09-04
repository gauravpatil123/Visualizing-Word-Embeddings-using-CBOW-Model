import numpy as np
import DatasetProcessing as DP
import Model as M
import PlotCode as PC
import logging
from utilities import pca

logging.basicConfig(format="%(message)s", level=logging.INFO)

# 1. Loading and processing dataset and extarcting parameters

data = DP.ProcessData("data/data.txt") # TODO: change dataset to something more appropriate
# prints dataset summary
data.summary()
processed_data = data.get_processed_data()
freq_dist = data.get_freq_distribution()
word_to_index, index_to_word = data.get_dicts()
V = data.get_vocab_size()

# 2. Initializing and defining training fxn for the model
model = M.Model(N=50, V=V)

def train_model(model, data, w2i, iterations, batch_size, selected_words,
                most_common_words, indices, select_indices,
                create_plot=False, verbose=False):
    num_iter = 0
    Model = model
    while(iterations != num_iter):
        Model.gradient_descent(data, w2i, num_iters=1, batch_size=batch_size)
        num_iter += 1
        
        if verbose:
            if (num_iter % 10 == 0):
                cost = Model.get_cost()
                cost_log = f"iteration: {num_iter} cost: {cost:.6f}"
                logging.info(cost_log)
        
        if create_plot and (num_iter % 10 == 0):
            # extracting word embeddings from trained model
            embeddings = Model.get_word_embeddings()

            # defining datapoints for plot
            X = embeddings[indices, :]
            XS = embeddings[select_indices, :]

            # reducing the dimensions of embeddings to 2
            X_2D = pca(X, 2)
            XS_2D = pca(XS, 2)

            # reducing the dimentions of embeddings to 3
            X_3D = pca(X, 3)
            XS_3D = pca(XS, 3)

            plot = PC.Plot(X_2D,XS_2D, most_common_words, selected_words, 
                            "blue", "green", "o", "o", num_iter)
            plot.plot_2D()
    
    model = Model

# 3. Getting most common words to plot embeddings
most_common_words, _ = data.most_common_words_and_counts(num=200)
indices = data.get_indices_of_words(most_common_words)
selected_words = ["lord", "king", "good", "sir", "she", "most", "mine", 
                    "speak", "queen", "men", "lady", "god", "great", "art", "prince",
                    "fear", "heaven", "blood", "brother", "poor", "noble", "caesar", "wife"]
selected_indices = data.get_indices_of_words(selected_words)

for i in selected_indices:
    if i in indices:
        indices.remove(i)

for word in selected_words:
    if word in most_common_words:
        most_common_words.remove(word)

# 4. Training model and plotting word embeddings
history = train_model(model=model, data=processed_data, w2i=word_to_index, iterations=200,
                        batch_size=128, selected_words=selected_words, most_common_words=most_common_words, 
                        indices=indices, select_indices=selected_indices, create_plot=True, verbose=True)



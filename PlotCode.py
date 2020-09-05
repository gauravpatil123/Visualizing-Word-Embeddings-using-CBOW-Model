import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

class Plot:

    def __init__(self, data_points, select_data_points, 
                    words, select_words, color1, color2, marker1, marker2, iters):
        self.data_points = data_points
        self.select_data_points = select_data_points
        self.words = words
        self.color1 = color1
        self.color2 = color2
        self.marker1 = marker1
        self.marker2 = marker2
        self.iters = iters
        self.select_words = select_words

    def plot_2D(self):
        X, XS = self.data_points, self.select_data_points
        color1, color2 = self.color1, self.color2
        words, marker1, marker2 = self.words, self.marker1, self.marker2
        iters, select_words = self.iters, self.select_words
        plt.figure(figsize=(50, 30), dpi=120)
        plt.scatter(X[:, 0], X[:, 1], s=100, c=color1, marker=marker1)
        plt.scatter(XS[:, 0], XS[:, 1], s=100, c=color2, marker=marker2)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        for i, word in enumerate(words):
            plt.annotate(word, xy=(X[i, 0], X[i, 1]), fontsize=20)
        for i, word in enumerate(select_words):
            plt.annotate(word, xy=(XS[i, 0], XS[i, 1]), fontsize=20)
        plt.legend(["most common words", "select words"], loc='upper right', fontsize=24)
        plt.title(r"Word embeddings of most common words after " + str(iters) + r" iterations", fontsize=37)
        plt.xlabel(r"Dimension 1", fontsize=24)
        plt.ylabel(r"Dimension 2", fontsize=24)
        plt.savefig("Images/2D/2D_word_embeddings_num_iters_"+str(iters)+".png")

    
    def plot_3D(self):
        X, XS = self.data_points, self.select_data_points
        color1, color2 = self.color1, self.color2
        words, marker1, marker2 = self.words, self.marker1, self.marker2
        iters, select_words = self.iters, self.select_words
        plt.figure(figsize=(50, 30), dpi=120)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        ax = plt.axes(projection="3d")
        ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], s=100, c=color1, marker=marker1)
        ax.scatter3D(XS[:, 0], XS[:, 1], XS[:, 2], s=100, c=color2, marker=marker2)
        for i, word in enumerate(words):
            ax.text(X[i, 0], X[i, 1], X[i, 2], '%s' % (str(word)), fontsize=20)
        for i, word in enumerate(select_words):
            ax.text(XS[i, 0], XS[i, 1], XS[i, 2], '%s' % (str(word)), fontsize=20)
        plt.legend(["most common words", "select words"], loc='upper right', fontsize=24)
        plt.title(r"Word embeddings of most common words after " + str(iters) + r" iterations", fontsize=37)
        ax.set_xlabel(r"Dimension 1", fontsize=24)
        ax.set_ylabel(r"Dimension 2", fontsize=24)
        ax.set_zlabel(r"Dimention 3", fontsize=24)
        plt.savefig("Images/3D/3D_word_embeddings_num_iters_"+str(iters)+".png")


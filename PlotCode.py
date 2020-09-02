import matplotlib.pyplot as plt

class Plot:

    def __init__(self, data_points, words, color1, color2, marker1, marker2, iters):
        self.data_points = data_points
        self.words = words
        self.color1 = color1
        self.color2 = color2
        self.marker1 = marker1
        self.marker2 = marker2
        self.iters = iters

    def plot_2D(self):
        X, color1, color2 = self.data_points, self.color1, self.color2
        words, marker1, marker2 = self.words, self.marker1, self.marker2
        iters = self.iters
        plt.figure(figsize=(50, 30), dpi=120)
        plt.scatter(X[:, 0], X[:, 1])
        for i , word in enumerate(words):
            plt.annotate(word, xy=(X[i, 0], X[i, 1]))
        plt.title(r"Word embeddings of most common words after " + str(iters) + r"iterations")
        plt.savefig("Images/2D_word_embeddings_num_iters_"+str(iters)+".png")

    def plot_3D(self):
        ###
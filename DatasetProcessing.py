import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import re
import logging
from utilities import get_dictionaries

class ProcessData:

    def __init__(self, dataset_file__dir):
        
        logging.basicConfig(format='%(message)s', level=logging.INFO)

        with open(dataset_file__dir) as file_handle:
            data = file_handle.read()

        data = re.sub(r'[,!?;-]', '.', data)

        data = nltk.word_tokenize(data)

        data = [char.lower() for char in data
                if char.isalpha()
                or char == '.']
        
        self.data = data

        # frequency distribution of words in the dataset
        self.freq_dist = nltk.FreqDist(word for word in data)

        # creating dictionary mappings from words to indices and vice-versa
        self.word_to_index, self.index_to_word = get_dictionaries(data)

        # size of vocabulary
        self.V = len(self.word_to_index)
    
    def get_processed_data(self):
        return self.data

    def get_freq_distribution(self):
        return self.freq_dist

    def get_dicts(self):
        return self.word_to_index, self.index_to_word

    def get_vocab_size(self):
        return self.V

    def most_common_words_and_counts(self, num=20):
        freq_dist = self.freq_dist
        common_words_with_count = freq_dist.most_common(num)
        most_common_words = [word for word, count in common_words_with_count]
        return most_common_words, common_words_with_count

    def get_indices_of_words(self, words):
        indices = [self.word_to_index[word] for word in words]
        return indices

    def summary(self, num_data_tokens=15, num_most_common_words=200):
        data_log = "\nNumber of tokens: " + str(len(self.data))
        logging.info(data_log)
        data_token_logs = "\n" + str(self.data[:num_data_tokens])
        logging.info(data_token_logs)
        vocab_size_log = "\nSize of vocabulary: " + str(len(self.freq_dist))
        logging.info(vocab_size_log)
        frequent_token_log = "\nMost frequenct tokens: " + str(self.freq_dist.most_common(num_most_common_words)) + "\n"
        logging.info(frequent_token_log)







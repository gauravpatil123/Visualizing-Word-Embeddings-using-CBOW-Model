"""
DatasetProcessing:
    1. Defines ProcessData class to preprocess the dataset
"""

import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import re
import logging
from utilities import get_dictionaries

class ProcessData:

    """
    class to process the dataset
    """

    def __init__(self, dataset_file_dir):

        """
        Input:
            dataset_file_dir: directory path of the dataset file

        Action:
            1. configures standard logging format and level
            2. Opens the data file from the dataset_file_dir
            3. Processes the data file 
            4. Initializes the datafile
            5. Initializes freq_dist a dictionary of words and their frequency in corpus
            6. Initializes word_to_index and index_to_word dictionaries
            7. Initializes V the length of unique word sin dataset 
        """
        
        logging.basicConfig(format='%(message)s', level=logging.INFO)

        with open(dataset_file_dir) as file_handle:
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
        """
        Returns the processed data
        """
        return self.data

    def get_freq_distribution(self):
        """
        Returns the freq_dict frequency dictionary
        """
        return self.freq_dist

    def get_dicts(self):
        """
        Returns the word_to_index and index_to_word dictionary
        """
        return self.word_to_index, self.index_to_word

    def get_vocab_size(self):
        """
        Returns the vocabulary size
        """
        return self.V

    def most_common_words_and_counts(self, num=20):
        """
        Input:
            num: number of most common words to choose from corpus , defaults to 20
        
        Output:
            most_common_words: list of 'num' most_common_words
            common_words_with_count: list of tuples of 'num' most common words and their frequency
        """
        freq_dist = self.freq_dist
        common_words_with_count = freq_dist.most_common(num)
        most_common_words = [word for word, count in common_words_with_count]
        return most_common_words, common_words_with_count

    def get_indices_of_words(self, words):
        """
        Input:
            words: list of words to get their indices

        Output:
            indices: list of index of the words in from the vocabulary 
        """
        indices = [self.word_to_index[word] for word in words]
        return indices

    def summary(self, num_data_tokens=15, num_most_common_words=200):
        """
        Input:
            num_data_tokens: number of data tokens to print, defaults to 15
            num_most_common_words: number of most common words to print, defaults to 200

        Action:
            prints all the chosen info
        """
        data_log = "\nNumber of tokens: " + str(len(self.data))
        logging.info(data_log)
        data_token_logs = "\n" + str(self.data[:num_data_tokens])
        logging.info(data_token_logs)
        vocab_size_log = "\nSize of vocabulary: " + str(len(self.freq_dist))
        logging.info(vocab_size_log)
        frequent_token_log = "\nMost frequenct tokens: " + str(self.freq_dist.most_common(num_most_common_words)) + "\n"
        logging.info(frequent_token_log)







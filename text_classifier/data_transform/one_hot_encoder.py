import numpy as np
import nltk

from nltk.tokenize import RegexpTokenizer
from collections import Counter

nltk.download('stopwords')

class OneHotEncoder(object):
    """ Class dedicated to transform text data into one-hot encoding feature set. """

    def __init__(self, raw_train_data, raw_test_data, target_classes):
        """
        Construct the data transformation object.

        raw_data    A list with all text documents that will be used to build the vector space (train + test data).
        """
        self.raw_train_data = raw_train_data
        self.raw_test_data = raw_test_data
        self.target_classes = target_classes
        self._get_vocabulary()
        self._get_word_2_index()
    
    def _get_tokens(self, text):
        """ Generates a list with all tokens found in a text after removing stop words. """
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text.lower())
        stopwords = nltk.corpus.stopwords.words("english")
        return [token for token in tokens if token not in stopwords]
    
    def _get_vocabulary(self):
        """ Split a raw text into a Counter indexed by word. """
        self.vocabulary = Counter()
        raw_data = self.raw_train_data + self.raw_test_data
        for text in raw_data:
            for token in self._get_tokens(text):
                self.vocabulary[token] += 1
    
    def _get_word_2_index(self):
        """ Generate an word:index dictionary. """
        self.word2index = {}
        for i, word in enumerate(self.vocabulary):
            self.word2index[word.lower()] = i
    
    def get_vocab_length(self):
        """ Returns the number of words that compose the vocabulary. """
        return len(self.vocabulary)

    def _get_binary_input(self, text):
        """ Returns the transformed text in one-hot format. """
        matrix = np.zeros(self.get_vocab_length(), dtype=float)
        for word in self._get_tokens(text):
            matrix[self.word2index[word.lower()]] += 1
        return matrix
    
    def transform_input(self, data):
        """ Returns the transformed data to be used as input by the classifier. """
        return [self._get_binary_input(text) for text in data]

    def _get_binary_target(self, target):
        """
        Transforms the text category into binary one-hot encoding representation.
        
        target  Category name to be transformed into one-hot encoding (binary).
        """
        return np.array([float(target == tgt) for tgt in self.target_classes])
    
    def transform_output(self, targets):
        """
        Transforms the text category into binary one-hot encoding representation.
        
        targets  Iterable with all target to be transformed into one-hot encoding (binary).
        """
        return np.array([self._get_binary_target(tgt) for tgt in targets])
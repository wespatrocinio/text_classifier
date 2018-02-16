from settings import *

from load.data import *
from data_transform.one_hot_encoder import OneHotEncoder
from classifier.nn import Perceptron
from model.neural_network import NeuralNetwork

import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    train_text, test_text = get_train_test_data(DATA_CATEGORIES)

    parameters = {
        'n_hidden_layers': N_HIDDEN,
        'size_hidden_layers': SIZE_HIDDEN,
        'learning_rate': LEARNING_RATE,
        'loss_threshold': 0.01
    }

    model = NeuralNetwork(
        parameters,
        list(train_text.data),
        list(train_text.target),
        list(test_text.data),
        list(test_text.target)
    )

    model.train()
    model.test()

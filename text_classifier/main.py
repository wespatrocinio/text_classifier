from settings import *

from load.data import *
from data_transform.one_hot_encoder import OneHotEncoder
from classifier.nn import Perceptron
from model.neural_network import NeuralNetwork

import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    train_text, test_text = get_train_test_data(DATA_CATEGORIES)

    target_classes = np.unique(train_text.target)
    data = OneHotEncoder(train_text.data + test_text.data, target_classes)

    train_data = data.transform_input(train_text.data)
    train_target = data.transform_output(train_text.target)

    test_data = data.transform_input(test_text.data)
    test_target = data.transform_output(test_text.target)

    parameters = {
        'input_size': data.get_vocab_length(),
        'output_size': len(data.target_classes),
        'n_hidden_layers': N_HIDDEN,
        'size_hidden_layers': SIZE_HIDDEN,
        'learning_rate': LEARNING_RATE,
        'loss_threshold': 0.01
    }

    model = NeuralNetwork(parameters, train_data, train_target, test_data, test_target)

    model.train()
    model.test()

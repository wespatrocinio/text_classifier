# text_classifier
*A little bit more encapsulated library to classify your texts.*

## Why?

This library was built with educational focus to show how to build a Machine Learning based application which encapsulates all the data prepataration, feature generation and algorithm choice, once these steps were already defined.

In this library, you don't need to worry about removing points and special characters, stop words, etc. You'll only need to pass your raw text documents to the Model building. 

The remaining parts of the Model bulding are:

- The hyper-parameters choice, that still manual once the optimal settings depends on several document characteristics (as document average size, number of categories, training/testing datasets size, classes balance, etc.);
- The splitting of the dataset between train/test (for the same reason of the topic above).

## Limitations

- English only;

## How to use it

Currently, the `text_classifier` has only one model available: a Multi-layer Perceptron running One-hot encoded features. To use it, you'll need to do:

```Python

from text_classifier import NeuralNetwork

parameters = {
    'n_hidden_layers': <N_HIDDEN>,
    'size_hidden_layers': <SIZE_HIDDEN>,
    'learning_rate': <LEARNING_RATE>,
    'loss_threshold': <THRESHOLD>
}

train_data = [] # List of text documents to be used as the training dataset
train_target = [] # List of categories of each text document to be used as the training target
test_data = [] # List of text documents to be used as the testing dataset
test_target = [] # List of categories of each text document to be used as the testing target

model = NeuralNetwork(
        parameters,
        train_data,
        train_target,
        test_data,
        test_target
    )
```

You can see a full and working example in the `sample.py` file.
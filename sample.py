from text_classifier import NeuralNetwork

from sklearn.datasets import fetch_20newsgroups

def get_train_test_data(categories):
    """ Generate train and test dataset from Scikit-learn datasets package. """
    train_data = fetch_20newsgroups(subset='train', categories=categories)
    print('Total texts in train:', len(train_data.data))
    test_data = fetch_20newsgroups(subset='test', categories=categories)
    print('Total texts in test:', len(test_data.data))
    return train_data, test_data

if __name__ == '__main__':

    # Perceptron settings
    N_HIDDEN = 2
    SIZE_HIDDEN = {
        'h0': 200,
        'h1': 200
    }

    # Optimization settings
    LEARNING_RATE = 0.01

    # Dataset settings
    DATA_CATEGORIES = [
        "comp.graphics",
        "sci.space",
        "rec.sport.baseball"
    ]

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


    input_text = "Close your eyes and I kiss you. Tomorrow I'll miss you"
    prediction = model.predict([input_text])
    print(prediction)
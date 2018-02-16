from sklearn.datasets import fetch_20newsgroups

def get_train_test_data(categories):
    """ Generate train and test dataset from Scikit-learn datasets package. """
    train_data = fetch_20newsgroups(subset='train', categories=categories)
    print('Total texts in train:', len(train_data.data))
    test_data = fetch_20newsgroups(subset='test', categories=categories)
    print('Total texts in test:', len(test_data.data))
    return train_data, test_data

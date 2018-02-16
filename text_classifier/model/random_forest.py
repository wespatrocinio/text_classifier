from model.base import Model

from sklearn.ensemble import RandomForestClassifier

class RandomForest(Model):
    """ """

    def __init__(self, classifier, parameters, train_data, train_target, test_data, test_target):
        """ """
        super(RandomForest, self).__init__(classifier, parameters, train_data, train_target, test_data, test_target)
    
    def train(self):
        """ """
        pass
    
    def test(self):
        """ """
        pass
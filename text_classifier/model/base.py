class Model(object):
    """ """
    
    def __init__(self, parameters, train_data, train_target, test_data, test_target):
        self.parameters = parameters
        self._get_treated_data(train_data, train_target, test_data, test_target)
        self._get_classifier()

    def _get_treated_data(self, train_data, train_target, test_data, test_target):
        """ """
        raise NotImplementedError("Must be implemented by subclass.")
    
    def _get_classifier(self):
        """ """
        raise NotImplementedError("Must be implemented by subclass.")

    def train(self):
        """ """
        pass
    
    def test(self):
        """ """
        pass
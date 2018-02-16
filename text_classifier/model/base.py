class Model(object):
    """ """
    
    def __init__(self, parameters, train_data, train_target, test_data, test_target):
        self.parameters = parameters
        self.train_data = train_data
        self.train_target = train_target
        self.test_data = test_data
        self.test_target = test_target
        self._get_classifier()

    def _get_classifier(self):
        """ """
        raise NotImplementedError("Must be implemented by subclass.")

    def train(self):
        """ """
        pass
    
    def test(self):
        """ """
        pass
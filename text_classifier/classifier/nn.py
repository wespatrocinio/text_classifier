from classifier.base_class import Classifier

import tensorflow as tf

class Perceptron(Classifier):

    def __init__(self, input_size, output_size, n_hidden, size_hidden):
        """
        Constructor of the Perceptron class

        n_hidden        Number of hidden layers of your perceptron (N)
        size_hidden     Dict with size of each hidden layer labeled as h0, h1 ... hN-1
        """
        super(Perceptron, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden = n_hidden
        self.size_hidden = size_hidden
        self.weights = {}
        self.biases = {}
        self._get_weights_and_biases()
    
    def _get_weights_and_biases(self):
        """ Generate random weights and biases for each layer (including hidden ones) in the NN."""
        current_size = self.input_size
        for i in range(0, self.n_hidden):
            self.weights.update({
                'h{}'.format(i): self._get_layer_weights(current_size, self.size_hidden.get('h{}'.format(i)))
            })
            self.biases.update({
                'h{}'.format(i): self._get_layer_biases(self.size_hidden.get('h{}'.format(i)))
            })
            current_size = self.size_hidden.get('h{}'.format(i))
        self.weights.update({'out': self._get_layer_weights(current_size, self.output_size)})
        self.biases.update({'out': self._get_layer_biases(self.output_size)})

    def _get_layer_weights(self, input_size, layer_size):
        """ Returns random weights for a specific layer. """
        return tf.Variable(tf.random_normal([input_size, layer_size]))
    
    def _get_layer_biases(self, layer_size):
        """ Returns random biases for a specific layer. """
        return tf.Variable(tf.random_normal([layer_size]))

    def predict(self, input_tensor):
        """ Gets an input tensor and generates a prediction of the output tensor. """
        current_input = input_tensor
        for i in range(0, self.n_hidden):
            current_mult = tf.matmul(current_input, self.weights.get('h{}'.format(i)))
            current_add = tf.add(current_mult, self.biases.get('h{}'.format(i)))
            current_actv = tf.nn.relu(current_add)
            current_input = current_actv
        output_mult = tf.matmul(current_input, self.weights.get('out'))
        return tf.add(output_mult, self.biases.get('out'))  

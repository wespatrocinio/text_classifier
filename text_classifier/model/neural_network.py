from text_classifier.model.base import Model
from text_classifier.data_transform.one_hot_encoder import OneHotEncoder
from text_classifier.classifier.nn import Perceptron

import tensorflow as tf
import numpy as np

class NeuralNetwork(Model):

    def __init__(self, parameters, train_data, train_target, test_data, test_target):
        super(NeuralNetwork, self).__init__(parameters, train_data, train_target, test_data, test_target)
        self.tf_session = tf.Session()
        self.input_tensor = tf.placeholder(
            tf.float32,
            [None, self.parameters.get('input_size')], name="input")
        self.output_tensor = tf.placeholder(
            tf.float32,
            [None, self.parameters.get('output_size')], name="output")
    
    def _get_treated_data(self, train_data, train_target, test_data, test_target):
        """ """
        target_classes = list(set(train_target + test_target))
        self.encoder = OneHotEncoder(train_data, test_data, target_classes)
        self.train_data = self.encoder.transform_input(train_data)
        self.test_data = self.encoder.transform_input(test_data)
        self.train_target = self.encoder.transform_output(train_target)
        self.test_target = self.encoder.transform_output(test_target)
        self.vocabulary_size = self.encoder.get_vocab_length()
        self.target_classes = self.encoder.target_classes

    
    def _get_classifier(self):
        """ """
        self.classifier = Perceptron(
            self.vocabulary_size,
            len(self.target_classes),
            self.parameters.get('n_hidden_layers'),
            self.parameters.get('size_hidden_layers')
        )


    def train(self, batch=True, batch_size=200):
        loss = self._get_entropy_loss(self.classifier.predict(self.input_tensor), self.output_tensor)
        optimizer = self._get_optimizer(loss, self.parameters.get('learning_rate'))

        # Initializing the variables
        init = tf.global_variables_initializer()

        training_epochs = 10
        # Launch the graph
        self.tf_session.run(init) # inits the variables
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(len(self.train_data) / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = self._get_batch(self.train_data, self.train_target, i, batch_size)
                # Run optimization op (back propagation) and cost optimization
                c, _ = self.tf_session.run(
                    [loss, optimizer],
                    feed_dict={
                        self.input_tensor: batch_x,
                        self.output_tensor: batch_y
                    }
                )
                avg_cost += c / total_batch
            # Display logs per epoch step
            print("Epoch", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_cost))
        print("Optimization Finished!")
        return True
    
    def test(self, input_data=None, output_target=None):
        """ """
        # Test model
        correct_prediction = tf.equal(
            tf.argmax(self.classifier.predict(self.input_tensor), 1),
            tf.argmax(self.output_tensor, 1)
        )
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        if (input_data is None) and (output_target is None):
            batch_x_test, batch_y_test = self._get_batch(self.test_data, self.test_target, 0, len(self.test_data))
        else:
            batch_x_test, batch_y_test = self._get_batch(input_data, output_target, 0, len(input_data))
        print(
            "Accuracy:",
            accuracy.eval(
                {self.input_tensor: batch_x_test, self.output_tensor: batch_y_test},
                session=self.tf_session
            )
        )
    
    def predict(self, input_data):
        """ """
        treated_data = self.encoder.transform_input(input_data)
        return self.tf_session.run(
            tf.argmax(self.classifier.predict(self.input_tensor), 1),
            feed_dict= {self.input_tensor: treated_data}
        )
    
    def _get_batch(self, data, target, iteration, batch_size):
        """ Defines the size of each batch to be processed. """
        input_batch = data[iteration * batch_size : iteration * batch_size + batch_size]
        target_batch = target[iteration * batch_size : iteration * batch_size + batch_size] 
        return np.array(input_batch), np.array(target_batch)

    def _get_entropy_loss(self, prediction, output_tensor):
        """ Calculate the mean loss based in the difference between the prediction and the ground truth. """
        entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor)
        return tf.reduce_mean(entropy_loss)

    def _get_optimizer(self, loss, learning_rate):
        """ Update all the variables bases on the Adaptive Moment Estimation (Adam) method. """
        return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
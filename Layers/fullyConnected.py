import numpy as np

"""
Fully Connected Layer
Use: it acts as the decision-making part of the network, combining features to make predictions.
"""

class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.biases = np.zeros((output_size, 1))

    def forward(self, input):
        """
        Forward Pass:
            Flattens the feature map and applies a linear transformation followed by an activation function.
            Outputs high-level representations or predictions based on the extracted features.
        """
        self.input = input.reshape(input.shape[0], -1).T  # Flatten and transpose
        self.output = np.dot(self.weights, self.input) + self.biases
        return self.output.T

    def backward(self, d_output, learning_rate=0.001):
        """
        Backward Pass:
            Here we compute gradients of the loss with respect to:
                Weights and Biases: Updates the connections between neurons for better decision-making.
                Input: Passes the error signal to earlier layers.
        """
        batch_size = d_output.shape[0]
        d_output = d_output.T

        d_weights = np.dot(d_output, self.input.T) / batch_size
        d_biases = np.sum(d_output, axis=1, keepdims=True) / batch_size
        d_input = np.dot(self.weights.T, d_output)

        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return d_input.T

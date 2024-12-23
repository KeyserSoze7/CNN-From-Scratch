import numpy as np
from Layers.conv import Conv
from Layers.maxpool import MaxPool
from Layers.fullyConnected import FullyConnectedLayer
from utils.activation import relu, relu_derivative, softmax
from utils.loss import cross_entropy_loss, cross_entropy_derivative
from utils.data_loader import load_data, split_data

# Model Definition
class CNN:
    def __init__(self):
        # Calling the Layer functions
        self.conv1 = Conv(input_channels=1, output_channels=6, kernel_size=3, stride=1, padding=1)
        self.pool1 = MaxPool(pool_size=2, stride=2)
        self.fc1 = FullyConnectedLayer(input_size=14*14*6, output_size=128)
        self.fc2 = FullyConnectedLayer(input_size=128, output_size=10)  # 10 classes for digits

    def forward(self, X):
        self.x_conv1 = self.conv1.forward(X)
        self.x_relu1 = relu(self.x_conv1)
        self.x_pool1 = self.pool1.forward(self.x_relu1)
        self.x_fc1 = self.fc1.forward(self.x_pool1)
        self.x_relu2 = relu(self.x_fc1)
        self.x_fc2 = self.fc2.forward(self.x_relu2)
        self.output = softmax(self.x_fc2)
        return self.output

    def backward(self, y_true, learning_rate):
        # Compute loss gradient
        grad_output = cross_entropy_derivative(y_true, self.output)

        # Fully Connected Layer 2
        grad_fc2 = self.fc2.backward(grad_output, learning_rate)
        grad_relu2 = relu_derivative(self.x_fc1) * grad_fc2

        # Fully Connected Layer 1
        grad_fc1 = self.fc1.backward(grad_relu2, learning_rate)
        grad_pool1 = grad_fc1.reshape(self.x_pool1.shape)

        # Max Pool Layer
        grad_relu1 = self.pool1.backward(grad_pool1)
        grad_conv1 = relu_derivative(self.x_conv1) * grad_relu1

        # Convolutional Layer
        self.conv1.backward(grad_conv1, learning_rate)

# Train Function
def train(model, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate):
    num_batches = X_train.shape[0] // batch_size

    for epoch in range(epochs):
        total_loss = 0

        for i in range(num_batches):
            X_batch = X_train[i * batch_size:(i + 1) * batch_size]
            y_batch = y_train[i * batch_size:(i + 1) * batch_size]

            # Forward Pass
            predictions = model.forward(X_batch)
            loss = cross_entropy_loss(y_batch, predictions)
            total_loss += loss

            # Backward Pass
            model.backward(y_batch, learning_rate)

        # Validation
        val_predictions = model.forward(X_val)
        val_loss = cross_entropy_loss(y_val, val_predictions)
        val_accuracy = np.mean(np.argmax(val_predictions, axis=1) == y_val)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / num_batches:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")

# Main Code
if __name__ == "__main__":
    # Load and preprocess data
    print("Loading MNIST dataset...")
    X, y, X_test, y_test = load_data()
    X_train, X_val, y_train, y_val = split_data(X, y)

    # Initialize and train the model
    print("Initializing CNN model...")
    model = CNN()
    epochs = 10
    batch_size = 32
    learning_rate = 0.01

    print("Training started...")
    train(model, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate)

    # Test accuracy
    print("Testing the model...")
    test_predictions = model.forward(X_test)
    test_accuracy = np.mean(np.argmax(test_predictions, axis=1) == y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

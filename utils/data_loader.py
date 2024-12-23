import numpy as np
import gzip
import os

def load_data():
    """
    Load and preprocess the MNIST dataset without using external libraries.
    MNIST data is assumed to be downloaded in raw format (gzip files).

    Returns:
        X_train: Training images (normalized to [0, 1])
        y_train: Training labels
        X_test: Test images (normalized to [0, 1])
        y_test: Test labels
    """
    # File paths for MNIST raw data (downloaded manually)
    path = "/home/adityasr/Code/AI/CNN From Scratch /Data"
    files = {
        "train_images": os.path.join(path, "train-images-idx3-ubyte.gz"),
        "train_labels": os.path.join(path, "train-labels-idx1-ubyte.gz"),
        "test_images": os.path.join(path, "t10k-images-idx3-ubyte.gz"),
        "test_labels": os.path.join(path, "t10k-labels-idx1-ubyte.gz"),
    }

    # Reading MNIST gzip files
    def read_images(filename):
        with gzip.open(filename, 'rb') as f:
            f.read(16)  # Skip the header
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(-1, 28, 28)  # Shape: (num_samples, height, width)

    def read_labels(filename):
        with gzip.open(filename, 'rb') as f:
            f.read(8)  # Skip the header
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data

    print("Loading MNIST dataset...")
    X_train = read_images(files["train_images"])
    y_train = read_labels(files["train_labels"])
    X_test = read_images(files["test_images"])
    y_test = read_labels(files["test_labels"])

    # Normalizing pixels values b/w [0,1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Reshaping :  (batch_size, channels, height, width)
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)

    print(f"Data loaded successfully! Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    return X_train, y_train, X_test, y_test


def split_data(X, y, validation_ratio=0.1):
    """
    Split the training data into training and validation sets manually.

    Args:
        X: Input data (numpy array)
        y: Corresponding labels
        validation_ratio: Proportion of data to use for validation

    Returns:
        X_train: Training data after split
        X_val: Validation data
        y_train: Training labels after split
        y_val: Validation labels
    """
    print("Splitting data into training and validation sets...")

    # Calculate split index
    num_samples = X.shape[0]
    split_index = int(num_samples * (1 - validation_ratio))

    # Shuffle the data before splitting
    indices = np.arange(num_samples)
    np.random.seed(42)  # Ensures reproducibility
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    # Split the data
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
    return X_train, X_val, y_train, y_val

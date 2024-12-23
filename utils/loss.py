import numpy as np

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true])
    return np.sum(log_likelihood) / m

def cross_entropy_derivative(y_true, y_pred):
    m = y_true.shape[0]
    grad = y_pred
    grad[range(m), y_true] -= 1
    grad = grad / m
    return grad

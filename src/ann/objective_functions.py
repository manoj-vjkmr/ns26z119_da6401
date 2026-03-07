"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np
from .activations import softmax

def cross_entropy_loss(logits, y):
    probs= softmax(logits)
    m= y.shape[0]
    loss= -np.log(probs[range(m), y])
    return np.mean(loss)

def cross_entropy_grad(y, pred):
    batch_size = pred.shape[0]
    y_idx = y.astype(int).flatten()
    
    grad = pred.copy()
    grad[np.arange(batch_size), y_idx] -= 1
    
    return grad / batch_size

def mse_loss(pred, y):
    if y.ndim == 1:
        y = np.eye(pred.shape[1])[y]
    return np.mean((pred - y) ** 2)

def mse_grad(y, pred):
    y_one_hot = np.zeros_like(pred)
    y_idx = y.astype(int).flatten()
    
    y_one_hot[np.arange(len(y_idx)), y_idx] = 1
    
    return 2 * (pred - y_one_hot) / y.size
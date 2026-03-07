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

def cross_entropy_grad(logits, y):
    probs= softmax(logits)
    m= y.shape[0]
    probs[range(m), y] -= 1
    return probs / m

def mse_loss(pred, y):
    if y.ndim == 1:
        y = np.eye(pred.shape[1])[y]
    return np.mean((pred - y) ** 2)

def mse_grad(pred, y):
    if y.ndim == 1:
        y = np.eye(pred.shape[1])[y]
    return 2 * (pred - y) / y.size
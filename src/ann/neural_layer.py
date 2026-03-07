"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np

class neural_layer:

    def __init__(self, n_input, n_output, weight_init="xavier"):
        if weight_init == "xavier":
            limit= np.sqrt(6 / (n_input + n_output))
            self.W= np.random.uniform(-limit, limit, (n_input, n_output))
        else:
            self.W= np.random.randn(n_input, n_output) * 0.01

        self.b= np.zeros((1, n_output))

    def forward_pass(self, X):
        self.x= X
        self.z= np.dot(X, self.W) + self.b
        return self.z

    def backward(self, grad):
        self.grad_W= np.dot(self.x.T, grad)
        self.grad_b= np.sum(grad, axis=0, keepdims=True)
        return np.dot(grad, self.W.T)
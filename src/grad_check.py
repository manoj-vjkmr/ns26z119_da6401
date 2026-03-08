import numpy as np
from ann.neural_network import NeuralNetwork

def numerical_gradient(model, X, y, layer_idx, param_name, epsilon=1e-7):
    layer = model.layers[layer_idx]
    param = getattr(layer, param_name)
    grad_num = np.zeros_like(param)
    
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_val = param[ix]
        
        # f(x + eps)
        param[ix] = old_val + epsilon
        out_plus = model.forward(X)
        if model.loss == "cross_entropy":
            # Manual Cross Entropy for scalar loss
            exps = np.exp(out_plus - np.max(out_plus, axis=1, keepdims=True))
            probs = exps / np.sum(exps, axis=1, keepdims=True)
            loss_plus = -np.mean(np.log(probs[np.arange(len(y)), y.astype(int)] + 1e-12))
        else:
            y_oh = np.zeros_like(out_plus)
            y_oh[np.arange(len(y)), y.astype(int)] = 1
            loss_plus = np.mean(np.square(out_plus - y_oh))

        # f(x - eps)
        param[ix] = old_val - epsilon
        out_minus = model.forward(X)
        if model.loss == "cross_entropy":
            exps = np.exp(out_minus - np.max(out_minus, axis=1, keepdims=True))
            probs = exps / np.sum(exps, axis=1, keepdims=True)
            loss_minus = -np.mean(np.log(probs[np.arange(len(y)), y.astype(int)] + 1e-12))
        else:
            y_oh = np.zeros_like(out_minus)
            y_oh[np.arange(len(y)), y.astype(int)] = 1
            loss_minus = np.mean(np.square(out_minus - y_oh))

        grad_num[ix] = (loss_plus - loss_minus) / (2 * epsilon)
        param[ix] = old_val
        it.iternext()
    return grad_num

class DummyArgs:
    dataset = "mnist"
    hidden_size = [16]
    num_layers = 1
    activation = "sigmoid"
    weight_init = "random"
    loss = "cross_entropy"
    optimizer = "sgd"
    learning_rate = 0.01

args = DummyArgs()
model = NeuralNetwork(args)
X = np.random.randn(2, 784)
y = np.array([1, 5])

# Get Analytical Gradients
logits = model.forward(X)
model.backward(y, logits)

# Compare Layer 0 Weights
grad_analytical = model.grad_W[0]
grad_numerical = numerical_gradient(model, X, y, 0, 'W')

rel_error = np.linalg.norm(grad_analytical - grad_numerical) / (np.linalg.norm(grad_analytical) + np.linalg.norm(grad_numerical))
print(f"Mean Relative Error: {rel_error}")

if rel_error < 1e-7:
    print("SUCCESS: Gradients match!")
else:
    print("FAILURE: Check your objective_functions.py scaling.")
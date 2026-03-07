import numpy as np
from ann.neural_network import NeuralNetwork
from argparse import Namespace

OLD_MODEL_PATH = "./best_model.npy"
NEW_MODEL_PATH = "./best_model_dict.npy"

# Load old model (array format)
data = np.load(OLD_MODEL_PATH, allow_pickle=True)

# Convert array to dict
weights = {}
for i in range(len(data)//2):
    weights[f"W{i}"] = data[2*i]
    weights[f"b{i}"] = data[2*i + 1]

# Save new dictionary format
np.save(NEW_MODEL_PATH, weights)
print(f"Converted weights saved to {NEW_MODEL_PATH}")

# --- Optional test ---
# Create dummy args
args = Namespace(dataset="mnist", hidden_size=[128,64,32,16], activation="relu", weight_init="xavier", learning_rate=0.001)

# Load NeuralNetwork
model = NeuralNetwork(args)
model.set_weights(weights)
print("Weights successfully loaded. Layer 0 W shape:", model.layers[0].W.shape)
import wandb
import numpy as np
import os
import json
import sys
from train import parse_arguments, NeuralNetwork, load_dataset, model_path

sweep_config = {
    "method": "random",
    "metric": {"name": "test_accuracy", "goal": "maximize"},
    "parameters": {
        "dataset": {"values": ["mnist"]},
        "epochs": {"values": [10]},
        "batch_size": {"values": [32, 64, 128]},
        "learning_rate": {"values": [0.0005, 0.001, 0.002, 0.005, 0.01]},
        "weight_decay": {"values": [0.0, 1e-5, 1e-4]}, 
        "loss": {"values": ["cross_entropy", "mse"]},
        "optimizer": {"values": ["sgd", "momentum", "nag", "rmsprop"]},
        "num_layers": {"values": [1, 2, 3, 4, 5, 6]},
        "hidden_size": {"values": [32, 64, 96, 128]},
        "activation": {"values": ["relu", "tanh", "sigmoid"]},
        "weight_init": {"values": ["random", "xavier"]}
    }
}

def sweep_train():
    wandb.init()
    config = wandb.config

    class Args:
        pass
    args = Args()
    args.dataset = config.dataset
    args.epochs = config.epochs
    args.batch_size = config.batch_size
    args.learning_rate = config.learning_rate
    args.weight_decay = config.weight_decay
    args.loss = config.loss
    args.optimizer = config.optimizer
    args.num_layers = config.num_layers
    args.hidden_size = [config.hidden_size] * config.num_layers
    args.activation = config.activation
    args.weight_init = config.weight_init
    args.wandb_project = "da6401"
    args.model_save_path = None
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(args.dataset)
    
    model = NeuralNetwork(args)
    model.train(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)
    
    val_acc = model.evaluate(X_val, y_val)
    test_acc = model.evaluate(X_test, y_test)

    wandb.log({
        "validation_accuracy": val_acc,
        "test_accuracy": test_acc
    })
    
    path = model_path(args)
    weights = model.get_weights()
    np.save(path, weights)
    
    config_dict = vars(args)
    config_save_path = path.replace(".npy", ".json")
    with open(config_save_path, "w") as f:
        json.dump(config_dict, f, indent=4)
    
    if test_acc >= 0.97:
        os._exit(0)

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="da6401")
    wandb.agent(sweep_id, function=sweep_train)
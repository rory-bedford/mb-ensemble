"""
In this experiment, we have a changing environment,
and perform a grid search over a set of hyperparameters.
"""

import torch
import sys
sys.path.append("model")
from model import MushroomBody
from dataloader import DynamicOlfactoryValences
import numpy as np
from sklearn.model_selection import ParameterGrid
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# environment parameters
sparsity = 0.2 # sparsity
patterns = 32*64 # number of patterns
kenyon_cells = 2000 # number of neurons
positive_ratio = 0.5 # probability of positive valence
change_rate = [10] # number of epochs before changing valences
change_fraction = [0.1] # fraction of valences to change
noise_level = 0.1 # standard deviation of the Gaussian noise

dataloader = DynamicOlfactoryValences(patterns, kenyon_cells, sparsity, positive_ratio, change_rate, change_fraction, noise_level, device)

epochs = 1000

# grid search parameters
param_grid = {
    "n_positive": list(range(1, 8, 3)),
    "n_negative": list(range(1, 8, 3)),
    "lower_lr": range(-6, 2),
    "upper_lr": range(-6, 2),
    "activation": ["sigmoid","relu","linear"]
}
grid_search = ParameterGrid(param_grid)

# run grid search
for params in grid_search:
    n_positive = params["n_positive"]
    n_negative = params["n_negative"]
    lower_lr = params["lower_lr"]
    upper_lr = params["upper_lr"]
    activation = params["activation"]
    valences = ["positive"] * n_positive + ["negative"] * n_negative
    learning_rates = np.append(np.logspace(lower_lr, upper_lr, n_positive), np.logspace(lower_lr, upper_lr, n_negative)).astype(float)

    mushroom_body = MushroomBody(activation, kenyon_cells, valences, learning_rates, sparsity).to(device)
    losses = mushroom_body.group_train(dataloader, epochs)
    mean_loss = np.array(losses[50:]).mean()
    result = {**params, "mean_loss": mean_loss}
    results_df = pd.DataFrame([result])
    results_df.to_csv("results/experiment2.csv", mode='a', header=not pd.io.common.file_exists("grid_search_results.csv"), index=False)

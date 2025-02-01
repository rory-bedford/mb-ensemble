"""
In this experiment, we have a changing environment with
changes over multiple timescales, and perform a grid search
over learning rates.
"""

import torch
import sys
sys.path.append("model")
from model import MushroomBody
from dataloader import DynamicOlfactoryValences
import numpy as np
from sklearn.model_selection import ParameterGrid
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# environment parameters
sparsity = 0.2 # sparsity
patterns = 32*32 # number of patterns
kenyon_cells = 2000 # number of neurons
positive_ratio = 0.5 # probability of positive valence
change_rate = [3,15,75] # number of epochs before changing valences
change_fraction = [0.1,0.1,0.1] # fraction of valences to change
noise_level = 0.1 # standard deviation of the Gaussian noise

dataloader = DynamicOlfactoryValences(patterns, kenyon_cells, sparsity, positive_ratio, change_rate, change_fraction, noise_level, device)

epochs = 1000

# fixed parameters
n_positive = 8
n_negative = 7
valences = ["positive"] * n_positive + ["negative"] * n_negative
activation = "relu"

# grid search parameters
param_grid = {
    "lower_lr": range(-5, 2),
    "upper_lr": range(-5, 2),
}
grid_search = ParameterGrid(param_grid)

# run grid search
for i, params in enumerate(grid_search):

    print(f"Running experiment {i+1}/{len(grid_search)}")

    lower_lr = params["lower_lr"]
    upper_lr = params["upper_lr"]
    learning_rates = np.append(np.logspace(lower_lr, upper_lr, n_positive), np.logspace(lower_lr, upper_lr, n_negative)).astype(float)

    mushroom_body = MushroomBody(activation, kenyon_cells, valences, learning_rates, sparsity).to(device)
    losses = mushroom_body.group_train(dataloader, epochs)
    mean_loss = np.array(losses[50:]).mean()
    result = {**params, "mean_loss": mean_loss}
    results_df = pd.DataFrame([result])
    results_df.to_csv("results/experiment3.csv", mode='a', header=not pd.io.common.file_exists("grid_search_results.csv"), index=False)

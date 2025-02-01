"""
Here we just check the best models and rerun.
"""

import numpy as np
import torch
import pandas as pd
import sys
sys.path.append("model")
from model import MushroomBody
from dataloader import DynamicOlfactoryValences

results_df = pd.read_csv("results/experiment2.csv")

best_models = results_df[results_df['activation'] != "linear"].sort_values("total_loss").head(20)
print(best_models)

# Get the parameters from the best model
best_model_params = best_models.iloc[0]
activation = best_model_params['activation']
n_positive = best_model_params['n_positive']
n_negative = best_model_params['n_negative']
lower_lr = best_model_params['lower_lr']
upper_lr = best_model_params['upper_lr']
total_loss = best_model_params['total_loss']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# environment parameters
sparsity = 0.2 # sparsity
patterns = 32*32 # number of patterns
kenyon_cells = 2000 # number of neurons
positive_ratio = 0.5 # probability of positive valence
change_rate = [10] # number of epochs before changing valences
change_fraction = [0.1] # fraction of valences to change
noise_level = 0.1 # standard deviation of the Gaussian noise

environment = DynamicOlfactoryValences(patterns, kenyon_cells, sparsity, positive_ratio, change_rate, change_fraction, noise_level, device)
epochs = 1000

learning_rates = np.append(np.logspace(lower_lr, upper_lr, n_positive), np.logspace(lower_lr, upper_lr, n_negative)).astype(float)
valences = ["positive"] * n_positive + ["negative"] * n_negative

mushroom_body = MushroomBody(activation, kenyon_cells, valences, learning_rates, sparsity).to(device)

losses = mushroom_body.group_train(environment, epochs)

losses = np.array(losses)
np.save("results/experiment2-bestrun.npy", losses)

# Additional 5 runs with randomly selected parameters
for i in range(1, 6):
    random_params = results_df.sample(n=1).iloc[0]
    activation = random_params['activation']
    n_positive = random_params['n_positive']
    n_negative = random_params['n_negative']
    lower_lr = random_params['lower_lr']
    upper_lr = random_params['upper_lr']
    
    learning_rates = np.append(np.logspace(lower_lr, upper_lr, n_positive), np.logspace(lower_lr, upper_lr, n_negative)).astype(float)
    valences = ["positive"] * n_positive + ["negative"] * n_negative
    
    mushroom_body = MushroomBody(activation, kenyon_cells, valences, learning_rates, sparsity).to(device)
    losses = mushroom_body.group_train(environment, epochs)
    
    losses = np.array(losses)
    np.save(f"results/experiment2-run{i}.npy", losses)
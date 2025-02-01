"""
In this experiment, we test our model with a constant environment,
and compare multiple compartments with one compartment.
"""

import torch
import sys
sys.path.append("model")
from model import MushroomBody
from dataloader import DynamicOlfactoryValences
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# environment parameters
sparsity = 0.2 # sparsity
patterns = 32*32 # number of patterns
kenyon_cells = 2000 # number of neurons
positive_ratio = 0.5 # probability of positive valence
change_rate = [] # number of epochs before changing valences
change_fraction = [] # fraction of valences to change
noise_level = 0.1 # standard deviation of the Gaussian noise

dataloader = DynamicOlfactoryValences(patterns, kenyon_cells, sparsity, positive_ratio, change_rate, change_fraction, noise_level, device)

# multi compartments
# model parameters
n_positive = 8
n_negative = 7
lower_lr = -1 # log10 of the lower bound of the learning rate
upper_lr = -1 # log10 of the upper bound of the learning rate
valences = ["positive"] * n_positive + ["negative"] * n_negative
learning_rates = np.append(np.logspace(lower_lr, upper_lr, n_positive), np.logspace(lower_lr, upper_lr, n_negative)).astype(float)
activation = "relu"

mushroom_body = MushroomBody(activation, kenyon_cells, valences, learning_rates, sparsity).to(device)

# run training and save losses
epochs = 100
losses = mushroom_body.group_train(dataloader, epochs)
loss_array = np.array(losses)
np.save("results/experiment1-multiple.npy", loss_array)

# single compartments
# model parameters
n_positive = 1
n_negative = 1
lower_lr = -1 # log10 of the lower bound of the learning rate
upper_lr = -1 # log10 of the upper bound of the learning rate
valences = ["positive"] * n_positive + ["negative"] * n_negative
learning_rates = np.append(np.logspace(lower_lr, upper_lr, n_positive), np.logspace(lower_lr, upper_lr, n_negative)).astype(float)
activation = "relu"

mushroom_body = MushroomBody(activation, kenyon_cells, valences, learning_rates, sparsity).to(device)

# run training and save losses
epochs = 100
losses = mushroom_body.group_train(dataloader, epochs)
loss_array = 0.5*np.array(losses)
np.save("results/experiment1-single.npy", loss_array)
import torch
from torch.utils.data import DataLoader, TensorDataset


def generate_data(patterns, kenyon_cells, sparsity, positive_ratio):
    """
    Generate random sparse binary vectors with valences

    Args:
        patterns: int, number of patterns
        kenyon_cells: int, number of neurons
        sparsity: float, sparsity
        positive_ratio: float, probability of positive valence

    Returns:
        X: torch.Tensor, shape (patterns, kenyon_cells), random binary vectors
        Y: torch.Tensor, shape (patterns, 1), valences of the vectors
    """

    probabilities = torch.full((patterns, kenyon_cells), sparsity)
    X = torch.bernoulli(probabilities)

    Y = 2*torch.bernoulli(torch.full((patterns, 1), positive_ratio)) - 1

    return X, Y


def add_gaussian_noise(X, noise_level):
    """
    Add Gaussian noise to the input data

    Args:
        X: torch.Tensor, shape (N, D), input data
        noise_level: float, standard deviation of the Gaussian noise

    Returns:
        X_noisy: torch.Tensor, shape (N, D), noisy input data
    """
    noise = torch.randn_like(X) * noise_level
    X_noisy = X + noise
    return X_noisy


class DynamicOlfactoryValences(DataLoader):
    """
    A custom DataLoader that dynamically changes valences and adds Gaussian noise to the data.

    Args:
        patterns (int): Number of patterns.
        kenyon_cells (int): Number of neurons.
        sparsity (float): Sparsity of the input data.
        positive_ratio (float): Probability of positive valence.
        change_rate (list of int): List of epochs before changing valences.
        change_fraction (list of float): List of fractions of valences to change.
        noise_level (float): Standard deviation of the Gaussian noise.
        device (torch.device): Device to move the data to.
        batch_size (int, optional): Size of each batch. Default is 32.
        shuffle (bool, optional): Whether to shuffle the data. Default is True.

    Methods:
        __iter__():
            Returns an iterator for the DataLoader.
        __next__():
            Returns the next batch of data with added Gaussian noise.
        current_data():
            Returns the current data (X, Y).
    """
    def __init__(self, patterns, kenyon_cells, sparsity, positive_ratio, change_rate, change_fraction, noise_level, device, batch_size=32, shuffle=True):
        self.patterns = patterns
        self.kenyon_cells = kenyon_cells
        self.sparsity = sparsity
        self.positive_ratio = positive_ratio
        self.change_rate = change_rate
        self.change_fraction = change_fraction
        self.noise_level = noise_level
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.X, self.Y = generate_data(patterns, kenyon_cells, sparsity, positive_ratio)
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)
        self.dataset = TensorDataset(self.X, self.Y)
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle)

        self.epochs = 0

    def __iter__(self):
        # dynamically change a subset of valences at determined points
        for change_rate, change_fraction in zip(self.change_rate, self.change_fraction):
            if self.epochs % change_rate == 0:
                indices = torch.randint(0, self.patterns, (int(change_fraction * self.patterns),)).to(self.device)
                self.Y[indices] = 2*torch.bernoulli(torch.full((int(change_fraction * self.patterns), 1), self.positive_ratio).to(self.device)) - 1
                self.dataset.tensors = (self.X, self.Y)
        self.epochs += 1
        return super().__iter__()

    def __next__(self):
        batch_X, batch_Y = next(super().__iter__())
        batch_X = add_gaussian_noise(batch_X, self.noise_level)
        return batch_X, batch_Y

    def current_data(self):
        return self.X, self.Y
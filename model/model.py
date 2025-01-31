import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from losses import positive_valence_loss, negative_valence_loss, fraction_incorrect
import numpy as np
from tqdm import tqdm


class MBCompartment(nn.Module):
    """
    A simple perceptron with logit outputs

    Args:
        input_dim (int): dimension of input vectors
        valence (str): valence of the perceptron, either "positive" or "negative"
        lr (float): learning rate
        sparsity (float): sparsity of the input data

    Methods:
        forward(x):
            Forward pass of the perceptron
        train(X, Y, epochs=100, lr=0.01, batch_size=32):
            Train the perceptron on the given data
        initialize_weights():
            Initialize weights using Xavier uniform initialization
    """

    def __init__(self, input_dim, valence, lr, sparsity, activation):
        super(MBCompartment, self).__init__()
        assert valence in ["positive", "negative"]
        assert isinstance(lr, (float, np.float32, np.float64))
        self.input_dim = input_dim
        self.valence = valence
        self.lr = lr
        self.sparsity = sparsity
        self.linear = nn.Linear(input_dim, 1)
        self.initialize_weights()
        if valence == "positive":
            self.loss_fn = positive_valence_loss
        else:
            self.loss_fn = negative_valence_loss
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.activation = activation

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu') * (1 - self.sparsity))

    def forward(self, x):
        if self.activation == 'sigmoid':
            return torch.sigmoid(self.linear(x))
        elif self.activation == 'relu':
            return torch.relu(self.linear(x))
        elif self.activation == 'linear':
            return self.linear(x)

    def step(self, X, Y):
        """
        Perform one step of optimization

        Args:
            X: torch.Tensor, shape (N, D), input data
            Y: torch.Tensor, shape (N, 1), target valences
            optimizer: torch.optim.Optimizer, optimizer
        """
        self.optimizer.zero_grad()
        outputs = self(X)
        loss = self.loss_fn(outputs, Y)
        loss.backward()
        self.optimizer.step()


class MushroomBody(nn.Module):
    """
    An ensemble compartment that averages the outputs of multiple compartments.

    Args:
        kenyon_cells: int, dimension of input vectors
        valences: list of str, valences of the compartments
        learning_rates: list of float, learning rates of the compartments
        sparsity: float, sparsity of the input data

    Methods:
        forward(x):
            Forward pass of the ensemble compartment.
        group_train(dataloader, epochs):
            Train the ensemble compartment on the given data
        initialize_all_weights():
            Initialize weights of all compartments using Xavier uniform initialization
        lateral_horn(alpha, beta):
            Outputs the beta distribution log prior representing the lateral horn pathway
    """

    def __init__(self, activation, kenyon_cells, valences, learning_rates, sparsity):
        super(MushroomBody, self).__init__()
        assert len(valences) == len(learning_rates), "Number of valences must match number of learning rates"
        assert all(valence in ["positive", "negative"] for valence in valences), "Valences must be 'positive' or 'negative'"
        assert all(isinstance(lr, (float, np.float32, np.float64)) for lr in learning_rates), "Learning rates must be floats"

        num_compartments = len(valences)
        base_size = kenyon_cells // num_compartments
        remainder = kenyon_cells % num_compartments
        compartment_sizes = [base_size] * num_compartments
        compartment_sizes[-1] += remainder

        self.activation = activation
        compartments = []
        for valence, lr, compartment_size in zip(valences, learning_rates, compartment_sizes):
            compartments.append(MBCompartment(compartment_size, valence, lr, sparsity, activation))
        self.compartments = nn.ModuleList(compartments)
        self.initialize_all_weights()

    def initialize_all_weights(self):
        """
        Initialize weights of all compartments using Xavier uniform initialization.
        """
        for compartment in self.compartments:
            compartment.initialize_weights()

    def lateral_horn(self, alpha, beta):
        """
        Outputs the beta distribution log prior representing the lateral horn pathway
        
        Args:
            alpha: float, shape parameter of the beta distribution
            beta: float, shape parameter of the beta distribution

        Returns:
            log_prior: torch.Tensor, shape (1,), log prior
        """
        pass

    def forward(self, x):
        """
        Provides a Bayesian integration of compartment outputs

        Args:
            x: torch.Tensor, shape (N, D), input data

        Returns:
            output: torch.Tensor, shape (N, 1), integrated output
        """

        sum_log_posterior = torch.zeros(x.size(0), 1).to(x.device)

        index = 0

        for compartment in self.compartments:
            outputs = compartment(x[:, index:index + compartment.input_dim])
            index += compartment.input_dim
            if compartment.valence == "positive":
                sum_log_posterior += outputs
            elif compartment.valence == "negative":
                sum_log_posterior -= outputs
        return torch.sign(sum_log_posterior)

    def group_train(self, dataloader, epochs):
        """
        Train the ensemble model on the given data

        Args:
            dataloader: DataLoader, dynamically changing valences
            epochs: int, number of epochs to train the ensemble compartment

        Returns:
            losses: list of float, fraction of incorrect predictions over epochs
        """

        losses = []

        for _ in tqdm(range(epochs)):
            for batch_X, batch_Y in dataloader:
                index = 0
                for i, compartment in enumerate(self.compartments):
                    compartment.step(batch_X[:, index:index + compartment.input_dim], batch_Y)
                    index += compartment.input_dim

            with torch.no_grad():
                X_test, Y_test = dataloader.current_data()
                Y_pred = self(X_test)
                loss = fraction_incorrect(Y_pred, Y_test)
                losses.append(loss)

        return losses

    def run_individual_compartments(self, x):
        """
        Run the compartments individually and return their stacked answers.

        Args:
            x: torch.Tensor, shape (N, D), input data

        Returns:
            stacked_outputs: torch.Tensor, shape (N, num_compartments), outputs of individual compartments
        """
        outputs_list = []
        index = 0

        for compartment in self.compartments:
            outputs = compartment(x[:, index:index + compartment.input_dim])
            outputs_list.append(outputs.squeeze(1))  # Ensure outputs are 1D before stacking
            index += compartment.input_dim

        stacked_outputs = torch.stack(outputs_list, dim=1)
        return stacked_outputs
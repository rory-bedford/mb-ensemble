import torch
import torch.nn as nn

def positive_valence_loss(outputs, targets):
    """
    Compute the binary cross-entropy loss for positive valences.

    Args:
        outputs: torch.Tensor, shape (N, 1), predicted valences
        targets: torch.Tensor, shape (N, 1), target valences

    Returns:
        loss: torch.Tensor, shape (1,), binary cross-entropy loss
    """
    positive_targets = (targets + 1) / 2
    return nn.BCEWithLogitsLoss()(outputs, positive_targets)


def negative_valence_loss(outputs, targets):
    """
    Compute the binary cross-entropy loss for negative valences.

    Args:
        outputs: torch.Tensor, shape (N, 1), predicted valences
        targets: torch.Tensor, shape (N, 1), target valences

    Returns:
        loss: torch.Tensor, shape (1,), binary cross-entropy loss
    """
    negative_targets = (1 - targets) / 2
    return nn.BCEWithLogitsLoss()(outputs, negative_targets)


def fraction_incorrect(outputs, targets):
    """
    Compute the fraction of incorrect predictions.

    Args:
        outputs: torch.Tensor, shape (N, 1), predicted valences
        targets: torch.Tensor, shape (N, 1), target valences

    Returns:
        fraction: float, fraction of incorrect predictions
    """
    incorrect = (outputs != targets).float().mean().item()
    return incorrect
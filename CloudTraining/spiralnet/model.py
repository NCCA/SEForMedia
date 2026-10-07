"""
A small multi layer perceptron for the spiral dataset.
"""

import torch
from torch import nn


class SpiralNet(nn.Module):
    """
    Two hidden layer MLP mapping a 2D point to class logits.

    Attributes
    ----------
        layers : nn.Sequential
            the linear / ReLU stack
    """

    def __init__(
        self, in_features: int = 2, hidden: int = 64, classes: int = 3
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def build_model(hidden: int = 64, classes: int = 3) -> SpiralNet:
    """
    Build a fresh model from a config.

    The config is saved alongside the weights so whoever loads the model (the
    notebook, or another job) can rebuild exactly the same architecture.

    Parameters
    ----------
        hidden : int
            width of the hidden layers
        classes : int
            number of output classes
    """
    return SpiralNet(hidden=hidden, classes=classes)

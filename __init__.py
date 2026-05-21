"""
neural-network-from-scratch
A pure NumPy neural network library built from first principles.
"""

from src.network import NeuralNetwork, Layer
from src.activations import get_activation, list_activations
from src.losses import get_loss, list_losses
from src.optimizers import get_optimizer, list_optimizers
from src import utils

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    "NeuralNetwork",
    "Layer",
    "get_activation",
    "list_activations",
    "get_loss",
    "list_losses",
    "get_optimizer",
    "list_optimizers",
    "utils",
]

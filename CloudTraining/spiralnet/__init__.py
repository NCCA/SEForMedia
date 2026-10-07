"""
Shared code for the cloud training demo.

Everything here is imported by both the notebook and the training job (train.py)
so that the code we prototype with is exactly the code that runs in the container.
"""

from .data import load_dataset, make_spiral, save_dataset, train_val_split
from .model import SpiralNet, build_model
from .training import fit, predict_classes

__all__ = [
    "SpiralNet",
    "build_model",
    "fit",
    "load_dataset",
    "make_spiral",
    "predict_classes",
    "save_dataset",
    "train_val_split",
]

#!/usr/bin/env python

"""
Module to provide utility functions for working with PyTorch.
"""

try:
    import torch
except ImportError:
    print("torch not found")
    print(
        "refer to this URL for install instructions https://pytorch.org/get-started/locally/ "
    )


def get_device() -> torch.device:
    """
    Returns the appropriate device for the current environment.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # mac metal backend
        return torch.device("mps")
    else:
        return torch.device("cpu")


if __name__ == "__main__":
    print(f"found device {get_device()}")

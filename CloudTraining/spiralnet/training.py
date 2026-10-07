"""
The training loop shared by the notebook and the cloud job.
"""

from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _loader(
    points: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(points), torch.from_numpy(labels))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator
    )


def fit(
    model: nn.Module,
    train: tuple[np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray],
    epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 64,
    device: torch.device | None = None,
    seed: int = 1234,
    on_epoch: Callable[[dict], None] | None = None,
) -> list[dict]:
    """
    Train the model and return the per epoch history.

    Nothing in here knows whether it is running in a notebook or a container,
    the caller decides how to report progress via on_epoch.

    Parameters
    ----------
        model : nn.Module
            model to train (moved to device in place)
        train : tuple[np.ndarray, np.ndarray]
            training points and labels
        val : tuple[np.ndarray, np.ndarray]
            validation points and labels
        epochs : int
            number of passes over the training data
        lr : float
            Adam learning rate
        batch_size : int
            mini batch size
        device : torch.device | None
            where to train, the CPU if not given
        seed : int
            seed for the data shuffle
        on_epoch : Callable[[dict], None] | None
            called with the metrics dictionary at the end of each epoch

    Returns
    -------
        list[dict]
            one dictionary per epoch with epoch, train_loss, val_loss and val_accuracy
    """
    device = device or torch.device("cpu")
    model.to(device)
    train_loader = _loader(*train, batch_size, shuffle=True, seed=seed)
    val_points = torch.from_numpy(val[0]).to(device)
    val_labels = torch.from_numpy(val[1]).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for points, labels in train_loader:
            points, labels = points.to(device), labels.to(device)
            optimiser.zero_grad()
            loss = loss_fn(model(points), labels)
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * len(points)
        model.eval()
        with torch.no_grad():
            logits = model(val_points)
            val_loss = loss_fn(logits, val_labels).item()
            val_accuracy = (logits.argmax(dim=1) == val_labels).float().mean().item()
        metrics = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader.dataset),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(metrics)
        if on_epoch is not None:
            on_epoch(metrics)
    return history


def predict_classes(model: nn.Module, points: np.ndarray) -> np.ndarray:
    """
    Run the model on a numpy array of points and return the predicted class for each.

    Parameters
    ----------
        model : nn.Module
            trained model
        points : np.ndarray
            (N, 2) float32 array
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(points.astype(np.float32)).to(device))
    return logits.argmax(dim=1).cpu().numpy()

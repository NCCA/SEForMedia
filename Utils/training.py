#!/usr/bin/env python

"""
Training and evaluation helpers shared by the notebooks.

These are the same train_epoch / evaluate functions I build up in the Free Spoken
Digits and MNIST notebooks, moved here once the loop has been explained so the
later notebooks can concentrate on the model and the data.
"""

from collections.abc import Callable, Iterable

import torch


def argmax_predict(logits: torch.Tensor) -> torch.Tensor:
    """Pick the class with the highest score (class scores live in dimension 1)."""
    return logits.argmax(dim=1)


def binary_predict(logits: torch.Tensor) -> torch.Tensor:
    """A single logit above zero means class 1 (sigmoid(0) is 0.5)."""
    return (logits > 0).float()


class Metrics:
    """
    Running totals for loss and accuracy over one pass of the data.

    Each batch is weighted by the number of predictions it holds (targets.numel()),
    so a short final batch doesn't count as much as a full one. For a classifier
    this is the batch size, for a sequence model it is batch * sequence length.

    Parameters
    ----------
        predict : Callable[[torch.Tensor], torch.Tensor], optional
            turns logits into predicted labels with the same shape as the targets,
            defaults to argmax_predict, use binary_predict for a single-logit model

    Attributes
    ----------
        total_loss : float
            sum of the loss over every prediction seen so far
        correct : int
            number of correct predictions so far
        count : int
            number of predictions so far
    """

    def __init__(
        self, predict: Callable[[torch.Tensor], torch.Tensor] = argmax_predict
    ) -> None:
        self.predict = predict
        self.total_loss, self.correct, self.count = 0.0, 0, 0

    def update(
        self, loss: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """
        Add one batch to the totals.

        Parameters
        ----------
            loss : torch.Tensor
                the mean loss for the batch (the default for PyTorch loss functions)
            logits : torch.Tensor
                the raw model output for the batch
            targets : torch.Tensor
                the labels for the batch
        """
        n = targets.numel()
        self.count += n
        self.total_loss += loss.item() * n
        self.correct += (self.predict(logits) == targets).sum().item()

    def result(self) -> tuple[float, float]:
        """Return (mean loss, accuracy) where accuracy is in the range 0-1."""
        return self.total_loss / self.count, self.correct / self.count


def train_epoch(
    model: torch.nn.Module,
    loader: Iterable,
    loss_fn: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    predict: Callable[[torch.Tensor], torch.Tensor] = argmax_predict,
) -> tuple[float, float]:
    """
    One pass over the training data, updating the weights after each batch.

    Parameters
    ----------
        model : torch.nn.Module
            the model to train, already on device
        loader : Iterable
            yields (inputs, targets) batches, usually a DataLoader. It can be
            wrapped in mo.status.progress_bar to show progress for each batch
        loss_fn : torch.nn.Module
            the loss function, for example nn.CrossEntropyLoss()
        optimiser : torch.optim.Optimizer
            the optimiser holding the model parameters
        device : torch.device
            where to send each batch
        transform : Callable, optional
            applied to the inputs before the forward pass, used for data augmentation
        predict : Callable, optional
            turns logits into labels for the accuracy, see Metrics

    Returns
    -------
        tuple[float, float]
            mean loss and accuracy for the epoch
    """
    model.train()
    metrics = Metrics(predict)
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        if transform is not None:
            inputs = transform(inputs)
        optimiser.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimiser.step()
        metrics.update(loss, logits, targets)
    return metrics.result()


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: Iterable,
    loss_fn: torch.nn.Module,
    device: torch.device,
    predict: Callable[[torch.Tensor], torch.Tensor] = argmax_predict,
) -> tuple[float, float]:
    """
    One pass over the data without gradients or weight updates.

    Parameters
    ----------
        model : torch.nn.Module
            the model to evaluate, already on device
        loader : Iterable
            yields (inputs, targets) batches
        loss_fn : torch.nn.Module
            the loss function
        device : torch.device
            where to send each batch
        predict : Callable, optional
            turns logits into labels for the accuracy, see Metrics

    Returns
    -------
        tuple[float, float]
            mean loss and accuracy
    """
    model.eval()
    metrics = Metrics(predict)
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        metrics.update(loss_fn(logits, targets), logits, targets)
    return metrics.result()


def copy_weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    Take a CPU copy of the model weights, used to keep the best epoch.

    state_dict() returns references to the live tensors, so without the clone
    the "best" weights would keep changing as training carries on.
    """
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }

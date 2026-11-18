#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Checkpoints

    Training a real model can take hours or days, and at some point it will be interrupted (a lab machine reboots, a cluster job hits its time limit, or the laptop lid gets closed). A checkpoint is a snapshot of the training state written to disk, so we can pick up where we left off rather than starting again from scratch.

    In this notebook we will

    1. look at what has to go into a checkpoint, and why the weights on their own are not enough,
    2. train a small classifier, saving a checkpoint every few epochs and whenever the validation loss improves,
    3. load one of those checkpoints back and resume training from it,
    4. save a model for inference, and load saved weights into a model with a different architecture.

    Everything uses [`torch.save`](https://docs.pytorch.org/docs/stable/generated/torch.save.html), [`torch.load`](https://docs.pytorch.org/docs/stable/generated/torch.load.html) and the model's [`state_dict`](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict). The PyTorch tutorial [Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) covers the same principles and is worth reading alongside this.

    First we will define a model and toy dataset to train on.
    """)
    return


@app.cell
def _():
    import io
    import sys

    import matplotlib.pyplot as plt
    import torch
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split
    from torch import nn

    sys.path.append("../")
    from Utils import get_device

    device = get_device()
    print(f"Using {device}")
    return device, io, make_blobs, nn, plt, torch, train_test_split


@app.cell
def _(make_blobs, plt):
    X, y = make_blobs(n_samples=2000, n_features=2, centers=12, random_state=1234)
    print(X.shape, y.shape)
    _fig, _ax = plt.subplots(figsize=(5, 4), layout="constrained")
    _ax.scatter(X[:, 0], X[:, 1], c=y, s=0.4, cmap="tab20")
    _ax.set(xlabel="x0", ylabel="x1", title="12 blobs")
    plt.close(_fig)
    _fig
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can use sklearn's [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) to split the data into training and validation sets, then convert the NumPy arrays to tensors on our device. The validation set is what we use to decide which checkpoint is the best one.
    """)
    return


@app.cell
def _(X, device, torch, train_test_split, y):
    _X_train, _X_val, _y_train, _y_val = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    X_train = torch.tensor(_X_train, dtype=torch.float32, device=device)
    X_val = torch.tensor(_X_val, dtype=torch.float32, device=device)
    y_train = torch.tensor(_y_train, dtype=torch.long, device=device)
    y_val = torch.tensor(_y_val, dtype=torch.long, device=device)
    print(f"train {tuple(X_train.shape)}, validation {tuple(X_val.shape)}")
    return X_train, X_val, y_train, y_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The model

    This is a small fully connected classifier. I make new models with a `build_model()` function rather than a single module-level instance, because we are going to need several copies: one to train, one to resume into, and one to load the inference weights into. Seeding inside `build_model()` means every copy starts from the same random weights.
    """)
    return


@app.cell
def _(device, nn, torch):
    class Classify(nn.Module):
        def __init__(
            self,
            input_features: int = 2,
            output_features: int = 1,
            hidden_size: int = 8,
        ) -> None:
            super().__init__()
            self.linear_layer = nn.Sequential(
                nn.Linear(in_features=input_features, out_features=hidden_size),
                nn.ReLU(),
                nn.Linear(in_features=hidden_size, out_features=hidden_size),
                nn.ReLU(),
                nn.Linear(in_features=hidden_size, out_features=output_features),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear_layer(x)

    def build_model() -> Classify:
        """Make a fresh, seeded Classify model on the current device."""
        torch.manual_seed(1234)
        return Classify(input_features=2, output_features=12).to(device)

    print(build_model())
    return (build_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What goes in a checkpoint

    It is tempting to think that all we need is the model weights, however to carry on training exactly where we stopped we need everything that changes during training:

    | Item | Why we need it |
    | --- | --- |
    | `model.state_dict()` | the learned weights and biases |
    | `optimizer.state_dict()` | the optimiser's own running values (see below) |
    | `epoch` | so we know where to restart the loop and how many epochs are left |
    | `loss` | the validation loss at this point, handy for picking the best checkpoint |
    | `history` | the loss curve so far, so the plots carry on rather than restarting |

    The optimiser state is the one people forget as some optimzers such as a plain SGD has no state, but SGD with momentum keeps a velocity for every parameter, and [Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html) keeps two running averages (`exp_avg` and `exp_avg_sq`) plus a step count. These control how big the next update is. If we throw them away and build a new optimiser, training restarts with a cold optimiser and the loss usually jumps.

    We can see this state by taking a single training step with Adam and looking inside the optimiser.
    """)
    return


@app.cell
def _(X_train, build_model, nn, torch, y_train):
    _model = build_model()
    _optimiser = torch.optim.Adam(_model.parameters(), lr=0.01)
    print(f"Before a step: {_optimiser.state_dict()['state']}")

    _loss = nn.functional.cross_entropy(_model(X_train), y_train)
    _optimiser.zero_grad()
    _loss.backward()
    _optimiser.step()

    _state = _optimiser.state_dict()["state"]
    print(f"After a step there is state for {len(_state)} parameter tensors")
    for _name, _value in _state[0].items():
        print(f"  {_name}: shape {tuple(_value.shape)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before the first step the state is empty; afterwards there is an entry for each of the six parameter tensors (three weight matrices and three bias vectors). The `exp_avg` and `exp_avg_sq` tensors are the same shape as the parameter they belong to, so for a large model the optimiser state can be twice the size of the weights.

    If your training uses randomness that changes between epochs (shuffling in a `DataLoader`, dropout, data augmentation) you would also save the random number generator state with [`torch.get_rng_state()`](https://docs.pytorch.org/docs/stable/generated/torch.get_rng_state.html). Here we train on the whole dataset each epoch with no dropout, so training is deterministic and we can leave it out.

    ## Saving and loading helpers

    The two functions below are the core of this notebook. `save_checkpoint` packs everything from the table into a dictionary and writes it with `torch.save`. The dictionary keys follow the names used in the PyTorch tutorial, which is why they say `optimizer` rather than `optimiser`.
    """)
    return


@app.cell
def _(torch):
    from pathlib import Path

    def save_checkpoint(
        path: Path,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        history: list[tuple[float, float, float]],
    ) -> None:
        """
        Write a training checkpoint to disk.

        Parameters
        ----------
            path : Path
                file to write, the folder must already exist
            model : torch.nn.Module
                the model being trained
            optimiser : torch.optim.Optimizer
                the optimiser being used to train the model
            epoch : int
                the number of epochs completed so far
            loss : float
                the validation loss after this epoch
            history : list[tuple[float, float, float]]
                (train loss, validation loss, validation accuracy) for each epoch so far
        """
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimiser.state_dict(),
                "loss": loss,
                "history": list(history),
            },
            path,
        )

    def load_checkpoint(
        path: Path,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        device: torch.device,
    ) -> dict:
        """
        Load a training checkpoint into a model and optimiser.

        Parameters
        ----------
            path : Path
                the checkpoint file written by save_checkpoint
            model : torch.nn.Module
                a model with the same architecture as the saved one
            optimiser : torch.optim.Optimizer
                an optimiser of the same type, built on model.parameters()
            device : torch.device
                the device to load the tensors on to

        Returns
        -------
            dict
                the whole checkpoint, so the caller can read epoch, loss and history
        """
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint

    return load_checkpoint, save_checkpoint


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are two arguments to `torch.load` worth knowing about.

    `map_location` says which device the tensors should be loaded on to. A checkpoint remembers the device each tensor was saved from, so one written on a CUDA machine in the lab will fail to load on a Mac unless we say `map_location="cpu"` (or `"mps"`). Passing our `device` means the same code works wherever it runs.

    `weights_only=True` tells PyTorch to only unpickle tensors and basic Python types (dictionaries, lists, numbers and strings). A `.pth` file is a Python [pickle](https://docs.python.org/3/library/pickle.html), and loading an arbitrary pickle can run arbitrary code, so this is the safe choice for any file you didn't write yourself. It has been the default since PyTorch 2.6, but I pass it explicitly so it is clear what is happening. Our checkpoint dictionary only holds tensors, numbers and lists, so it loads without any problems.

    Note that the order matters when loading. We build the model and optimiser first (with the optimiser pointing at the new model's parameters) and then load the saved state into both.

    ## Training with checkpoints

    The training loop below is a standard full-batch loop with two additions. Every `save_every` epochs it writes `epoch_NNN.pth`, and whenever the validation loss is the lowest seen so far it overwrites `best.pth`. The first is what you resume from after an interruption, the second is the model you keep at the end. The files go in a `checkpoints` folder next to this notebook, which is cleared at the start of each run so you don't mix up files from different runs. (`*.pth` is in the repository's `.gitignore`, so none of these get committed.)

    Set the number of epochs and the learning rate, then press Train.
    """)
    return


@app.cell
def _(mo):
    training_settings = mo.ui.dictionary(
        {
            "epochs": mo.ui.number(
                start=10, stop=1000, step=10, value=200, label="Epochs"
            ),
            "learning_rate": mo.ui.number(
                start=0.001,
                stop=0.1,
                step=0.001,
                value=0.01,
                label="Learning rate",
            ),
            "save_every": mo.ui.number(
                start=5,
                stop=200,
                step=5,
                value=50,
                label="Save every N epochs",
            ),
        }
    ).form(submit_button_label="Train")
    training_settings
    return (training_settings,)


@app.cell
def _(nn, torch):
    def train_step(
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> tuple[float, float, float]:
        """
        Train for one epoch on the whole training set, then evaluate.

        Returns
        -------
            tuple[float, float, float]
                (train loss, validation loss, validation accuracy)
        """
        model.train()
        loss = nn.functional.cross_entropy(model(X_train), y_train)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        model.eval()
        with torch.inference_mode():
            logits = model(X_val)
            val_loss = nn.functional.cross_entropy(logits, y_val).item()
            val_accuracy = (logits.argmax(dim=1) == y_val).float().mean().item()
        return loss.item(), val_loss, val_accuracy

    return (train_step,)


@app.cell
def _(
    X_train,
    X_val,
    build_model,
    device,
    mo,
    save_checkpoint,
    torch,
    train_step,
    training_settings,
    y_train,
    y_val,
):
    mo.stop(training_settings.value is None, mo.md("Press Train above to begin."))

    checkpoint_dir = mo.notebook_dir() / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    for _old in checkpoint_dir.glob("*.pth"):
        _old.unlink()

    _model = build_model()
    _optimiser = torch.optim.Adam(
        _model.parameters(), lr=training_settings.value["learning_rate"]
    )
    _save_every = training_settings.value["save_every"]

    history = []
    _best_loss = float("inf")
    best_epoch = 0
    for _epoch in mo.status.progress_bar(
        range(1, training_settings.value["epochs"] + 1), title="Training"
    ):
        _train_loss, _val_loss, _val_accuracy = train_step(
            _model, _optimiser, X_train, y_train, X_val, y_val
        )
        history.append((_train_loss, _val_loss, _val_accuracy))

        if _epoch % _save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"epoch_{_epoch:03d}.pth",
                _model,
                _optimiser,
                _epoch,
                _val_loss,
                history,
            )
        if _val_loss < _best_loss:
            _best_loss = _val_loss
            best_epoch = _epoch
            save_checkpoint(
                checkpoint_dir / "best.pth",
                _model,
                _optimiser,
                _epoch,
                _val_loss,
                history,
            )

    mo.md(
        f"Trained for **{len(history)}** epochs on **{device}**. "
        f"Final validation accuracy **{history[-1][2]:.1%}**, "
        f"best validation loss **{_best_loss:.3f}** at epoch **{best_epoch}**."
    )
    return best_epoch, checkpoint_dir, history


@app.cell
def _(best_epoch, history, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3), layout="constrained")
    _epochs = range(1, len(history) + 1)
    _axes[0].plot(_epochs, [_row[0] for _row in history], label="Training")
    _axes[0].plot(_epochs, [_row[1] for _row in history], label="Validation")
    _axes[0].set(xlabel="Epoch", ylabel="Loss")
    _axes[1].plot(_epochs, [_row[2] for _row in history], label="Validation")
    _axes[1].set(xlabel="Epoch", ylabel="Accuracy")
    for _ax in _axes:
        _ax.axvline(best_epoch, color="grey", linestyle="--", label="Best checkpoint")
        _ax.legend()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's see what was written to disk. Each file is small because the model only has a few hundred parameters, but notice that each checkpoint is around three times the size of the weights alone. Most of that is the optimiser state, and the files also grow a little each time as the history gets longer.
    """)
    return


@app.cell
def _(checkpoint_dir, io, torch):
    for _path in sorted(checkpoint_dir.glob("*.pth")):
        _checkpoint = torch.load(_path, map_location="cpu", weights_only=True)
        print(
            f"{_path.name:14} epoch {_checkpoint['epoch']:4d}  "
            f"loss {_checkpoint['loss']:.3f}  {_path.stat().st_size:6d} bytes"
        )

    _checkpoint = torch.load(
        checkpoint_dir / "best.pth", map_location="cpu", weights_only=True
    )
    _buffer = io.BytesIO()
    torch.save(_checkpoint["model_state_dict"], _buffer)
    print(f"\nWeights alone: {_buffer.getbuffer().nbytes} bytes")
    print(f"Keys in a checkpoint: {list(_checkpoint.keys())}")
    print(f"History entries: {len(_checkpoint['history'])}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Resuming training

    Now we pretend training was interrupted. Pick one of the periodic checkpoints and a number of extra epochs, and the cell below will build a brand new model and optimiser, load the checkpoint into them, and carry on from the saved epoch.

    To show why the optimiser state matters it actually resumes twice:

    - **full resume** loads both the model and the optimiser state,
    - **weights only** loads the model weights but builds a fresh Adam optimiser, which is what happens if you only save `model.state_dict()`.

    As training here is deterministic, a full resume should follow the original run exactly (up to tiny floating point differences on a GPU), so you can resume from a checkpoint part way through and compare it with the curve you already have. By default I pick the last checkpoint that still has some of the original run after it.
    """)
    return


@app.cell
def _(checkpoint_dir, history, mo):
    _choices = sorted(_path.name for _path in checkpoint_dir.glob("epoch_*.pth"))
    mo.stop(
        not _choices,
        mo.md(
            "No periodic checkpoints yet, train for longer than *Save every N epochs*."
        ),
    )
    resume_settings = mo.ui.dictionary(
        {
            "checkpoint": mo.ui.dropdown(
                options=_choices,
                value=_choices[-2] if len(_choices) > 1 else _choices[0],
                label="Checkpoint",
            ),
            "epochs": mo.ui.number(
                start=10,
                stop=1000,
                step=10,
                value=min(100, len(history)),
                label="Extra epochs",
            ),
        }
    ).form(submit_button_label="Resume")
    resume_settings
    return (resume_settings,)


@app.cell
def _(
    X_train,
    X_val,
    build_model,
    checkpoint_dir,
    device,
    load_checkpoint,
    mo,
    resume_settings,
    torch,
    train_step,
    training_settings,
    y_train,
    y_val,
):
    mo.stop(
        resume_settings.value is None,
        mo.md("Press Resume above to continue training."),
    )

    _path = checkpoint_dir / resume_settings.value["checkpoint"]
    _learning_rate = training_settings.value["learning_rate"]

    # full resume: model and optimiser state
    _model = build_model()
    _optimiser = torch.optim.Adam(_model.parameters(), lr=_learning_rate)
    _checkpoint = load_checkpoint(_path, _model, _optimiser, device)
    resume_epoch = _checkpoint["epoch"]
    resumed_history = list(_checkpoint["history"])

    # weights only: a fresh optimiser knows nothing about the previous training
    _cold_model = build_model()
    _cold_model.load_state_dict(_checkpoint["model_state_dict"])
    _cold_optimiser = torch.optim.Adam(_cold_model.parameters(), lr=_learning_rate)
    cold_history = list(_checkpoint["history"])

    for _epoch in mo.status.progress_bar(
        range(
            resume_epoch + 1,
            resume_epoch + resume_settings.value["epochs"] + 1,
        ),
        title="Resuming",
    ):
        resumed_history.append(
            train_step(_model, _optimiser, X_train, y_train, X_val, y_val)
        )
        cold_history.append(
            train_step(_cold_model, _cold_optimiser, X_train, y_train, X_val, y_val)
        )

    mo.md(
        f"Resumed from **{_path.name}** (epoch {resume_epoch}, validation loss "
        f"{_checkpoint['loss']:.3f}) and trained to epoch **{len(resumed_history)}**."
    )
    return cold_history, resume_epoch, resumed_history


@app.cell
def _(cold_history, history, plt, resume_epoch, resumed_history):
    _fig, _ax = plt.subplots(figsize=(9, 4), layout="constrained")
    _ax.plot(
        range(1, len(history) + 1),
        [_row[1] for _row in history],
        label="Original run",
        linewidth=4,
        alpha=0.4,
    )
    _resumed_epochs = range(resume_epoch + 1, len(resumed_history) + 1)
    _ax.plot(
        _resumed_epochs,
        [_row[1] for _row in resumed_history[resume_epoch:]],
        label="Full resume",
    )
    _ax.plot(
        _resumed_epochs,
        [_row[1] for _row in cold_history[resume_epoch:]],
        label="Weights only (fresh optimiser)",
        linestyle="--",
    )
    _ax.axvline(resume_epoch, color="grey", linestyle=":", label="Resumed here")
    _ax.set(xlabel="Epoch", ylabel="Validation loss", yscale="log")
    _ax.legend()
    plt.close(_fig)

    _overlap = range(resume_epoch, min(len(history), len(resumed_history)))
    _difference = max(
        (abs(history[_i][1] - resumed_history[_i][1]) for _i in _overlap),
        default=0.0,
    )
    print(f"Largest difference between original and full resume: {_difference:.2e}")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The full resume sits on top of the original run, which is exactly what we want: as far as the model is concerned the interruption never happened. The weights only line starts from the same point but behaves differently, as the fresh Adam optimiser has lost its running averages and its step count. With Adam the first few steps after a reset are roughly `lr` in size for every parameter, regardless of how close to the minimum we are. Late in training, when the model is close to a minimum and the real updates have become small, this kicks it away again and you see a bump in the loss before it settles. Early in training it can look harmless, and sometimes the loss even drops faster, but either way it is no longer the same training run, so your results can't be reproduced. Try resuming from an early and a late checkpoint, and with a learning rate of 0.05, to see how big the bump gets.

    ## Saving for inference

    Once training is finished we don't need the optimiser any more. For inference (or to give the model to someone else) we only save the weights, and load them into a fresh model built from the same class. Here I load the weights out of `best.pth`; in a real project you would usually write them to their own small file with `torch.save(model.state_dict(), "model.pth")`.

    Remember to call `model.eval()` before making predictions. It doesn't matter for this model, but layers such as dropout and batch normalisation behave differently in training and evaluation mode.
    """)
    return


@app.cell
def _(X_val, build_model, checkpoint_dir, device, torch, y_val):
    _checkpoint = torch.load(
        checkpoint_dir / "best.pth", map_location=device, weights_only=True
    )
    best_model = build_model()
    best_model.load_state_dict(_checkpoint["model_state_dict"], strict=True)
    best_model.eval()
    with torch.inference_mode():
        _accuracy = (best_model(X_val).argmax(dim=1) == y_val).float().mean().item()
    print(
        f"Best model from epoch {_checkpoint['epoch']}: validation accuracy {_accuracy:.1%}"
    )
    return (best_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `strict=True` option (the default) makes sure the keys in the state dictionary match the keys in the model exactly. If they don't match an error is raised, which is usually what you want, as it catches loading the wrong file into the wrong model.

    ## Full model save and load

    It is also possible to save the whole model object rather than its `state_dict`. This pickles the model, so you must still have the class definition available when you load it back, and it can only be loaded with `weights_only=False`. That is the unsafe mode described above, so only do this with files you have written yourself. To avoid writing yet another file I save into an in-memory [`io.BytesIO`](https://docs.python.org/3/library/io.html#io.BytesIO) buffer, which `torch.save` and `torch.load` accept in place of a file name.
    """)
    return


@app.cell
def _(X_val, best_model, io, torch, y_val):
    _buffer = io.BytesIO()
    torch.save(best_model, _buffer)
    _buffer.seek(0)

    try:
        torch.load(_buffer, weights_only=True)
    except Exception as _error:
        print(
            f"weights_only=True refuses the full model:\n  {str(_error).splitlines()[0]}\n"
        )

    _buffer.seek(0)
    _full_model = torch.load(_buffer, weights_only=False)
    with torch.inference_mode():
        _accuracy = (_full_model(X_val).argmax(dim=1) == y_val).float().mean().item()
    print(
        f"{type(_full_model).__name__} loaded with weights_only=False: accuracy {_accuracy:.1%}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Loading into a different architecture

    Because a `state_dict` is just a dictionary of named tensors, we can load it into a model that is not quite the same as the one that saved it. Below is a new model with ELU activations and a different set of layer sizes.
    """)
    return


@app.cell
def _(best_model, device, nn, torch):
    class ClassifyELU(nn.Module):
        def __init__(
            self,
            input_features: int = 2,
            output_features: int = 1,
            hidden_size: int = 8,
        ) -> None:
            super().__init__()
            self.linear_layer = nn.Sequential(
                nn.Linear(in_features=input_features, out_features=8),
                nn.ELU(),
                nn.Linear(in_features=8, out_features=hidden_size // 2),
                nn.ELU(),
                nn.Linear(in_features=hidden_size // 2, out_features=hidden_size // 2),
                nn.ELU(),
                nn.Linear(in_features=hidden_size // 2, out_features=output_features),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear_layer(x)

    model_elu = ClassifyELU(input_features=2, output_features=12).to(device)
    try:
        model_elu.load_state_dict(best_model.state_dict(), strict=False)
    except RuntimeError as _error:
        print(f"Failed to load the model:\n{_error}")
    return (model_elu,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Even with `strict=False` this fails. `strict=False` only forgives keys that are missing or unexpected; it doesn't forgive a key that exists in both models with a different shape. We can get round this by copying the parameters across one at a time, keeping only those whose name and shape match.
    """)
    return


@app.cell
def _(best_model, model_elu, torch):
    def transfer_weights(
        source_state: dict[str, torch.Tensor], new_model: torch.nn.Module
    ) -> None:
        """
        Copy every parameter whose name and shape match into new_model.

        Parameters
        ----------
            source_state : dict[str, torch.Tensor]
                a state_dict, for example from a checkpoint
            new_model : torch.nn.Module
                the model to copy the matching parameters into
        """
        new_state = new_model.state_dict()
        for name, param in source_state.items():
            if name not in new_state:
                print(f"Skipping {name} as it is not in the new model.")
            elif new_state[name].shape != param.shape:
                print(f"Skipping {name} due to size mismatch.")
            else:
                new_state[name].copy_(param)
                print(f"Copied {name}")
        new_model.load_state_dict(new_state)

    transfer_weights(best_model.state_dict(), model_elu)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Only the first layer matches, so that is all we keep. The rest of the new model starts from its random weights and would need training.

    ## Use cases for changing architectures

    - Fine-tuning: add or replace layers to adapt the model for a new task.
    - Model compression: prune or reduce the number of layers to make the model lighter.
    - Transfer learning: load pre-trained weights into a similar architecture (see the PreTrainedModels notebooks).

    # Exercises

    1. Change *Save every N epochs* to 10 and train again. Resume from `epoch_010.pth` and from the last periodic checkpoint. Is the bump in the weights only curve bigger early or late in training? Why do you think that is?
    2. Swap Adam for `torch.optim.SGD` with no momentum in the training and resume cells. Before you run it, predict what the weights only curve will look like. Then try `momentum=0.9`.
    3. The resume cell doesn't save any checkpoints of its own. Add the same `save_every` and `best.pth` logic to it, so that a resumed run can itself be resumed.
    4. Add the random number generator state (`torch.get_rng_state()`) to the checkpoint, and restore it with `torch.set_rng_state()` in `load_checkpoint`. Then add a `nn.Dropout(0.2)` to `Classify` and check the full resume still matches the original run.
    5. Add the date and time of the save (from `datetime.datetime.now()`) to the dictionary in `save_checkpoint` and train again. What happens when the checkpoint is loaded with `weights_only=True`, and how could you store the date so it still loads?

    # Conclusion

    A checkpoint for resuming training needs more than the weights: the optimiser state, the epoch and any history you want to keep all go in one dictionary, saved every few epochs and whenever the validation loss improves. When loading, build the model and optimiser first, then load into them with `map_location` set to the current device and `weights_only=True`. For inference only the `state_dict` is needed, and because it is just named tensors it can be loaded selectively into a different architecture.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

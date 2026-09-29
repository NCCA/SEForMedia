import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import torch

    # grab the cell from notebook one to get the data directory
    from FSDDMarimoPt1 import download_digits
    import matplotlib.pyplot as plt

    output, definitions = download_digits.run()
    dataset_path = definitions["dataset_path"]
    print(dataset_path)
    return Path, dataset_path, mo, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Free Spoken Digits Dataset Part 3 Training

    Now we have a dataloader for our train and validation data we can build our network and train it.


    We will build a [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) that treats the mel spectrogram as a single-channel image. It learns patterns across time and frequency, then uses those patterns to predict which digit was spoken. The colours in our plot are only for display. The network receives the underlying numbers.

    ## Tensor Shape

    With our preprocessing, each recording becomes a tensor of shape [1, 64, 101]:

                        101 time frames →
                     ┌───────────────────────┐
                     │                       │
        64 mel bands │  Each value describes │
                  ↑  │  normalised log-power │
                     │  at a band and time   │
                     └───────────────────────┘

                      1 input channel

    At 8 kHz, a hop of 80 samples places frames 10 ms apart. The 256-sample analysis window covers 32 ms. The default centred spectrogram gives us 101 frames for 8,000 samples.
    The loader stacks recordings into batches:

    |batch size| channels| mel bands| time frames|
    |-------|------|------|------|
    |    B|         1|        64     |    101    |

    We will then build our CNN as follows
    """)
    return


@app.cell
def _(mo):
    mo.mermaid(
        """
        flowchart TD
        A["Mel spectrograms<br/>B × 1 × 64 × 101"]
        B["Conv2d: 1 → 16 channels<br/>3 × 3 kernels, padding 1<br/>B × 16 × 64 × 101"]
        C["ReLU, then MaxPool2d 2 × 2<br/>B × 16 × 32 × 50"]
        D["Conv2d: 16 → 32 channels<br/>3 × 3 kernels, padding 1<br/>B × 32 × 32 × 50"]
        E["ReLU, then MaxPool2d 2 × 2<br/>B × 32 × 16 × 25"]
        F["AdaptiveAvgPool2d 4 × 5<br/>B × 32 × 4 × 5"]
        G["Flatten<br/>B × 640"]
        H["Linear: 640 → 10<br/>B × 10 digit scores"]
        A --> B --> C --> D --> E --> F --> G --> H
        """,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can implement this using the torch.nn module as follows (again this is loaded in from a file so I can re-use it later).
    """)
    return


@app.cell
def _(Path, mo):
    _source = Path("DigitCNN.py").read_text(encoding="utf-8")
    mo.md(f"```python\n{_source}\n```")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As before we can load in our data and prepare it ready for the model to train on. This was explained in the previous notebook, we will generate a data loader for our validation and test data which will be used for each epoch to train on.
    """)
    return


@app.cell
def _(Path, dataset_path, torch):
    from torch.utils.data import DataLoader

    from FSDDataLoader import split_recordings, SpokenDigits

    root = Path(f"{dataset_path}/recordings")
    recordings = sorted(root.glob("*.wav"))

    train_paths, validation_paths, test_paths = split_recordings(recordings)
    train_data = SpokenDigits(train_paths)
    validation_data = SpokenDigits(validation_paths)
    test_data = SpokenDigits(test_paths)
    # gnerate DataLoader for batches
    validation_loader = DataLoader(validation_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)
    # Note this is set to shuffle
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )

    # Find the device and set manual seed
    torch.manual_seed(42)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return device, test_data, test_loader, train_loader, validation_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training and Evauluation

    For ease we can generate functions that train a single epoch and evaluate the model. As both need to generate metrics for loss we can use a simple class for this. The code is outlined below, with the class _Metrics used for the purpose. Note the leading _ to denote a "internal" class not used outside of the module (see PEP8).
    """)
    return


@app.cell
def _(torch):
    class _Metrics:
        """Running totals for sample-weighted loss and accuracy."""

        def __init__(self) -> None:
            self.total_loss, self.correct, self.count = 0.0, 0, 0

        def update(
            self,
            loss: torch.Tensor,
            logits: torch.Tensor,
            labels: torch.Tensor,
        ) -> None:
            n = labels.size(0)
            self.count += n
            self.total_loss += loss.item() * n
            self.correct += (logits.argmax(1) == labels).sum().item()

        def result(self) -> tuple[float, float]:
            return self.total_loss / self.count, self.correct / self.count

    def train_epoch(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        device: torch.device,
    ) -> tuple[float, float]:
        """One pass over the data with weight updates; returns (loss, accuracy)."""
        model.train()
        metrics = _Metrics()
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimiser.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimiser.step()
            metrics.update(loss, logits, labels)
        return metrics.result()

    @torch.inference_mode()
    def evaluate(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device,
    ) -> tuple[float, float]:
        """One pass over the data without gradients; returns (loss, accuracy)."""
        model.eval()
        metrics = _Metrics()
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            metrics.update(loss_fn(logits, labels), logits, labels)
        return metrics.result()

    return evaluate, train_epoch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You will see that the evaluate function uses evaluate uses the decorator ```@torch.inference_mode()``` rather than ```set_grad_enabled(False)```. It's a stricter, slightly faster form of no_grad, and it's safe here because the outputs never go back into autograd. Applied as a decorator, the whole function is visibly gradient-free before you read the loop.

    ## Training the CNN

    The loader stacks features into `[batch, 1, 64, 101]`. The CNN learns
    small patterns in time and frequency and returns ten scores (logits).
    `CrossEntropyLoss` takes logits directly, so there is no softmax in the
    model. Adam adjusts the weights after each batch.

    I use a fixed seed and shuffle only the training loader. Results can still
    vary between devices. Features are cached in memory after their first
    decode; the first epoch will therefore take longer.

    Choose the settings and press **Train**. Submitting again starts a fresh
    model. We use CUDA or Apple MPS when available, otherwise CPU.
    """)
    return


@app.cell
def _(mo):
    training_settings = mo.ui.dictionary(
        {
            "epochs": mo.ui.number(start=1, stop=100, value=10, label="Epochs"),
            "learning_rate": mo.ui.number(
                start=0.0001,
                stop=0.1,
                step=0.0001,
                value=0.001,
                label="Learning rate",
            ),
        }
    ).form(submit_button_label="Train")
    training_settings
    return (training_settings,)


@app.cell
def _(
    device,
    evaluate,
    mo,
    torch,
    train_epoch,
    train_loader,
    training_settings,
    validation_loader,
):
    from DigitCNN import DigitCNN

    mo.stop(training_settings.value is None, mo.md("Press Train above to begin."))

    model = DigitCNN().to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    _optimiser = torch.optim.Adam(
        model.parameters(), lr=training_settings.value["learning_rate"]
    )
    history = []
    _best_loss = float("inf")
    best_epoch = 0
    for _epoch in mo.status.progress_bar(
        range(training_settings.value["epochs"]), title="Training"
    ):
        _train_loss, _train_accuracy = train_epoch(
            model, train_loader, loss_fn, _optimiser, device
        )
        _val_loss, _val_accuracy = evaluate(model, validation_loader, loss_fn, device)
        history.append((_train_loss, _val_loss, _train_accuracy, _val_accuracy))
        if _val_loss < _best_loss:
            _best_loss = _val_loss
            best_epoch = _epoch + 1
            _best_weights = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        print(
            f"Epoch {_epoch + 1}: train loss {_train_loss:.3f}, validation loss {_val_loss:.3f}, validation accuracy {_val_accuracy:.1%}"
        )
    model.load_state_dict(_best_weights)
    mo.md(
        f"Training finished on **{device}**. Restored weights from epoch **{best_epoch}**."
    )
    return best_epoch, history, loss_fn, model


@app.cell
def _(history, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3), layout="constrained")
    _epochs = range(1, len(history) + 1)
    for _index, _name in enumerate(["Loss", "Accuracy"]):
        for _offset, _label in enumerate(["Training", "Validation"]):
            _axes[_index].plot(
                _epochs,
                [_row[2 * _index + _offset] for _row in history],
                label=_label,
            )
        _axes[_index].set(xlabel="Epoch", ylabel=_name)
        _axes[_index].legend()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Testing and listening to a prediction

    Validation helped us choose the weights. Now we evaluate those weights on
    the separate test recordings. The confusion matrix shows where one digit
    is mistaken for another; rows are the true labels and columns are predictions.
    If we repeatedly tune the model using this test score, it stops being an
    independent check.
    """)
    return


@app.cell
def _(device, evaluate, loss_fn, mo, model, plt, test_loader, torch):
    test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    with torch.inference_mode():
        for _features, _labels in test_loader:
            _predictions = model(_features.to(device)).argmax(1).cpu()
            confusion += torch.bincount(
                _labels * 10 + _predictions, minlength=100
            ).reshape(10, 10)
    _fig, _ax = plt.subplots(figsize=(6, 5), layout="constrained")
    _image = _ax.imshow(confusion.numpy(), cmap="Blues")
    for _row in range(10):
        for _column in range(10):
            _ax.text(
                _column,
                _row,
                str(confusion[_row, _column].item()),
                ha="center",
                va="center",
                color="red",
            )
    _ax.set(
        xlabel="Predicted digit",
        ylabel="True digit",
        xticks=range(10),
        yticks=range(10),
    )
    _fig.colorbar(_image, ax=_ax)
    plt.close(_fig)
    mo.vstack(
        [
            mo.md(
                f"Test accuracy: **{test_accuracy:.1%}**, loss: **{test_loss:.3f}**."
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(mo, test_data):
    recording_index = mo.ui.slider(
        start=0, stop=len(test_data) - 1, value=0, label="Test recording"
    )
    recording_index
    return (recording_index,)


@app.cell
def _(device, mo, model, recording_index, test_data, torch):
    _feature, _label = test_data[recording_index.value]
    with torch.inference_mode():
        _probabilities = model(_feature.unsqueeze(0).to(device)).softmax(1)[0].cpu()
    _prediction = int(_probabilities.argmax())
    mo.vstack(
        [
            mo.audio(test_data.paths[recording_index.value].read_bytes()),
            mo.md(
                f"True digit: **{_label}**. Prediction: **{_prediction}** ({_probabilities[_prediction]:.1%} softmax score). This score is not a calibrated probability of being correct."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    ckpt_path = mo.ui.text(value="digits.pt", label="checkpoint file")
    save_btn = mo.ui.run_button(label="Save model")
    mo.hstack([ckpt_path, save_btn], justify="start")
    return ckpt_path, save_btn


@app.cell
def _(
    best_epoch,
    ckpt_path,
    history,
    mo,
    model,
    save_btn,
    torch,
    training_settings,
):
    mo.stop(not save_btn.value)
    _path = mo.notebook_dir() / ckpt_path.value
    torch.save(
        {
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "best_epoch": best_epoch,
            "history": history,
            "settings": dict(training_settings.value),
            "torch_version": torch.__version__,
        },
        _path,
    )
    mo.md(f"Saved to `{_path}` (weights from epoch {best_epoch}).")
    return


if __name__ == "__main__":
    app.run()

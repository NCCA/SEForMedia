#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Point Clouds for Machine Learning, Part 4: classification

    We will train the classifier from [Part 3](PointCloudForMLPart3PointNet.py),
    select weights using validation loss, then inspect held-out predictions.
    The generated data lets us test the full loop before using ModelNet10.

    The figures use Plotly. Drag a 3D view to rotate it and scroll to zoom.
    Camera controls do not change the input coordinates or repeat training.
    Run this notebook with `uv run --with plotly marimo edit` followed by its path.

    By the end we should be able to train from logits, keep the splits separate,
    and explain a confusion matrix and per-class accuracy. This builds on
    [PyTorch for ML](../PyTorchForML/README.md).
    """)
    return


@app.cell
def _():
    import inspect
    import sys
    from pathlib import Path
    import marimo as mo
    import numpy as np
    import torch
    from torch import nn

    # This also works when marimo starts from the notebook's own folder.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pointcloud as pc

    return (inspect, mo, nn, np, pc, torch, Path)


@app.cell
def _(mo):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ModuleNotFoundError:
        mo.stop(
            True,
            mo.md(
                "Plotly is missing. Add --with plotly to the uv run command for this notebook."
            ),
        )
    return go, make_subplots


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data and training controls

    Use generated sphere/cube/rod surfaces for a quick CPU experiment, or enter the
    path to an extracted [ModelNet10](https://modelnet.cs.princeton.edu/) directory.
    The expected layout is `ModelNet10/chair/train/*.off` and `chair/test/*.off`,
    with matching folders for every category. Download and extract the dataset yourself.
    For ModelNet run marimo with `uv run --with plotly --with 'trimesh>=4.6' marimo edit ...`.

    We reserve 20% of each training category for validation, keeping each source
    mesh in only one split. The official test directory is separate. Generated
    training and test shapes use different seeds. Sampling is cached for this run;
    we do not resample a mesh and pretend it is a new independent object.

    The limit is **per category**, so a short run remains balanced. Zero loads all
    meshes. The small default budget demonstrates the loop; it is not a benchmark.
    The form controls data loading and training together. No download happens automatically.
    """)
    return


@app.cell
def _(mo):
    training_form = mo.ui.dictionary(
        {
            "root": mo.ui.text(
                value="", label="ModelNet root (blank for generated shapes)"
            ),
            "points": mo.ui.number(
                32, 2048, value=128, step=32, label="Points per shape"
            ),
            "limit": mo.ui.number(
                0, 1000, value=24, label="Training meshes per category (0 = all)"
            ),
            "epochs": mo.ui.number(1, 200, value=12, label="Epochs"),
            "lr": mo.ui.number(
                0.0001, 0.1, value=0.003, step=0.0001, label="Learning rate"
            ),
            "seed": mo.ui.number(0, 10000, value=42, label="Seed"),
        }
    ).form(submit_button_label="Load data and train")
    training_form
    return (training_form,)


@app.cell
def _(training_form, mo, pc):
    mo.stop(
        training_form.value is None, mo.md("Submit the form to load data and train.")
    )
    settings = training_form.value
    try:
        if settings["root"].strip():
            all_x, all_y, names = pc.load_modelnet(
                settings["root"],
                "train",
                int(settings["points"]),
                int(settings["limit"]),
                int(settings["seed"]),
            )
            test_x, test_y, test_names = pc.load_modelnet(
                settings["root"],
                "test",
                int(settings["points"]),
                0,
                int(settings["seed"]) + 1,
            )
            if names != test_names:
                raise ValueError("Train and test categories must agree")
        else:
            all_x, all_y, names = pc.make_shapes(
                max(4, int(settings["limit"])),
                int(settings["points"]),
                int(settings["seed"]),
            )
            test_x, test_y, _test_names = pc.make_shapes(
                12, int(settings["points"]), int(settings["seed"]) + 1000
            )
        train_indices, val_indices = pc.split_indices(all_y, int(settings["seed"]))
    except (ModuleNotFoundError, OSError, ValueError) as _error:
        mo.stop(
            True,
            mo.md(
                f"Data could not be loaded: {_error}. Check the path and optional trimesh dependency."
            ),
        )
    train_x, train_y = all_x[train_indices], all_y[train_indices]
    val_x, val_y = all_x[val_indices], all_y[val_indices]
    print(
        "Training:",
        tuple(train_x.shape),
        "validation:",
        tuple(val_x.shape),
        "test:",
        tuple(test_x.shape),
    )
    print("Categories:", names)
    return (settings, train_x, train_y, val_x, val_y, test_x, test_y, names)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train, validate and keep the best weights

    Cross entropy expects raw logits and integer class indices. We use small batches
    and a fresh model for every submitted run. Validation is evaluated without
    gradients after each epoch; it selects the checkpoint. Test data is only used
    below, after selection. All calculations here use CPU, so the lesson does not
    need an accelerator. Changing display controls does not repeat training.
    """)
    return


@app.cell
def _(settings, train_x, train_y, val_x, val_y, names, pc, torch, nn, mo):
    with torch.random.fork_rng():
        torch.manual_seed(int(settings["seed"]))
        trained_model = pc.PointNet(len(names))
    _optimizer = torch.optim.Adam(trained_model.parameters(), lr=float(settings["lr"]))
    _loss_fn = nn.CrossEntropyLoss()
    _generator = torch.Generator().manual_seed(int(settings["seed"]))
    history = []
    _best_loss = float("inf")
    for _epoch in mo.status.progress_bar(
        range(int(settings["epochs"])), title="Training classifier"
    ):
        trained_model.train()
        _order = torch.randperm(len(train_x), generator=_generator)
        _total_loss = 0.0
        for _indices in _order.split(16):
            _optimizer.zero_grad()
            _loss = _loss_fn(trained_model(train_x[_indices]), train_y[_indices])
            _loss.backward()
            _optimizer.step()
            _total_loss += _loss.item() * len(_indices)
        trained_model.eval()
        with torch.no_grad():
            _val_logits = torch.cat(
                [trained_model(_batch) for _batch in val_x.split(32)]
            )
            _val_loss = _loss_fn(_val_logits, val_y).item()
            _val_accuracy = (_val_logits.argmax(1) == val_y).float().mean().item()
        history.append(
            {
                "epoch": _epoch + 1,
                "train_loss": _total_loss / len(train_x),
                "val_loss": _val_loss,
                "val_accuracy": _val_accuracy,
            }
        )
        if _val_loss < _best_loss:
            _best_loss = _val_loss
            _best_state = {
                name: value.detach().clone()
                for name, value in trained_model.state_dict().items()
            }
    trained_model.load_state_dict(_best_state)
    trained_model.eval()
    print("Best validation loss:", round(_best_loss, 4))
    return (trained_model, history)


@app.cell
def _(history, go, make_subplots, mo):
    _epochs = [item["epoch"] for item in history]
    classification_curve = make_subplots(
        rows=1, cols=2, subplot_titles=("Learning curves", "Validation accuracy")
    )
    for _key, _name in [("train_loss", "Training"), ("val_loss", "Validation")]:
        classification_curve.add_trace(
            go.Scatter(
                x=_epochs, y=[item[_key] for item in history], mode="lines", name=_name
            ),
            row=1,
            col=1,
        )
    classification_curve.add_trace(
        go.Scatter(
            x=_epochs,
            y=[item["val_accuracy"] for item in history],
            mode="lines",
            name="Validation accuracy",
        ),
        row=1,
        col=2,
    )
    classification_curve.update_xaxes(title_text="Epoch")
    classification_curve.update_yaxes(title_text="Cross entropy", row=1, col=1)
    classification_curve.update_yaxes(title_text="Accuracy", range=[0, 1], row=1, col=2)
    classification_curve.update_layout(height=400)
    mo.ui.plotly(classification_curve, config={"displaylogo": False})
    return (classification_curve,)


@app.cell
def _(trained_model, test_x, test_y, names, torch, mo, go):
    with torch.no_grad():
        test_logits = torch.cat([trained_model(_batch) for _batch in test_x.split(32)])
    predictions = test_logits.argmax(1)
    confusion = torch.bincount(
        test_y * len(names) + predictions, minlength=len(names) ** 2
    ).reshape(len(names), len(names))
    print("Test accuracy:", (predictions == test_y).float().mean().item())
    print("Test shapes:", len(test_y))
    _rows = [
        {
            "class": name,
            "shapes": int(confusion[i].sum()),
            "accuracy": float(confusion[i, i] / confusion[i].sum().clamp_min(1)),
        }
        for i, name in enumerate(names)
    ]
    confusion_plot = go.Figure(
        go.Heatmap(
            z=confusion.numpy(),
            x=names,
            y=names,
            colorscale="Blues",
            colorbar=dict(title="Shape count"),
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Shapes: %{z}<extra></extra>",
        )
    )
    confusion_plot.update_layout(
        title="Held-out confusion matrix",
        height=550,
        xaxis=dict(title="Predicted class"),
        yaxis=dict(title="Actual class", autorange="reversed"),
    )
    mo.vstack(
        [
            mo.ui.table(_rows),
            mo.ui.plotly(confusion_plot, config={"displaylogo": False}),
        ]
    )
    return predictions, test_logits, confusion_plot


@app.cell
def _(mo, test_x):
    example_index = mo.ui.slider(
        0, len(test_x) - 1, value=0, label="Test shape", show_value=True
    )
    example_index
    return (example_index,)


@app.cell
def _(example_index, predictions, test_x, test_y, names, pc, go, mo):
    _index = example_index.value
    print(
        "Actual:",
        names[int(test_y[_index])],
        "predicted:",
        names[int(predictions[_index])],
    )
    print(
        "Misclassified indices:", (predictions != test_y).nonzero().flatten().tolist()
    )
    prediction_view = pc.cloud_figure(
        test_x[_index].numpy(),
        f"Actual: {names[int(test_y[_index])]} — predicted: {names[int(predictions[_index])]}",
    )
    mo.ui.plotly(prediction_view, config={"displaylogo": False, "scrollZoom": True})
    return (prediction_view,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save a reproducible checkpoint

    The checkpoint includes the class names, sampling settings and validation
    history. It is written only when we press Save, under this notebook folder's
    `outputs` directory. Reusing the button overwrites the previous classifier.
    Weights do not replace recording the data version and software environment.
    """)
    return


@app.cell
def _(mo):
    save_button = mo.ui.run_button(label="Save classifier checkpoint")
    save_button
    return (save_button,)


@app.cell
def _(save_button, mo, Path, torch, trained_model, names, settings, history):
    mo.stop(not save_button.value)
    _output = Path(__file__).resolve().parent / "outputs"
    _output.mkdir(exist_ok=True)
    torch.save(
        {
            "state_dict": trained_model.state_dict(),
            "classes": names,
            "settings": settings,
            "history": history,
        },
        _output / "classifier.pt",
    )
    print("Saved:", _output / "classifier.pt")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Train the flattened baseline from Part 2 on the same splits and compare it fairly.
    2. Find one confused class pair. What geometric evidence separates them?
    3. Increase the point count whilst keeping the source meshes and split seed fixed.
    4. Explain why selecting an epoch using test accuracy would invalidate the test.

    [Part 5](PointCloudForMLPart5Robustness.py) compares the same training recipe with and without rotation augmentation.
    """)
    return


if __name__ == "__main__":
    app.run()

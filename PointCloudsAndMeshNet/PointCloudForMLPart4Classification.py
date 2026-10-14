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

    By the end we should be able to train from logits, keep the splits separate,
    and explain a confusion matrix and per-class accuracy.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path
    import marimo as mo
    import torch
    from torch import nn

    # This also works when marimo starts from the notebook's own folder.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pointcloud as pc
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return Path, go, make_subplots, mo, nn, pc, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data and training controls

    We can train using generated sphere, cube and rod surfaces, or sample point clouds from meshes in ModelNet10. Start with the generated shapes for a quick CPU experiment. They need no downloaded assets and let us check the training loop before trying a larger dataset.

    ### Download and select ModelNet10

    ModelNet10 contains 3D CAD meshes from ten object categories, with training and test splits supplied in the archive. These are source meshes from which we will sample points, rather than pretrained network weights. The meshes use the `.off` format.

    To use them:

    1. Open the [official ModelNet download page](https://modelnet.cs.princeton.edu/download.html).
    2. Select **ModelNet10.zip** under **Download 10-Class Orientation-aligned Subset**.
    3. Extract the archive to a local folder.
    4. Enter the path to the extracted `ModelNet10` folder in the notebook’s dataset-path field.

    Select the folder containing the category directories, rather than the ZIP file, an individual mesh or a category’s `train` folder. The expected structure is:

    ```text
    ModelNet10/
    ├── chair/
    │   ├── train/
    │   │   └── *.off
    │   └── test/
    │       └── *.off
    ├── bed/
    │   ├── train/
    │   │   └── *.off
    │   └── test/
    │       └── *.off
    └── ... other category folders
    ```

    For example, if we extract the archive into a folder named `datasets`, the path we enter should end in `datasets/ModelNet10`. An absolute path avoids ambiguity about which working directory the notebook uses.

    The notebook does not download the dataset automatically. We supply the extracted files ourselves.

    ### Keep training, validation and testing separate

    For ModelNet10, we reserve 20% of each training category for validation. The remaining training examples are used to update the network’s weights. Validation examples let us inspect its behaviour on objects it has not trained on, whilst the official `test` directories remain a separate set for evaluation.

    We split by **source mesh**, so each object belongs to only one split. Sampling two clouds from the same mesh does not give us two independent objects. Putting one in training and the other in validation would allow the network to see the validation object’s geometry during training, making the results misleading.

    Sampling is cached for the current run, so we reuse each mesh’s sampled cloud during training. For the generated dataset, training and test examples use different random seeds.

    ### Choose a small experiment first

    The mesh limit applies **per category**. A limit of 20 requests up to 20 meshes from each category, rather than 20 meshes across the entire dataset. This keeps a short run balanced when each category has enough examples. A limit of zero loads all available meshes.

    The mesh limit and point count control different things: the limit sets how many objects we load, whilst the point count sets how many surface samples represent each object. Increasing either adds work; increasing the point count does not increase the number of independent training objects.

    Set the data and training options, then submit the form to load the data and start training. Begin with the small default budget to inspect the data flow and training results before increasing it. This simplified model and short run demonstrate the process; their scores should not be treated as a ModelNet benchmark.
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
                0,
                1000,
                value=24,
                label="Training meshes per category (0 = all)",
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
def _(mo, pc, training_form):
    mo.stop(
        training_form.value is None,
        mo.md("Submit the form to load data and train."),
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
    return names, settings, test_x, test_y, train_x, train_y, val_x, val_y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train, validate and keep the best weights

    Each submitted run starts with a fresh model and newly initialised weights. We train on small batches of clouds, using cross-entropy loss to compare the model’s predictions with the correct labels.

    Cross entropy expects **raw logits**: one score per class, before applying softmax. The target for each cloud is an integer class index. For a batch of \(B\) clouds and \(C\) classes, the logits have shape `(B, C)` and the targets have shape `(B,)`. We pass these directly to the loss function, which performs the required probability calculation internally.

    For each training batch, we clear the previous gradients, calculate the logits and loss, then backpropagate and update the weights. One *epoch* completes a pass through the training set.

    After each epoch, we evaluate the validation set with the model in evaluation mode and gradient calculation disabled. These examples do not update the weights. Instead, their results determine which checkpoint we keep: a saved copy of the weights from the epoch that performs best according to our validation criterion. The final epoch is not necessarily the best one, because performance on unseen examples can worsen even whilst training loss continues to fall.

    We restore the selected weights before evaluating the test set below. Keeping test data out of checkpoint selection gives us a separate assessment of the chosen model. If we repeatedly change our settings in response to test scores, the test set also becomes part of our selection process.

    All calculations run on the CPU, so the lesson does not require an accelerator. Submitting the training form starts a new run; changing the display controls lets us inspect the existing results without repeating training.
    """)
    return


@app.cell
def _(mo, names, nn, pc, settings, torch, train_x, train_y, val_x, val_y):
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
    return history, trained_model


@app.cell
def _(go, history, make_subplots, mo):
    _epochs = [item["epoch"] for item in history]
    classification_curve = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Learning curves", "Validation accuracy"),
    )
    for _key, _name in [
        ("train_loss", "Training"),
        ("val_loss", "Validation"),
    ]:
        classification_curve.add_trace(
            go.Scatter(
                x=_epochs,
                y=[item[_key] for item in history],
                mode="lines",
                name=_name,
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
    return


@app.cell
def _(go, mo, names, test_x, test_y, torch, trained_model):
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
    return (predictions,)


@app.cell
def _(mo, test_x):
    example_index = mo.ui.slider(
        0, len(test_x) - 1, value=0, label="Test shape", show_value=True
    )
    example_index
    return (example_index,)


@app.cell
def _(example_index, mo, names, pc, predictions, test_x, test_y):
    _index = example_index.value
    print(
        "Actual:",
        names[int(test_y[_index])],
        "predicted:",
        names[int(predictions[_index])],
    )
    print(
        "Misclassified indices:",
        (predictions != test_y).nonzero().flatten().tolist(),
    )
    prediction_view = pc.cloud_figure(
        test_x[_index].numpy(),
        f"Actual: {names[int(test_y[_index])]} — predicted: {names[int(predictions[_index])]}",
    )
    mo.ui.plotly(prediction_view, config={"displaylogo": False, "scrollZoom": True})
    return


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
def _(Path, history, mo, names, save_button, settings, torch, trained_model):
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

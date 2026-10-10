#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Point Clouds for Machine Learning, Part 7: identifying ModelNet10 objects

    We will download [ModelNet10](https://modelnet.cs.princeton.edu/download.html),
    sample its mesh surfaces, and train a PointNet to identify an object's category.
    We can then rotate a held-out point cloud and inspect its predicted label,
    class scores and mistakes. This uses real CAD objects from ten categories.

    [Part 4](PointCloudForMLPart4Classification.py) introduced the training loop.
    Here we connect download, preprocessing, training, evaluation and inference
    in one notebook. By the end we should be able to explain what evidence a
    classifier receives and evaluate its predictions on objects it did not train on.

    Run from the repository root:

    ```bash
    uv run --with plotly --with 'trimesh>=4.6' marimo edit PointCloudForML/PointCloudForMLPart7ModelNet.py
    ```

    Downloading waits for its button; training waits for its form. The archive
    is about 451 MiB and the extracted meshes require more disk space. An existing
    completed download is reused. Plotly camera controls only move the view;
    they do not rotate the coordinates passed to our classifier.
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
    import modelnet_data as dataset_tools

    return (inspect, mo, nn, np, pc, torch, Path, dataset_tools)


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


@app.cell
def _(Path):
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from Utils import in_lab as _in_lab

    dataset_cache = (
        Path("/transfer/modelnet10")
        if _in_lab()
        else Path(__file__).resolve().parent / "data"
    )
    print("Dataset cache:", dataset_cache)
    return (dataset_cache,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download once, then reuse the meshes

    The notebook downloads the official HTTPS archive and extracts it into
    `PointCloudForML/data/ModelNet10`. In the NCCA labs we use
    `/transfer/modelnet10` instead, following the other training demos.
    The archive and a completion marker stay in the cache so rerunning does
    not download or extract it again. Interrupted downloads use a temporary
    filename; an incomplete extraction does not receive a completion marker.

    To reclaim the local archive and meshes later, use the repository's
    cleanup script. Its dry run lists the cache before deleting anything:

    ```bash
    uv run scripts/clean_data.py --dry-run
    uv run scripts/clean_data.py
    ```

    The cleanup script leaves lab data under `/transfer` alone, and the saved
    model is in a separate `outputs` directory.
    """)
    return


@app.cell
def _(mo):
    download_button = mo.ui.run_button(
        label="Download and prepare ModelNet10 (about 451 MiB)"
    )
    download_button
    return (download_button,)


@app.cell
def _(download_button, dataset_cache, dataset_tools, mo):
    mo.stop(
        not download_button.value and not dataset_tools.modelnet10_ready(dataset_cache),
        mo.md("Press Download and prepare ModelNet10 to fetch the dataset."),
    )
    try:
        with mo.status.spinner(
            title="Preparing ModelNet10: downloading or extracting the cache"
        ):
            dataset_root = dataset_tools.prepare_modelnet10(dataset_cache)
    except (OSError, ValueError) as _error:
        mo.stop(
            True,
            mo.md(
                f"Dataset preparation failed: {_error}. Check the network and available disk space, then press the download button again."
            ),
        )
    dataset_counts = [
        {
            "category": name,
            "training meshes": len(list((dataset_root / name / "train").glob("*.off"))),
            "test meshes": len(list((dataset_root / name / "test").glob("*.off"))),
        }
        for name in dataset_tools.MODELNET10_CATEGORIES
    ]
    print("Prepared dataset:", dataset_root)
    mo.ui.table(dataset_counts)
    return dataset_root, dataset_counts


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sample surfaces and reserve validation objects

    Each category folder contains its official train and test splits. We reserve
    20% of the training meshes in each class for validation, then sample each
    selected mesh with an area-weighted surface sampler. We subtract the sampled
    centroid and divide by the largest radius, as in Part 1.

    The default uses 40 training meshes per category so we can try the pipeline
    on CPU. Zero removes the limit. The test limit defaults to zero, which uses
    all official test meshes. A small non-zero test limit is useful for checking
    the code; report it explicitly rather than treating it as a full benchmark.
    The seed controls mesh selection, sampling, validation split and initial weights.
    """)
    return


@app.cell
def _(mo):
    training_form = mo.ui.dictionary(
        {
            "points": mo.ui.number(
                32, 2048, value=256, step=32, label="Points per mesh"
            ),
            "limit": mo.ui.number(
                0, 1000, value=40, label="Training meshes per category (0 = all)"
            ),
            "test_limit": mo.ui.number(
                0, 1000, value=0, label="Test meshes per category (0 = all)"
            ),
            "epochs": mo.ui.number(1, 200, value=30, label="Epochs"),
            "lr": mo.ui.number(
                0.0001, 0.1, value=0.003, step=0.0001, label="Learning rate"
            ),
            "seed": mo.ui.number(0, 10000, value=42, label="Seed"),
        }
    ).form(submit_button_label="Sample ModelNet10 and train")
    training_form
    return (training_form,)


@app.cell
def _(training_form, dataset_root, dataset_tools, pc, mo):
    mo.stop(
        training_form.value is None,
        mo.md("Submit the training form after preparing the dataset."),
    )
    settings = training_form.value
    mo.stop(
        0 < int(settings["limit"]) < 2,
        mo.md("Use at least two training meshes per category, or zero for all."),
    )
    try:
        all_x, all_y, class_names = pc.load_modelnet(
            str(dataset_root),
            "train",
            int(settings["points"]),
            int(settings["limit"]),
            int(settings["seed"]),
        )
        test_x, test_y, _test_names, test_sources = pc.load_modelnet(
            str(dataset_root),
            "test",
            int(settings["points"]),
            int(settings["test_limit"]),
            int(settings["seed"]) + 1,
            return_sources=True,
        )
        if (
            class_names != list(dataset_tools.MODELNET10_CATEGORIES)
            or class_names != _test_names
        ):
            raise ValueError("Expected the same ten categories in both splits")
        train_indices, val_indices = pc.split_indices(all_y, int(settings["seed"]))
    except (ModuleNotFoundError, OSError, ValueError) as _error:
        mo.stop(
            True,
            mo.md(
                f"Could not prepare point clouds: {_error}. Check that trimesh is available."
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
    print("Categories:", class_names)
    print("Coordinate dtype:", train_x.dtype, "label dtype:", train_y.dtype)
    return (
        settings,
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        class_names,
        test_sources,
    )


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
def _(settings, train_x, train_y, val_x, val_y, class_names, pc, torch, nn, mo):
    with torch.random.fork_rng():
        torch.manual_seed(int(settings["seed"]))
        trained_model = pc.PointNet(len(class_names))
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
def _(trained_model, test_x, test_y, class_names, torch, mo, go):
    with torch.no_grad():
        test_logits = torch.cat([trained_model(_batch) for _batch in test_x.split(32)])
    predictions = test_logits.argmax(1)
    confusion = torch.bincount(
        test_y * len(class_names) + predictions, minlength=len(class_names) ** 2
    ).reshape(len(class_names), len(class_names))
    print("Test accuracy:", (predictions == test_y).float().mean().item())
    print("Test shapes:", len(test_y))
    _rows = [
        {
            "class": name,
            "shapes": int(confusion[i].sum()),
            "accuracy": float(confusion[i, i] / confusion[i].sum().clamp_min(1)),
        }
        for i, name in enumerate(class_names)
    ]
    confusion_plot = go.Figure(
        go.Heatmap(
            z=confusion.numpy(),
            x=class_names,
            y=class_names,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Identify a held-out object

    Select a test shape below. The model receives only its sampled xyz coordinates.
    The known category is shown alongside its prediction so we can inspect mistakes.
    The original mesh and the sampled cloud share the same centre and scale.
    Drag either view to inspect it and compare which surface details survive sampling.
    The classifier receives the cloud shown on the right, without the mesh faces.

    Softmax converts the logits into scores summing to one. The largest score
    selects the predicted category, but a high score is not a guarantee that
    the answer is correct or a calibrated probability. The class vocabulary is
    limited to the ten categories used for training.
    """)
    return


@app.cell
def _(
    example_index,
    predictions,
    test_logits,
    test_x,
    test_y,
    class_names,
    test_sources,
    np,
    make_subplots,
    go,
    mo,
    torch,
):
    import trimesh

    _index = example_index.value
    _actual = class_names[int(test_y[_index])]
    _predicted = class_names[int(predictions[_index])]
    identification_probabilities = torch.softmax(test_logits[_index], dim=0)
    print("Test object", _index, "actual:", _actual, "identified as:", _predicted)
    print(
        "Misclassified indices:", (predictions != test_y).nonzero().flatten().tolist()
    )
    _source = test_sources[_index]
    _mesh = trimesh.load_mesh(_source["path"], process=False)
    _vertices = (_mesh.vertices - _source["centre"]) / max(_source["radius"], 1e-12)
    _points = test_x[_index].numpy()
    identification_view = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Original mesh", "Sampled point cloud"),
    )
    identification_view.add_trace(
        go.Mesh3d(
            x=_vertices[:, 0],
            y=_vertices[:, 1],
            z=_vertices[:, 2],
            i=_mesh.faces[:, 0],
            j=_mesh.faces[:, 1],
            k=_mesh.faces[:, 2],
            color="lightsteelblue",
            flatshading=True,
            name="Original mesh",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    identification_view.add_trace(
        go.Scatter3d(
            x=_points[:, 0],
            y=_points[:, 1],
            z=_points[:, 2],
            mode="markers",
            marker=dict(size=3, color="#ff7f0e"),
            name="Classifier input",
        ),
        row=1,
        col=2,
    )
    _extent = float(max(np.abs(_vertices).max(), np.abs(_points).max()) * 1.1)
    _scene = dict(
        aspectmode="cube",
        dragmode="orbit",
        xaxis=dict(title="x", range=[-_extent, _extent]),
        yaxis=dict(title="y", range=[-_extent, _extent]),
        zaxis=dict(title="z", range=[-_extent, _extent]),
        camera=dict(up=dict(x=0, y=1, z=0), eye=dict(x=1.5, y=1.2, z=1.5)),
    )
    identification_view.update_layout(
        scene=_scene,
        scene2=_scene,
        height=600,
        title=f"Actual: {_actual} — identified as: {_predicted}",
        margin=dict(l=0, r=0, b=0, t=70),
    )
    print("Source mesh:", _source["path"].name)
    identification_scores = go.Figure(
        go.Bar(
            x=identification_probabilities.numpy(),
            y=class_names,
            orientation="h",
            marker_color=[
                "#ff7f0e" if name == _predicted else "#1f77b4" for name in class_names
            ],
        )
    )
    identification_scores.update_layout(
        title="Class scores for this object",
        xaxis=dict(title="Softmax score", range=[0, 1]),
        yaxis=dict(title="Category", autorange="reversed"),
        height=450,
    )
    mo.vstack(
        [
            mo.ui.plotly(
                identification_view, config={"displaylogo": False, "scrollZoom": True}
            ),
            mo.ui.plotly(identification_scores, config={"displaylogo": False}),
        ]
    )
    return identification_view, identification_scores, identification_probabilities


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save a reproducible checkpoint

    The checkpoint includes the class names, sampling settings and validation
    history and source URL. It is written only when we press Save, under this notebook folder's
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
def _(
    save_button,
    mo,
    Path,
    torch,
    trained_model,
    class_names,
    settings,
    history,
    dataset_tools,
):
    mo.stop(not save_button.value)
    _output = Path(__file__).resolve().parent / "outputs"
    _output.mkdir(exist_ok=True)
    torch.save(
        {
            "state_dict": trained_model.state_dict(),
            "classes": class_names,
            "dataset_url": dataset_tools.MODELNET10_URL,
            "settings": settings,
            "history": history,
        },
        _output / "modelnet10_classifier.pt",
    )
    print("Saved:", _output / "modelnet10_classifier.pt")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Find a confused class pair and inspect five examples. Which surface features survive sampling?
    2. Increase the training-mesh limit and compare validation curves with the short default run.
    3. Repeat training with another seed. Report the selected epoch, point count and data limits alongside accuracy.
    4. Revisit Part 5 and add rotation augmentation. Does it help on ModelNet10's aligned objects?
    5. Save the classifier, reload its state dictionary and check that a held-out cloud gives the same logits.

    We have connected a downloadable mesh dataset to object identification.
    Our simplified model and limited training budget give us an experiment to
    investigate; they do not reproduce the original PointNet benchmark.
    """)
    return


if __name__ == "__main__":
    app.run()

#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Point Clouds for Machine Learning, Part 5: robustness and rotations

    Turning an object changes its coordinates, but usually leaves its class unchanged: a chair is still a chair when placed on its side. A classifier trained only on upright objects may rely on their orientation and struggle when we rotate them.

    In this lesson we will investigate whether rotation augmentation improves that behaviour. We will train two copies of [Part 4’s model](PointCloudForMLPart4Classification.py), starting from identical weights and using the same batches and training budget. One model sees the original clouds, whilst the other sees clouds transformed by random rotations in full 3D. Keeping the other conditions matched helps us assess the effect of augmentation.

    The shared mapping and pooling make our model invariant to point order, but they do not make it invariant to rotation. Shuffling rows preserves each point’s coordinates; rotating the cloud changes them. Augmentation gives the model examples from different orientations so it can learn to tolerate these changes, without guaranteeing identical scores for every rotation.

    We will compare the models on rotated inputs and repeat evaluations with random perturbations. Reporting variation across these evaluations helps us see whether an apparent improvement is consistent or depends on a particular random draw.

    The figures use Plotly. Drag a 3D view to rotate it and scroll to zoom. Moving the camera changes our view of the cloud; it does not rotate the coordinates passed to the model or repeat training.

    By the end we should be able to distinguish architectural invariance from learned tolerance, design a controlled comparison, and report the variation introduced by random perturbations. Results from the generated shapes describe our toy dataset; they do not establish performance on ModelNet.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path
    import marimo as mo
    import numpy as np
    import torch
    from torch import nn
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # This also works when marimo starts from the notebook's own folder.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pointcloud as pc

    return go, make_subplots, mo, nn, np, pc, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rotations preserve geometry, but change coordinates

    We store points as row vectors, so a rotation uses `points @ R.T`. Y-axis
    rotations change heading whilst preserving up; full rotations also tilt the
    object. We generate full rotations using normalised Gaussian quaternions,
    which sample SO(3) uniformly. Uniform Euler angles do not do this.
    """)
    return


@app.cell
def _(mo, pc, torch):
    _probe, _labels, _names = pc.make_shapes(1, 64, 5)
    _rotated = pc.rotate(_probe, torch.Generator().manual_seed(9), full=True)
    print(
        "Maximum pairwise distance change:",
        (
            torch.cdist(_probe, _probe, compute_mode="donot_use_mm_for_euclid_dist")
            - torch.cdist(
                _rotated, _rotated, compute_mode="donot_use_mm_for_euclid_dist"
            )
        )
        .abs()
        .max()
        .item(),
    )
    rotation_view = pc.cloud_figure(_rotated[2].numpy(), "A rotated rod")
    mo.ui.plotly(rotation_view, config={"displaylogo": False, "scrollZoom": True})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data and training controls

    Use generated sphere/cube/rod surfaces for a quick CPU experiment, or enter the
    path to an extracted [ModelNet10](https://modelnet.cs.princeton.edu/) directory discussed in the previous notebook.

    As before I reserve 20% of each training category for validation, keeping each source
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
                0,
                1000,
                value=24,
                label="Training meshes per category (0 = all)",
            ),
            "epochs": mo.ui.number(1, 200, value=30, label="Epochs"),
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
    ## A paired training experiment

    The two models start with identical parameters. Separate random generators
    keep batch selection independent of augmentation. We select each model's best
    weights using a fixed validation set containing both clean and rotated clouds.
    This validation choice is part of our experiment and should be recorded.

    Augmentation is applied to training batches only. We do not rotate points and
    then recompute normalisation: rotation already preserves centroid and radius.
    """)
    return


@app.cell
def _(mo, names, nn, pc, settings, torch, train_x, train_y, val_x, val_y):
    with torch.random.fork_rng():
        torch.manual_seed(int(settings["seed"]))
        _initial = pc.PointNet(len(names)).state_dict()
    comparison_models = {}
    comparison_history = {}
    _val_rotated = pc.rotate(val_x, torch.Generator().manual_seed(812), full=True)
    _validation_x = torch.cat([val_x, _val_rotated])
    _validation_y = torch.cat([val_y, val_y])
    for _label, _augment in [
        ("No augmentation", False),
        ("Rotation augmentation", True),
    ]:
        _model = pc.PointNet(len(names))
        _model.load_state_dict(_initial)
        _optimizer = torch.optim.Adam(_model.parameters(), lr=float(settings["lr"]))
        _batch_rng = torch.Generator().manual_seed(int(settings["seed"]))
        _rotation_rng = torch.Generator().manual_seed(int(settings["seed"]) + 1)
        _best_loss = float("inf")
        _history = []
        for _epoch in mo.status.progress_bar(
            range(int(settings["epochs"])), title=_label
        ):
            _model.train()
            _order = torch.randperm(len(train_x), generator=_batch_rng)
            for _indices in _order.split(16):
                _batch = train_x[_indices]
                if _augment:
                    _batch = pc.rotate(_batch, _rotation_rng, full=True)
                _optimizer.zero_grad()
                _loss = nn.functional.cross_entropy(_model(_batch), train_y[_indices])
                _loss.backward()
                _optimizer.step()
            _model.eval()
            with torch.no_grad():
                _scores = torch.cat(
                    [_model(_batch) for _batch in _validation_x.split(32)]
                )
                _val_loss = nn.functional.cross_entropy(_scores, _validation_y).item()
            _history.append(_val_loss)
            if _val_loss < _best_loss:
                _best_loss = _val_loss
                _best_state = {
                    name: value.detach().clone()
                    for name, value in _model.state_dict().items()
                }
        _model.load_state_dict(_best_state)
        comparison_models[_label] = _model.eval()
        comparison_history[_label] = _history
    print("Two models trained with the same initial weights and batch orders.")
    return comparison_history, comparison_models


@app.cell
def _(comparison_history, go, mo):
    augmentation_curve = go.Figure()
    for _name, _losses in comparison_history.items():
        augmentation_curve.add_trace(
            go.Scatter(
                x=list(range(1, len(_losses) + 1)),
                y=_losses,
                name=_name,
                mode="lines",
            )
        )
    augmentation_curve.update_layout(
        title="Clean + rotated validation loss",
        xaxis_title="Epoch",
        yaxis_title="Cross entropy",
        height=400,
    )
    mo.ui.plotly(augmentation_curve, config={"displaylogo": False})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluate one change at a time

    We vary one input condition at a time so we can see how it affects classification. Both models receive **exactly the same perturbed clouds**, giving us a paired comparison: differences in their scores cannot be explained by one model receiving an easier random example.

    We repeat each condition with five perturbation seeds. Each seed produces a different realisation of the perturbation, letting us measure how much the results depend on the particular random changes applied.

    For noise experiments, the standard deviation is expressed in the normalised, unit-sphere coordinate system. It therefore describes noise relative to the cloud’s scale, rather than a distance in metres or the mesh’s original units. Adding noise can move points outside the unit sphere; normalising the original cloud does not constrain later perturbations.

    For point-count experiments, we use nested subsets of each held-out cloud. A smaller subset is contained in the larger ones, so we progressively remove points from the same sampled object. This measures the effect of losing points without also introducing a fresh surface sample.

    We do not fill the missing rows with zeros. The model would interpret each zero row as a point at the origin, changing the cloud we are testing. Padding is only appropriate if the pooling operation explicitly masks those rows so they cannot contribute.

    The error bars show the **standard deviation across the five perturbation seeds**. They describe variation in evaluation results for these fixed trained models. They are not confidence intervals, and they do not measure uncertainty caused by initial weights or other randomness during training.

    For a stronger study, we would also repeat the paired training experiment with several training seeds. That would let us assess whether the observed difference persists across independently trained model pairs, as well as across different perturbations.
    """)
    return


@app.cell
def _(comparison_models, mo, np, pc, test_x, test_y, torch):
    robustness_rows = []
    _conditions = [
        ("Clean", 0),
        ("Shuffle", 0),
        ("Y rotation", 0),
        ("Full rotation", 0),
    ]
    _conditions += [("Noise", value) for value in (0.01, 0.03, 0.08)]
    _conditions += [
        ("Points", value)
        for value in sorted(set([32, 64, 128, test_x.shape[1]]))
        if value <= test_x.shape[1]
    ]
    _conditions += [("Missing region", 0)]
    for _condition, _amount in mo.status.progress_bar(
        _conditions, title="Evaluating perturbations"
    ):
        _accuracies = {name: [] for name in comparison_models}
        for _repeat in range(5):
            _rng = torch.Generator().manual_seed(2000 + _repeat)
            _changed = test_x.clone()
            if _condition == "Shuffle":
                _changed = _changed[
                    :, torch.randperm(_changed.shape[1], generator=_rng)
                ]
            elif _condition in {"Y rotation", "Full rotation"}:
                _changed = pc.rotate(_changed, _rng, full=_condition == "Full rotation")
            elif _condition == "Noise":
                _changed += torch.randn(_changed.shape, generator=_rng) * _amount
            elif _condition == "Points":
                _changed = _changed[
                    :,
                    torch.randperm(_changed.shape[1], generator=_rng)[: int(_amount)],
                ]
            elif _condition == "Missing region":
                # Keep the half with smaller x coordinates: a simple spatial crop.
                _indices = _changed[:, :, 0].argsort(dim=1)[:, : _changed.shape[1] // 2]
                _changed = _changed.gather(1, _indices.unsqueeze(-1).expand(-1, -1, 3))
            for _name, _model in comparison_models.items():
                with torch.no_grad():
                    _predicted = torch.cat(
                        [_model(_batch) for _batch in _changed.split(32)]
                    ).argmax(1)
                _accuracies[_name].append((_predicted == test_y).float().mean().item())
        for _name, _values in _accuracies.items():
            robustness_rows.append(
                {
                    "model": _name,
                    "condition": _condition,
                    "amount": _amount,
                    "accuracy": float(np.mean(_values)),
                    "std": float(np.std(_values, ddof=1)),
                }
            )
    mo.ui.table(robustness_rows)
    return (robustness_rows,)


@app.cell
def _(comparison_models, go, make_subplots, mo, robustness_rows):
    robustness_plot = make_subplots(rows=1, cols=2, subplot_titles=("Noise", "Points"))
    for _name in comparison_models:
        for _column, _condition, _xlabel in [
            (1, "Noise", "Noise standard deviation"),
            (2, "Points", "Retained points"),
        ]:
            _rows = [
                row
                for row in robustness_rows
                if row["model"] == _name and row["condition"] == _condition
            ]
            robustness_plot.add_trace(
                go.Scatter(
                    x=[row["amount"] for row in _rows],
                    y=[row["accuracy"] for row in _rows],
                    error_y=dict(
                        type="data",
                        array=[row["std"] for row in _rows],
                        visible=True,
                    ),
                    mode="lines+markers",
                    line=dict(
                        color="#1f77b4" if _name == "No augmentation" else "#ff7f0e"
                    ),
                    name=_name,
                    legendgroup=_name,
                    showlegend=_column == 1,
                ),
                row=1,
                col=_column,
            )
            robustness_plot.update_xaxes(title_text=_xlabel, row=1, col=_column)
            robustness_plot.update_yaxes(
                title_text="Test accuracy", range=[0, 1.05], row=1, col=_column
            )
    robustness_plot.update_layout(height=400)
    mo.ui.plotly(robustness_plot, config={"displaylogo": False})
    return


@app.cell
def _(mo):
    rotation_angle = mo.ui.slider(
        0,
        180,
        value=90,
        label="Rod rotation about x (degrees)",
        show_value=True,
    )
    rotation_angle
    return (rotation_angle,)


@app.cell
def _(comparison_models, mo, np, pc, rotation_angle, torch):
    _probe, _target, _names = pc.make_shapes(1, 128, 901)
    _angle = np.deg2rad(rotation_angle.value)
    _rotation = torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(_angle), -np.sin(_angle)],
            [0, np.sin(_angle), np.cos(_angle)],
        ],
        dtype=torch.float32,
    )
    _rotated = _probe[2:3] @ _rotation.T
    for _name, _model in comparison_models.items():
        with torch.no_grad():
            print(_name, "raw logits:", _model(_rotated).numpy().round(3))
    print(
        "This generated rod is a diagnostic probe; for ModelNet models its class vocabulary is different."
    )
    probe_view = pc.cloud_figure(
        _rotated[0].numpy(), "Change orientation whilst keeping the geometry"
    )
    mo.ui.plotly(probe_view, config={"displaylogo": False, "scrollZoom": True})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Add a third model trained with y-axis rotations only. Compare it with full rotation augmentation.
    2. Repeat the whole experiment with three training seeds. Separate training variation from perturbation variation.
    3. Compare spatial cropping with randomly removing the same number of points.
    4. Explain a case where augmentation reduces clean accuracy. Is that an acceptable trade-off for your application?

    [Part 6](PointCloudForMLPart6Segmentation.py) keeps the per-point features to label object parts.
    """)
    return


if __name__ == "__main__":
    app.run()

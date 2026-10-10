#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Point Clouds for Machine Learning, Part 6: part segmentation

    Classification produces one answer per object. We will now retain the local
    features from [Part 3](PointCloudForMLPart3PointNet.py) and combine them with the
    global shape features to produce an answer for every point.

    The figures use Plotly. Drag a 3D view to rotate it and scroll to zoom.
    Camera controls do not change the input coordinates or repeat training.
    Run this notebook with `uv run --with plotly marimo edit` followed by its path.

    By the end we should be able to implement local/global concatenation, keep
    labels aligned during sampling, test permutation equivariance, and evaluate
    segmentation using per-shape part IoU. We begin with generated chairs, then
    provide a route to labelled ShapeNet Part chairs.

    The loss and IoU examples run immediately. For a complete generated-data
    experiment, leave the ShapeNet path blank and press **Load labelled chairs
    and train** below. Learning curves, held-out predictions and checkpoint
    saving appear after that run. Opening the notebook does not start training.
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
    ## Give each point the same global context

    Local features have shape `(B, 64, N)`. Pooling produces `(B, 128, 1)`;
    `expand` gives every point the same global context without copying the data.
    Concatenation along the channel axis produces `(B, 192, N)`, then a shared
    head produces `(B, parts, N)` logits. Our "local" features are per-point
    features; this network does not explicitly aggregate spatial neighbourhoods.
    """)
    return


@app.cell
def _(inspect, pc, mo):
    mo.md("```python\n" + inspect.getsource(pc.SegmentNet) + "\n```")
    return


@app.cell
def _(pc, torch, go, mo):
    example_chairs, example_parts, part_names = pc.make_parts(2, 192, 7)
    with torch.random.fork_rng():
        torch.manual_seed(7)
        segment_probe = pc.SegmentNet(3).eval()
    with torch.no_grad():
        local_features = segment_probe.local(example_chairs.transpose(1, 2))
        global_context = segment_probe.features(local_features).amax(2, keepdim=True)
        combined = torch.cat(
            [local_features, global_context.expand(-1, -1, example_chairs.shape[1])],
            dim=1,
        )
    print("Local:", tuple(local_features.shape), "global:", tuple(global_context.shape))
    print(
        "Concatenated:",
        tuple(combined.shape),
        "logits:",
        tuple(segment_probe(example_chairs).shape),
    )
    example_view = pc.cloud_figure(
        example_chairs[0].numpy(),
        "Generated chair parts",
        example_parts[0].numpy(),
        labels=part_names,
    )
    mo.ui.plotly(example_view, config={"displaylogo": False, "scrollZoom": True})
    return (example_chairs, example_parts, segment_probe)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Shuffle coordinates and labels together

    Segmentation should obey `s(PX) = P s(X)`. Classification's invariance would
    be the wrong test: each prediction belongs to a particular input point.
    A silent data error is to shuffle coordinates whilst leaving labels unchanged.
    The arrays still have the right shape, so cross entropy cannot detect this mistake.
    """)
    return


@app.cell
def _(example_chairs, example_parts, segment_probe, torch):
    _order = torch.randperm(
        example_chairs.shape[1], generator=torch.Generator().manual_seed(8)
    )
    with torch.no_grad():
        _original = segment_probe(example_chairs)
        _shuffled = segment_probe(example_chairs[:, _order])
    print(
        "Equivariant logits:",
        torch.allclose(_shuffled, _original[:, :, _order], atol=1e-6),
    )
    print("Wrong labels still have the right shape:", tuple(example_parts.shape))
    print(
        "Labels mismatched if we forget the shuffle:",
        (example_parts != example_parts[:, _order]).float().mean().item(),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A loss for each point

    Before training a chair model, we can check the loss on three points with
    two possible part labels. The first two predictions are confident and
    correct; the third is less certain. We expect it to contribute more loss.

    [`cross_entropy`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)
    takes `(B, parts, N)` logits and `(B, N)` integer targets. With
    `reduction="none"` it returns `(B, N)`, so we can inspect every point.
    We pass raw logits, then use log-softmax separately to check the calculation:

    $$\ell_i = -\log\left(\frac{\exp(z_{y_i,i})}{\sum_k\exp(z_{k,i})}\right).$$
    """)
    return


@app.cell
def _(torch, nn):
    _demo_logits = torch.tensor(
        [[[3.0, 0.0, 1.0], [0.0, 3.0, 2.0]]], requires_grad=True
    )
    _demo_labels = torch.tensor([[0, 1, 1]], dtype=torch.long)
    point_losses = nn.functional.cross_entropy(
        _demo_logits, _demo_labels, reduction="none"
    )
    _log_probabilities = _demo_logits.log_softmax(dim=1)
    manual_point_losses = -_log_probabilities.gather(
        1, _demo_labels.unsqueeze(1)
    ).squeeze(1)
    print("Logits (batch, parts, points):", tuple(_demo_logits.shape))
    print("Targets (batch, points):", tuple(_demo_labels.shape))
    print(
        "Part probabilities for each point:\n",
        _demo_logits.softmax(1).detach().transpose(1, 2),
    )
    print("Loss per point:", point_losses.detach())
    print("Manual negative log-probabilities:", manual_point_losses.detach())
    point_losses.mean().backward()
    print("Gradients for both part scores at each point:\n", _demo_logits.grad)
    point_losses = point_losses.detach()
    manual_point_losses = manual_point_losses.detach()
    return point_losses, manual_point_losses


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The mean loss gives each point equal weight. That is not the same as giving
    each part equal weight: a part containing many points contributes more terms.
    We will return to this when comparing point accuracy and part IoU.

    We can also flatten the prediction for cross entropy, but we must move
    the part axis to the end first. Reshaping `(B, parts, N)` directly into
    `(B*N, parts)` silently groups unrelated scores. The shapes still match!
    """)
    return


@app.cell
def _(torch, nn, point_losses):
    _logits = torch.tensor([[[3.0, 0.0, 1.0], [0.0, 3.0, 2.0]]])
    _labels = torch.tensor([[0, 1, 1]])
    _point_rows = _logits.transpose(1, 2).reshape(-1, 2)
    flattened_point_loss = nn.functional.cross_entropy(_point_rows, _labels.reshape(-1))
    _wrong_loss = nn.functional.cross_entropy(
        _logits.reshape(-1, 2), _labels.reshape(-1)
    )
    print("Direct per-point mean:", point_losses.mean().item())
    print("Transpose then flatten:", flattened_point_loss.item())
    print("Reshape without moving the part axis:", _wrong_loss.item())
    try:
        nn.functional.cross_entropy(_logits, _labels.float())
    except RuntimeError as _error:
        print("Floating-point class indices:", str(_error).split("\n")[0])
    return (flattened_point_loss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generated chairs or ShapeNet Part

    The generated chairs have three labelled parts and are deliberately simple.
    They exercise the pipeline, not the ShapeNet benchmark.

    For real data, use [PyTorch Geometric's ShapeNet dataset](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.ShapeNet.html).
    Prepare a local cache explicitly before opening the training example:

    ```bash
    uv run --with torch-geometric python -c "from torch_geometric.datasets import ShapeNet; ShapeNet('data/ShapeNetPart', categories=['Chair'], split='train', include_normals=False)"
    uv run --with plotly --with torch-geometric marimo edit PointCloudForML/PointCloudForMLPart6Segmentation.py
    ```

    The first command downloads/processes data and may take some time. Check the
    dataset's access conditions before preparing the cache. Enter that cache path below.
    The notebook loader requires a complete prepared cache and uses separate train, val
    and test splits. It samples point indices once and applies those same indices
    to coordinates and labels. It remaps Chair's labels 12–15 to 0–3 and uses xyz
    only. The real-data plots retain the original part IDs in their names; inspect the annotations before assigning semantic names.
    """)
    return


@app.cell
def _(mo):
    segmentation_form = mo.ui.dictionary(
        {
            "root": mo.ui.text(
                value="", label="ShapeNet cache (blank for generated chairs)"
            ),
            "points": mo.ui.number(
                48, 2048, value=192, step=16, label="Points per shape"
            ),
            "limit": mo.ui.number(4, 1000, value=24, label="Maximum training chairs"),
            "epochs": mo.ui.number(1, 200, value=12, label="Epochs"),
            "lr": mo.ui.number(
                0.0001, 0.1, value=0.003, step=0.0001, label="Learning rate"
            ),
            "seed": mo.ui.number(0, 10000, value=42, label="Seed"),
        }
    ).form(submit_button_label="Load labelled chairs and train")
    segmentation_form
    return (segmentation_form,)


@app.cell
def _(segmentation_form, mo, pc):
    mo.stop(
        segmentation_form.value is None,
        mo.md("Submit the form to train a segmentation model."),
    )
    seg_settings = segmentation_form.value
    try:
        if seg_settings["root"].strip():
            seg_train_x, seg_train_y, seg_names = pc.load_shapenet(
                seg_settings["root"],
                "train",
                int(seg_settings["points"]),
                int(seg_settings["limit"]),
                int(seg_settings["seed"]),
            )
            seg_val_x, seg_val_y, _val_names = pc.load_shapenet(
                seg_settings["root"],
                "val",
                int(seg_settings["points"]),
                0,
                int(seg_settings["seed"]) + 1,
            )
            seg_test_x, seg_test_y, _test_names = pc.load_shapenet(
                seg_settings["root"],
                "test",
                int(seg_settings["points"]),
                0,
                int(seg_settings["seed"]) + 2,
            )
        else:
            seg_train_x, seg_train_y, seg_names = pc.make_parts(
                int(seg_settings["limit"]),
                int(seg_settings["points"]),
                int(seg_settings["seed"]),
            )
            seg_val_x, seg_val_y, _val_names = pc.make_parts(
                8, int(seg_settings["points"]), int(seg_settings["seed"]) + 1
            )
            seg_test_x, seg_test_y, _test_names = pc.make_parts(
                8, int(seg_settings["points"]), int(seg_settings["seed"]) + 2
            )
    except (ModuleNotFoundError, OSError, ValueError) as _error:
        mo.stop(True, mo.md(f"Could not load labelled data: {_error}"))
    print("Coordinates:", tuple(seg_train_x.shape), "labels:", tuple(seg_train_y.shape))
    print("Part names:", seg_names)
    return (
        seg_settings,
        seg_train_x,
        seg_train_y,
        seg_val_x,
        seg_val_y,
        seg_test_x,
        seg_test_y,
        seg_names,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cross entropy at every point

    PyTorch accepts `(B, parts, N)` logits and `(B, N)` integer labels. We do not
    flatten away the correspondence. Validation loss selects the checkpoint.
    The loop follows Part 4, but each batch now contains many labelled predictions.
    """)
    return


@app.cell
def _(
    seg_settings,
    seg_train_x,
    seg_train_y,
    seg_val_x,
    seg_val_y,
    seg_names,
    pc,
    torch,
    nn,
    mo,
):
    with torch.random.fork_rng():
        torch.manual_seed(int(seg_settings["seed"]))
        segmentation_model = pc.SegmentNet(len(seg_names))
    _optimizer = torch.optim.Adam(
        segmentation_model.parameters(), lr=float(seg_settings["lr"])
    )
    _generator = torch.Generator().manual_seed(int(seg_settings["seed"]))
    seg_history = []
    _best_loss = float("inf")
    for _epoch in mo.status.progress_bar(
        range(int(seg_settings["epochs"])), title="Training segmentation"
    ):
        segmentation_model.train()
        _order = torch.randperm(len(seg_train_x), generator=_generator)
        _total = 0.0
        for _indices in _order.split(8):
            _optimizer.zero_grad()
            _loss = nn.functional.cross_entropy(
                segmentation_model(seg_train_x[_indices]), seg_train_y[_indices]
            )
            _loss.backward()
            _optimizer.step()
            _total += _loss.item() * len(_indices)
        segmentation_model.eval()
        with torch.no_grad():
            _scores = torch.cat(
                [segmentation_model(_batch) for _batch in seg_val_x.split(16)]
            )
            _val_loss = nn.functional.cross_entropy(_scores, seg_val_y).item()
        seg_history.append((_total / len(seg_train_x), _val_loss))
        if _val_loss < _best_loss:
            _best_loss = _val_loss
            _best_state = {
                name: value.detach().clone()
                for name, value in segmentation_model.state_dict().items()
            }
    segmentation_model.load_state_dict(_best_state)
    segmentation_model.eval()
    print("Selected validation loss:", _best_loss)
    return (segmentation_model, seg_history)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Accuracy and intersection over union

    A model which labels every point as a large part can have misleading point
    accuracy. We also measure overlap for every part:

    $$\operatorname{IoU}_k = \frac{|P_k\cap T_k|}{|P_k\cup T_k|}.$$

    Calculate part IoUs separately for each shape, average its parts, then average
    shapes. If a part is absent in both prediction and target we assign IoU=1;
    if the model hallucinates an absent part it receives IoU=0. We state this rule
    because changing it changes the reported metric. This is a single-category
    experiment; a full ShapeNet benchmark needs its category-specific valid part sets.
    """)
    return


@app.cell
def _(torch, pc):
    _actual = torch.tensor([[0, 0, 1, 1]])
    _predicted = torch.tensor([[0, 1, 1, 1]])
    print("Hand-checkable IoUs:", pc.part_iou(_predicted, _actual, 3))
    print("Expected: 1/2, 2/3, 1 for the absent third part")
    return


@app.cell
def _(
    segmentation_model,
    seg_test_x,
    seg_test_y,
    seg_names,
    pc,
    torch,
    mo,
    go,
    seg_history,
):
    with torch.no_grad():
        seg_logits = torch.cat(
            [segmentation_model(_batch) for _batch in seg_test_x.split(16)]
        )
        seg_predictions = seg_logits.argmax(1)
    seg_ious = pc.part_iou(seg_predictions, seg_test_y, len(seg_names))
    print("Point accuracy:", (seg_predictions == seg_test_y).float().mean().item())
    print("Mean shape part IoU:", seg_ious.mean(dim=1).mean().item())
    segmentation_curve = go.Figure()
    for _column, _name in [(0, "Training"), (1, "Validation")]:
        segmentation_curve.add_trace(
            go.Scatter(
                x=list(range(1, len(seg_history) + 1)),
                y=[row[_column] for row in seg_history],
                mode="lines",
                name=_name,
            )
        )
    segmentation_curve.update_layout(
        title="Segmentation learning curves",
        xaxis_title="Epoch",
        yaxis_title="Cross entropy",
        height=400,
    )
    mo.vstack(
        [
            mo.ui.table(
                [
                    {"part": name, "mean IoU": float(seg_ious[:, i].mean())}
                    for i, name in enumerate(seg_names)
                ]
            ),
            mo.ui.plotly(segmentation_curve, config={"displaylogo": False}),
        ]
    )
    return seg_predictions, seg_ious, seg_logits, segmentation_curve


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Read the evaluation before inspecting a chair

    The learning curves show whether fitting the training chairs also improves
    validation loss. A falling training curve with a rising validation curve
    suggests overfitting. Our short run may still be under-trained; we should
    inspect the curves before deciding to change the architecture.

    The part table averages each part's IoU across held-out shapes. The shape
    table below averages parts within each chair. Neither is simply point
    accuracy. A thin leg can have poor overlap whilst contributing few errors
    to the overall point count. Sort the table by mean part IoU and use the
    shape number in the viewer below to inspect a difficult example.
    """)
    return


@app.cell
def _(seg_predictions, seg_test_y, seg_ious, mo):
    seg_shape_summary = [
        {
            "shape": index,
            "point accuracy": float(
                (seg_predictions[index] == seg_test_y[index]).float().mean()
            ),
            "mean part IoU": float(seg_ious[index].mean()),
            "incorrect points": int(
                (seg_predictions[index] != seg_test_y[index]).sum()
            ),
        }
        for index in range(len(seg_test_y))
    ]
    mo.ui.table(seg_shape_summary)
    return (seg_shape_summary,)


@app.cell
def _(mo, seg_test_x):
    chair_index = mo.ui.slider(
        0, len(seg_test_x) - 1, value=0, label="Held-out chair", show_value=True
    )
    chair_index
    return (chair_index,)


@app.cell
def _(
    chair_index,
    seg_test_x,
    seg_test_y,
    seg_predictions,
    seg_names,
    seg_ious,
    pc,
    go,
    make_subplots,
    mo,
):
    _index = chair_index.value
    _points = seg_test_x[_index].numpy()
    _truth = seg_test_y[_index].numpy()
    _prediction = seg_predictions[_index].numpy()
    print("Chair", _index, "mean part IoU:", float(seg_ious[_index].mean()))
    print("IoU by part:", dict(zip(seg_names, seg_ious[_index].tolist())))
    segmentation_plot = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}] * 3],
        subplot_titles=("Ground truth", "Predicted parts", "Incorrect points"),
    )
    for _column, _values, _labels in [
        (1, _truth, seg_names),
        (2, _prediction, seg_names),
        (3, (_truth != _prediction).astype(int), ["Correct", "Incorrect"]),
    ]:
        _view = pc.cloud_figure(_points, "", _values, labels=_labels)
        for _trace in _view.data:
            _trace.showlegend = _column != 2
            segmentation_plot.add_trace(_trace, row=1, col=_column)
    _scene = pc.cloud_figure(_points, "").layout.scene.to_plotly_json()
    segmentation_plot.update_layout(
        scene=_scene,
        scene2=_scene,
        scene3=_scene,
        height=600,
        title="Inspect labelled points and segmentation errors",
    )
    mo.ui.plotly(segmentation_plot, config={"displaylogo": False, "scrollZoom": True})
    return (segmentation_plot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save and reload the selected model

    We can save the checkpoint selected using validation loss, along with the
    part names, preprocessing convention, submitted settings and learning
    history. A state dictionary alone would not tell us what its output
    channels mean. Keep a record of the dataset version as well.

    The default path is `outputs/segmentation.pt` beside this notebook. Press
    **Save segmentation checkpoint** after training; saving again overwrites
    that path. Editing the path does not repeat training. We immediately reload
    our own checkpoint with `weights_only=True` and compare its held-out logits
    to check that the architecture and weights have survived the round trip.
    """)
    return


@app.cell
def _(mo, Path):
    seg_checkpoint_path = mo.ui.text(
        value=str(Path(__file__).resolve().parent / "outputs" / "segmentation.pt"),
        label="Segmentation checkpoint path",
        full_width=True,
    )
    seg_checkpoint_path
    return (seg_checkpoint_path,)


@app.cell
def _(mo):
    seg_save_button = mo.ui.run_button(label="Save segmentation checkpoint")
    seg_save_button
    return (seg_save_button,)


@app.cell
def _(
    seg_save_button,
    seg_checkpoint_path,
    segmentation_model,
    seg_names,
    seg_settings,
    seg_history,
    seg_test_x,
    pc,
    torch,
    Path,
    mo,
):
    mo.stop(not seg_save_button.value)
    mo.stop(
        not seg_checkpoint_path.value.strip(),
        mo.md("Enter a checkpoint path before saving."),
    )
    saved_segmentation_path = Path(seg_checkpoint_path.value).expanduser()
    try:
        saved_segmentation_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": segmentation_model.state_dict(),
                "parts": seg_names,
                "settings": seg_settings,
                "history": seg_history,
                "preprocessing": "xyz only; subtract sampled centroid; divide by maximum radius",
                "architecture": "SegmentNet: local 64 channels, pooled 128 channels, shared head",
            },
            saved_segmentation_path,
        )
        _checkpoint = torch.load(
            saved_segmentation_path, map_location="cpu", weights_only=True
        )
    except OSError as _error:
        mo.stop(True, mo.md(f"Could not save or reload the checkpoint: {_error}"))
    reloaded_segmentation_model = pc.SegmentNet(len(_checkpoint["parts"]))
    reloaded_segmentation_model.load_state_dict(_checkpoint["state_dict"])
    reloaded_segmentation_model.eval()
    with torch.no_grad():
        _before = segmentation_model(seg_test_x[:1])
        _after = reloaded_segmentation_model(seg_test_x[:1])
    torch.testing.assert_close(_before, _after)
    print("Saved:", saved_segmentation_path)
    print("Reloaded part names:", _checkpoint["parts"])
    print(
        "Largest logit difference after reloading:",
        (_before - _after).abs().max().item(),
    )
    return saved_segmentation_path, reloaded_segmentation_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now have a complete segmentation experiment: labelled points, a shared
    model with global context, validation-based selection, held-out overlap
    measurements, an error viewer and a reloadable checkpoint.

    The generated chairs make the plumbing easy to check. Their simple parts
    and consistent orientation also make the task easier than real annotations.
    For ShapeNet, inspect missing parts, thin structures and annotation boundaries
    before drawing conclusions from one average score. Increasing training time
    does not automatically fix a mismatch between our data and the task.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Remove the global context and compare local-only segmentation on the same data.
    2. Rotate held-out chairs. Does classification robustness imply correct part labels?
    3. Shuffle a test cloud and undo the permutation on its predictions. Compare logits.
    4. Find a chair with missing armrests and explain how our absent-part rule affects IoU.
    5. Extend to a second ShapeNet category. Decide how to handle its valid labels before training.

    For interactive inspection, [Polyscope](https://polyscope.run/py/) can register a
    point cloud and attach part IDs as a scalar quantity. We use Plotly here so the views can be rotated inside the notebook and its HTML export.
    """)
    return


if __name__ == "__main__":
    app.run()

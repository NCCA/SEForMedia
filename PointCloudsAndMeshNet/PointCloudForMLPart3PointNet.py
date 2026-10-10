#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Point Clouds for Machine Learning, Part 3: a small PointNet

    We will use the shared mapping and maximum from [Part 2](PointCloudForMLPart2Sets.py)
    to build a classifier. Our model follows the central idea in
    [PointNet](https://openaccess.thecvf.com/content_cvpr_2017/html/Qi_PointNet_Deep_Learning_CVPR_2017_paper.html).
    It omits the original input and feature transformation networks, batch normalisation
    and dropout so we can inspect the data flow first.

    The figures use Plotly. Drag a 3D view to rotate it and scroll to zoom.
    Camera controls do not change the input coordinates or repeat training.
    Run this notebook with `uv run --with plotly marimo edit` followed by its path.

    By the end we should be able to trace every tensor dimension, explain a
    kernel-size-one convolution, and identify the points contributing pooled maxima.
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
    ## A 1D convolution can be a shared linear layer

    [`Conv1d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
    expects `(batch, channels, points)`. With kernel size 1, each output uses only
    one point's channels. It does not mix neighbouring rows or search for nearby
    points in 3D. A kernel size of 3 would mix neighbouring *array positions*, which
    have no geometric meaning in an unordered cloud.

    We can copy a linear layer's weights into a convolution and check their outputs.
    """)
    return


@app.cell
def _(nn, torch):
    with torch.random.fork_rng():
        torch.manual_seed(4)
        linear = nn.Linear(3, 5)
        convolution = nn.Conv1d(3, 5, kernel_size=1)
        coordinates = torch.randn(2, 7, 3)
    with torch.no_grad():
        convolution.weight.copy_(linear.weight.unsqueeze(-1))
        convolution.bias.copy_(linear.bias)
        dense_result = linear(coordinates)
        conv_result = convolution(coordinates.transpose(1, 2)).transpose(1, 2)
    print("Linear output:", tuple(dense_result.shape))
    print("Conv output:", tuple(conv_result.shape))
    print("Largest difference:", (dense_result - conv_result).abs().max().item())
    try:
        convolution(coordinates)
    except RuntimeError as _error:
        print("Wrong layout:", str(_error).split("\n")[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The complete classifier

    The shared model lives in `pointcloud.py` so later notebooks use these same
    weights and layers. The source below is the actual implementation, rather than
    pseudocode. We will also write its forward calculation out in the next cell.
    Input is `(B, N, 3)` and output is `(B, classes)` raw logits.
    """)
    return


@app.cell
def _(inspect, mo, pc):
    mo.md("```python\n" + inspect.getsource(pc.PointNet) + "\n```")
    return


@app.cell
def _(pc, torch):
    clouds, targets, class_names = pc.make_shapes(2, 128, seed=8)
    with torch.random.fork_rng():
        torch.manual_seed(4)
        model = pc.PointNet(len(class_names)).eval()
    with torch.no_grad():
        local = model.local(clouds.transpose(1, 2))
        features = model.features(local)
        pooled, contributors = features.max(dim=2)
        logits = model.head(pooled)
    print("Coordinates:", tuple(clouds.shape))
    print("Local features:", tuple(local.shape))
    print("Features before pooling:", tuple(features.shape))
    print("Global representation:", tuple(pooled.shape))
    print("Class logits:", tuple(logits.shape))
    print("Matches forward:", torch.allclose(logits, model(clouds)))
    return (clouds, contributors, features, model)


@app.cell
def _(clouds, model, torch):
    _order = torch.randperm(clouds.shape[1], generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        print(
            "Shuffle score change:",
            (model(clouds) - model(clouds[:, _order])).abs().max().item(),
        )
        print("Fewer points accepted:", tuple(model(clouds[:, :32]).shape))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Which points supplied the maxima?

    Each feature channel can take its maximum from a different point. Highlight
    those contributors for one cloud, then select a channel to see its activation
    across the surface. This model is untrained, so the highlighted points are not
    yet evidence of useful recognition. Max contributors are also not a complete
    explanation of the prediction; the classifier combines all pooled channels.
    """)
    return


@app.cell
def _(mo):
    feature_channel = mo.ui.slider(
        0, 127, value=0, label="Feature channel", show_value=True
    )
    feature_channel
    return (feature_channel,)


@app.cell
def _(
    clouds, contributors, features, feature_channel, go, make_subplots, pc, torch, mo
):
    _points = clouds[0].numpy()
    _selected = torch.unique(contributors[0]).numpy()
    feature_plot = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Points supplying maxima", "One feature channel"),
    )
    feature_plot.add_trace(
        go.Scatter3d(
            x=_points[:, 0],
            y=_points[:, 1],
            z=_points[:, 2],
            mode="markers",
            name="All points",
            marker=dict(size=3, color="#1f77b4", opacity=0.3),
        ),
        row=1,
        col=1,
    )
    feature_plot.add_trace(
        go.Scatter3d(
            x=_points[_selected, 0],
            y=_points[_selected, 1],
            z=_points[_selected, 2],
            mode="markers",
            name="Max contributors",
            marker=dict(size=5, color="red"),
        ),
        row=1,
        col=1,
    )
    feature_plot.add_trace(
        go.Scatter3d(
            x=_points[:, 0],
            y=_points[:, 1],
            z=_points[:, 2],
            mode="markers",
            name="Feature activation",
            marker=dict(
                size=3,
                color=features[0, feature_channel.value].numpy(),
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Activation"),
            ),
        ),
        row=1,
        col=2,
    )
    _scene = pc.cloud_figure(_points, "").layout.scene.to_plotly_json()
    feature_plot.update_layout(
        scene=_scene,
        scene2=_scene,
        height=550,
        title="Inspect pooling and per-point features",
    )
    mo.ui.plotly(feature_plot, config={"displaylogo": False, "scrollZoom": True})
    return (feature_plot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Count the trainable parameters and explain why N does not appear in the count.
    2. Replace max pooling with mean. What else needs to change?
    3. Explain why our per-point features do not explicitly describe local neighbourhoods.
    4. Find the original paper's transformation networks. What extra problem do they address?

    [Part 4](PointCloudForMLPart4Classification.py) trains this model and examines its mistakes.
    """)
    return


if __name__ == "__main__":
    app.run()

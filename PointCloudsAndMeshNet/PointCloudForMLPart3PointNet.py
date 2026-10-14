#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Point Clouds for Machine Learning, Part 3: a small PointNet

    In [Part 2](PointCloudForMLPart2Sets.py) we applied the same mapping to every point, then used maximum pooling to combine the point features without depending on their row order. We will now use these operations to build a small point-cloud classifier.

    Our model follows the central idea in [PointNet](https://openaccess.thecvf.com/content_cvpr_2017/html/Qi_PointNet_Deep_Learning_CVPR_2017_paper.html): learn features for each point independently, pool them into one representation of the cloud, then use that representation to predict a class. We will trace this data flow from the input coordinates to the output scores, checking the tensor shape at each stage.

    We will implement the shared mapping using one-dimensional convolutions with a kernel size of one. Each convolution processes one point at a time using the same weights at every point. It combines feature channels without combining neighbouring points, so their order does not acquire a geometric meaning.

    Maximum pooling then keeps the largest response in each feature channel. Different points may supply different maxima. We will identify these contributing points to see which parts of the cloud supply the features passed to the classifier.

    This is a simplified teaching model compared to the original paper, I omit the original input and feature transformation networks, batch normalisation and dropout so we can inspect the shared mapping, pooling and classification stages first.

    By the end we should be able to trace every tensor dimension, explain how a kernel size one convolution implements a shared per-point mapping, and identify the points contributing to the pooled maxima.
    """)
    return


@app.cell
def _():
    import inspect
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

    return go, inspect, make_subplots, mo, nn, pc, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A 1D convolution can be a shared linear layer

    We can use [`Conv1d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) to apply the same learned mapping to every point. The main difference from `Linear` is how we arrange the tensor dimensions.

    Our cloud starts with shape `(B, N, 3)`: a batch of \(B\) clouds, each containing \(N\) points with three coordinates. `Conv1d` expects `(batch, channels, points)`, so we transpose it to `(B, 3, N)`. The coordinates become the three input channels.

    With `kernel_size=1`, the convolution processes one point at a time. Each output channel is a weighted combination of that point’s input channels, plus a bias:

    \[
    y_{b,k,i}
    =
    \sum_{j=1}^{C_{\text{in}}} W_{k,j}\,x_{b,j,i}
    +
    \beta_k.
    \]

    Here, \(i\) identifies the point, \(j\) the input channel and \(k\) the output channel. The weights \(W\) and biases \(\beta\) are shared across all points. For example, `Conv1d(3, 8, kernel_size=1)` maps each point’s three coordinates to eight features, changing `(B, 3, N)` into `(B, 8, N)`.

    This is the same calculation as applying `Linear(3, 8)` to every point in a tensor with shape `(B, N, 3)`. The outputs contain the same features, arranged along different dimensions.

    The convolution does not combine different points or search for neighbours in 3D. A kernel size of three would combine neighbouring **array positions**. In an unordered cloud, consecutive rows need not contain points that are close together in space. Shuffling those rows would change the groups processed by the convolution, even though the geometry remained unchanged.

    To check the equivalence, we can copy a linear layer’s weights and bias into a kernel-size-one convolution. The linear weights have shape `(8, 3)`, whilst the convolution weights have shape `(8, 3, 1)`, so we add a final dimension when copying them. After transposing the input and output, the two layers should produce matching values within floating-point tolerance.
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
    return clouds, contributors, features, model


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

    Maximum pooling reduces the per-point features to one feature vector for the whole cloud. For each channel, it keeps the largest activation across all points. Different channels can take their maxima from different points, and one point may supply the maximum for several channels.

    We will highlight these contributing points for one cloud. We can then select a feature channel and colour every sampled point by its activation. This lets us see where that channel responds strongly and which point supplies the value retained by pooling.

    The model is untrained, so these responses come from randomly initialised weights. A highlighted point tells us where a maximum occurred; it does not yet show that the network has learnt a useful feature for recognising the object.

    The contributors also give us only part of the explanation for a prediction. They show which points supplied the pooled values, but the classifier combines all those values to produce its output scores. A large activation does not, by itself, tell us how strongly it affects a particular class.

    If several points have exactly the same maximum activation, the displayed index may identify just one of them. The pooled value remains the same whichever tied point we select.
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
    clouds,
    contributors,
    feature_channel,
    features,
    go,
    make_subplots,
    mo,
    pc,
    torch,
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
    return


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

#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Point Clouds for Machine Learning, Part 2: a cloud is a set

    In [Part 1](PointCloudForMLPart1Sampling.py) we sampled a surface. The row
    numbers were an accident of sampling, rather than a property of the object.
    We will shuffle those rows and compare an MLP which flattens the cloud with a
    shared per-point mapping followed by pooling.

    By the end we should be able to explain permutation invariance and test it
    using output scores, rather than only looking at the predicted class.
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
def _(torch, nn):
    points = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]]
    )
    order = torch.tensor([2, 0, 3, 1])
    with torch.random.fork_rng():
        torch.manual_seed(7)
        flat_mlp = nn.Sequential(
            nn.Flatten(), nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 3)
        )
        shared_mlp = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 5))
    print("Original shape:", tuple(points.shape))
    print("Shuffled rows:\n", points[:, order])
    return (flat_mlp, points, order, shared_mlp)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Flattening gives each row a different job

    `Flatten` turns `(batch, points, xyz)` into `(batch, points * xyz)`.
    The following linear layer has different weights for point 0 and point 1.
    Changing their order therefore changes which weights process their coordinates.
    We can observe this before training; the architecture itself makes no promise about order.
    """)
    return


@app.cell
def _(flat_mlp, order, points, torch):
    with torch.no_grad():
        original_scores = flat_mlp(points)
        shuffled_scores = flat_mlp(points[:, order])
    print("Original logits:", original_scores)
    print("Shuffled logits:", shuffled_scores)
    print("Largest change:", (original_scores - shuffled_scores).abs().max().item())
    try:
        flat_mlp(points[:, :3])
    except RuntimeError as _error:
        print("Fewer points:", str(_error).split("\n")[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Shared features and a symmetric reduction

    `Linear` acts on the final dimension, so the same weights can process every
    point in `(B, N, 3)`. We then pool over N, independently for each feature channel:

    $$f(X)=g\left(\max_i h(x_i)\right).$$

    Maximum does not depend on order. If P permutes the input rows, then
    $f(PX)=f(X)$: the classification output is permutation invariant.
    Mean and sum also provide symmetric reductions, but have different responses
    to duplicate points and changing point counts.
    """)
    return


@app.cell
def _(shared_mlp, points, order, torch):
    with torch.no_grad():
        local_features = shared_mlp(points)
        shuffled_features = shared_mlp(points[:, order])
        global_features = local_features.amax(dim=1)
    print("Local feature shape:", tuple(local_features.shape))
    print("Global feature shape:", tuple(global_features.shape))
    print(
        "Local features follow the shuffle:",
        torch.allclose(shuffled_features, local_features[:, order]),
    )
    print(
        "Max pooling ignores order:",
        torch.allclose(global_features, shuffled_features.amax(dim=1)),
    )
    print("One pooled value per feature, not one selected point:", global_features)
    return


@app.cell
def _(mo):
    shuffle_seed = mo.ui.slider(0, 50, value=1, label="Shuffle seed", show_value=True)
    shuffle_seed
    return (shuffle_seed,)


@app.cell
def _(points, flat_mlp, shared_mlp, shuffle_seed, torch):
    _permutation = torch.randperm(
        points.shape[1], generator=torch.Generator().manual_seed(shuffle_seed.value)
    )
    with torch.no_grad():
        print("Permutation:", _permutation.tolist())
        print(
            "Flattened MLP score change:",
            (flat_mlp(points) - flat_mlp(points[:, _permutation])).abs().max().item(),
        )
        print(
            "Shared MLP + max change:",
            (shared_mlp(points).amax(1) - shared_mlp(points[:, _permutation]).amax(1))
            .abs()
            .max()
            .item(),
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Invariance and equivariance

    Classification gives one answer for the object, so shuffling should leave it
    unchanged. Segmentation gives one answer for each point, so its answers should
    follow the shuffle: `s(PX) = P s(X)`. This is permutation equivariance.

    Neither property implies rotation invariance. A shared MLP still sees xyz
    coordinates, and a rotation changes them. A model may also accept fewer
    points whilst giving a different prediction: accepting a shape is not a guarantee of accuracy.

    ## Exercises

    1. Replace maximum with mean and sum. Which survive shuffling?
    2. Duplicate one point and compare all three reductions.
    3. Rotate the cloud by 90 degrees. Does the pooled vector stay the same?
    4. Why is an unchanged argmax class weaker evidence than unchanged logits?

    [Part 3](PointCloudForMLPart3PointNet.py) turns this construction into a classifier.
    """)
    return


if __name__ == "__main__":
    app.run()

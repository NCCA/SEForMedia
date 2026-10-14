#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Point Clouds for Machine Learning, Part 2: a cloud is a set

    In [Part 1](PointCloudForMLPart1Sampling.py) we sampled points from a surface and stored them as rows in a NumPy array. Each row contains a position, but its row number simply records when that point was generated. The object has no first or last point, and points in consecutive rows need not be close together in space.

    We can shuffle these rows without moving, adding or removing any points. The array changes, but the shape it describes stays the same. For a network working with a point cloud as a set, we want the same output whichever order we use. This property is called *permutation invariance*.

    We will compare two ways of processing the cloud. First, we will flatten the array into one long vector and pass it to a multilayer perceptron (MLP). This gives each row a particular place in the input, so shuffling the points can change the output.

    We will then apply the same mapping to every point independently and combine the resulting features using pooling. Shuffling the input shuffles the per-point features, whilst a symmetric pooling operation, such as a sum, mean or maximum, gives the same result regardless of their order.

    By the end we should be able to explain permutation invariance and test it by comparing output scores before and after shuffling. Checking only the predicted class is not enough: the scores could change substantially whilst the highest-scoring class remains the same.
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

    return mo, nn, torch


@app.cell
def _(nn, torch):
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
    return flat_mlp, order, points, shared_mlp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The code above does the following, we start with four 3D points:

    ```python
    points = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]]
    )
    ```

    The extra outer pair of brackets adds a batch dimension. The tensor has shape `(1, 4, 3)`:

    - `1`: one cloud in the batch.
    - `4`: four points in the cloud.
    - `3`: the `x`, `y` and `z` coordinates of each point.

    We then specify a new order for the points:

    ```python
    order = torch.tensor([2, 0, 3, 1])
    ```

    This means take point 2 first, then point 0, point 3 and point 1. In `points[:, order]`, the colon selects every cloud in the batch and `order` rearranges the point rows. Each point keeps its coordinates together, so the shuffled cloud describes exactly the same shape.

    ### Repeatable initial weights

    ```python
    with torch.random.fork_rng():
        torch.manual_seed(7)
    ```

    Our linear layers start with randomly initialised weights. Setting the seed makes that initialisation repeatable.

    `fork_rng()` saves the surrounding random-number state and restores it when we leave the block. This lets us create repeatable networks without changing the random sequence used by later code. The networks remain available after the block ends.

    ### Flatten the cloud

    ```python
    flat_mlp = nn.Sequential(
        nn.Flatten(), nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 3)
    )
    ```

    This network processes the whole cloud as one vector. The shapes passing through its layers are:

    ```text
    (batch, 4, 3)
        → (batch, 12)  Flatten
        → (batch, 8)   Linear
        → (batch, 8)   ReLU
        → (batch, 3)   Linear
    ```

    `nn.Flatten()` keeps the batch dimension and joins the four points’ coordinates into 12 values. The first linear layer maps these values to eight features. `ReLU()` replaces negative values with zero, and the final linear layer produces three outputs.

    Each position in the flattened vector has its own learned weights. Moving a point to another row changes which weights process its coordinates. The network can therefore produce different outputs when we shuffle the points, even though their positions in space have not changed.

    This setup also fixes the input size: the first linear layer expects exactly 12 values, corresponding to four 3D points.

    ### Apply a shared mapping to every point

    ```python
    shared_mlp = nn.Sequential(
        nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 5)
    )
    ```

    `nn.Linear` operates on the last dimension of its input. Here it processes each point’s three coordinates independently:

    ```text
    (batch, 4, 3)
        → (batch, 4, 8)  Linear
        → (batch, 4, 8)  ReLU
        → (batch, 4, 5)  Linear
    ```

    Each point becomes a vector of five features. All points pass through the same layers with the same weights; we do not create a separate network for each row.

    If we shuffle the input points, their feature vectors are shuffled in the same way. This property is called *permutation equivariance*: the output follows the permutation of the input.

    The shared mapping alone does not produce one representation of the whole cloud. We still need to combine the point features using a symmetric pooling operation, such as a sum, mean or maximum across the point dimension. Pooling gives the same result regardless of row order, producing *permutation invariance*.

    ### Inspect the shuffled cloud

    ```python
    print("Original shape:", tuple(points.shape))
    print("Shuffled rows:\n", points[:, order])
    ```

    These lines print:

    ```text
    Original shape: (1, 4, 3)
    Shuffled rows:
     tensor([[[0., 2., 0.],
             [0., 0., 0.],
             [0., 0., 3.],
             [1., 0., 0.]]])
    ```

    At this stage we have created the networks and inspected the reordered input. We have not yet passed the cloud through either network or compared their outputs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Flattening makes the network depend on row order

    `Flatten` turns a cloud with shape `(batch, points, xyz)` into a vector with shape `(batch, points * xyz)`. For our four-point cloud, each batch item becomes 12 values:

    ```text
    [x₀, y₀, z₀, x₁, y₁, z₁, x₂, y₂, z₂, x₃, y₃, z₃]
    ```

    The following linear layer learns separate weights for each position in this vector. The coordinates of point 0 use one set of weights, whilst the coordinates of point 1 use another.

    If we swap those points, their positions in space stay the same, but their coordinates move to different positions in the input vector. Different weights now process them, so the network can produce different output scores for exactly the same cloud.

    We can observe this before training, using the randomly initialised weights. The dependence on row order comes from the architecture. Training might reduce that dependence, but flattening followed by a linear layer does not guarantee permutation invariance.
    """)
    return


@app.cell
def _(flat_mlp, order, points, torch):
    with torch.no_grad():
        original_scores = flat_mlp(points)
        shuffled_scores = flat_mlp(points[:, order])
    print("Original logits:", original_scores)
    print("Shuffled logits:", shuffled_scores)
    print(
        "Largest change:",
        (original_scores - shuffled_scores).abs().max().item(),
    )
    try:
        flat_mlp(points[:, :3])
    except RuntimeError as _error:
        print("Fewer points:", str(_error).split("\n")[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Shared features and a symmetric reduction

    Instead of flattening the cloud, we apply the same small network to every point. For an input with shape `(B, N, 3)`, `Linear` acts on the final dimension: each point’s three coordinates are processed using the same weights. Here, \(B\) is the batch size and \(N\) is the number of points.

    We call this shared mapping \(h\). If it produces \(K\) features per point, its output has shape `(B, N, K)`. Shuffling the input rows shuffles these feature vectors in the same way, without changing their values.

    We then pool across the point dimension to produce one feature vector for each cloud. With maximum pooling, we take the largest value independently in each feature channel:

    \[
    z_k=\max_{i=1,\ldots,N}h(x_i)_k.
    \]

    The result has shape `(B, K)`. Different channels can take their maximum from different points; we are combining features rather than selecting one point to represent the whole object.

    A final network \(g\) maps this pooled vector to classification scores:

    \[
    f(X)=g\left(\max_i h(x_i)\right).
    \]

    Reordering the points cannot change the maximum in any channel. If \(P\) permutes the input rows, then:

    \[
    f(PX)=f(X).
    \]

    The complete network is therefore *permutation invariant*: the same points produce the same classification scores regardless of their row order.

    Mean and sum pooling also ignore row order, but they respond differently to repeated points and changes in point count:

    - **Maximum** is unchanged by exact duplicates. Additional points affect it only if they introduce a larger value in a feature channel.
    - **Mean** averages the point features. Duplicating selected points gives them more influence, whilst repeating the entire cloud leaves the mean unchanged.
    - **Sum** accumulates the point features. Duplicates contribute again, and repeating the entire cloud multiplies the pooled vector by the number of repetitions.

    All three remove dependence on order. Our choice determines how the network responds to the number and distribution of sampled points.
    """)
    return


@app.cell
def _(order, points, shared_mlp, torch):
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
    print(
        "One pooled value per feature, not one selected point:",
        global_features,
    )
    return


@app.cell
def _(mo):
    shuffle_seed = mo.ui.slider(0, 50, value=1, label="Shuffle seed", show_value=True)
    shuffle_seed
    return (shuffle_seed,)


@app.cell
def _(flat_mlp, points, shared_mlp, shuffle_seed, torch):
    _permutation = torch.randperm(
        points.shape[1],
        generator=torch.Generator().manual_seed(shuffle_seed.value),
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
    ## Why permutation invariance matters

    A point cloud describes a shape through the positions of its points. The order of those points in an array usually has no meaning. Loading a file, shuffling training data or changing how we collect samples can rearrange the rows without changing the geometry.

    We want our network to recognise the same object after such a rearrangement. If its scores depend on row order, it can respond to an arbitrary detail of how we stored the data. Two arrays containing exactly the same points could then produce different predictions.

    This also matters during training. A network that depends on row order may learn associations between particular input positions and the training labels. Those associations may disappear when we present the points in another order. We could train with shuffled examples to encourage the network to ignore order, but shared point processing followed by symmetric pooling gives us that property directly.

    The shared mapping also lets us process different numbers of points. Each point passes through the same function, and pooling reduces the resulting `(B, N, K)` features to `(B, K)`. The final classifier therefore receives a fixed-size representation even when \(N\) changes. Batching clouds with different point counts still needs care, such as padding and masking, but the network’s input layers no longer require a fixed number of coordinates.

    There is an important distinction between **shuffling** and **resampling**. Shuffling preserves every point, so a permutation-invariant network should preserve its output. Resampling creates a different set of positions on the same surface. A sparse sample might miss a thin part or a small feature, so the output can change even though the source mesh is unchanged. Permutation invariance removes dependence on order; it does not guarantee that every sample contains enough information to recognise the shape.

    Pooling also discards information. With maximum pooling, we retain the strongest response in each feature channel, but lose which point produced it and how often that response occurred. The shared mapping must therefore learn features whose pooled values are useful for the task. An invariant architecture gives us a consistent way to combine points, but we still need suitable features, representative samples and training data.

    We test this property by comparing the complete output scores before and after a shuffle. Checking only the predicted class can hide changes: the scores may move whilst the same class remains highest. Small floating-point differences can occur, especially with sum or mean reductions, so we compare scores using a numerical tolerance.

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

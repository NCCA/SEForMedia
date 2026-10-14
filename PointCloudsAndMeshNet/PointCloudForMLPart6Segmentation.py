#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Point Clouds for Machine Learning, Part 6: part segmentation

    Classification gives us one prediction for a whole object. Part segmentation gives us a prediction for every point. For a chair, we want to identify which points belong to parts such as the seat, back and legs.

    We will extend the shared point mapping from [Part 3](PointCloudForMLPart3PointNet.py). Instead of keeping only the pooled representation, we retain each point’s local features and combine them with the global features of the cloud. This gives the prediction layers information about both the individual point and the object it belongs to.

    To do this, we repeat the global feature vector for every point and concatenate it with the local features. The resulting tensor contains one combined feature vector per point. A shared prediction network then produces part scores for each of them.

    The output must remain aligned with the input. If we shuffle the points, their predictions should follow the same shuffle. This is *permutation equivariance*: unlike the single output of a classifier, the segmentation output changes order to match the points it describes. We must also preserve this correspondence when sampling or selecting points, carrying each point’s label with it.

    We will begin with generated chairs, then provide a route to labelled ShapeNet Part chairs. We will evaluate predictions using per-shape part intersection over union (IoU), which compares the predicted and labelled point sets for each part. This helps us inspect performance on individual parts rather than relying only on overall point accuracy.

    The figures use Plotly. Drag a 3D view to rotate it and scroll to zoom. These camera controls change our view; they do not change the input coordinates or repeat training.

    By the end we should be able to combine local and global features, keep labels aligned with sampled points, test permutation equivariance, and evaluate segmentation using per-shape part IoU.

    The loss and IoU examples run when the notebook opens. To run the complete generated-data experiment, leave the ShapeNet path blank and press **Load labelled chairs and train** below. Learning curves, held-out predictions and checkpoint saving appear after that run. Opening the notebook does not start training.
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
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # This also works when marimo starts from the notebook's own folder.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pointcloud as pc

    return Path, go, inspect, make_subplots, mo, nn, pc, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Give each point the same global context

    For segmentation, we need to retain information about each point whilst also giving it context from the whole object. Our local features have shape `(B, 64, N)`: \(B\) clouds, 64 feature channels and \(N\) points.

    A deeper shared mapping produces 128 channels per point. Maximum pooling across the point dimension reduces these to `(B, 128, 1)`. This is one global feature vector per cloud, summarising the strongest response in each channel.

    We use `expand` to make that vector available at every point:

    ```text
    (B, 128, 1) → (B, 128, N)
    ```

    Every point in a cloud receives the same global context. `expand` creates a view that reuses the underlying values rather than storing \(N\) separate copies.

    We then concatenate the local and global features along the channel dimension:

    ```text
    Local features:   (B,  64, N)
    Global context:   (B, 128, N)
    Combined:         (B, 192, N)
    ```

    Each point now has 192 features: 64 describing that point and 128 providing context about the whole cloud. Although `expand` avoids copying the global vector, concatenation creates a new tensor containing the combined features.

    A shared prediction head processes these vectors and produces logits with shape `(B, parts, N)`. For every point, we have one raw score for each possible part. Selecting the highest-scoring part along the channel dimension gives predicted labels with shape `(B, N)`.

    Here, *local* means **per-point**. These features are calculated from each point independently; the network does not explicitly gather nearby points or describe a spatial neighbourhood. Points exchange information through the pooled global vector, which is shared across the cloud.
    """)
    return


@app.cell
def _(inspect, mo, pc):
    mo.md("```python\n" + inspect.getsource(pc.SegmentNet) + "\n```")
    return


@app.cell
def _(mo, pc, torch):
    example_chairs, example_parts, part_names = pc.make_parts(2, 192, 7)
    with torch.random.fork_rng():
        torch.manual_seed(7)
        segment_probe = pc.SegmentNet(3).eval()
    with torch.no_grad():
        local_features = segment_probe.local(example_chairs.transpose(1, 2))
        global_context = segment_probe.features(local_features).amax(2, keepdim=True)
        combined = torch.cat(
            [
                local_features,
                global_context.expand(-1, -1, example_chairs.shape[1]),
            ],
            dim=1,
        )
    print(
        "Local:",
        tuple(local_features.shape),
        "global:",
        tuple(global_context.shape),
    )
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
    return example_chairs, example_parts, segment_probe


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Shuffle coordinates and labels together

    Segmentation produces one prediction for every input point, so the order of the predictions must follow the order of the points. If \(P\) permutes the point rows and \(s\) is our segmentation network, we expect:

    \[
    s(PX)=P\,s(X).
    \]

    This is *permutation equivariance*. Shuffling the input should rearrange the output in exactly the same way. To test it, we compare the scores from the shuffled cloud with the original scores rearranged using that same permutation.

    Classification uses a different test because it produces one answer for the whole cloud. Its output should remain unchanged after shuffling. For segmentation, keeping the output order unchanged would attach predictions to the wrong points.

    The ground-truth labels must also move with their coordinates. If row 12 contains a seat point and its label, moving that point to row 3 must move its label to row 3 as well. We should generate one permutation and apply it to both arrays.

    Shuffling coordinates whilst leaving labels unchanged creates a silent training error. The arrays still have the expected shapes, and the labels still contain valid class indices, so cross entropy can calculate a loss. It cannot check whether a label belongs to the point at that position. The network would be trained against incorrectly matched coordinates and labels.

    The same rule applies when selecting a subset of points: use the same indices for the coordinates and their labels so the correspondence is preserved.
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

    Before training a chair model, we will check the loss using three points and two possible part labels. The first two predictions assign a high score to the correct part. The third is less certain, giving the correct part a lower probability, so we expect it to contribute more loss.

    For segmentation, [`cross_entropy`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html) takes logits with shape `(B, parts, N)` and integer targets with shape `(B, N)`. Each target identifies the correct part for the point at the same batch and point position.

    We pass **raw logits**, without applying softmax first. Cross entropy calculates the negative log-probability of the correct label internally. For one point \(i\), omitting the batch index, this is:

    \[
    \ell_i
    =
    -\log\left(
    \frac{\exp(z_{y_i,i})}
    {\sum_k \exp(z_{k,i})}
    \right),
    \]

    where \(z_{k,i}\) is the logit for part \(k\) at point \(i\), and \(y_i\) is that point’s correct part label.

    The loss depends on the probability assigned to the correct label. A probability close to one gives a loss close to zero. A lower probability gives a larger loss, with confident incorrect predictions penalised strongly. The predicted label alone therefore does not tell us the loss: two points can both be classified correctly but contribute different amounts.

    Using `reduction="none"` returns a loss tensor with shape `(B, N)`, allowing us to inspect each point separately. The default mean reduction instead averages these losses into the single scalar normally used for backpropagation.

    We will check the calculation by applying `log_softmax` across the part dimension, selecting the log-probability of each point’s correct label, and negating it. This should match the unreduced cross-entropy result. Using `log_softmax` also avoids explicitly calculating potentially large exponentials in the formula above.
    """)
    return


@app.cell
def _(nn, torch):
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
    return (point_losses,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Point weighting and reshaping the logits

    With the default mean reduction, each point contributes equally to the loss. This does not give each **part** equal weight. A seat containing 600 sampled points contributes 600 loss terms, whilst a leg containing 100 points contributes only 100. The larger part therefore has more influence on the average.

    This matters when interpreting our results. A model can achieve high point accuracy by predicting the parts containing most points correctly whilst performing poorly on smaller parts. We will compare point accuracy with part IoU to examine this difference.

    We can also calculate cross entropy by treating all points in the batch as separate examples. Starting with logits of shape `(B, parts, N)`, we first move the part axis to the end:

    ```python
    flat_logits = logits.permute(0, 2, 1).reshape(-1, num_parts)
    flat_targets = targets.reshape(-1)
    ```

    The shapes change as follows:

    ```text
    Logits:   (B, parts, N) → (B, N, parts) → (B*N, parts)
    Targets:  (B, N)                       → (B*N,)
    ```

    Each row of `flat_logits` now contains all part scores for one point, and the corresponding entry in `flat_targets` contains that point’s correct label.

    Reshaping `(B, parts, N)` directly into `(B*N, parts)` does not perform this rearrangement. It groups values according to their existing layout, potentially combining scores from different points into one row. The resulting shapes look valid, so cross entropy can run without reporting an error, even though the scores and targets no longer correspond correctly.

    The rule is to put all class scores for one point together before flattening the batch and point dimensions. With that ordering preserved, the flattened calculation gives the same loss as passing `(B, parts, N)` logits and `(B, N)` targets directly.
    """)
    return


@app.cell
def _(nn, point_losses, torch):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generated chairs or ShapeNet Part

    We can run this lesson using generated chairs or labelled chair point clouds from ShapeNet Part. Start with the generated data: leave the ShapeNet path blank and press **Load labelled chairs and train**. These chairs have three labelled parts and deliberately simple geometry. They let us check sampling, label alignment, training and evaluation; their results are not ShapeNet benchmark scores.

    ### Prepare the ShapeNet Part data

    For real data, we use [PyTorch Geometric’s ShapeNet dataset loader](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.ShapeNet.html). This provides point clouds with a part label for each point, rather than unlabelled meshes that we must sample ourselves.

    Prepare the local cache explicitly before running the training example. Check the dataset’s access and usage conditions, then run these commands from the repository root:

    ```bash
    uv run --with torch-geometric python -c "from torch_geometric.datasets import ShapeNet; ShapeNet('data/ShapeNetPart', categories=['Chair'], split='train', include_normals=False)"

    ```

    The first command downloads the source archive if needed and prepares the chair data. This may take some time. Although we request the training split, the loader’s processing step prepares the train, validation and test caches, along with a combined train/validation cache. The download itself contains more than just chairs; `categories=['Chair']` selects the category to process. See the [loader implementation](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/shapenet.html) for these details.

    Once preparation finishes, enter `data/ShapeNetPart` in the notebook’s ShapeNet path field, or use its absolute path. Select the cache root containing `raw` and `processed`, rather than either subfolder or an individual file. The notebook requires a complete prepared cache.

    ### Keep points and labels together

    The notebook uses separate training, validation and test splits. For each cloud, it selects point indices once and applies the same indices to both coordinates and labels. This preserves the correspondence between a point and its target part.

    We use only the `x`, `y` and `z` coordinates as input. Setting `include_normals=False` excludes the supplied surface normals from the input features.

    ShapeNet Chair uses four part IDs: `12`, `13`, `14` and `15`. Our loader remaps these to `0`, `1`, `2` and `3`, giving us consecutive class indices for a four-output prediction head. This differs from the three-part generated chairs.

    The real-data plots retain the original part IDs in their names so we can trace predictions back to the dataset annotations. Inspect those annotations before assigning semantic names: a numeric ID identifies a labelled region, but does not explain what that region represents.
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
def _(mo, pc, segmentation_form):
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
    print(
        "Coordinates:",
        tuple(seg_train_x.shape),
        "labels:",
        tuple(seg_train_y.shape),
    )
    print("Part names:", seg_names)
    return (
        seg_names,
        seg_settings,
        seg_test_x,
        seg_test_y,
        seg_train_x,
        seg_train_y,
        seg_val_x,
        seg_val_y,
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
    mo,
    nn,
    pc,
    seg_names,
    seg_settings,
    seg_train_x,
    seg_train_y,
    seg_val_x,
    seg_val_y,
    torch,
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
                segmentation_model(seg_train_x[_indices]),
                seg_train_y[_indices],
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
    return seg_history, segmentation_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Accuracy and intersection over union

    Point accuracy measures the fraction of points assigned the correct label. It is useful, but parts containing many points have more influence on the result. If most points belong to the seat, a model that labels everything as seat could achieve a reasonable accuracy whilst missing the back and legs entirely.

    We therefore also calculate *intersection over union* (IoU) for each part. Let \(P_k\) be the set of points predicted as part \(k\), and \(T_k\) the set whose true label is part \(k\):

    \[
    \operatorname{IoU}_k
    =
    \frac{|P_k\cap T_k|}{|P_k\cup T_k|}.
    \]

    The intersection counts points correctly assigned to that part. The union includes every point predicted or labelled as that part, so it accounts for both missed points and incorrect additions. An IoU of one means a perfect match; zero means there is no overlap.

    For example, suppose a part contains 100 labelled points. The model predicts 90 points as that part, of which 80 are correct. There are 20 missed points and 10 incorrect additions, giving:

    \[
    \operatorname{IoU}
    =
    \frac{80}{100+90-80}
    =
    \frac{80}{110}
    \approx 0.73.
    \]

    We calculate IoU separately for each valid part of each shape. We then average the part IoUs within that shape, followed by an average across shapes. This gives each valid part equal weight within a shape and each shape equal weight in the final score, regardless of its point count.

    We also need a rule for absent parts. If a part appears in neither the prediction nor the target, the union is empty and the formula would divide by zero. We assign **IoU = 1** in this case, treating the correct absence as a match. If the target contains no points for a part but the model predicts some, the intersection is empty and the union is not, giving **IoU = 0**.

    We state this convention explicitly because excluding absent parts or assigning them another value changes the reported score.

    This lesson evaluates one object category. A full ShapeNet Part evaluation must use the valid part set for each category when calculating these averages; it should not average every shape over all part IDs in the dataset.
    """)
    return


@app.cell
def _(pc, torch):
    _actual = torch.tensor([[0, 0, 1, 1]])
    _predicted = torch.tensor([[0, 1, 1, 1]])
    print("Hand-checkable IoUs:", pc.part_iou(_predicted, _actual, 3))
    print("Expected: 1/2, 2/3, 1 for the absent third part")
    return


@app.cell
def _(
    go,
    mo,
    pc,
    seg_history,
    seg_names,
    seg_test_x,
    seg_test_y,
    segmentation_model,
    torch,
):
    with torch.no_grad():
        seg_logits = torch.cat(
            [segmentation_model(_batch) for _batch in seg_test_x.split(16)]
        )
        seg_predictions = seg_logits.argmax(1)
    seg_ious = pc.part_iou(seg_predictions, seg_test_y, len(seg_names))
    print(
        "Point accuracy:",
        (seg_predictions == seg_test_y).float().mean().item(),
    )
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
    return seg_ious, seg_predictions


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
def _(mo, seg_ious, seg_predictions, seg_test_y):
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
    return


@app.cell
def _(mo, seg_test_x):
    chair_index = mo.ui.slider(
        0,
        len(seg_test_x) - 1,
        value=0,
        label="Held-out chair",
        show_value=True,
    )
    chair_index
    return (chair_index,)


@app.cell
def _(
    chair_index,
    make_subplots,
    mo,
    pc,
    seg_ious,
    seg_names,
    seg_predictions,
    seg_test_x,
    seg_test_y,
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
    return


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
def _(Path, mo):
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
    Path,
    mo,
    pc,
    seg_checkpoint_path,
    seg_history,
    seg_names,
    seg_save_button,
    seg_settings,
    seg_test_x,
    segmentation_model,
    torch,
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now have a complete segmentation experiment: labelled points, a shared
    model with global context, validation-based selection, held-out overlap
    measurements, an error viewer and a reloadable checkpoint.

    The generated chairs make the setup easy to check. Their simple parts
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

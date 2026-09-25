#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PyTorch for Machine Learning, Part 4: evaluation and inference

    Once we have trained a model, we need to evaluate its predictions and measure how well it performs. This notebook covers how we turn model outputs into predictions and prepare the model for evaluation.

    Our classification models output **logits**: raw scores that can take any real value. We will look at converting these scores into probabilities and choosing a predicted class for binary and multi-class problems. When we only need the class label, we can often work directly with the logits.

    We will then look at two separate controls. `model.eval()` changes the behaviour of layers such as dropout and batch normalisation. `torch.no_grad()` and `torch.inference_mode()` disable gradient tracking, avoiding work we do not need during inference.

    Setting evaluation mode does not disable gradient tracking, and disabling gradient tracking does not set evaluation mode. We normally use both when evaluating a trained model.
    """)
    return


@app.cell
def _():
    import torch
    from torch import nn

    torch.manual_seed(42)
    return nn, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [model.eval()](https://pytorch.org/docs/stable/generated/torch.nn.Module.eval.html) and [model.train()](https://pytorch.org/docs/stable/generated/torch.nn.Module.train.html)

    These methods set the mode on the model and all its submodules. They __do not__ run a training or evaluation step.

    ```python
    model.train()  # enable training mode
    model.eval()   # equivalent to model.train(False)
    ```

    Some layers change their behaviour according to this mode. Two common examples are:

    - **Dropout:** randomly zeroes activations during training and passes them through unchanged during evaluation.
    - **BatchNorm:** uses the current batch’s statistics during training and, by default, stored running statistics during evaluation.

    Layers such as `Linear` and `ReLU` behave the same in both modes. If these are the only layers in our model, calling `eval()` changes the mode flags but not the output calculation. Forgetting it may therefore go unnoticed until we add a layer such as dropout.

    Neither method controls gradient tracking. For evaluation, we normally combine `model.eval()` with `torch.no_grad()` or `torch.inference_mode()`.

    The example below passes the same input through the same dropout model twice in each mode. Training mode can produce different outputs because of the random dropout masks; evaluation mode leaves the activations unchanged.
    """)
    return


@app.cell
def _(nn, torch):
    dropout_model = nn.Sequential(nn.Linear(4, 6), nn.Dropout(p=0.5))
    fixed_input = torch.randn(1, 4)

    dropout_model.train()
    print("train mode, same input twice:")
    print("  ", dropout_model(fixed_input).detach().numpy().round(3))
    print("  ", dropout_model(fixed_input).detach().numpy().round(3))
    print("  different every time, and about half the values are zero")

    dropout_model.eval()
    print()
    print("eval mode, same input twice:")
    print("  ", dropout_model(fixed_input).detach().numpy().round(3))
    print("  ", dropout_model(fixed_input).detach().numpy().round(3))
    print("  identical, and nothing dropped")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice the scale of the surviving activations as well as the zeros. During training, dropout divides each surviving value by \(1-p\), preserving its expected value. At \(p=0.5\), half the activations are dropped on average and the surviving values are doubled.

    This is called *inverted dropout*. The scaling happens during training, so evaluation can pass the activations through unchanged.

    Batch normalisation also behaves differently between training and evaluation. Using the wrong mode can be harder to spot here, because the outputs may still look reasonable.
    """)
    return


@app.cell
def _(nn, torch):
    bn = nn.BatchNorm1d(3)

    print("running stats before seeing any data:")
    print(
        "  mean",
        bn.running_mean.numpy().round(3),
        "var",
        bn.running_var.numpy().round(3),
    )

    bn.train()
    for _ in range(50):
        bn(torch.randn(16, 3) * 5 + 10)  # mean about 10, sd about 5

    print()
    print("after 50 training batches:")
    print(
        "  mean",
        bn.running_mean.numpy().round(3),
        "var",
        bn.running_var.numpy().round(3),
    )

    bn.eval()
    print()
    print("eval output uses those stored stats, so one sample is fine:")
    print("  ", bn(torch.randn(1, 3) * 5 + 10).detach().numpy().round(3))

    bn.train()
    try:
        bn(torch.randn(1, 3))  # train mode needs a batch to compute statistics from
    except ValueError as e:
        print()
        print("but in train mode a batch of one raises:")
        print("  ValueError:", str(e).split("\n")[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [torch.no_grad](https://pytorch.org/docs/stable/generated/torch.no_grad.html) and [torch.inference_mode](https://pytorch.org/docs/stable/generated/torch.inference_mode.html)

    Both are context managers that stop autograd recording, so no graph is built. That saves the memory the graph would occupy and the time spent building it.

    ```python
    with torch.no_grad():
        ...
    with torch.inference_mode():
        ...
    ```

    They are **not** the same thing as `eval()`. `eval()` changes what the layers compute; these change whether the computation is recorded. You need both, and one does not imply the other:

    | | `model.eval()` | `no_grad` / `inference_mode` |
    | --- | --- | --- |
    | Dropout and BatchNorm behaviour | changes it | no effect |
    | Graph recording | no effect | switches it off |
    | Memory used | unchanged | lower |

    The difference between the two context managers is how strict they are. `inference_mode` is newer and faster; tensors created inside it are permanently marked and cannot later be used in a graph at all. `no_grad` is the looser one, and the one to use if the result will feed back into something trainable.

    The demos here use both `inference_mode` in the from-scratch training loops, `no_grad` in the transfer-learning validation. Either is fine for ordinary evaluation.
    """)
    return


@app.cell
def _(nn, torch):
    probe_model = nn.Linear(4, 2)
    probe_input = torch.randn(3, 4)

    normal = probe_model(probe_input)
    with torch.no_grad():
        no_grad_out = probe_model(probe_input)
    with torch.inference_mode():
        inference_out = probe_model(probe_input)

    print("normal          grad_fn:", normal.grad_fn)
    print("no_grad         grad_fn:", no_grad_out.grad_fn)
    print("inference_mode  grad_fn:", inference_out.grad_fn)
    return (inference_out,)


@app.cell
def _(inference_out, torch):
    # the extra strictness of inference_mode, which is the only practical difference
    trainable = torch.randn(3, 2, requires_grad=True)

    try:
        (inference_out * trainable).sum().backward()
    except RuntimeError as e:
        print("RuntimeError:", str(e).split("\n")[0])
        print()
        print("a no_grad tensor would have been allowed back into the graph here")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From logits to predictions

    Our classification models output logits: raw scores that can take any real value. To interpret them as probabilities or choose a class, we use different operations for binary and multi-class classification.

    ### Binary: sigmoid and a threshold

    For binary classification with one logit per sample, [`torch.sigmoid`](https://pytorch.org/docs/stable/generated/torch.sigmoid.html) converts the score into a probability for class 1. We then apply a threshold to choose the predicted class:

    ```python
    probabilities = torch.sigmoid(y_logits)
    y_pred = (probabilities >= 0.5).long()
    ```

    `Classification/BinaryClassification.py` uses:

    ```python
    y_pred = torch.round(torch.sigmoid(y_logits))
    ```

    This gives the same labels except at exactly `0.5`: `torch.round()` rounds that value to `0`, whereas our explicit `>= 0.5` threshold assigns class `1`. Writing the comparison makes the choice clear.

    We use this conversion when inspecting predictions, either during or after training. For the loss calculation, we pass logits directly to `BCEWithLogitsLoss`, which already includes the sigmoid calculation.
    """)
    return


@app.cell
def _(torch):
    binary_logits = torch.tensor([2.5, -1.0, 0.3, -0.05, 4.0])

    probs_b = torch.sigmoid(binary_logits)
    preds_b = torch.round(probs_b)

    print("logits     ", binary_logits.numpy().round(3))
    print("sigmoid    ", probs_b.numpy().round(3))
    print("round      ", preds_b.numpy())
    print()
    print(
        "note the logit's sign already decides it:",
        (binary_logits > 0).int().numpy(),
    )
    print("so sigmoid is only needed when you want the confidence")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Multi-class: [softmax](https://pytorch.org/docs/stable/generated/torch.softmax.html) and [argmax](https://pytorch.org/docs/stable/generated/torch.argmax.html)

    For multi-class classification, softmax converts logits into probabilities that sum to one across the classes. `argmax` returns the index of the largest value.

    ```python
    torch.softmax(input, dim)
    torch.argmax(input, dim=None, keepdim=False)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `dim` (softmax) | required | Dimension over which to calculate probabilities. Use `1` for logits shaped `(batch, classes)`. |
    | `dim` (argmax) | `None` | Dimension over which to find the largest value. If omitted, returns an index into the flattened tensor. |
    | `keepdim` | `False` | Retains the reduced dimension with size 1. |

    `Checkpoints/CheckPoints.py` combines the two operations:

    ```python
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    ```

    Softmax preserves the ordering of the scores within each sample, so we can choose the predicted class directly from the logits:

    ```python
    y_pred = y_logits.argmax(dim=1)
    ```

    We only need softmax when we want class probabilities. These express the model’s confidence, but they are not automatically calibrated: a probability of `0.9` does not guarantee that the model is correct 90% of the time on predictions with that score.
    """)
    return


@app.cell
def _(torch):
    multi_logits = torch.tensor(
        [
            [2.0, 1.0, 0.1],
            [0.5, 3.0, 0.2],
            [0.1, 0.2, 4.0],
            [1.5, 1.4, 0.3],
        ]
    )

    print("softmax(dim=1), each row sums to 1:")
    print(torch.softmax(multi_logits, dim=1).numpy().round(3))
    print()
    print("argmax on logits:", torch.argmax(multi_logits, dim=1).numpy())
    print(
        "argmax on probs: ",
        torch.softmax(multi_logits, dim=1).argmax(dim=1).numpy(),
    )
    print("identical, as promised")
    return (multi_logits,)


@app.cell
def _(multi_logits, torch):
    # the dim trap
    print(
        "argmax(dim=1) ",
        torch.argmax(multi_logits, dim=1).numpy(),
        " one per sample",
    )
    print(
        "argmax()      ",
        torch.argmax(multi_logits).item(),
        "      a flat index, useless here",
    )
    print()
    print(
        "and the wrong dim on softmax normalises down the batch instead of across classes:"
    )
    print(torch.softmax(multi_logits, dim=0).numpy().round(3))
    print("columns sum to 1, which means nothing")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### More than one answer: [topk](https://pytorch.org/docs/stable/generated/torch.topk.html)

    `argmax` returns the index of the highest score. When we want several candidates, `topk` returns the highest \(k\) values and their indices. We can use this to display a shortlist or calculate top-5 accuracy.

    ```python
    torch.topk(input, k, dim=None, largest=True, sorted=True)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `k` | required | Number of entries to return. |
    | `dim` | `None` | Dimension to select along. Defaults to the last dimension. |
    | `largest` | `True` | Selects the largest values; use `False` for the smallest. |
    | `sorted` | `True` | Returns the selected values in order. |

    The result is a named tuple containing `values` and `indices`. For predictions shaped `(batch, classes)`, we can write:

    ```python
    probabilities = torch.softmax(y_logits, dim=1)
    top_probs, top_classes = torch.topk(probabilities, k=5, dim=1)
    ```

    Both outputs have shape `(batch, 5)`. The values are probabilities here because we applied softmax first. Calling `topk` directly on logits returns the highest raw scores instead.

    With `k=1`, the indices have the same shape as `argmax(dim=1, keepdim=True)`. They identify the same class when the maximum is unique; tied scores may be handled differently.

    For the classifiers in `PreTrainedModels/`, top-5 accuracy measures how often the target class appears among the five highest-scoring classes. Reporting it alongside top-1 accuracy helps distinguish cases where the correct class narrowly misses first place from those where it falls outside the shortlist.
    """)
    return


@app.cell
def _(torch):
    five_class_logits = torch.tensor(
        [
            [2.0, 1.0, 0.1, 3.2, 0.5],
            [0.5, 3.0, 0.2, 0.1, 2.8],
            [0.1, 0.2, 4.0, 1.1, 0.3],
        ]
    )
    five_class_probs = torch.softmax(five_class_logits, dim=1)

    top_values, top_indices = torch.topk(five_class_probs, k=3, dim=1)

    for row in range(len(five_class_logits)):
        ranked = ", ".join(
            f"class {top_indices[row, j].item()} ({top_values[row, j].item():.1%})"
            for j in range(3)
        )
        print(f"sample {row}: {ranked}")

    print()
    print("row 1 is the interesting one - classes 1 and 4 are almost tied,")
    print("and argmax alone would hide that entirely")
    print()
    print(
        "topk(k=1) is argmax:",
        torch.equal(
            torch.topk(five_class_logits, k=1, dim=1).indices.squeeze(1),
            five_class_logits.argmax(dim=1),
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The accuracy idiom

    `Utils/functions.py` counts correct predictions using these two lines:

    ```python
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    ```

    Each operation has a specific job:

    | Call | What it does |
    | --- | --- |
    | `argmax(dim=1, keepdim=True)` | Selects the predicted class for each sample, giving shape `(n, 1)`. |
    | [`view_as(pred)`](https://pytorch.org/docs/stable/generated/torch.Tensor.view_as.html) | Reshapes the labels from `(n,)` to `(n, 1)`. |
    | [`eq`](https://pytorch.org/docs/stable/generated/torch.eq.html) | Compares each prediction with its target, producing Boolean values. |
    | `sum()` | Counts the `True` values. |
    | `item()` | Returns the count as a Python integer. |

    Matching the shapes matters. Without `view_as(pred)`, PyTorch broadcasts predictions of shape `(n, 1)` against labels of shape `(n,)`, producing an `(n, n)` result. This compares every prediction with every label instead of comparing each sample with its own target.

    We can also keep both tensors one-dimensional:

    ```python
    pred = output.argmax(dim=1)
    correct = pred.eq(y).sum().item()
    ```

    Here, both `pred` and `y` have shape `(n,)`. Dividing `correct` by the number of samples gives the accuracy.

    This is the same broadcasting issue we saw in `NumPyForML/NumPyForMLPart2Shapes.py`. When a result looks wrong, checking the shapes is a useful first step.
    """)
    return


@app.cell
def _(multi_logits, torch):
    labels = torch.tensor([0, 1, 2, 1])  # the last one is wrong: model says 0
    pred_keep = multi_logits.argmax(dim=1, keepdim=True)

    print("predictions", tuple(pred_keep.shape), "labels", tuple(labels.shape))
    print()

    wrong_way = pred_keep.eq(labels)
    print("without view_as: comparison shape", tuple(wrong_way.shape))
    print(
        "                 'accuracy'      ",
        wrong_way.sum().item() / len(labels),
    )

    right_way = pred_keep.eq(labels.view_as(pred_keep))
    print("with view_as:    comparison shape", tuple(right_way.shape))
    print(
        "                 accuracy        ",
        right_way.sum().item() / len(labels),
    )
    print()
    print(
        "3 of 4 correct is the truth. The first number is not even wrong in a useful way."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Putting it together

    Here is an evaluation function with everything in it, and quite typical for most tasks.
    """)
    return


@app.cell
def _(nn, torch):
    def evaluate(model, X, y, loss_fn):
        """Return average loss and accuracy over the data. No training happens here."""
        model.eval()  # 1. switch dropout and batchnorm to inference behaviour
        with torch.inference_mode():  # 2. and stop recording the graph
            logits = model(X)
            loss = loss_fn(logits, y)
            pred = logits.argmax(dim=1, keepdim=True)
            correct = pred.eq(y.view_as(pred)).sum().item()
        return loss.item(), correct / len(y)

    torch.manual_seed(0)
    demo_model = nn.Sequential(
        nn.Linear(5, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 3)
    )
    X_demo = torch.randn(200, 5)
    y_demo = (X_demo[:, 0] + X_demo[:, 1] * 2).round().clamp(0, 2).long()

    eval_loss, eval_acc = evaluate(demo_model, X_demo, y_demo, nn.CrossEntropyLoss())
    print(f"untrained model: loss {eval_loss:.4f}  accuracy {eval_acc:.1%}")
    print(
        "(three classes, so chance is about 33% - though these classes are not balanced)"
    )
    print()
    print("class counts:", torch.bincount(y_demo).tolist())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before interpreting an accuracy score, check the class balance. If 80% of the samples belong to one class, always predicting that class gives 80% accuracy without using the input at all. This majority-class baseline helps us judge whether the model’s accuracy represents a useful result.

    The `evaluate` function also leaves the model in evaluation mode. If we call it during training, we need to restore training mode before continuing. This is why `LinearModel.ipynb` calls `model.train()` at the start of every epoch: it resets the mode after the earlier call to `model.eval()`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try it

    `torch.round` thresholds at 0.5, and because that is what the demos here do it is easy to read 0.5 as part of the maths. It is not. It is a convention, it is only the right answer when the classes are balanced and the two kinds of mistake cost the same, and it is the one line of an inference pipeline you can change without retraining anything.

    The data below is 200 samples with about 30% positives — deliberately unbalanced, which is the normal case. Drag the threshold and watch the three numbers move against each other:

    - **precision** — of the ones you flagged, how many were right
    - **recall** — of the ones that were actually positive, how many you caught
    - **accuracy** — how many of all 200 you got right

    Lower the threshold and you flag everything, so recall goes to 1.0 and precision collapses. Raise it and the opposite. There is no setting that maximises both, which is the whole point: you are choosing which mistake you would rather make, and that is a question about the problem rather than about the model.

    Note where accuracy peaks. It is not at 0.5.
    """)
    return


@app.cell
def _(mo):
    threshold_slider = mo.ui.slider(
        0.05,
        0.95,
        step=0.05,
        value=0.5,
        label="Decision threshold",
        show_value=True,
    )
    threshold_slider
    return (threshold_slider,)


@app.cell
def _(torch):
    torch.manual_seed(1)
    thr_labels = (torch.rand(200) < 0.3).float()  # about 30% positive
    thr_logits = (
        torch.randn(200) * 1.5 + thr_labels * 2.5
    )  # overlapping, as real data is
    thr_probs = torch.sigmoid(thr_logits)

    print("positives in the data:", int(thr_labels.sum().item()), "of 200")
    return thr_labels, thr_probs


@app.cell
def _(thr_labels, thr_probs, threshold_slider):
    flagged = (thr_probs >= threshold_slider.value).float()

    true_pos = ((flagged == 1) & (thr_labels == 1)).sum().item()
    false_pos = ((flagged == 1) & (thr_labels == 0)).sum().item()
    false_neg = ((flagged == 0) & (thr_labels == 1)).sum().item()

    precision = true_pos / (true_pos + false_pos) if true_pos + false_pos else 0.0
    recall = true_pos / (true_pos + false_neg) if true_pos + false_neg else 0.0
    thr_accuracy = (flagged == thr_labels).float().mean().item()

    print(f"threshold {threshold_slider.value:.2f}")
    print(
        f"  flagged as positive  {true_pos + false_pos:3d}  ({true_pos} right, {false_pos} wrong)"
    )
    print(f"  missed               {false_neg:3d}")
    print()
    print(f"  precision {precision:.2f}")
    print(f"  recall    {recall:.2f}")
    print(f"  accuracy  {thr_accuracy:.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Build a model with dropout, run the same input through it 100 times in train mode, and average the results. How close is the average to what eval mode gives?
    2. Take the `evaluate` function and remove `model.eval()`. Run it ten times on the same data and report the spread of accuracies.
    3. Time a forward pass over 10,000 samples with and without `inference_mode`. Now compare peak memory.
    4. Write the binary equivalent of `evaluate`, using `sigmoid` and `round`. What shape do the labels need to be?
    5. Construct data where accuracy is 90% and the model has learned nothing. How would you have spotted it?
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NumPy for Machine Learning, Part 3: maths and reductions

    Parts 1 and 2 built arrays and got their shapes right. This notebook is about doing something with them.

    There are only three kinds of operation here :

    - **elementwise** :- one input value in, one output value out, shape unchanged (`np.exp`, `np.sin`)
    - **matrix products** :- combine two arrays along a shared dimension (`@`)
    - **reductions** :- collapse an axis away, so the output has fewer dimensions than the input (`sum`, `mean`, `argmax`)

    A neural network is elementwise operations and matrix products all the way forward, with one reduction at the end to turn a batch of losses into the single number you are minimising.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    print("numpy", np.__version__)
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Universal functions

    A [ufunc](https://numpy.org/doc/stable/reference/ufuncs.html) applies a scalar operation to every element of an array, in compiled code, with no Python loop. `np.sin`, `np.exp`, `np.log`, `np.sqrt`, `np.abs` and the arithmetic operators are all ufuncs.

    They share a signature:

    ```python
    np.exp(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `x` | required | the input array |
    | `out` | `None` | write the result into an existing array instead of allocating one |
    | `where` | `True` | a boolean mask; positions that are `False` are left alone |
    | `dtype` | `None` | force the working type |

    In practice you write `np.exp(x)` and ignore the rest, but `out` is worth knowing about for a training loop where you are allocating the same array thousands of times.
    """)
    return


@app.cell
def _(np):
    angles = np.linspace(0, np.pi, 5)

    print("input ", angles)
    print("sin   ", np.sin(angles))
    print("exp   ", np.exp(angles))
    return


@app.cell
def _(np):
    # comparing speeds (SIMD from numpy vs normal python)
    import math
    import time

    big = np.linspace(0, np.pi, 2_000_000)

    _t0 = time.perf_counter()
    _by_loop = [math.sin(v) for v in big]
    _t1 = time.perf_counter()
    _by_ufunc = np.sin(big)
    _t2 = time.perf_counter()

    print(f"python loop {_t1 - _t0:.3f}s")
    print(f"np.sin      {_t2 - _t1:.3f}s")
    print(f"speed up    {(_t1 - _t0) / (_t2 - _t1):.0f}x")
    print("same answer:", np.allclose(_by_loop, _by_ufunc))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### In machine learning

    Most of the activation and loss functions we use are built from ufuncs. In `Neuron/nn_from_scratch.py` the sigmoid is `1 / (1 + np.exp(-z))` and the binary cross-entropy loss uses `np.log`. As ufuncs work element by element, the same expression works for one sample or a batch of ten thousand.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    I think the sigmoid is worth experimenting with as it shows what happens to the values and gradients as they pass through a network. The last layer produces a raw score called a *logit*. For binary classification, the sigmoid maps this from any real value into the range zero to one, which we can treat as a probability.

    Move the slider and watch the probability and gradient. The curve is steepest near zero, where the model is least certain and a small change to the logit has the largest effect. Towards either end the curve flattens out, so changing the logit has very little effect on the probability. The gradient is the slope of the curve, so it also becomes very small. If this happens across several layers, the gradients passed back through the network can shrink towards zero. This is one cause of the vanishing gradient problem, and we can see it happening here.
    """)
    return


@app.cell
def _(mo):
    logit = mo.ui.slider(
        -8.0, 8.0, step=0.25, value=0.0, label="logit", show_value=True
    )
    logit
    return (logit,)


@app.cell
def _(logit, mo, np, plt):
    def sigmoid_curve(z):
        return 1.0 / (1.0 + np.exp(-z))

    _z = np.linspace(-8, 8, 201)
    _p = sigmoid_curve(logit.value)

    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot(_z, sigmoid_curve(_z))
    _ax.axhline(0.5, color="grey", linewidth=0.8, linestyle=":")
    _ax.scatter([logit.value], [_p], s=80, zorder=3, color="tab:orange")
    _ax.set(xlabel="logit", ylabel="probability", ylim=(-0.05, 1.05))
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md(
                f"sigmoid output **{_p:.4f}**, gradient at this point **{_p * (1 - _p):.4f}**"
            ),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## [np.clip](https://numpy.org/doc/stable/reference/generated/numpy.clip.html)

    `np.clip` keeps values within a specified range. Values below the
    lower bound become the lower bound, and values above the upper bound
    become the upper bound. Anything already within the range stays unchanged.

    ```python
    np.clip(a, a_min, a_max, out=None)
    np.clip(a, min=0, max=2)  # available from NumPy 2.1
    ```

    | Parameter | Default | What it does |
    | --------- | ------- | ------------ |
    | `a` | Required | Input array. |
    | `a_min` / `min` | — | Lower bound; `None` means no lower bound. |
    | `a_max` / `max` | — | Upper bound; `None` means no upper bound. |
    | `out` | `None` | Where to store the result. Use `out=a` to modify the input. |

    We can limit values to the range `[0, 2]`:
    """)
    return


@app.cell
def _(np):
    values = np.array([-3, -1, 0, 1, 5])
    print(np.clip(values, 0, 2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We can also apply just one bound. Here, negative values become zero,
    whilst positive values stay unchanged:
    """)
    return


@app.cell
def _(np):
    floor_values = np.array([-3.0, 0.5, 9.0])
    print(np.clip(floor_values, 0.0, None))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    In machine learning, clipping can help prevent numerical problems.
    A loss calculation involving `log(p)` or `log(1 - p)` encounters
    `log(0)` when a probability is exactly `0` or `1`.

    We can keep probabilities slightly away from those endpoints:
    """)
    return


@app.cell
def _(np):
    probabilities = np.array([0.0, 0.5, 1.0])
    epsilon = 1e-7
    safe_probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)

    print("Original:", probabilities)
    print("Clipped:", safe_probabilities)
    print("log(p):", np.log(safe_probabilities))
    print("log(1 - p):", np.log1p(-safe_probabilities))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Both logarithms now have positive inputs and produce finite results.
    Clipping changes the extreme values, so we need to choose bounds
    that make sense for the calculation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""

    """)
    return


@app.cell
def _(np):
    # cross-entropy takes the log of a predicted probability.
    # a confident, wrong prediction can put a probability at zero.
    p_confident = np.array([1e-12, 0.0, 0.3])

    with np.errstate(divide="ignore"):
        print("unclipped log:", np.log(p_confident))

    eps = 1e-7
    print("clipped log:  ", np.log(np.clip(p_confident, eps, 1.0 - eps)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A probability of exactly zero gives `-inf`, and once one `-inf` is in the loss the whole thing becomes `inf` or `nan` and every gradient after it is meaningless. Clipping to a small epsilon either side bounds the worst case instead. `nn_from_scratch.py` does this before its `np.log`, which is why its loss stays finite when the network gets confident.

    Worth knowing that PyTorch's `CrossEntropyLoss` solves the same problem a different way — it takes raw logits and folds the softmax inside, so the dangerous intermediate never exists. That is a better answer than clipping, but you only appreciate why once you have seen the `-inf`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [Matrix multiplication](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html): the `@` operator

    ```python
    A @ B                  # the operator, added in Python 3.5
    np.matmul(A, B)        # identical
    np.dot(A, B)           # the same for 2D, different for higher dimensions
    ```

    For two 2D arrays, `(n, k) @ (k, m)` gives `(n, m)`. The inner dimensions must match and they vanish; the outer ones survive. If you remember nothing else, remember that the two numbers facing each other have to be equal.

    Lecture 5 introduces `@` as the operator form of `np.matmul`. Prefer it to `np.dot` — for 2D they agree, but for stacks of matrices `np.dot` does something else entirely and the difference will not announce itself.
    """)
    return


@app.cell
def _(np):
    A = np.arange(6).reshape(2, 3)  # (2, 3)
    B = np.arange(12).reshape(3, 4)  # (3, 4)

    print(f"{A.shape} @ {B.shape} -> {(A @ B).shape}")
    print(A @ B)

    try:
        _ = B @ A @ B  # (3,4) @ (2,3) will not line up
    except ValueError as e:
        print()
        print("ValueError:", e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### In machine learning

    This is what a neural network is. A layer is `z = X @ w + b` and nothing more, the matrix product mixes every input into every output, the bias shifts it, and an activation bends it. `nn_from_scratch.py:52` writes exactly that, and its two-layer version chains two of them.

    Going backwards uses the same operation transposed. The gradients in that file are `dW1 = X.T @ dz1 / n` and `da1 = dz2 @ W2.T`  still matrix products, just with the axes swapped so the shapes line up the other way.

    Here is a forward pass for a batch, written out with the shapes printed at each step. Every layer you meet this term is this cell with bigger numbers.
    """)
    return


@app.cell
def _(np):
    rng = np.random.default_rng(42)

    X = rng.normal(size=(8, 3))  # 8 samples, 3 features
    W1 = rng.normal(size=(3, 5)) * 0.1  # 3 features -> 5 hidden units
    b1 = np.zeros(5)
    W2 = rng.normal(size=(5, 1)) * 0.1  # 5 hidden -> 1 output
    b2 = np.zeros(1)

    z1 = X @ W1 + b1
    a1 = np.maximum(0, z1)  # ReLU, elementwise
    z2 = a1 @ W2 + b2

    print(f"X  {X.shape} @ W1 {W1.shape} + b1 {b1.shape} -> z1 {z1.shape}")
    print(f"a1 {a1.shape} @ W2 {W2.shape} + b2 {b2.shape} -> z2 {z2.shape}")
    print()
    print("one output per sample:", z2.ravel().round(3))
    return (rng,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reductions and the `axis` parameter

    A reduction combines values along one or more axes to produce a summary, such as a sum, mean, minimum, maximum or standard deviation. Functions including [`np.sum`](https://numpy.org/doc/stable/reference/generated/numpy.sum.html), `np.mean`, `np.min`, `np.max` and `np.std` use the same `axis` and `keepdims` rules.

    ```python
    np.sum(a, axis=None, keepdims=False)
    np.mean(a, axis=None, keepdims=False)
    ```

    | Parameter | Default | What it does |
    | --------- | ------- | ------------ |
    | `axis` | `None` | Axis or tuple of axes to reduce. `None` reduces over all elements. |
    | `keepdims` | `False` | When `True`, retains each reduced axis with length one. |

    With `keepdims=False`, **the axis we specify disappears from the result’s shape**. For an array of shape `(3, 4)`:

    | Operation | Result shape | Meaning |
    | --------- | ------------ | ------- |
    | `np.sum(a, axis=0)` | `(4,)` | Combine the three rows, giving one total per column. |
    | `np.sum(a, axis=1)` | `(3,)` | Combine the four columns, giving one total per row. |
    | `np.sum(a)` | `()` | Combine all elements into a scalar total. |

    Setting `keepdims=True` preserves the reduced axis as a dimension of length one:

    ```python
    a = np.arange(12).reshape(3, 4)

    print(np.mean(a, axis=1).shape)                 # (3,)
    print(np.mean(a, axis=1, keepdims=True).shape)  # (3, 1)
    ```

    This is useful when we want to broadcast the result back against the original array. For example, we can subtract each row’s mean from that row:

    ```python
    row_means = np.mean(a, axis=1, keepdims=True)
    centred = a - row_means
    ```
    """)
    return


@app.cell
def _(np):
    table = np.arange(12).reshape(3, 4)
    print(table)
    print()
    print("sum()        ", table.sum(), "  everything, a scalar")
    print("sum(axis=0)  ", table.sum(axis=0), "  the 3 collapsed, 4 left")
    print("sum(axis=1)  ", table.sum(axis=1), "  the 4 collapsed, 3 left")
    return (table,)


@app.cell
def _(table):
    # keepdims leaves the axis in place so the result still broadcasts against the input
    print("without keepdims", table.sum(axis=1).shape)
    print("with keepdims   ", table.sum(axis=1, keepdims=True).shape)
    print()
    print("which means this works, subtracting each row's mean from that row:")
    print(table - table.mean(axis=1, keepdims=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### In machine learning

    In machine learning, the meaning of a reduction depends on which axis we reduce.

    Reducing over the **batch axis** combines results from multiple samples. If we have one loss value per sample, `mean` gives us the average batch loss used for training. Similarly, taking the mean of boolean correctness flags gives the fraction of samples classified correctly.

    Reducing over the **feature axis** summarises each sample separately. For an array shaped `(batch_size, n_features)`, `mean(axis=1)` calculates one mean per sample. If the columns contain class scores, `argmax(axis=1)` returns the index of the highest-scoring class for each sample.

    `keepdims=True` is useful whenever we need to broadcast a reduced result back against the original array. One example is normalising each sample: we calculate its mean and standard deviation across the features, then subtract the mean and divide by the standard deviation. Keeping these results shaped `(batch_size, 1)` ensures that each sample uses its own statistics. We also need to handle zero standard deviations to avoid division by zero.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### [min](https://numpy.org/doc/stable/reference/generated/numpy.min.html) and [max](https://numpy.org/doc/stable/reference/generated/numpy.max.html), and why features get scaled

    `np.min` and `np.max` find the smallest and largest values, using the same `axis` and `keepdims` rules as the other reductions. One use is **feature scaling**.

    Features often arrive in different units and ranges. An age might be around 30, whilst a salary might be around 30,000. With similarly sized weights, the larger input contributes more to a neuron’s weighted sum. Input scale also affects the gradients of those weights, which can make training slower or more sensitive to the learning rate. Bringing features onto comparable scales can help.

    Min-max scaling maps each non-constant feature’s observed minimum to zero and its maximum to one:

    ```python
    feature_min = X.min(axis=0)
    feature_max = X.max(axis=0)
    feature_range = feature_max - feature_min

    safe_range = np.where(feature_range == 0, 1, feature_range)
    scaled = (X - feature_min) / safe_range
    ```

    Here, `X` has shape `(n_samples, n_features)`. We use `axis=0` to reduce across samples, leaving one minimum and maximum per feature. These values broadcast across the rows. Replacing zero ranges with one avoids division by zero and maps constant columns to zero.

    Using `axis=1` with `keepdims=True` would instead scale each sample using its own smallest and largest feature values. That is a different operation: it can remove differences in overall magnitude between samples and mix measurements with unrelated units.

    Calculate the scaling values from the training data, then reuse them for validation, test and future inputs. New values outside the training range can produce scaled values below zero or above one; min-max scaling does not automatically clip them. See [scikit-learn’s MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).
    """)
    return


@app.cell
def _(np):
    raw_features = np.array(
        [
            [25.0, 22000.0],
            [41.0, 58000.0],
            [33.0, 31000.0],
            [58.0, 90000.0],
        ]
    )

    feature_min = raw_features.min(axis=0, keepdims=True)
    feature_max = raw_features.max(axis=0, keepdims=True)

    print("per feature min", feature_min, feature_min.shape)
    print("per feature max", feature_max)
    print()
    scaled_features = (raw_features - feature_min) / (feature_max - feature_min)
    print(scaled_features.round(3))
    return feature_max, feature_min, raw_features


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two things to be careful about, and both of them are the kind of mistake that quietly costs you marks rather than raising.

    **The range can be zero.** A feature that is the same value in every sample gives `max - min == 0`, and you divide by nought. The fix is in the `np.where` section below.

    **Work out the min and max on the training data only, then reuse those exact numbers on the validation and test sets.** If you scale the test set by its own min and max you have let the test data influence how the test data is prepared, which is a small leak but a real one, and your reported accuracy is then a little better than the truth. It also breaks the moment you deploy, because a single incoming sample has no range to scale by.
    """)
    return


@app.cell
def _(feature_max, feature_min, np):
    # a new sample scaled with the *training* statistics, not its own
    new_sample = np.array([[37.0, 45000.0]])
    print((new_sample - feature_min) / (feature_max - feature_min))
    print("note it lands inside 0 to 1 here, but nothing guarantees that")

    outlier = np.array([[70.0, 120000.0]])
    print(((outlier - feature_min) / (feature_max - feature_min)).round(3))
    print("an unseen extreme goes past 1, which is correct and fine")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [np.argmax](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html)

    Returns the *index* of the largest value rather than the value itself. There is an `argmin` that does the opposite.

    ```python
    np.argmax(a, axis=None, out=None, keepdims=False)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `axis` | `None` | flattens first if left as `None`, which is rarely what you want |
    | `keepdims` | `False` | keep the collapsed axis with length 1 |

    The `axis=None` default is problematic in machine learning code. On a batch of class scores it gives you one number for the whole batch, the position in the flattened array, rather than one prediction per sample. Name the axis to ensure you don't get this error.
    """)
    return


@app.cell
def _(np):
    # 4 samples, 3 classes, as a network would produce them
    logits = np.array(
        [
            [2.0, 1.0, 0.1],
            [0.5, 3.0, 0.2],
            [0.1, 0.2, 4.0],
            [1.5, 1.4, 0.3],
        ]
    )

    print("argmax()        ", logits.argmax(), "  the flat index - not useful")
    print(
        "argmax(axis=1)  ",
        logits.argmax(axis=1),
        "  one prediction per sample",
    )
    print(
        "argmax(axis=0)  ",
        logits.argmax(axis=0),
        "  which sample scored each class highest",
    )
    return (logits,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One point that is worth making to save confusion later. In the PyTorch demos you will see `softmax` applied before `argmax`:

    ```python
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    ```

    Softmax is strictly increasing, so it cannot change which element is largest, the `argmax` gives the same answer with or without it. The softmax is there when you want a probability to report as a confidence, not to pick the winner. Below is the NumPy version showing the two agree.
    """)
    return


@app.cell
def _(logits, np):
    def softmax(x, axis=1):
        shifted = x - x.max(axis=axis, keepdims=True)  # for stability
        e = np.exp(shifted)
        return e / e.sum(axis=axis, keepdims=True)

    probs = softmax(logits)
    print("probabilities per row (each sums to 1):")
    print(probs.round(3))
    print("row sums", probs.sum(axis=1))
    print()
    print("argmax on logits", logits.argmax(axis=1))
    print("argmax on probs ", probs.argmax(axis=1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note the `x - x.max(...)` on the first line of that softmax. Subtracting the largest value does not change the result mathematically, because the constant cancels between the numerator and denominator, but it keeps `np.exp` away from overflow. It is the same numerical stability problem as the `clip` earlier, with a neater solution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [np.where](https://numpy.org/doc/stable/reference/generated/numpy.where.html)

    A vectorised `if`. Given a boolean array, pick from one array where it is true and another where it is false.

    ```python
    np.where(condition, x, y)    # elementwise choice
    np.where(condition)          # with one argument, the indices where it is true
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `condition` | required | boolean array |
    | `x` | — | values used where the condition is true |
    | `y` | — | values used where it is false |

    All three broadcast against each other, so `x` and `y` are often plain scalars.
    """)
    return


@app.cell
def _(np, rng):
    scores = rng.normal(size=10)

    print("scores   ", scores.round(2))
    print("threshold", np.where(scores > 0, 1, 0))
    print(
        "clamped  ",
        np.where(scores > 0, scores, 0).round(2),
        " - this is ReLU",
    )
    print()
    print("indices above zero:", np.where(scores > 0)[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### In machine learning

    `np.where(x > 0, x, 0)` *is* ReLU, and writing it out that way once makes the activation much less mysterious than the name suggests. It is also how you threshold probabilities into labels, and how you build a mask to ignore padding in a batch of unequal-length sequences.

    `np.maximum(0, x)` does the same ReLU job more directly, and I used it in the forward pass earlier, `where` is the one to reach for when the two branches are not so simple.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `np.where` does not skip the unused calculation

    Python evaluates a function’s arguments before calling it. This means `np.where` selects between values that have already been calculated. An `if` statement, by comparison, only executes the selected branch.

    This matters when we try to avoid division by zero:

    ```python
    np.where(denominator == 0, 0.0, numerator / denominator)
    ```

    Python calculates `numerator / denominator` before passing the result to `np.where`. Any zero denominators are therefore still used in the division, potentially producing warnings and intermediate `inf` or `nan` values. Selecting zero afterwards hides those results, but does not prevent the invalid calculation. If floating-point errors are configured to raise exceptions, execution can stop before `np.where` runs.

    We can make the denominator safe first, then explicitly set the output to zero where the original denominator was zero:

    ```python
    zero_denominator = denominator == 0
    safe_denominator = np.where(zero_denominator, 1.0, denominator)

    result = np.where(
        zero_denominator,
        0.0,
        numerator / safe_denominator,
    )
    ```

    The division still runs across the whole array, but it no longer divides by zero. The second `np.where` gives those positions the output value we intended. This handles zero denominators; any other invalid inputs, such as existing `nan` values, still need separate consideration.
    """)
    return


@app.cell
def _(np, raw_features):
    # a third feature that never varies, so its range is zero
    constant_column = np.full((4, 1), 7.0)
    with_constant = np.hstack([raw_features, constant_column])

    spread = with_constant.max(axis=0) - with_constant.min(axis=0)
    print("ranges", spread, " - the third one is zero")

    # the guard that does not guard
    with np.errstate(invalid="ignore", divide="ignore"):
        looks_safe = np.where(
            spread == 0,
            0.0,
            (with_constant[0] - with_constant.min(axis=0)) / spread,
        )
        print(
            "still divided by zero on the way:",
            (with_constant[0] - with_constant.min(axis=0)) / spread,
        )
    print("but the result looks fine:", looks_safe)

    # the guard that does
    safe_spread = np.where(spread == 0, 1.0, spread)
    actually_safe = (with_constant[0] - with_constant.min(axis=0)) / safe_spread
    print()
    print("safe denominators:", safe_spread)
    print("no nan was ever created:", actually_safe)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cyclical features with [sin](https://numpy.org/doc/stable/reference/generated/numpy.sin.html) and [cos](https://numpy.org/doc/stable/reference/generated/numpy.cos.html)

    A last pair of ufuncs, included because of one problem they solve neatly and because it is a good illustration of what feature engineering actually is.

    Suppose one of your features is the hour of the day. Feed in the number 23 and the number 0 and the model sees them as far apart as two numbers in that range can be, when in fact they are an hour apart. Scaling does not help; the discontinuity is still there, just between 0.0 and 1.0 instead.

    The fix is to stop representing the hour as one number and represent it as a position on a circle, which takes two. Map the hour onto an angle, then take the sine and the cosine of it. Midnight and 23:00 are now next to each other, because on a circle they are.
    """)
    return


@app.cell
def _(np):
    hours = np.array([0, 6, 12, 18, 23])
    hour_angles = 2 * np.pi * hours / 24

    cyclical = np.c_[np.sin(hour_angles), np.cos(hour_angles)]

    print("hour   sin      cos")
    for _h, (_s, _c) in zip(hours, cyclical, strict=False):
        print(f"{_h:>4}  {_s: .3f}  {_c: .3f}")

    print()
    print("distance from 23:00 to 00:00, as one number:", abs(23 - 0))
    print(
        "distance from 23:00 to 00:00, on the circle:",
        np.linalg.norm(cyclical[4] - cyclical[0]).round(3),
    )
    print(
        "distance from 00:00 to 12:00, on the circle:",
        np.linalg.norm(cyclical[0] - cyclical[2]).round(3),
    )
    return cyclical, hours


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note the last two numbers. Eleven at night to midnight is a short hop; midnight to midday is the full diameter of the circle. That is the behaviour we wanted and could not get from a single number.

    The same trick works for any repeating quantity, day of the week, month of the year, compass bearing, the angle of a joint in a character rig. `np.sin` and `np.cos` also turn up in rotation matrices when you are augmenting a training set by rotating images, which is the other place you will meet them this term.
    """)
    return


@app.cell
def _(cyclical, hours, plt):
    _fig, _ax = plt.subplots(figsize=(4, 4))
    _ax.plot(cyclical[:, 1], cyclical[:, 0], "o", markersize=9)
    for _h, (_s, _c) in zip(hours, cyclical, strict=False):
        _ax.annotate(f"{_h}:00", (_c, _s), xytext=(8, 4), textcoords="offset points")
    _ax.set(
        xlim=(-1.6, 1.6),
        ylim=(-1.6, 1.6),
        xlabel="cos",
        ylabel="sin",
        aspect="equal",
    )
    _ax.set_title("the hour of the day, as two features", loc="left")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Build a `(5, 3)` array of random scores. Find the highest score in the whole array, the highest in each row, and which row held the overall highest.
    2. Write the sigmoid function using ufuncs. Check that `sigmoid(0)` is 0.5 and that a large negative input does not produce a warning.
    3. Take the `(8, 3)` input `X` from the forward pass above and normalise each *feature* to zero mean and unit standard deviation. Which axis do you reduce over, and where do you need `keepdims`?
    4. Given `logits` from the argmax section and labels `[0, 1, 2, 1]`, compute the accuracy in one line. Then check the shape of your comparison before you trust the number.
    5. Min-max scale `raw_features` using `axis=1` instead of `axis=0` and look at what comes out. Explain, in one sentence, why every row is now nought and one.
    6. Add the day of the week to the cyclical encoding, so a sample carries four features rather than two. Check that Sunday and Monday come out adjacent.

    Part 4 covers random numbers and the business of comparing floats, which is what lets you test any of this.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

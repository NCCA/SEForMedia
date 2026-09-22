#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NumPy for Machine Learning, Part 4: random numbers and comparing floats

    This notebook covers two practical parts of machine learning: controlling randomness and checking calculations with floating-point numbers.

    We use random numbers to initialise model weights, shuffle training data and to choose which activations dropout disables when training.

    Controlling the random number generator helps us repeat experiments and compare changes under the same conditions. A fixed seed is a useful starting point, although reproducibility also depends on the software, hardware and operations we use.

    Floating-point calculations introduce rounding errors, so mathematically equivalent expressions may produce slightly different results. Exact comparison with `==` is therefore often unsuitable for checking numerical calculations. We need to decide how much difference is acceptable and compare using a tolerance.

    In the final section, we use these comparisons to check a hand-derived gradient against a numerical estimate. This technique, called *gradient checking*, helps us find mistakes in derivatives and backpropagation code. It brings together the array operations, shape handling and numerical calculations from the previous notebooks.

    See NumPy’s [random sampling documentation](https://numpy.org/doc/stable/reference/random/index.html) for the generator API and available distributions.
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
    ## NumPy's two API's.

    NumPy has two random number interfaces and you will meet both in this repository, so it is worth knowing why.

    The old one is a set of functions in `np.random` — `np.random.rand`, `np.random.randint`, `np.random.seed` these all draw from one global generator hidden inside the module. It still works and will remain in the library, but NumPy calls it legacy.

    The new one, from NumPy 1.17 onwards, is [`np.random.default_rng`](https://numpy.org/doc/stable/reference/random/generator.html), provides a generator object of your own. It has a better algorithm behind it, and more importantly the state is in something you are holding rather than in module-level global state that anything else in the process can reach and change.

    Use `default_rng` in new code. I have covered the legacy functions too because the older demos use them and you will read them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [np.random.default_rng](https://numpy.org/doc/stable/reference/random/generator.html)

    ```python
    rng = np.random.default_rng(seed=None)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `seed` | `None` | an int for a repeatable stream; `None` seeds from the operating system |

    The generator it returns has the methods you want on it:

    ```python
    rng.random(size=None)                              # uniform in [0, 1)
    rng.normal(loc=0.0, scale=1.0, size=None)          # gaussian
    rng.integers(low, high=None, size=None, endpoint=False)
    rng.permutation(x)                                 # a shuffled copy
    rng.shuffle(x)                                     # shuffle in place
    rng.choice(a, size=None, replace=True)             # sample from a
    ```

    Note `endpoint=False` on `integers` — the high value is excluded, matching Python's `range`. Set `endpoint=True` if you want it included.
    """)
    return


@app.cell
def _(np):
    rng = np.random.default_rng(42)

    print("uniform  ", rng.random(4).round(3))
    print("normal   ", rng.normal(size=4).round(3))
    print("normal(mean=10, sd=2)", rng.normal(10, 2, size=4).round(3))
    print("integers ", rng.integers(0, 10, size=8), " - 10 is excluded")
    print("shuffled ", rng.permutation(np.arange(8)))
    return (rng,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reproducibility

    Two generators made with the same seed produce the same stream. This is the whole point.
    """)
    return


@app.cell
def _(np):
    first = np.random.default_rng(1234).random(3)
    second = np.random.default_rng(1234).random(3)
    third = np.random.default_rng(99).random(3)

    print("seed 1234 ", first.round(4))
    print("seed 1234 ", second.round(4), " identical")
    print("seed 99   ", third.round(4), "  different")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A seeded generator helps us make a fair comparison between experiments. If we change the model and its random initialisation at the same time, we cannot easily tell which caused the change in loss. Fixing the seed lets us control that source of variation whilst testing an idea. We should then repeat promising experiments with several seeds to check that the improvement holds across different initialisations.

    The PyTorch demos here use `torch.manual_seed(42)` for this purpose. The value [`42`](https://en.wikipedia.org/wiki/Phrases_from_The_Hitchhiker's_Guide_to_the_Galaxy#The_Answer_to_the_Ultimate_Question_of_Life,_the_Universe,_and_Everything_is_42) is simply a choice; it has no special effect. NumPy and PyTorch use separate random number generators, so seeding one does not seed the other. If our pipeline uses both, we need to seed both.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The legacy functions

    You will see these in the older notebooks, so here is what they do.

    ```python
    np.random.rand(d0, d1, ...)                  # uniform [0,1), shape as separate arguments
    np.random.randn(d0, d1, ...)                 # standard normal, same calling style
    np.random.randint(low, high=None, size=None) # ints, high excluded
    np.random.seed(n)                            # seed the global generator
    ```

    The thing that catches people is the calling convention. `np.random.rand(2, 3)` takes the dimensions as separate arguments, while every modern function takes a shape tuple — `rng.random((2, 3))`. Mixing the two up gives you either an error or, worse, an array of the wrong shape that broadcasts anyway.
    """)
    return


@app.cell
def _(np):
    np.random.seed(0)

    print("rand(2, 3)   - separate arguments")
    print(np.random.rand(2, 3).round(3))
    print()
    print("randint(0, 255, size=5)", np.random.randint(0, 255, size=5))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why weights start random

    We want the hidden units in a neural network to learn different features. If we initialise them identically, with the same incoming weights, biases and outgoing connections, they produce the same outputs and receive the same gradients. Under ordinary gradient descent, each update preserves that symmetry, so the units continue doing the same job.

    Random initialisation gives the units different starting points. Their outputs and gradients can then differ, allowing them to learn different features during training. The scale of those initial weights also matters: values that are too large or too small can make training difficult.

    The cell below demonstrates the symmetry problem with two hidden units. Both start with identical parameters, and we can see that their gradients give them identical updates.
    """)
    return


@app.cell
def _(np, rng):
    inputs = rng.normal(size=(6, 3))

    same = np.full((3, 2), 0.5)  # both units identical
    varied = rng.normal(size=(3, 2)) * 0.5  # each unit different

    print("identical weights, the two hidden units produce:")
    print((inputs @ same).round(3))
    print("the two columns are the same, so they will always be the same")
    print()
    print("random weights:")
    print((inputs @ varied).round(3))
    print("the two columns differ, so the units can specialise")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `* 0.5` on that second line is also doing something. The *scale* of the initial weights matters as much as the randomness: too large and the activations saturate and gradients vanish, too small and the signal dies out through the layers. Real initialisers such as Xavier and He scale by the number of inputs to the layer, and `nn.Linear` in PyTorch applies a sensible default so you rarely set it by hand — but when a deep network refuses to train at all, initialisation scale is one of the first things to suspect.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The slider below draws a thousand weights from `rng.normal` and shows you what that scale actually buys. The seed stays fixed, so nothing here is randomness — the shape is the same bell each time and only its width changes.

    The second plot is the part that matters. It pushes those weights through a sigmoid and shows where the activations land. At a small spread everything piles up near 0.5, which is the steep part of the curve where there is plenty of gradient. Wind the spread up and the activations pile up at nought and one instead, which is the flat part, and that layer has stopped learning before training has started.
    """)
    return


@app.cell
def _(mo):
    spread = mo.ui.slider(
        0.1,
        8.0,
        step=0.1,
        value=0.5,
        label="initialisation scale",
        show_value=True,
    )
    spread
    return (spread,)


@app.cell
def _(np, plt, spread):
    _draws = np.random.default_rng(21).normal(0.0, spread.value, size=1000)
    _activations = 1.0 / (1.0 + np.exp(-_draws))

    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(10, 3))
    _left.hist(_draws, bins=np.linspace(-20, 20, 61))
    _left.set(xlabel="weight", ylabel="count", xlim=(-20, 20), title="the weights")
    _right.hist(_activations, bins=np.linspace(0, 1, 41), color="tab:orange")
    _right.set(xlabel="sigmoid(weight)", xlim=(0, 1), title="where they land")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [np.isclose](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html) and [np.allclose](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html)

    Floats do not compare exactly. We met this in [Lecture 2](https://nccastaff.bournemouth.ac.uk/jmacey/SEForMedia/lectures/Lecture2/) with `0.1 + 0.2 != 0.3`, and it matters more here because every number in a network has been through thousands of operations, each one losing a little precision.

    ```python
    np.isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)   # elementwise, returns an array
    np.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)  # a single bool for the lot
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `rtol` | `1e-05` | relative tolerance, scaled by the size of `b` |
    | `atol` | `1e-08` | absolute tolerance, which is what saves you near zero |
    | `equal_nan` | `False` | whether two `nan` values count as equal |

    The test applied is `|a - b| <= atol + rtol * |b|`. Two things follow from that formula and both are worth knowing. It is not symmetric, because only `b` is scaled. And near zero the relative term contributes nothing, so `atol` is the only thing doing any work, this is why you often need to raise `atol` explicitly when comparing small numbers such as gradients.
    """)
    return


@app.cell
def _(np):
    print("0.1 + 0.2 == 0.3          ", 0.1 + 0.2 == 0.3)
    print("np.isclose(0.1 + 0.2, 0.3)", np.isclose(0.1 + 0.2, 0.3))
    print("the actual difference     ", (0.1 + 0.2) - 0.3)
    return


@app.cell
def _(np):
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0001, 3.0])

    print("isclose ", np.isclose(a, b), "  which element differs")
    print("allclose", np.allclose(a, b), "  one answer for the array")
    print("allclose with a looser tolerance:", np.allclose(a, b, atol=1e-3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [np.nditer](https://numpy.org/doc/stable/reference/generated/numpy.nditer.html)

    Most examples in these notebooks use array operations to avoid Python loops. `np.nditer` gives us a way to visit array elements individually, with control over how we access them.

    We will use it for numerical gradient checking. For each parameter, we make a small change, evaluate the loss, then restore the original value. A loop makes this process straightforward, and tracking the parameter’s position lets us store its estimated gradient in the corresponding output element.

    ```python
    np.nditer(op, flags=None, op_flags=None, order="K")
    ```

    | Parameter | Default | What it does |
    | --------- | ------- | ------------ |
    | `op` | Required | The array, or sequence of arrays, to iterate over. |
    | `flags` | `None` | Iterator options. Use `["multi_index"]` to track the current coordinates. |
    | `op_flags` | `None` | Access options for each array; read-only by default. |
    | `order` | `"K"` | Follow the order of elements in memory as closely as possible. |

    With `multi_index` enabled, `it.multi_index` gives us an index tuple such as `(1, 2)`. We can use that tuple to access both the parameter and its position in a gradient array of the same shape.
    """)
    return


@app.cell
def _(np):
    small_layer = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    walker = np.nditer(small_layer, flags=["multi_index"])
    for _value in walker:
        print(walker.multi_index, "->", float(_value))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compare that with `for i in range(len(w))`, which only works if `w` is one dimensional. A real layer's weights are a matrix, and the layer after that is a different sized matrix, and a convolution is four dimensional. `nditer` with `multi_index` is the same three lines for all of them, which is exactly what you want for a check you are going to run against every layer in a network.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Putting it together: checking a gradient

    A mistake in a hand-derived gradient does not necessarily stop the code running. The loss may even decrease, so observing some improvement is not enough to show that the derivative is correct.

    We can check our calculation by estimating the gradient numerically. We increase one weight by a small amount `h`, evaluate the loss, then repeat with that weight decreased by `h`. All other parameters stay fixed:

    ```text
    numerical gradient ≈ (loss(w + h) - loss(w - h)) / (2 * h)
    ```

    This is called a *central difference*. We restore the weight afterwards and repeat for the other parameters.

    Checking every weight requires two loss evaluations per parameter, making this expensive for training but useful for testing a small network. The estimate uses the forward calculation without relying on our derivative formula. Agreement with the analytic gradient provides evidence that the derivative is implemented correctly at the points we test; it does not prove correctness for every input.

    The choice of `h` matters. Too large, and the estimate is inaccurate; too small, and floating-point rounding can dominate. We therefore compare using a tolerance and keep the calculation deterministic, using the same data and random choices for both evaluations.

    This example brings together the tools from the four notebooks: `default_rng` initialises the weights, `@` performs the matrix operations, `np.exp` and `np.log` calculate the sigmoid and loss, `mean` reduces the sample losses, and `allclose` checks whether the analytic and numerical gradients agree within our chosen tolerances.

    This is exactly what `Neuron/nn_from_scratch.py` does at line 315, and it needs every function from all four notebooks: `default_rng` to make the weights, `@` for the forward pass, `np.exp` and `np.log` for the sigmoid and the loss, `mean` to reduce, and `allclose` to make the verdict.
    """)
    return


@app.cell
def _(np):
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def loss_fn(w, b, X, y):
        """Binary cross entropy for a single logistic neuron."""
        p = sigmoid(X @ w + b)
        p = np.clip(p, 1e-12, 1.0 - 1e-12)  # keep the log finite, as in Part 3
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def analytic_grad(w, b, X, y):
        """The hand-derived gradient. This is the thing under test."""
        p = sigmoid(X @ w + b)
        dz = (p - y) / len(y)
        return X.T @ dz, dz.sum()

    return analytic_grad, loss_fn


@app.cell
def _(analytic_grad, loss_fn, np):
    check_rng = np.random.default_rng(0)

    X_check = check_rng.normal(size=(20, 3))
    y_check = (X_check[:, 0] + X_check[:, 1] > 0).astype(np.float64)
    w_check = check_rng.normal(size=3) * 0.5
    b_check = 0.1

    dw_analytic, db_analytic = analytic_grad(w_check, b_check, X_check, y_check)

    # now the same gradient, estimated by nudging each weight in turn.
    # nditer rather than range() so this code is unchanged for a weight matrix.
    h = 1e-5
    dw_numeric = np.zeros_like(w_check)
    it = np.nditer(w_check, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        w_up, w_down = w_check.copy(), w_check.copy()
        w_up[idx] += h
        w_down[idx] -= h
        dw_numeric[idx] = (
            loss_fn(w_up, b_check, X_check, y_check)
            - loss_fn(w_down, b_check, X_check, y_check)
        ) / (2 * h)

    print("analytic ", dw_analytic.round(8))
    print("numerical", dw_numeric.round(8))
    print("difference", np.abs(dw_analytic - dw_numeric).max())
    print()
    print("do they agree?", np.allclose(dw_analytic, dw_numeric, atol=1e-4))
    return X_check, b_check, dw_numeric, w_check, y_check


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note the `atol=1e-4`. The default of `1e-08` would fail here, and not because the maths is wrong as the numerical estimate is itself approximate, and subtracting two nearly equal losses and dividing by a small `h` loses precision badly. A tolerance around `1e-4` is the usual choice for a gradient check, and `nn_from_scratch.py` uses exactly that.

    The last cell shows what a *failing* check looks like, which is the more useful thing to recognise. I have introduced a plausible bug: forgetting to divide by the batch size. The loss still goes down if you train with it, the shapes are all correct, and nothing raises. Only the check catches it.
    """)
    return


@app.cell
def _(X_check, b_check, dw_numeric, np, w_check, y_check):
    def buggy_grad(w, b, X, y):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        dz = p - y  # the bug: no / len(y)
        return X.T @ dz, dz.sum()

    dw_buggy, _ = buggy_grad(w_check, b_check, X_check, y_check)

    print("buggy    ", dw_buggy.round(6))
    print("numerical", dw_numeric.round(6))
    print("do they agree?", np.allclose(dw_buggy, dw_numeric, atol=1e-4))
    print()
    print("the ratio between them:", (dw_buggy / dw_numeric).round(2))
    print(
        "which is the batch size,",
        len(y_check),
        "- the check tells you where to look",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here, the buggy gradient is larger than the numerical gradient by a factor equal to the batch size. Our loss is a **mean** over the samples, but `buggy_grad` adds their contributions without dividing by `len(y)`. It therefore calculates the gradient of the summed loss. Dividing `dz` by the batch size fixes both the weight and bias gradients.

    The pattern of a mismatch can help us investigate. A constant ratio suggests a scaling error, such as a missing division. Opposite signs suggest checking subtraction order, whilst a mismatch in one component suggests checking its indexing. These are clues rather than guarantees, and ratios are unreliable when the numerical gradient is close to zero.

    When we move to PyTorch, `loss.backward()` calculates gradients automatically by following the recorded operations and applying the chain rule. It does not perform this numerical check or verify that we wrote the intended loss. Having derived and checked a gradient ourselves helps us understand what backpropagation computes and where to investigate when training behaves unexpectedly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Make two generators with the same seed, draw from one, then draw from the other. Do they still agree? Now do it again drawing different amounts from each — what does that tell you about where the state lives?
    2. `np.allclose(a, b)` and `np.allclose(b, a)` can disagree. Find two numbers where they do, and explain it from the formula in the tolerance section.
    3. Extend the gradient check to cover the bias as well as the weights. You will need to nudge `b` the same way.
    4. Break `analytic_grad` in a different way from mine — a transpose the wrong way round, or a sign flip — and see what the ratio tells you.
    5. Run the gradient check with `h = 1e-10`. It should get *worse*, not better. Why?
    6. The check above only tests the weights. Extend the `nditer` loop to cover a weight *matrix* rather than a vector, and convince yourself you did not have to change the body of the loop.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

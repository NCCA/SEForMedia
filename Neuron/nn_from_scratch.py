#!/usr/bin/env uv run marimo edit --watch

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", app_title="Neural Networks From Scratch")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell
def _(mo):
    mo.md("""
    # Neural Networks From Scratch

    A student-led walkthrough: from a single neuron to a tiny network that
    solves XOR. Everything is built with plain NumPy so you can see exactly
    what's happening at every step — no frameworks hiding the math.

    **How to use this notebook:** run cells top to bottom. Cells marked
    **TODO** contain a function you need to finish. Each one is followed by
    a *check* cell that compares your analytical gradient against a
    numerical gradient computed by finite differences — a real technique
    used to debug backprop in practice. You'll see ✅ when it's correct.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Stage 1 — A Single Neuron (fixed weights)

    <img src="public/Neuron.png" width="1000" />
    """)
    return


@app.cell
def _(np):
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def neuron(X, w, b):
        z = X @ w + b
        return sigmoid(z)

    return neuron, sigmoid


@app.cell
def _(mo):
    mo.md(r"""
    A single neuron multiplies each input by a weight, adds a bias, and
    squashes the result through the **sigmoid** to get an output between
    0 and 1:
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(np):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    gates = {
        "AND": np.array([0, 0, 0, 1], dtype=float),
        "OR": np.array([0, 1, 1, 1], dtype=float),
        "XOR": np.array([0, 1, 1, 0], dtype=float),
    }
    return X, gates


@app.cell
def _(mo):
    mo.md("""
    Tune the weights below by hand to make a single neuron solve **AND**
    or **OR** — you just need to find a line that separates the 1s from
    the 0s. Then switch to **XOR** and try as long as you like: no
    setting of `w1`, `w2`, `b` will work, because a single neuron can
    only draw one straight line through the input space.
    """)
    return


@app.cell
def _(mo):
    gate_picker_1 = mo.ui.dropdown(
        options=["AND", "OR", "XOR"], value="AND", label="Gate"
    )
    w1_slider = mo.ui.slider(-10, 10, step=0.5, value=1, label="w1")
    w2_slider = mo.ui.slider(-10, 10, step=0.5, value=1, label="w2")
    b_slider = mo.ui.slider(-10, 10, step=0.5, value=-1.5, label="b")
    return b_slider, gate_picker_1, w1_slider, w2_slider


@app.cell
def _(
    X,
    b_slider,
    gate_picker_1,
    gates,
    mo,
    neuron,
    np,
    plot_boundary,
    w1_slider,
    w2_slider,
):
    w_manual = np.array([w1_slider.value, w2_slider.value])
    b_manual = b_slider.value
    y_manual = gates[gate_picker_1.value]
    preds_manual = neuron(X, w_manual, b_manual)
    accuracy_manual = np.mean((preds_manual > 0.5) == y_manual) * 100

    rows = "\n".join(
        f"| {x[0]:.0f} | {x[1]:.0f} | {y:.0f} | {p:.3f} |"
        for x, y, p in zip(X, y_manual, preds_manual, strict=False)
    )
    table_manual = mo.md(
        f"""
    **Gate:** {gate_picker_1.value} &nbsp;&nbsp; **Accuracy:** {accuracy_manual:.0f}%

    | x1 | x2 | target | neuron output |
    |----|----|--------|----------------|
    {rows}
    """
    )

    ax_manual = plot_boundary(
        w_manual,
        b_manual,
        X,
        y_manual,
        f"{gate_picker_1.value} (manual weights)",
    )

    controls_manual = mo.hstack(
        [gate_picker_1, w1_slider, w2_slider, b_slider],
        justify="center",
    )
    mo.vstack(
        [
            mo.hstack(
                [table_manual, ax_manual.figure],
                justify="center",
                align="center",
                widths="equal",
            ),
            controls_manual,
        ]
    )
    return


@app.cell
def _(np, plt, sigmoid):
    def plot_boundary(w, b, X, y, title, ax=None):
        """Shade the region where the neuron predicts class 1 vs class 0."""
        if ax is None:
            _, ax = plt.subplots(figsize=(4, 4))
        xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        zz = sigmoid(grid @ w + b).reshape(xx.shape)
        ax.contourf(
            xx,
            yy,
            zz,
            levels=[0, 0.5, 1],
            colors=["#fde0dd", "#deebf7"],
            alpha=0.8,
        )
        for (px, py), target in zip(X, y):
            is_true = target > 0.5
            ax.text(
                px,
                py,
                "T" if is_true else "F",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color="white",
                bbox={
                    "boxstyle": "circle",
                    "facecolor": "#d6604d" if is_true else "#4393c3",
                    "edgecolor": "k",
                },
                zorder=3,
            )
        ax.set_title(title)
        return ax

    return (plot_boundary,)


@app.cell
def _(mo):
    mo.md("""
    ## Stage 2 — Learning the weights (gradient descent)
    """)
    return


@app.cell
def _(np):
    def bce_loss(a, y):
        """Binary cross-entropy loss, averaged over examples."""
        eps = 1e-9
        a = np.clip(a, eps, 1 - eps)
        return -np.mean(y * np.log(a) + (1 - y) * np.log(1 - a))

    def numerical_gradient(f, param, eps=1e-5):
        """Finite-difference gradient of scalar function f w.r.t. array param."""
        grad = np.zeros_like(param, dtype=float)
        it = np.nditer(param, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            original = param[idx]
            param[idx] = original + eps
            plus = f(param)
            param[idx] = original - eps
            minus = f(param)
            param[idx] = original
            grad[idx] = (plus - minus) / (2 * eps)
        return grad

    return bce_loss, numerical_gradient


@app.cell
def _(mo):
    mo.md(r"""
    ### TODO: implement `compute_gradient`

    For a single sigmoid neuron trained with binary cross-entropy loss,
    the gradient with respect to the *pre-activation* `z` simplifies to
    a clean expression:

    $$\frac{\partial L}{\partial z} = a - y$$

    From there, for a batch of examples:

    $$\frac{\partial L}{\partial w} = \text{mean}\big((a - y)\, x\big)
    \qquad
    \frac{\partial L}{\partial b} = \text{mean}(a - y)$$

    Implement this below. `X` has shape `(n, 2)`, `y` has shape `(n,)`.
    """)
    return


@app.function
def compute_gradient(X, y, w, b):
    """
    TODO: return (dw, db), the gradient of the binary cross-entropy loss
    with respect to w and b, for a single sigmoid neuron.

    Steps:
      1. Run the forward pass to get predictions `a = neuron(X, w, b)`.
      2. Compute `error = a - y`.
      3. dw = mean over examples of error * x   (shape matches w)
         db = mean over examples of error       (a single number)
    """
    raise NotImplementedError("Implement compute_gradient")


@app.cell
def _(mo):
    mo.accordion(
        {
            "🔑 Reveal solution — `compute_gradient`": mo.md(
                r"""
                ```python
                def compute_gradient(X, y, w, b):
                    a = neuron(X, w, b)
                    error = a - y
                    dw = np.mean(error[:, None] * X, axis=0)
                    db = np.mean(error)
                    return dw, db
                ```
                """
            )
        }
    )
    return


@app.cell
def _(X, bce_loss, gates, neuron, np, numerical_gradient):
    def _check_single_neuron_gradient():
        y_check = gates["OR"]
        w_check = np.array([0.3, -0.7])
        b_check = 0.1

        def loss_of(w, b):
            return bce_loss(neuron(X, w, b), y_check)

        dw_num = numerical_gradient(lambda w: loss_of(w, b_check), w_check.copy())
        db_num = numerical_gradient(
            lambda b: loss_of(w_check, b[0]), np.array([b_check])
        )[0]
        dw_analytic, db_analytic = compute_gradient(X, y_check, w_check, b_check)

        assert np.allclose(dw_analytic, dw_num, atol=1e-4), (
            "dw doesn't match the numerical gradient"
        )
        assert np.isclose(db_analytic, db_num, atol=1e-4), (
            "db doesn't match the numerical gradient"
        )
        return "✅ compute_gradient matches the numerical gradient!"

    _check_single_neuron_gradient()
    return


@app.cell
def _(bce_loss, neuron, np):
    def train_neuron(X, y, lr, epochs, seed=0):
        rng = np.random.default_rng(seed)
        w = rng.normal(size=X.shape[1]) * 0.5
        b = 0.0
        history = []
        for _ in range(epochs):
            a = neuron(X, w, b)
            history.append(bce_loss(a, y))
            dw, db = compute_gradient(X, y, w, b)
            w = w - lr * dw
            b = b - lr * db
        return w, b, history

    return (train_neuron,)


@app.cell
def _(mo):
    gate_picker_2 = mo.ui.dropdown(
        options=["AND", "OR", "XOR"], value="OR", label="Gate"
    )
    lr_slider = mo.ui.slider(0.01, 3.0, step=0.01, value=0.5, label="Learning rate")
    epochs_slider = mo.ui.slider(50, 3000, step=50, value=500, label="Epochs")
    mo.hstack([gate_picker_2, lr_slider, epochs_slider])
    return epochs_slider, gate_picker_2, lr_slider


@app.cell
def _(
    X,
    epochs_slider,
    gate_picker_2,
    gates,
    lr_slider,
    plot_boundary,
    plt,
    train_neuron,
):
    y_train = gates[gate_picker_2.value]
    w_trained, b_trained, history = train_neuron(
        X, y_train, lr_slider.value, epochs_slider.value
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.plot(history)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("Training loss")
    plot_boundary(
        w_trained,
        b_trained,
        X,
        y_train,
        f"{gate_picker_2.value}: final boundary",
        ax=ax2,
    )
    fig.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Stage 3 — Where a single neuron breaks

    Switch the dropdown above to **XOR** and re-run training. Watch the
    loss plateau well above zero, and the boundary plot show a single
    straight line cutting through the points — it's structurally
    incapable of separating XOR's two classes, no matter the learning
    rate or how long you train. This is *why* we stack neurons.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Stage 4 — A two-layer network solves XOR
    """)
    return


@app.cell
def _(sigmoid):
    def forward_mlp(X, W1, b1, W2, b2):
        """Forward pass through one hidden layer and a sigmoid output."""
        z1 = X @ W1 + b1
        a1 = sigmoid(z1)
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)
        return a1, a2

    return (forward_mlp,)


@app.cell
def _(mo):
    mo.md(r"""
    ### TODO: implement `compute_gradients_mlp`

    This is backprop applied via the chain rule, one layer at a time.
    With `y` reshaped to column shape `(n, 1)`:

    $$dz_2 = a_2 - y$$
    $$dW_2 = \frac{1}{n} a_1^T dz_2 \qquad db_2 = \text{mean}(dz_2)$$
    $$da_1 = dz_2\, W_2^T \qquad dz_1 = da_1 \odot a_1 \odot (1 - a_1)$$
    $$dW_1 = \frac{1}{n} X^T dz_1 \qquad db_1 = \text{mean}(dz_1,\ \text{axis}=0)$$

    Notice `dz_1` reuses `a1 * (1 - a1)`, the derivative of sigmoid —
    the same building block from Stage 2, just propagated one layer back.
    """)
    return


@app.function
def compute_gradients_mlp(X, y, W1, b1, W2, b2):
    """
    TODO: return (dW1, db1, dW2, db2), the gradients of the binary
    cross-entropy loss with respect to every parameter of a 2-layer
    sigmoid network. See the markdown cell above for the formulas.

    X: (n, 2)   W1: (2, h)   b1: (h,)   W2: (h, 1)   b2: (1,)
    y: (n,)  -- reshape to (n, 1) before using it in matrix ops.
    """
    raise NotImplementedError("Implement compute_gradients_mlp")


@app.cell
def _(mo):
    mo.accordion(
        {
            "🔑 Reveal solution — `compute_gradients_mlp`": mo.md(
                r"""
                ```python
                def compute_gradients_mlp(X, y, W1, b1, W2, b2):
                    n = X.shape[0]
                    y_col = y.reshape(-1, 1)

                    a1, a2 = forward_mlp(X, W1, b1, W2, b2)

                    # Output layer
                    dz2 = a2 - y_col          # (n, 1)
                    dW2 = a1.T @ dz2 / n      # (h, 1)
                    db2 = np.mean(dz2, axis=0)  # (1,)

                    # Hidden layer — chain rule back through sigmoid
                    da1 = dz2 @ W2.T          # (n, h)
                    dz1 = da1 * a1 * (1 - a1)  # (n, h)  sigmoid derivative
                    dW1 = X.T @ dz1 / n       # (2, h)
                    db1 = np.mean(dz1, axis=0)  # (h,)

                    return dW1, db1, dW2, db2
                ```
                """
            )
        }
    )
    return


@app.cell
def _(X, bce_loss, forward_mlp, gates, np, numerical_gradient):
    def _check_mlp_gradients():
        rng = np.random.default_rng(1)
        X_check = X[:3]
        y_check = gates["XOR"][:3]
        hidden = 3
        W1_c = rng.normal(size=(2, hidden)) * 0.5
        b1_c = rng.normal(size=hidden) * 0.5
        W2_c = rng.normal(size=(hidden, 1)) * 0.5
        b2_c = rng.normal(size=1) * 0.5
        y_col = y_check.reshape(-1, 1)

        def loss_of(W1=W1_c, b1=b1_c, W2=W2_c, b2=b2_c):
            _, a2 = forward_mlp(X_check, W1, b1, W2, b2)
            return bce_loss(a2, y_col)

        dW1_a, db1_a, dW2_a, db2_a = compute_gradients_mlp(
            X_check, y_check, W1_c, b1_c, W2_c, b2_c
        )

        dW1_n = numerical_gradient(lambda w: loss_of(W1=w), W1_c.copy())
        db1_n = numerical_gradient(lambda b: loss_of(b1=b), b1_c.copy())
        dW2_n = numerical_gradient(lambda w: loss_of(W2=w), W2_c.copy())
        db2_n = numerical_gradient(lambda b: loss_of(b2=b), b2_c.copy())

        assert np.allclose(dW1_a, dW1_n, atol=1e-4), (
            "dW1 doesn't match the numerical gradient"
        )
        assert np.allclose(db1_a, db1_n, atol=1e-4), (
            "db1 doesn't match the numerical gradient"
        )
        assert np.allclose(dW2_a, dW2_n, atol=1e-4), (
            "dW2 doesn't match the numerical gradient"
        )
        assert np.allclose(db2_a, db2_n, atol=1e-4), (
            "db2 doesn't match the numerical gradient"
        )
        return "✅ compute_gradients_mlp matches the numerical gradient!"

    _check_mlp_gradients()
    return


@app.cell
def _(forward_mlp, np, plt):
    def plot_boundary_mlp(W1, b1, W2, b2, X, y, title, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(4, 4))
        xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        _, a2 = forward_mlp(grid, W1, b1, W2, b2)
        zz = a2.reshape(xx.shape)
        ax.contourf(
            xx,
            yy,
            zz,
            levels=[0, 0.5, 1],
            colors=["#fde0dd", "#deebf7"],
            alpha=0.8,
        )
        for (px, py), target in zip(X, y):
            is_true = target > 0.5
            ax.text(
                px,
                py,
                "T" if is_true else "F",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color="white",
                bbox={
                    "boxstyle": "circle",
                    "facecolor": "#d6604d" if is_true else "#4393c3",
                    "edgecolor": "k",
                },
                zorder=3,
            )
        ax.set_title(title)
        return ax

    return (plot_boundary_mlp,)


@app.cell
def _(bce_loss, forward_mlp, np):
    def train_mlp(X, y, hidden_units, lr, epochs, seed=0):
        rng = np.random.default_rng(seed)
        W1 = rng.normal(size=(X.shape[1], hidden_units)) * 0.5
        b1 = np.zeros(hidden_units)
        W2 = rng.normal(size=(hidden_units, 1)) * 0.5
        b2 = np.zeros(1)
        y_col = y.reshape(-1, 1)
        history = []
        for _ in range(epochs):
            _, a2 = forward_mlp(X, W1, b1, W2, b2)
            history.append(bce_loss(a2, y_col))
            dW1, db1, dW2, db2 = compute_gradients_mlp(X, y, W1, b1, W2, b2)
            W1 = W1 - lr * dW1
            b1 = b1 - lr * db1
            W2 = W2 - lr * dW2
            b2 = b2 - lr * db2
        return W1, b1, W2, b2, history

    return (train_mlp,)


@app.cell
def _(mo):
    hidden_slider = mo.ui.slider(1, 8, step=1, value=2, label="Hidden units")
    lr_slider_2 = mo.ui.slider(0.01, 3.0, step=0.01, value=1.0, label="Learning rate")
    epochs_slider_2 = mo.ui.slider(200, 8000, step=200, value=3000, label="Epochs")
    mo.hstack([hidden_slider, lr_slider_2, epochs_slider_2])
    return epochs_slider_2, hidden_slider, lr_slider_2


@app.cell
def _(
    X,
    epochs_slider_2,
    gates,
    hidden_slider,
    lr_slider_2,
    plot_boundary_mlp,
    plt,
    train_mlp,
):
    y_xor = gates["XOR"]
    W1_t, b1_t, W2_t, b2_t, history_mlp = train_mlp(
        X, y_xor, hidden_slider.value, lr_slider_2.value, epochs_slider_2.value
    )

    fig_mlp, (ax1_mlp, ax2_mlp) = plt.subplots(1, 2, figsize=(9, 4))
    ax1_mlp.plot(history_mlp)
    ax1_mlp.set_xlabel("epoch")
    ax1_mlp.set_ylabel("loss")
    ax1_mlp.set_title("Training loss (XOR)")
    plot_boundary_mlp(
        W1_t, b1_t, W2_t, b2_t, X, y_xor, "XOR: final boundary", ax=ax2_mlp
    )
    fig_mlp.tight_layout()
    fig_mlp
    return


@app.cell
def _(mo):
    mo.md("""
    ## Wrap-up

    A single neuron is a linear classifier — it can only separate data
    with one straight cut. Stacking a hidden layer lets the network
    combine several straight cuts into a curved boundary, which is
    exactly what XOR requires. Everything that follows in deep learning
    (more layers, more units, different activations) is the same idea
    scaled up.

    **Extension ideas:** try increasing hidden units to see how much the
    boundary shape changes; try a stricter `atol` in the gradient checks;
    or generalize `forward_mlp` / `compute_gradients_mlp` into a `Layer`
    class that can be stacked to arbitrary depth.
    """)
    return


if __name__ == "__main__":
    app.run()

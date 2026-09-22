#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NumPy for Machine Learning, Part 5: putting it together

    In the first four notebooks we looked at the individual NumPy functions.
    We will now use them together to train a single neuron to learn a logic gate.

    I have kept the example small so we can follow the calculations. There are
    four rows of data and three parameters, so we can check the results by hand
    if needed.

    We will use the following from the previous notebooks:

    | From | What we use it for |
    | --- | --- |
    | Part 1 | `np.array` and `np.zeros` for the data and parameter storage |
    | Part 2 | `np.c_` and `reshape` to arrange arrays, and `meshgrid` for plotting |
    | Part 3 | `@` for the forward pass, `np.exp` and `np.log` for the calculations, `mean` for the loss, and `np.where` for predictions |
    | Part 4 | `default_rng` to initialise the weights, and gradient checking |

    Once we have trained the neuron we will plot its decision boundary.
    We will then try XOR to see where this model stops being useful.
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
    ## The data

    A logic gate takes two inputs, each either zero or one, and produces one
    output. There are only four possible input combinations, so we can write
    out the whole dataset.

    Start with AND, then try OR once you have worked through the notebook.
    We will come back to XOR later.
    """)
    return


@app.cell
def _(mo):
    gate = mo.ui.radio(
        options=["AND", "OR", "XOR"],
        value="AND",
        label="which gate to learn",
        inline=True,
    )
    gate
    return (gate,)


@app.cell
def _(gate, np):
    # each row is one sample with two input features
    training_inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

    _truth_tables = {
        "AND": [0.0, 0.0, 0.0, 1.0],
        "OR": [0.0, 1.0, 1.0, 1.0],
        "XOR": [0.0, 1.0, 1.0, 0.0],
    }
    training_targets = np.array(_truth_tables[gate.value])

    print(f"the {gate.value} truth table")
    print("  a    b  ->  target")
    for _row, _target in zip(training_inputs, training_targets, strict=False):
        print(f"{_row[0]:>3.0f}  {_row[1]:>3.0f}  ->  {_target:.0f}")
    print()
    print("inputs ", training_inputs.shape, " targets", training_targets.shape)
    return training_inputs, training_targets


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The model

    Our neuron calculates a weighted sum of the inputs and adds a bias.
    We then pass this through a sigmoid to get a value between zero and one,
    which we interpret as the probability of the output being one.

    ```python
    z = X @ w + b            # (4, 2) @ (2,) + scalar -> (4,)
    p = 1 / (1 + np.exp(-z))
    ```

    This is the forward pass. We have two weights in `w` and one bias in `b`,
    giving us three parameters to train.

    Look at the shapes in the first line. `X @ w` gives us one value for each
    sample, so the result has shape `(4,)`. NumPy then broadcasts the scalar
    `b`, adding it to all four values.

    We do not need to loop over the samples. If we add more rows to `X`, the
    calculation stays the same.
    """)
    return


@app.cell
def _(np):
    def forward(X, w, b):
        """
        Calculate the output probability for each sample.

        Parameters
        ----------
            X : np.ndarray
                the (n, 2) feature matrix
            w : np.ndarray
                the (2,) weights
            b : float
                the bias

        Returns
        -------
            np.ndarray
                an (n,) array of probabilities
        """
        z = X @ w + b
        return 1.0 / (1.0 + np.exp(-z))

    def loss(p, y):
        """Calculate the mean binary cross-entropy loss."""
        # avoid taking log(0), as in Part 3
        safe = np.clip(p, 1e-12, 1.0 - 1e-12)
        return -np.mean(y * np.log(safe) + (1 - y) * np.log(1 - safe))

    def gradients(X, p, y):
        """Calculate the weight and bias gradients."""
        error = (p - y) / len(y)
        return X.T @ error, error.sum()

    return forward, gradients, loss


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Where the gradient comes from

    There is quite a lot of differentiation behind the two lines in
    `gradients`. We can work through it using the chain rule.

    Differentiating binary cross entropy with respect to `p` gives a term
    with `p * (1 - p)` in the denominator. The derivative of the sigmoid
    with respect to `z` is `p * (1 - p)`. These cancel, leaving:

    ```python
    dL_dz = (p - y) / n
    ```

    The division by `n` comes from taking the mean loss over the batch.
    We then multiply by `X.T` to get the weight gradients. For the bias we
    sum the values, as the same bias was added to every sample.

    This cancellation is useful. If we use squared error with a sigmoid,
    the gradient still contains the sigmoid derivative. This gets small
    near zero and one, so learning can be slow even when the prediction
    is wrong.

    Here we use the gradient of sigmoid cross entropy. The clipping in
    `loss` keeps the printed loss finite, but we are not differentiating
    through that clipping. Keep this in mind when checking gradients at
    extreme probabilities.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training

    We will use gradient descent to update the weights and bias:

    1. Run the forward pass.
    2. Calculate and store the loss.
    3. Calculate the gradients.
    4. Subtract the learning rate multiplied by each gradient.
    5. Repeat.

    The learning rate controls the size of each update. A small value can
    make training slow, whilst a value that is too large can cause the loss
    to oscillate or increase.

    Try changing the learning rate and the number of training steps.
    Watch the loss curve to see what happens for this example.
    """)
    return


@app.cell
def _(mo):
    learning_rate = mo.ui.slider(
        0.05, 5.0, step=0.05, value=1.0, label="learning rate", show_value=True
    )
    training_steps = mo.ui.slider(
        50, 4000, step=50, value=1500, label="training steps", show_value=True
    )
    mo.vstack([learning_rate, training_steps])
    return learning_rate, training_steps


@app.cell
def _(
    forward,
    gradients,
    learning_rate,
    loss,
    np,
    training_inputs,
    training_steps,
    training_targets,
):
    # keep the starting weights the same when comparing slider settings
    _rng = np.random.default_rng(4)
    _w = _rng.normal(0.0, 0.5, size=2)
    _b = 0.0

    loss_history = np.zeros(training_steps.value)

    for _step in range(training_steps.value):
        _p = forward(training_inputs, _w, _b)
        loss_history[_step] = loss(_p, training_targets)

        _dw, _db = gradients(training_inputs, _p, training_targets)
        _w = _w - learning_rate.value * _dw
        _b = _b - learning_rate.value * _db

    trained_weights = _w
    trained_bias = _b

    print("learned weights", trained_weights.round(4))
    print("learned bias   ", round(trained_bias, 4))
    print()
    print(
        "loss went from",
        loss_history[0].round(4),
        "to",
        loss_history[-1].round(4),
    )
    return loss_history, trained_bias, trained_weights


@app.cell
def _(loss_history, plt):
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot(loss_history)
    _ax.set(xlabel="training step", ylabel="mean cross entropy", ylim=(0, None))
    _ax.set_title("loss before each update", loc="left")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What it learned

    We can now run the four inputs through the trained neuron and compare
    the predictions with the truth table.

    The output is a probability. We use `np.where` from Part 3 to turn this
    into a gate output: values greater than 0.5 become one, and everything
    else becomes zero.
    """)
    return


@app.cell(hide_code=True)
def _(
    forward,
    np,
    trained_bias,
    trained_weights,
    training_inputs,
    training_targets,
):
    trained_probabilities = forward(training_inputs, trained_weights, trained_bias)
    trained_predictions = np.where(trained_probabilities > 0.5, 1.0, 0.0)

    print("  a    b   target   probability   predicted")
    for _row, _t, _p, _pred in zip(
        training_inputs,
        training_targets,
        trained_probabilities,
        trained_predictions,
        strict=False,
    ):
        _tick = "ok" if _t == _pred else "WRONG"
        print(
            f"{_row[0]:>3.0f}  {_row[1]:>3.0f}   {_t:>6.0f}   {_p:>11.4f}   {_pred:>9.0f}  {_tick}"
        )

    print()
    print("accuracy", (trained_predictions == training_targets).mean())
    return (trained_predictions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The decision boundary

    To see what the neuron has learned, we will sample a grid around the
    four training points and plot the output probabilities.

    The gate only uses zero and one as inputs, but the model will accept
    values between and beyond them. This lets us see where its prediction
    changes from one class to the other.

    We will use the array operations from Part 2:

    1. Create an axis with `linspace`, then use `meshgrid` to make two
       coordinate arrays with shape `(r, r)`.
    2. Flatten both arrays with `ravel` and join them with `np.c_`.
       This gives us `(r * r, 2)` input coordinates.
    3. Pass these coordinates through the model to get `(r * r,)`
       probabilities.
    4. Use `reshape` to put the probabilities back into an `(r, r)` grid.
    5. Plot the grid using `contourf`.

    The forward pass works on this larger batch without any changes.
    Each row still contains two input features.
    """)
    return


@app.cell
def _(mo):
    grid_resolution = mo.ui.slider(
        5, 121, step=4, value=61, label="samples per axis", show_value=True
    )
    grid_resolution
    return (grid_resolution,)


@app.cell
def _(forward, grid_resolution, np, trained_bias, trained_weights):
    _axis = np.linspace(-0.4, 1.4, grid_resolution.value)
    grid_x, grid_y = np.meshgrid(_axis, _axis)

    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    _flat = forward(grid_points, trained_weights, trained_bias)
    boundary_probabilities = _flat.reshape(grid_x.shape)

    print("lattice       ", grid_x.shape)
    print("as a batch    ", grid_points.shape)
    print("predictions   ", _flat.shape)
    print("folded back to", boundary_probabilities.shape)
    return boundary_probabilities, grid_x, grid_y


@app.cell
def _(
    boundary_probabilities,
    gate,
    grid_x,
    grid_y,
    np,
    plt,
    trained_predictions,
    training_inputs,
    training_targets,
):
    _fig, _ax = plt.subplots(figsize=(5.5, 4.5))

    _shading = _ax.contourf(
        grid_x,
        grid_y,
        boundary_probabilities,
        levels=np.linspace(0, 1, 21),
        cmap="coolwarm",
        vmin=0,
        vmax=1,
    )
    _ax.contour(grid_x, grid_y, boundary_probabilities, levels=[0.5], colors="black")

    _ax.scatter(
        training_inputs[:, 0],
        training_inputs[:, 1],
        c=training_targets,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        s=220,
        edgecolors="black",
        linewidths=1.5,
        zorder=3,
    )
    for _row, _t, _pred in zip(
        training_inputs, training_targets, trained_predictions, strict=False
    ):
        _label = f"{_t:.0f}" if _t == _pred else f"{_t:.0f} (missed)"
        _ax.annotate(_label, _row, xytext=(10, 8), textcoords="offset points")

    _ax.set(xlabel="input a", ylabel="input b")
    _ax.set_title(f"one neuron on {gate.value}", loc="left")
    _fig.colorbar(_shading, ax=_ax, label="probability of 1")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Now try XOR

    Go back to the radio buttons and select XOR. Increase the number of
    training steps and watch the loss. As training converges, the probabilities
    approach 0.5 and the mean loss approaches 0.693.

    At exactly 0.5 our threshold predicts zero for every sample, giving two
    correct answers. Whilst training is approaching this value, small
    differences around the threshold can change the reported accuracy.
    Look at the probabilities as well as the final predictions.

    We can see the problem by looking at the decision boundary. The sigmoid
    gives 0.5 when its input is zero, so the boundary satisfies:

    ```python
    X @ w + b = 0
    ```

    With two input features and non-zero weights, this describes a straight
    line. AND and OR can each be separated by a straight line, so our neuron
    can learn them.

    For XOR, the ones are on opposite corners and the zeros are on the other
    two. We cannot draw one straight line that separates these classes.
    Changing the learning rate or training for longer cannot remove this
    limitation.

    We need a model that can produce a non-linear boundary. One option is
    to add a hidden layer with a non-linear activation, then train an output
    neuron to combine its results. The non-linear activation matters:
    stacking linear layers alone still gives us a linear model.

    We can use this small network to learn XOR. The next step is to work
    through the forward pass and gradients for both layers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Train AND with a learning rate of 5, then try 0.05 with 4000 steps.
       Describe what happens to the loss curve in each case.
    2. Replace the random starting weights with `np.zeros(2)` and train AND.
       Explain why this works for our single neuron. How does this differ
       from giving all neurons in a hidden layer the same starting weights?
    3. Add NAND to the truth table dictionary and the radio button options.
       Before running it, draw a line that separates its two classes.
    4. Calculate the loss when all four probabilities are exactly 0.5.
       Compare this with the loss after training on XOR.
    5. Use the gradient check from Part 4 to check both the weights and bias.
       Choose values that keep the probabilities away from the clipping
       limits. Then remove `/ len(y)` from `gradients` and explain the
       difference.
    6. Add a hidden layer of two neurons and train the network on XOR.
       You will need weights and biases for both layers, a non-linear
       activation in the hidden layer, and gradients through both layers.

    We have now used NumPy to build the data, run a model, calculate its
    gradients and update its parameters. Keep this example to hand when
    moving on to PyTorch so we can compare the calculations.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

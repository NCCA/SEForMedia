import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training a Linear Model by Hand

    This simple gui setup show how we can simulate adjusting the weights and biases of a simple $y=mx+c$ model to fit a line.

    The sliders allow us to ajust $x$ and $c$ and the steps slider is intended to show how learning rate affects how close we can get to the answer.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import random

    print(torch.__version__)
    return mo, plt, random, torch


@app.cell
def _(random, torch):
    weight = random.uniform(0, 1.0)
    bias = random.uniform(-1, 1.0)

    # Create data
    start = 0
    end = 1
    step = 0.01
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias
    return X, bias, weight, y


@app.cell
def _(plt):
    def plot_data(x, y, user=None):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(x, y, c="b", s=4, label="equation")

        if user is not None:
            ax.scatter(x, user, c="r", s=4, label="Predictions")

        ax.legend(prop={"size": 12})
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_ylim(-1, 1)

        return fig

    return (plot_data,)


@app.cell
def _():
    return


@app.cell
def _(mo):
    steps_slider = mo.ui.slider(
        0.01, 1.0, step=0.01, value=0.05, label="steps", show_value=True
    )
    return (steps_slider,)


@app.cell
def _(mo, steps_slider):
    weight_slider = mo.ui.slider(
        0.0,
        1.0,
        step=steps_slider.value,
        value=0.1,
        label="weight",
        show_value=True,
    )
    bias_slider = mo.ui.slider(
        -1.0,
        1.0,
        step=steps_slider.value,
        label="bias",
        value=0.1,
        show_value=True,
    )
    return bias_slider, weight_slider


@app.cell
def _(
    X,
    bias,
    bias_slider,
    mo,
    plot_data,
    steps_slider,
    weight,
    weight_slider,
    y,
):
    user = weight_slider.value * X + bias_slider.value
    fig = plot_data(X, y, user)
    print(f"{weight=:0.3f},{bias=:0.3f}")
    mo.vstack(
        [
            steps_slider,
            weight_slider,
            bias_slider,
            fig,
        ]
    )
    return


if __name__ == "__main__":
    app.run()

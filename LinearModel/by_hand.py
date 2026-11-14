import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch

    print(torch.__version__)
    return mo, plt, torch


@app.cell
def _(torch):
    weight = 0.7
    bias = 0.3

    # Create data
    start = 0
    end = 1
    step = 0.01
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias
    return X, y


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
        ax.set_ylim(0, 1)

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
        0.0, 1.0, step=steps_slider.value, label="weight", show_value=True
    )
    bias_slider = mo.ui.slider(
        0.0, 1.0, step=steps_slider.value, label="bias", show_value=True
    )
    return bias_slider, weight_slider


@app.cell
def _(X, bias_slider, mo, plot_data, steps_slider, weight_slider, y):
    user = weight_slider.value * X + bias_slider.value
    fig = plot_data(X, y, user)
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

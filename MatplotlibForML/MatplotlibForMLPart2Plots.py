#!/usr/bin/env uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Matplotlib for Machine Learning, Part 2: plots for training and results

    In this notebook we will use four plot types: lines to show changes over epochs, scatter plots to explore relationships between two variables, histograms to examine distributions, and bars to compare counts across categories. We will also add legends and look at a common plotting mistake.

    We need to choose a plot that suits the data and what we want to show. Connecting points with a line suggests an order and a relationship between neighbouring values, which may not make sense for categories. Bars use length to represent values, so their numerical axis should start at zero. Truncating it can make small differences look much larger than they are.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(42)
    return np, plt, rng


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First we will generate some synthetic data to plot for our examples.
    """)
    return


@app.cell
def _(np, rng):
    # a plausible training run to plot throughout
    n_epochs = 80
    epochs = np.arange(n_epochs)
    train_loss = 2.3 * np.exp(-epochs / 14) + 0.05 + rng.normal(0, 0.015, n_epochs)
    valid_loss = 2.3 * np.exp(-epochs / 16) + 0.22 + rng.normal(0, 0.03, n_epochs)
    valid_loss[45:] += np.linspace(0, 0.25, n_epochs - 45)  # starts overfitting
    valid_acc = 1 - valid_loss / 3.2

    print(
        "epochs",
        n_epochs,
        "| final train loss",
        round(train_loss[-1], 3),
        "| final valid",
        round(valid_loss[-1], 3),
    )
    return epochs, train_loss, valid_acc, valid_loss


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [plt.plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html)

    We use `plt.plot()` or `ax.plot()` to draw lines, optionally with markers at each data point:

    ```python
    ax.plot(x, y, color="red", linestyle="--", marker="o", label="Training")
    ```

    The main options are:

    | Parameter | What it controls |
    | --- | --- |
    | `x`, `y` | Coordinates of the points. If we supply only `y`, the x values are `0, 1, 2, ...`. |
    | `fmt` | An optional shorthand string combining colour, line style and marker. |
    | `color` | The line colour. If omitted, Matplotlib uses the next colour in its configured cycle. |
    | `linestyle` | The line pattern: `"-"`, `"--"`, `":"` or `"-."`. Use `"none"` for markers without a line. |
    | `linewidth` | Line width in points. |
    | `marker` | A symbol at each point, such as `"o"` for circles or `"s"` for squares. |
    | `markersize` | Marker size in points. |
    | `label` | Text to use when we add a legend. |
    | `alpha` | Opacity, from `0` (transparent) to `1` (opaque). |

    The format string provides a shorter way to specify the appearance:

    ```python
    ax.plot(x, y, "r--o")
    ```

    This draws a red dashed line with circle markers. Explicit keywords take more space, but make the code easier to read without remembering the shorthand.

    ### The training curve

    Training curves show how a metric changes over epochs. We plot training and validation values for the same metric on the same `Axes` so we can compare their progress and see whether a gap develops between them.
    """)
    return


@app.cell
def _(epochs, plt, train_loss, valid_loss):
    fig_curve, ax_curve = plt.subplots(figsize=(6, 3.2), layout="constrained")

    ax_curve.plot(epochs, train_loss, label="train", linewidth=2)
    ax_curve.plot(epochs, valid_loss, label="validation", linewidth=2)

    ax_curve.set(
        xlabel="epoch",
        ylabel="cross-entropy loss (nats)",
        title="Training and validation loss",
    )
    ax_curve.legend()
    ax_curve.grid(alpha=0.3)  # recessive, so it helps reading without competing
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are four choices worth noting in this cell:

    - **Both series use the same `Axes`.** Sharing a scale makes it easier to compare the losses and see them diverge after epoch 45.
    - **Each line has a `label`.** Calling `ax.legend()` then adds the legend without needing any arguments.
    - **`linewidth=2` makes the lines easier to see**, particularly when projecting the plot.
    - **`grid(alpha=0.3)` keeps the grid faint.** It helps us estimate values without distracting from the data.

    The divergence is what we are looking for. Training loss continues to fall whilst validation loss rises, suggesting that the model is overfitting. The lowest validation loss identifies a candidate checkpoint to keep. In practice, we usually allow several epochs without improvement before stopping, as validation loss can fluctuate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The colour cycle

    If we do not specify a colour, Matplotlib chooses the next one from the `Axes` colour cycle. The default cycle contains the ten colours from `tab10`, which we can refer to as `"C0"` through `"C9"`. A style or configuration change can replace these colours.

    Colours are assigned in plotting order. Adding a line before the others can therefore change their colours, and the same colour may represent different things in different figures. If a colour has a particular meaning, such as blue for training loss, we should set it explicitly and use it consistently.

    After ten lines, the default cycle repeats. With many overlapping lines, colour alone becomes difficult to follow. We can split the series across smaller plots, highlight a few and draw the rest in grey, or summarise groups where that makes sense.
    """)
    return


@app.cell
def _(plt):
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    print("the default cycle has", len(cycle), "colours:")
    for _i, _c in enumerate(cycle):
        print(f"  C{_i}  {_c}")

    fig_cycle, ax_cycle = plt.subplots(figsize=(6, 2), layout="constrained")
    for _i in range(12):
        ax_cycle.plot([0, 1], [_i, _i], linewidth=6)
    ax_cycle.set(title="12 series, 10 colours - two pairs are now identical", yticks=[])
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Log scale

    ```python
    ax.set_yscale("log")
    ```

    A loss curve drops by orders of magnitude and then crawls. On a linear axis the interesting last 90% of training is squashed into a line along the bottom. A log y-axis gives each order of magnitude equal space and is usually the more honest view of a loss.
    """)
    return


@app.cell
def _(epochs, plt, train_loss, valid_loss):
    fig_log, (ax_lin, ax_log) = plt.subplots(
        1, 2, figsize=(7.5, 2.8), layout="constrained"
    )

    for _ax, _scale, _title in [
        (ax_lin, "linear", "linear y"),
        (ax_log, "log", "log y"),
    ]:
        _ax.plot(epochs, train_loss, label="train", linewidth=2)
        _ax.plot(epochs, valid_loss, label="validation", linewidth=2)
        _ax.set_yscale(_scale)
        _ax.set(xlabel="epoch", ylabel="loss (nats)", title=_title)
        _ax.grid(alpha=0.3)
    ax_lin.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparing metrics with different scales

    Matplotlib lets us add a second y-axis that shares the same x-axis:

    ```python
    ax2 = ax.twinx()
    ```

    This can make comparisons misleading. Each y-axis has its own scale, so changing the limits on either axis changes where the lines appear to cross, even though the data stays the same.

    The cell below plots the same loss and accuracy values twice, changing only the limits of the second y-axis. The curves cross early in one plot and late in the other. That crossing has no useful meaning: loss and accuracy measure different quantities, and their apparent intersection depends on the chosen scales.

    For training metrics, separate plots with a shared epoch axis make it easier to follow both trends without suggesting that their heights are directly comparable.
    """)
    return


@app.cell
def _(epochs, plt, valid_acc, valid_loss):
    fig_twin, axes_twin = plt.subplots(1, 2, figsize=(8, 2.8), layout="constrained")

    for _ax, _lo, _hi in [
        (axes_twin[0], 0.0, 1.0),
        (axes_twin[1], 0.55, 0.78),
    ]:
        _ax.plot(epochs, valid_loss, color="C0", linewidth=2)
        _ax.set(xlabel="epoch", ylabel="loss (nats)", ylim=(0, 2.5))
        _twin = _ax.twinx()
        _twin.plot(epochs, valid_acc, color="C1", linewidth=2)
        _twin.set(ylabel="accuracy", ylim=(_lo, _hi))
        _ax.set_title(f"same data, right axis {_lo}-{_hi}", fontsize=9)

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Both plots contain identical data, but the curves cross at different epochs. Neither crossing tells us anything useful. Loss in nats and accuracy as a fraction measure different quantities, so it makes no sense to say that accuracy has “overtaken” loss.

    We can show them in two vertically stacked plots with a shared x-axis. This lets us compare how both metrics change over epochs whilst giving each its own clearly labelled y-axis.
    """)
    return


@app.cell
def _(epochs, plt, valid_acc, valid_loss):
    fig_stack, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6, 3.6), sharex=True, layout="constrained"
    )

    ax_top.plot(epochs, valid_loss, color="C0", linewidth=2)
    ax_top.set(ylabel="loss (nats)", title="Validation loss and accuracy")

    ax_bot.plot(epochs, valid_acc, color="C1", linewidth=2)
    ax_bot.set(ylabel="accuracy", xlabel="epoch")

    for _ax in (ax_top, ax_bot):
        _ax.grid(alpha=0.3)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [plt.scatter](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html)

    We use `plt.scatter()` or `ax.scatter()` to plot individual points. Unlike `plot()`, a scatter plot lets us vary the size and colour of each marker:

    ```python
    ax.scatter(x, y, s=36, color="blue", alpha=0.6, edgecolors="white")
    ```

    | Parameter | What it controls |
    | --- | --- |
    | `s` | Marker size in points squared. This can be one value or a value for each point. |
    | `c` | A single colour, a sequence of colours, or numerical values mapped through a colormap. |
    | `marker` | Marker shape, such as `"o"` for circles. |
    | `cmap` | The colormap used when `c` contains numerical values. |
    | `vmin`, `vmax` | The value range used for colour mapping with the default normalisation. |
    | `alpha` | Opacity, from `0` (transparent) to `1` (opaque). Lower values can help reveal overlapping points. |
    | `edgecolors` | Marker outline colours. White outlines can help distinguish nearby points. |

    The size parameter `s` controls area rather than diameter. Doubling it doubles the nominal marker area; to double the diameter of the same marker, we need to multiply `s` by four. If size represents a quantity, we should scale the area in proportion to that quantity.

    For a single colour, `color="blue"` makes our intention clear. To colour points by numerical values, we pass an array to `c` and choose a colormap:

    ```python
    ax.scatter(x, y, c=values, cmap="viridis")
    ```

    Class labels need different treatment. Numerical labels such as `0`, `1` and `2` identify categories; they do not necessarily represent an ordered quantity. In Part 3 we will choose distinct colours for these classes rather than imply a progression with a continuous colour scale.
    """)
    return


@app.cell
def _(np, plt, rng):
    n_pts = 300
    cluster_a = rng.normal([-1.2, -0.6], 0.55, (n_pts, 2))
    cluster_b = rng.normal([1.1, 0.8], 0.55, (n_pts, 2))

    fig_sc, axes_sc = plt.subplots(1, 2, figsize=(7.5, 3), layout="constrained")

    axes_sc[0].scatter(*cluster_a.T, s=36)
    axes_sc[0].scatter(*cluster_b.T, s=36)
    axes_sc[0].set_title("opaque, overplotted", fontsize=9)

    axes_sc[1].scatter(
        *cluster_a.T, s=36, alpha=0.4, edgecolors="none", label="class 0"
    )
    axes_sc[1].scatter(
        *cluster_b.T, s=36, alpha=0.4, edgecolors="none", label="class 1"
    )
    axes_sc[1].set_title("alpha=0.4, density visible", fontsize=9)
    axes_sc[1].legend(markerscale=1.5)

    for _ax in axes_sc:
        _ax.set(xlabel="feature 1", ylabel="feature 2")
    plt.show()
    print(
        "overlapping points:",
        np.sum(np.abs(cluster_a[:, 0]) < 0.5),
        "near the boundary",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Without `alpha`, 600 points on top of each other look like 600 points wherever they are dense and you cannot tell a crowded region from a sparse one. With it, the density is the information. On anything above a few hundred points, set it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [plt.hist](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html)

    A histogram groups numerical values into intervals, called bins, and shows how many fall into each one. We use `plt.hist()` or `ax.hist()` to examine the shape and spread of a distribution:

    ```python
    ax.hist(weights, bins="auto")
    ```

    | Parameter | Default | What it controls |
    | --- | --- | --- |
    | `bins` | `10` | The number of bins, an explicit sequence of bin edges, or a strategy such as `"auto"`. |
    | `density` | `False` | Normalises the histogram so its total area is 1. Useful for comparing distribution shapes across groups of different sizes. |
    | `histtype` | `"bar"` | How bins are drawn. `"step"` draws outlines, which can make overlapping distributions easier to compare. |
    | `range` | `None` | The lower and upper limits used for binning. Values outside this range are excluded. |
    | `alpha` | `None` | Opacity, useful when overlaying filled histograms. |
    | `label` | `None` | Text to use in the legend. |

    In this unit, we will use histograms to inspect weight and activation distributions. A concentration of weights near zero or activations near a saturation limit can help us investigate a network’s behaviour, although the histogram alone does not establish the cause. For class balance, a bar chart of counts per class is usually clearer because class labels are categories.

    The choice of bins affects what we see. Too few can hide structure; too many can make random variation look significant. `bins="auto"` is a useful starting point, but we should still check whether the result shows the distribution clearly. When comparing groups, use the same bin edges.
    """)
    return


@app.cell
def _(np, plt, rng):
    healthy = rng.normal(0, 0.35, 4000)
    saturated = np.concatenate([rng.normal(-1, 0.06, 2000), rng.normal(1, 0.06, 2000)])

    fig_h, axes_h = plt.subplots(1, 2, figsize=(7.5, 2.8), layout="constrained")

    axes_h[0].hist(healthy, bins="auto", color="C0")
    axes_h[0].set_title("healthy weight distribution", fontsize=9)

    axes_h[1].hist(saturated, bins="auto", color="C3")
    axes_h[1].set_title("saturated nothing in the middle", fontsize=9)

    for _ax in axes_h:
        _ax.set(xlabel="weight value", ylabel="count")
    plt.show()
    return


@app.cell
def _(plt, rng):
    # overlaying two distributions: histtype='step' rather than transparent bars
    before = rng.normal(0, 1.0, 3000)
    after = rng.normal(0.4, 0.6, 3000)

    fig_ov, axes_ov = plt.subplots(1, 2, figsize=(7.5, 2.8), layout="constrained")

    for _label, _data in [("before", before), ("after", after)]:
        axes_ov[0].hist(_data, bins=40, alpha=0.5, label=_label)
        axes_ov[1].hist(_data, bins=40, histtype="step", linewidth=2, label=_label)

    axes_ov[0].set_title("alpha bars  the overlap is a third colour", fontsize=9)
    axes_ov[1].set_title("histtype='step'  both shapes readable", fontsize=9)
    for _ax in axes_ov:
        _ax.set(xlabel="activation", ylabel="count")
        _ax.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [plt.bar](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html)

    We use `plt.bar()` or `ax.bar()` to compare values across categories, such as the number of samples in each class or the accuracy of several models:

    ```python
    ax.bar(model_names, accuracies, width=0.8)
    ```

    For a standard bar chart, the numerical axis should start at zero. We judge the values by the lengths of the bars, so truncating the axis exaggerates their relative differences.

    If small differences are difficult to see, we can use a dot plot with clearly labelled limits or plot the differences from a reference value directly.

    The cell below shows the same three accuracy values twice. Only the y-axis limits change, but this changes how large the differences appear.
    """)
    return


@app.cell
def _(plt):
    models = ["baseline", "augmented", "pretrained"]
    accuracy = [0.882, 0.901, 0.927]

    fig_bar, axes_bar = plt.subplots(1, 2, figsize=(7.5, 2.8), layout="constrained")

    axes_bar[0].bar(models, accuracy, color="C0")
    axes_bar[0].set(ylim=(0, 1), ylabel="accuracy", title="honest: from zero")

    axes_bar[1].bar(models, accuracy, color="C3")
    axes_bar[1].set(ylim=(0.87, 0.94), ylabel="accuracy", title="misleading: truncated")

    for _ax in axes_bar:
        _ax.set_title(_ax.get_title(), fontsize=9)
    plt.show()

    print(
        "the real difference, baseline to pretrained:",
        f"{accuracy[2] - accuracy[0]:.3f}",
    )
    print("the right-hand chart makes it look like roughly a 7x improvement")
    return accuracy, models


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Labelling values directly

    When exact values matter, we can label them on the plot. For bars, pass the container returned by `ax.bar()` to `ax.bar_label()`:

    ```python
    bars = ax.bar(model_names, accuracies)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ```

    Here, `fmt="%.3f"` displays three decimal places, and `padding=3` leaves a gap of three points between each bar and its label.

    For individual points, `ax.annotate()` places text at a chosen location, with an optional arrow:

    ```python
    ax.annotate(
        "Lowest validation loss",
        xy=(best_epoch, best_loss),
        xytext=(20, 30),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )
    ```

    Choose labels that help explain the plot, such as an endpoint, the best epoch or an unusual value. Labelling every point can make the data harder to see.
    """)
    return


@app.cell
def _(accuracy, epochs, models, np, plt, valid_loss):
    fig_lab, axes_lab = plt.subplots(1, 2, figsize=(8, 3), layout="constrained")

    _bars = axes_lab[0].bar(models, accuracy, color="C0")
    axes_lab[0].bar_label(_bars, fmt="%.3f", padding=3)
    axes_lab[0].set(ylim=(0, 1.08), ylabel="accuracy", title="bar_label")

    best = int(np.argmin(valid_loss))
    axes_lab[1].plot(epochs, valid_loss, linewidth=2)
    axes_lab[1].scatter([best], [valid_loss[best]], s=60, zorder=3, color="C3")
    axes_lab[1].annotate(
        f"best: epoch {best}\nloss {valid_loss[best]:.3f}",
        xy=(best, valid_loss[best]),
        xytext=(best + 8, valid_loss[best] + 0.5),
        arrowprops=dict(arrowstyle="->", linewidth=1.2),
        fontsize=9,
    )
    axes_lab[1].set(xlabel="epoch", ylabel="loss (nats)", title="annotate")
    plt.show()

    print("early stopping would have stopped at epoch", best)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note `zorder=3` on that scatter. Artists are drawn in order and the default puts lines above points, so without it the marker hides behind the curve. It is a small thing that comes up whenever you overlay one plot type on another.

    ## Exercises

    1. Plot the training and validation curves with a log y-axis and mark the best epoch on both. At what epoch would early stopping with patience 5 have fired?
    2. Take the twin-axis figure and find limits that make accuracy appear to *lead* the loss. How far can you push the story before it looks wrong?
    3. Plot 5,000 scatter points with no `alpha` and then with `alpha=0.1`. Where is the actual density peak? Could you tell from the first one?
    4. Make a bar chart of class counts for a dataset that is 80% one class. Then compute what accuracy a model that always predicts that class would score.
    5. Plot eleven series with the default cycle and find the two that share a colour. Now rework the figure so it is readable — you are not allowed more colours.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

#!/usr/bin/env uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Matplotlib for Machine Learning, Part 1: figures and axes

    Matplotlib is a core part of machine learning in python. It allows us to plot graphs in 2D and 3D as well as display images easily. This set of notebooks are slit into 4 core sections.

    The four parts are:

    1. Figures, axes, and the two APIs (this notebook)
    2. Plots for training and results
    3. Images and 2D fields
    4. Getting figures out, and making them readable

    This follows on from `Lecture6/Matplotlib.ipynb`, which covers plot types, subplots and images. Where that one introduces matplotlib generally, these four are about the plots you will actually make in the machine learning part of the unit, and about the decisions that make a figure worth putting in a report.

    Documentation is at [matplotlib.org](https://matplotlib.org/stable/index.html).
    """)
    return


@app.cell
def _():
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    print("matplotlib", matplotlib.__version__)
    print("backend   ", matplotlib.get_backend())
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Two APIs for the same thing

    Nearly every matplotlib example you find online is written in one of two styles, and matplotlib itself is not consistent about which it shows. Knowing that there are two approches and how to use them is key to understanding how matplotlib works. Both methods are valid and you can use either, just try to be consitent when using matplot lib.

    **The pyplot style** is a state machine. There is a "current figure" and a "current axes" hidden in module level state, and each `plt.` call acts on whichever that happens to be.

    ```python
    plt.figure(figsize=(6, 4))
    plt.plot(x, y)
    plt.title("Loss")
    plt.show()
    ```

    **The object-oriented style**  you hold the objects and call methods on them.

    ```python
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y)
    ax.set_title("Loss")
    ```

    In the notebooks for this unit I mainly use the `plt.plot` rather than `ax.plot`. That is completely normal for notebook code and there is nothing wrong with it, the state machine is quicker to type and quicker to read for a single quick plot, which is most of what a notebook does.

    The official guidance is to prefer the object oriented style, and the reason is specific rather than stylistic: as soon as a figure has more than one axes, "the current axes" stops being obvious and starts being a thing you have to track.
    """)
    return


@app.cell
def _(np, plt):
    x = np.linspace(0, 10, 200)
    y = np.sin(x)

    # the pyplot style
    plt.figure(figsize=(5, 2.5))
    plt.plot(x, y)
    plt.title("pyplot style")
    plt.show()
    return x, y


@app.cell
def _(plt, x, y):
    # the same figure, object-oriented
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.plot(x, y)
    ax.set_title("object-oriented style")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The anatomy

    Matplotlib uses three similar names for different parts of a plot:

    | Name | What it represents |
    | --- | --- |
    | **Figure** | The whole figure, which we save to a file. |
    | **Axes** | One plot, with its own data area, title and labels. |
    | **Axis** | An individual axis, such as x or y, with its ticks and tick labels. |

    The confusing one is `Axes`: despite the name, it refers to a single plot. A figure containing four plots has one `Figure` and four `Axes` objects.

    Everything we draw, including lines, points, labels and legends, is represented by an [`Artist`](https://matplotlib.org/stable/tutorials/artists.html). Most belong to an `Axes`, although some, such as an overall title, belong to the `Figure`. We customise these objects by changing their properties, such as colour, size or line style.
    """)
    return


@app.cell
def _(plt, x, y):
    _fig, _ax = plt.subplots(figsize=(5, 2.5))
    _lines = _ax.plot(x, y)

    print("the figure   ", type(_fig).__name__)
    print("the axes     ", type(_ax).__name__)
    print("its x axis   ", type(_ax.xaxis).__name__)
    print(
        "the line     ",
        type(_lines[0]).__name__,
        "- plot returns a list, note",
    )
    print()
    print("artists on these axes:", len(_ax.get_children()))
    plt.close(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    plt.plot() returns a list of Line2D objects representing the lines it creates. In marimo, we display the plot by placing its Figure or Axes object at the end of the cell:

    ```python
    _fig, _ax = plt.subplots()
    _ax.plot(x, y)
    _fig
    ```

    You will notice if you comment out _fig in the following code it remains the same but it is always good to follow best practice.
    """)
    return


@app.cell
def _(plt, x, y):
    _fig, _ax = plt.subplots()
    _ax.plot(x, y)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we are using pyplot directly, plt.gca() returns the current Axes so we can display it:

    ```python
    plt.plot(x, y)
    plt.gca()
    ```
    """)
    return


@app.cell
def _(plt, x, y):
    plt.plot(x, y)
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    plt.show() also works, but displays the plot in marimo’s console area rather than the main cell output.
    """)
    return


@app.cell
def _(plt):
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Creating figures and plots

    We use [`plt.figure()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html) to create a figure, and [`plt.subplots()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html) to create a figure with one or more plots.

    ```python
    fig = plt.figure(figsize=(6, 4), dpi=100)

    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True)
    ```

    The main options are:

    | Parameter | Default | What it controls |
    | --- | --- | --- |
    | `figsize` | `(6.4, 4.8)`* | Figure width and height in inches. |
    | `dpi` | `100`* | Resolution in dots per inch. |
    | `nrows`, `ncols` | `1` | Number of rows and columns in the grid. |
    | `sharex`, `sharey` | `False` | Whether plots share their x or y axis, including its limits and scale. |
    | `squeeze` | `True` | Whether to remove dimensions of length one from the returned array of `Axes`. A single plot returns one `Axes` object. |
    | `layout` | `None` | Use `"constrained"` or `"tight"` to adjust spacing around plots and labels. |

    *The figure size and resolution come from Matplotlib’s settings, so a style or configuration change can override these defaults.*

    `figsize` uses inches, so we multiply by `dpi` to get the figure’s pixel dimensions. For example, `figsize=(6, 4), dpi=100` gives a 600 × 400 pixel figure. When saving, a different resolution or a tight crop can change the output dimensions.

    With several plots, `plt.subplots()` returns an array of `Axes` objects. For a two-by-two grid, we access them by row and column:

    ```python
    fig, axes = plt.subplots(2, 2, layout="constrained")
    axes[0, 0].plot(x, y)
    fig
    ```

    The similarly named [`plt.subplot()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html) selects or creates one plot within a grid on the current figure:

    ```python
    ax = plt.subplot(2, 2, 1)
    ```

    Here, `1` selects the top-left plot. Positions start at **1** and run across each row. `subplots()` creates the whole grid at once; `subplot()` works with one position at a time.
    """)
    return


@app.cell
def _(np, plt, x):
    fig_grid, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(7, 4), layout="constrained"
    )

    print("subplots returned:", type(axes).__name__, axes.shape)

    for i, a in enumerate(axes.ravel()):
        a.plot(x, np.sin(x * (i + 1)))
        a.set_title(f"sin({i + 1}x)", fontsize=9)

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Creating figures and plots

    We use [`plt.figure()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html) to create a figure, and [`plt.subplots()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html) to create a figure with one or more plots.

    ```python
    fig = plt.figure(figsize=(6, 4), dpi=100)

    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True)
    ```

    The main options are:

    | Parameter | Default | What it controls |
    | --- | --- | --- |
    | `figsize` | `(6.4, 4.8)`* | Figure width and height in inches. |
    | `dpi` | `100`* | Resolution in dots per inch. |
    | `nrows`, `ncols` | `1` | Number of rows and columns in the grid. |
    | `sharex`, `sharey` | `False` | Whether plots share their x or y axis, including its limits and scale. |
    | `squeeze` | `True` | Whether to remove dimensions of length one from the returned array of `Axes`. A single plot returns one `Axes` object. |
    | `layout` | `None` | Use `"constrained"` or `"tight"` to adjust spacing around plots and labels. |

    *The figure size and resolution come from Matplotlib’s settings, so a style or configuration change can override these defaults.*

    `figsize` uses inches, so we multiply by `dpi` to get the figure’s pixel dimensions. For example, `figsize=(6, 4), dpi=100` gives a 600 × 400 pixel figure. When saving, a different resolution or a tight crop can change the output dimensions.

    With several plots, `plt.subplots()` returns an array of `Axes` objects. For a two-by-two grid, we access them by row and column:

    ```python
    fig, axes = plt.subplots(2, 2, layout="constrained")
    axes[0, 0].plot(x, y)
    fig
    ```

    The similarly named [`plt.subplot()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html) selects or creates one plot within a grid on the current figure:

    ```python
    ax = plt.subplot(2, 2, 1)
    ```

    Here, `1` selects the top-left plot. Positions start at **1** and run across each row. `subplots()` creates the whole grid at once; `subplot()` works with one position at a time.
    """)
    return


@app.cell
def _(plt):
    for spec in [(), (1, 2), (2, 2)]:
        _f, _a = plt.subplots(*spec)
        print(
            f"plt.subplots{spec if spec else '()'}".ljust(24),
            "->",
            type(_a).__name__,
            getattr(_a, "shape", ""),
        )
        plt.close(_f)

    _f, _a = plt.subplots(squeeze=False)
    print(
        "plt.subplots(squeeze=False)".ljust(24),
        "->",
        type(_a).__name__,
        _a.shape,
    )
    plt.close(_f)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Losing track of the current Axes

    Pyplot keeps track of the current `Axes`, which is usually the one we created or selected most recently. If a helper function creates or selects another plot, subsequent pyplot calls will use that instead.

    In the cell below, we call a helper that creates another plot, then use `plt.title()` to add a title to our original plot. The title appears on the helper’s plot instead. No error is raised because the call is valid; it just changes the wrong `Axes`.
    """)
    return


@app.cell
def _(np, plt, x):
    def draw_a_helper_plot():
        """Innocent-looking helper. Creates its own figure and its own current axes."""
        plt.figure(figsize=(3, 1.5))
        plt.plot(x, np.cos(x))
        plt.title("the helper's plot")

    plt.figure(figsize=(5, 2))
    plt.plot(x, np.sin(x))

    draw_a_helper_plot()  # this quietly becomes the current figure

    plt.title("I meant this for the sine plot")  # lands on the helper's figure
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two figures came out, and the title went on the second one. In a notebook you spot it immediately because you can see both. In a script that saves the figure to a file, or in a loop over a dataset, you get a directory full of mislabelled plots and no error anywhere.

    The object-oriented version cannot go wrong in this way, because there is no "current" anything, each call names the axes it acts on.
    """)
    return


@app.cell
def _(np, plt, x):
    def draw_on(target_ax):
        """Takes the axes to draw on. No hidden state to disturb."""
        target_ax.plot(x, np.cos(x))
        target_ax.set_title("the helper's plot", fontsize=9)

    fig_safe, (ax_main, ax_helper) = plt.subplots(
        1, 2, figsize=(7, 2), layout="constrained"
    )

    ax_main.plot(x, np.sin(x))
    draw_on(ax_helper)
    ax_main.set_title("This is on the correct graph", fontsize=9)

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Try to **use pyplot for a throwaway look at something, and the object-oriented style the moment a figure has more than one axes or is produced by a function.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Labelling

    We can set titles, axis labels and limits through pyplot or directly on an `Axes` object:

    | Pyplot | Object interface |
    | --- | --- |
    | `plt.title(label)` | `ax.set_title(label)` |
    | `plt.xlabel(label)` | `ax.set_xlabel(label)` |
    | `plt.ylabel(label)` | `ax.set_ylabel(label)` |
    | `plt.xlim(left, right)` | `ax.set_xlim(left, right)` |
    | `plt.suptitle(label)` | `fig.suptitle(label)` |

    The pyplot calls act on the current plot or figure. Using `ax` or `fig` makes the target explicit. `suptitle()` adds a title to the whole figure, which is useful when it contains several plots.

    For several `Axes` properties, we can set them together using `ax.set()`:

    ```python
    ax.set(
        title="Training loss",
        xlabel="Epoch",
        ylabel="Cross-entropy loss (nats)",
        xlim=(0, 100),
    )
    ```

    Labels should describe the quantity and include units where applicable. For example, `"Training time (s)"` tells us more than `"Time"`. Use units that match the calculation: cross-entropy calculated using natural logarithms is measured in nats. Clear labels help the reader understand a plot without referring back to the code.
    """)
    return


@app.cell
def _(np, plt):
    epochs = np.arange(0, 60)
    loss = (
        2.3 * np.exp(-epochs / 12)
        + 0.08
        + np.random.default_rng(0).normal(0, 0.02, len(epochs))
    )

    fig_labelled, ax_labelled = plt.subplots(figsize=(5.5, 3), layout="constrained")
    ax_labelled.plot(epochs, loss)
    ax_labelled.set(
        title="Training loss",
        xlabel="epoch",
        ylabel="cross-entropy loss (nats)",
        xlim=(0, 60),
        ylim=(0, 2.5),
    )
    plt.show()
    return epochs, loss


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Layout

    ```python
    fig.tight_layout(pad=1.08)
    plt.subplots(..., layout="constrained")
    ```

    Default matplotlib lets axis labels and titles run off the edge of the figure or collide with each other, because it places axes before it knows how big the text will be. Both of the above fix it by measuring and re-placing.

    `tight_layout()` is the older one and is called after everything is drawn. `layout="constrained"` is passed when the figure is created, handles more cases including colourbars and shared legends, and is what I would use in new code.

    Get into the habit of one or the other. A figure whose y-label is cut off is the most common avoidable fault in a submitted report.
    """)
    return


@app.cell
def _(epochs, loss, plt):
    fig_cramped, axs_cramped = plt.subplots(1, 2, figsize=(6, 2.2))
    for _a, _t in zip(axs_cramped, ["no layout management", "labels may collide"]):
        _a.plot(epochs, loss)
        _a.set(title=_t, xlabel="epoch", ylabel="cross-entropy loss (nats)")
    plt.show()

    fig_roomy, axs_roomy = plt.subplots(1, 2, figsize=(6, 2.2), layout="constrained")
    for _a, _t in zip(axs_roomy, ["constrained layout", "everything fits"]):
        _a.plot(epochs, loss)
        _a.set(title=_t, xlabel="epoch", ylabel="cross-entropy loss (nats)")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [plt.show](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html) and marimo

    `plt.show()` means "display what has accumulated". In a script with a window-based backend it opens a window and blocks; in a notebook it renders the figure into the output and clears the current figure so the next cell starts fresh.

    In marimo there are two ways to get a figure to appear:

    - end the cell with `plt.show()`, which is what this repository does and what I have used here
    - end the cell with the figure object itself, since marimo displays a cell's last expression

    The second is the natural fit for the object oriented style of build `fig`, end the cell with `fig`. Either is fine; be consistent within a notebook.

    One thing to know either way: **figures are not closed automatically**, and matplotlib warns once you have more than 20 open. In a notebook that rarely matters, but in a loop that saves a figure per epoch it is a genuine memory leak. `plt.close(fig)` after saving fixes it.
    """)
    return


@app.cell
def _(plt):
    _start = len(plt.get_fignums())

    _leaked = [plt.figure() for _ in range(5)]  # opened and never closed
    print(f"opened 5 without closing: {len(plt.get_fignums()) - _start} still open")

    for _f in _leaked:
        plt.close(_f)
    print(f"after closing them:       {len(plt.get_fignums()) - _start} still open")
    print()
    print(
        "in a loop that saves one figure per epoch, the first number is your memory leak"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Write a function `plot_curve(ax, x, y, label)` that draws onto axes you pass it. Use it to build a 1x3 figure. Now write the pyplot equivalent and see which one you would rather extend to 2x3.
    2. `plt.subplots(3, 1, sharex=True)` — what does `sharex` change, and what does it do to the tick labels?
    3. Create a figure whose y-label is cut off. Fix it three ways: `tight_layout`, `layout="constrained"`, and by changing `figsize`.
    4. Take the state-machine trap above and add a third helper. Predict where the title lands before running it.
    5. `plt.subplots(2, 3)` then `axes[1, 2]` versus `axes.ravel()[5]` — confirm they are the same axes. Which reads better to you?
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

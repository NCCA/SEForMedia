#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Introduction to Matplotlib

    Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy, in this notebook we will learn how to use Matplotlib to create different types of plots.

    We will start by generating some data and then we will create different types of plots using Matplotlib.
    """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y)
    return np, plt, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Note we get some text looking like this [<matplotlib.lines.Line2D at 0x144699290>], this is because we are not using the show() function to display the plot but it still shows."""
    )
    return


@app.cell
def _(plt, x, y):
    ## change the size of the plot
    plt.figure(figsize=(2, 2))
    plt.plot(x, y)
    plt.show()
    return


@app.cell
def _(plt, x, y):
    ## changing the plot color and style
    plt.plot(x, y, color="red", linestyle="dashed")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Plot types

    There are many different plot types that can be created using Matplotlib, some of the most common ones are:
    1. Line plot
    2. Scatter plot
    3. Bar plot
    4. Histogram

    For a complete list of plot types that can be created using Matplotlib, you can refer to the official documentation [here](https://matplotlib.org/stable/gallery/index.html).

    Depending on the type of data you have, you can choose the appropriate plot type to visualize the data.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Subplots

    Subplots are used to create multiple plots in the same figure, this can be useful when you want to compare different plots side by side.

    The following code snippet shows how to create subplots using Matplotlib not the use of the OO interface to create subplots.
    """
    )
    return


@app.cell
def _(np, plt):
    x_1 = np.linspace(0, 10, 100)
    y_1 = np.random.rand(100)
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(x_1, y_1)
    axs[0, 1].scatter(x_1, y_1)
    axs[1, 0].bar(x_1, y_1)
    axs[1, 1].hist(y_1)
    plt.show()
    return x_1, y_1


@app.cell
def _(plt, x_1, y_1):
    plt.plot(x_1, y_1)
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.title("Title")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Images

    Matplotlib can also be used to display images, the following code snippet shows how to display an image using Matplotlib. In this case I am loading a series of images from the images folder and displaying in a grid.
    """
    )
    return


@app.cell
def _(plt):
    from PIL import Image

    _f, _ax = plt.subplots(3, 3, figsize=(7, 7), constrained_layout=True)
    [axi.set_axis_off() for axi in _ax.ravel()]
    _index = 0
    for row in range(0, 3):
        for col in range(0, 3):
            try:
                img = Image.open(f"images/test.{_index:04}.png")
                _index = _index + 1
                _ax[col, row].imshow(img)
                _ax[col, row].set_title(f"images/test.{_index:04}.png")
            except FileNotFoundError:
                pass
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""In this example I am generating a series of random colour values and plotting them as an image."""
    )
    return


@app.cell
def _(np, plt):
    def rand_colour():
        return (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        )

    colours = [rand_colour() for i in range(0, 20)]

    _f, _ax = plt.subplots(1, len(colours), figsize=(12, 12))
    _index = 0
    for c in colours:
        _ax[_index].set_axis_off()
        _ax[_index].imshow([[(c[0], c[1], c[2])]])
        _index = _index + 1
    plt.show()
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

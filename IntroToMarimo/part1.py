#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # An introduction to marimo

    I use [marimo](https://marimo.io) for interactive examples in lectures and
    labs. Each notebook is a Python file which we can edit in the browser.
    Run this one from the repository root using:

    ```bash
    uv run marimo edit IntroToMarimo/part1.py
    ```

    Try changing the examples as we go. The [documentation](https://docs.marimo.io),
    [source code](https://github.com/marimo-team/marimo) and
    [PyPI package](https://pypi.org/project/marimo/) have more details.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why use marimo?

    One problem with Jupyter notebooks is running cells out of order. We can
    change a value, forget to run a later cell and end up looking at an old
    result. Deleting a cell can also leave its variables in memory until we
    restart the kernel.

    marimo tracks which cells use each variable. When we change a value it
    re-runs the cells which depend on it, much like a spreadsheet. This is
    useful when teaching as we can change an example and see the result.
    The notebook is also stored as Python, which makes changes easier to
    read in git.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reactivity

    Move the slider below and watch the area change. The next cell uses
    `radius.value`, so marimo re-runs it when we move the slider. We don't
    need to write a callback for this example.
    """)
    return


@app.cell
def _(mo):
    radius = mo.ui.slider(start=1, stop=20, value=5, label="radius")
    radius
    return (radius,)


@app.cell
def _(mo, radius):
    import math

    area = math.pi * radius.value**2
    mo.md(f"A circle of radius {radius.value} has an area of {area:.2f}.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    The area cell depends on `radius`, which is defined in the cell above.
    The slider is a Python object and its `.value` gives us the selected number.

    We must define a shared variable in only one cell. This lets marimo track
    where it comes from and which cells need to run when it changes.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## A notebook is a Python file

    Open `IntroToMarimo/part1.py` in a text editor. Each cell is a function
    with an `@app.cell` decorator. The function arguments show the variables
    it uses from other cells, whilst the return values make variables
    available to the rest of the notebook.

    I find this easier to review in git than an `.ipynb` file, which stores
    JSON and can include cell outputs. We can also run the notebook as a script:

    ```bash
    uv run python IntroToMarimo/part1.py
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Running the notebook as an app

    We can run the same file with the code hidden, leaving the text, widgets
    and outputs visible:

    ```bash
    uv run marimo run IntroToMarimo/part1.py
    ```

    I can use this to share an interactive example with you. The widgets
    still work, but the page does not provide the notebook editor.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Try it yourself

    Change the slider range or use its value in another calculation. Watch
    which cells run when you move it.

    Next we will look at widgets in [part2_ui.py](part2_ui.py). You can also
    run the built-in tutorial with `uv run marimo tutorial intro`, or try the
    [online playground](https://marimo.app) in your browser.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

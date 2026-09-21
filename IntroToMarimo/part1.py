#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # An introduction to marimo

    This is a marimo notebook. It is a plain Python file — the one you are
    reading right now — that opens as an interactive, reactive notebook in the
    browser. I put it together as a quick intro for lectures and labs, so the
    best way to read it is to run it and change things.

    To run it yourself:

    ```bash
    uvx marimo edit intro_to_marimo.py
    ```

    [marimo.io](https://marimo.io) has the full docs; the source lives on
    [GitHub](https://github.com/marimo-team/marimo) and the package is on
    [PyPI](https://pypi.org/project/marimo/).
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Why not just use Jupyter?

    Jupyter is fine, and most of you already know it. The problem I keep
    hitting when teaching with it is hidden state: you run cells out of order,
    delete the cell that defined a variable, and the variable is still sitting
    in memory. The notebook looks like it works, you send it to someone else,
    and it falls over. The `.ipynb` file is also JSON with the outputs baked
    in, which makes it miserable to diff or review in git.

    marimo takes a different line. It works out the dependencies between your
    cells and keeps them consistent for you. Change a value in one cell and
    every cell that depends on it re-runs automatically — a bit like a
    spreadsheet. There is no run-order to get wrong because there is no hidden
    state to get wrong.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Reactivity, with a real example

    Below is a slider. Drag it. The cell underneath reads its value and
    re-runs on its own — I have not wired up any callback or "run" button. In
    marimo a cell that uses a variable automatically depends on the cell that
    defines it.
    """
    )
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
    mo.md(f"A circle of radius **{radius.value}** has an area of **{area:.2f}**.")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Notice what did *not* happen: I never told the area cell when to update.
    marimo saw that it reads `radius`, so it re-runs it whenever `radius`
    changes. The UI element and the Python value are the same thing.

    This is also why marimo stops you redefining the same variable in two
    different cells — if it let you, it could not know which one to trust. It
    feels strict at first and then you realise it is the thing quietly saving
    you from the class of bug that eats an afternoon.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## It is just a Python file

    Have a look at `intro_to_marimo.py` in a text editor. Each cell is an
    ordinary Python function decorated with `@app.cell`, and the arguments to
    that function are the variables the cell needs from elsewhere. That is how
    marimo tracks the dependencies — it reads them straight off the function
    signature.

    Because it is real Python and not JSON:

    - it diffs and reviews cleanly in git, so it works in a normal PR;
    - you can `import` it like any other module;
    - you can run it as a script with `python intro_to_marimo.py`;
    - your editor, linter and type checker all understand it.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Running it as an app

    The same file can be served as a read-only web app, with the code hidden
    and only the markdown, widgets and outputs on show:

    ```bash
    uvx marimo run intro_to_marimo.py
    ```

    So one file is both the teaching material I edit and the interactive demo
    I hand out. No separate export step, and no dashboard framework to learn.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Where to go next

    Pick something from the [docs](https://docs.marimo.io) and change it, or
    just start editing the cells above and watch what re-runs. A few things
    worth trying early:

    - the other UI elements — `mo.ui.dropdown`, `mo.ui.text`, `mo.ui.table`;
    - plotting, which reacts to widgets the same way the area cell did;
    - `uvx marimo tutorial intro`, the official built-in tour.

    The interactive [online playground](https://marimo.app) runs entirely in
    the browser if you want a look before installing anything.
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

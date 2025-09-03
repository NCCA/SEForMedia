#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full", app_title="Packages")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Packages

    In python we can package code into re-usable modules. This is know as a package. At a high level, Python packages are directories of modules that follow specific naming and structural conventions, allowing you to bundle multiple modules (individual .py files) under one namespace.

    ## Basic Structure

    At it's simplest level a package is a directory with a `__init__.py` file. This file can be empty, but it is required for Python to recognize the directory as a package.  This demo is going to explain how the package called Utils (which is a directory in this repository) is structured, we will then look at how we can use this pagage in our code / jupyter notebooks.

    ```
    Utils/
    ├── __init__.py
    ├── functions.py
    └── TorchUtils.py
    ```

    ```__init__.py```  is essential in making a directory a package. It can be empty or contain initialization code, and it allows Utils to be imported as a package.

    At present the package contains two modules, ```functions.py``` and ```TorchUtils.py```. These modules contain functions that we can use in our code.

    ## Python Path

    When we import a package, Python searches for the package in a list of directories. This list is stored in the `sys.path` variable. We can add a directory to this list by appending it to the `sys.path` variable. This is useful if we want to import a package that is not in the same directory as our code.

    As you can see if we try to import the Utils package without appending the path to the sys.path variable we get an error.
    """
    )
    return


@app.cell
def _():
    def _():
        """note we need to wrap in function else we get Utils re-definition error later"""
        try:
            import Utils  # noqa: F401
        except ModuleNotFoundError:
            print("can't find Utils")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To add the package to the path we can use the following code:

    ```python
    import sys
    sys.path.append('path_to_package')
    ```

    In our case we need to do this for the Utils package which is a directory below the current directory. We can do this by using the following code:
    """
    )
    return


@app.cell
def _():
    import sys

    sys.path.append("../")
    import Utils.TorchUtils as torchUtils

    print(torchUtils.get_device())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## ```__init__```

    It is possible to change the behavior of the package when it is imported by adding code to the `__init__.py` file. This file is executed when the package is imported. In our case we are going to add the following code to the `__init__.py` file:

    ```python
    # import functions into the package
    from .TorchUtils import get_device
    from .functions import in_lab, download

    __all__ = ["get_device", "in_lab", "download"]
    ```

    No we can import the package and use the functions in the modules.
    """
    )
    return


@app.cell
def _():
    import Utils

    print(Utils.get_device())
    print(Utils.in_lab())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    There are various other functions in the Utils module, we will be using them in the labs. If you wish to generate your own Package for the assignment hopefully this example demonstrates how you can do this.

    For more information on packages see the [Python documentation](https://docs.python.org/3/tutorial/modules.html#packages)
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

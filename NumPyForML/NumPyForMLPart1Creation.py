#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NumPy for Machine Learning, Part 1: creating arrays

    This is the first of five notebooks looking at the NumPy functions we actually use in the machine learning parts of this unit. It follows on from `Lecture5/IntroductionToNumpy.py` and the [Lecture 5 slides](https://nccastaff.bournemouth.ac.uk/jmacey/SEForMedia/lectures/Lecture5/), but where that notebook introduces NumPy in general, these four concentrate on the handful of functions that turn up over and over again once we get to PyTorch.

    I picked the functions by scanning every demo in this repository and counting what was actually called, so the ordering here reflects what you will meet rather than what a reference manual would list first.

    The five parts are:

    1. Creating arrays (this notebook)
    2. Shapes, views and broadcasting
    3. Maths and reductions
    4. Random numbers and comparing floats
    5. Putting it together — training one neuron

    The first four are a function at a time. The fifth uses the lot of them to train something and draw the result, which is the point of the exercise.

    Full documentation is at [numpy.org](https://numpy.org/doc/stable/index.html). I have linked each function to its own page as we go, and it is worth getting into the habit of opening those — the parameter lists are longer than anything I show here.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    print("numpy version", np.__version__)
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why use Arrays?

    A Python list can hold anything: a list of ints, a list of strings, a list of lists of different lengths. That flexibility whilst powerful is not computationally efficient. Every element is a separate Python object somewhere in memory, and every arithmetic operation goes through the interpreter.

    A NumPy array is the opposite, every element is the same type, the whole thing is one contiguous block of memory, and operations on it run in compiled code with no Python loop. That is the entire reason machine learning is built on arrays rather than lists, and a tensor in PyTorch is this same idea with gradient tracking bolted on.

    A Numpy array has two core attributes used to describe it :-

    - `shape` — a tuple giving the length along each dimension
    - `dtype` — the type of every element

    Most of the errors you will encounter when using NumPy and PyTorch are one of those two being the wrong size.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [np.array](https://numpy.org/doc/stable/reference/generated/numpy.array.html)

    Builds an array from something that already holds the data such as a Python list or a list of lists. This is one of the most used NumPy functions and understanding this is very important.

    ```python
    np.array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `object` | required | the list, tuple or other array to copy from |
    | `dtype` | `None` | element type; `None` means infer it from the data |
    | `copy` | `True` | copy the data; `False` avoids the copy where it can |
    | `ndmin` | `0` | pad the shape with leading 1s up to this many dimensions |

    The parameter to watch is `dtype`. If you leave NumPy to infer the type, a list of Python ints gives you `int64` and a list with one float in it gives you `float64` for every element. Unfortunatly GPU's / Nerual networks both work better with `float32`, so you will often see an explicit dtype or a cast straight afterwards to ensure compatability.
    """)
    return


@app.cell
def _(np):
    # the simplest case, a list of ints
    python_list = [2, 4, 8, 16, 32]
    arr_from_list = np.array(python_list)
    print(arr_from_list)
    print("shape", arr_from_list.shape, "dtype", arr_from_list.dtype)
    return (arr_from_list,)


@app.cell
def _(arr_from_list, np):
    # a nested list becomes a 2D array, as long as the rows are equal length
    arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
    print(arr_2d)
    print("shape", arr_2d.shape, "ndim", arr_2d.ndim)

    # and asking for a dtype explicitly
    arr_as_float = np.array(arr_from_list, dtype=np.float32)
    print(arr_as_float, arr_as_float.dtype)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note what happens if one element is a different type, NumPy promotes the whole array rather than keeping a mixture, because it cannot store a [hetrogenous](https://en.wikipedia.org/wiki/Homogeneity_and_heterogeneity) data set.
    """)
    return


@app.cell
def _(np):
    print(np.array([1, 2, 3]).dtype)  # int64
    print(np.array([1, 2, 3.0]).dtype)  # one float promotes all of them
    print(np.array([1, 2, "3"]).dtype)  # and a string promotes it to text
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### In machine learning

    In machine learning we need to represent our data as numbers. Pixels, labels, features and coordinates can all be stored in arrays, and `np.array` is one way to create them. Each array has a `shape`, which describes how the data is organised, and a `dtype`, which describes the type of each element. We will keep checking these when preparing data for a model: do we have the right dimensions, do we need to convert integers to floating point, and do we need to add a batch dimension?

    When we get to PyTorch, `torch.tensor` serves a similar purpose, but adds support for the way we train models. NumPy was designed around arrays in CPU memory and operations executed on the CPU; its own arrays still work this way. GPU computation needs support for device memory and operations implemented for the GPU, which NumPy leaves to other libraries. PyTorch provides this through its tensors, which can live on the CPU or a supported GPU. It also provides automatic differentiation, allowing us to calculate the gradients needed during training. We therefore need to consider a tensor’s `device` as well as its `shape` and `dtype`. The word *tensor* describes the multidimensional data structure; GPU support and gradient tracking are features of PyTorch’s implementation.

    When we get to PyTorch, `torch.tensor` is the same function with the same job.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [np.arange](https://numpy.org/doc/stable/reference/generated/numpy.arange.html)

    Makes a sequence of evenly spaced values, the same idea as Python's `range` but returning an array, unlike the Python range function `arange` can use float data too.

    ```python
    np.arange([start,] stop, [step,] dtype=None)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `start` | `0` | first value, included |
    | `stop` | required | stop before this value, **not** included |
    | `step` | `1` | gap between values |
    | `dtype` | `None` | inferred from the arguments |

    The arguments are positional in an awkward way: with one argument it is the stop, with two they are start and stop. This is the same as the python `range` function.
    """)
    return


@app.cell
def _(np):
    print(np.arange(10))  # 0 to 9
    print(np.arange(5, 10))  # 5 to 9
    print(np.arange(0, 1, 0.25))  # floats work
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Issues can occur with floats. `arange` works out how many values to produce by dividing, and floating point division does not always land where you expect, so the length can come out one longer or shorter than you counted on. This is the same float behaviour we saw in [Lecture 2](https://nccastaff.bournemouth.ac.uk/jmacey/SEForMedia/lectures/Lecture2/) and it is the reason `np.linspace` exists.
    """)
    return


@app.cell
def _(np):
    # we asked to stop *before* 1.3, in steps of 0.1, so we expect 1.0 1.1 1.2
    wobbly = np.arange(1, 1.3, 0.1)
    print(wobbly)
    print("length", len(wobbly), "- we expected 3")
    print("last value", repr(wobbly[-1]), "- which is past the stop we asked for")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [np.linspace](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html)

    Does the same job as `arange` from the opposite direction, you say how many values you want and NumPy works out the step. Typically I use this to build an x axis to plot something over, or generate a set range of data steps.

    ```python
    np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `start` | required | first value |
    | `stop` | required | last value, included by default |
    | `num` | `50` | how many values to produce |
    | `endpoint` | `True` | include `stop`; set `False` to exclude it like `arange` |
    | `retstep` | `False` | also return the calculated step |

    Because the count is fixed rather than derived, you always get exactly `num` values. If you are working in floats, use this one.
    """)
    return


@app.cell
def _(np):
    print(np.linspace(0, 1, 5))  # exactly 5 values, endpoint included
    print(np.linspace(0, 1, 5, endpoint=False))  # 5 values, endpoint dropped

    linspace_values, linspace_step = np.linspace(0, 1, 5, retstep=True)
    print("step was", linspace_step)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### In machine learning

    Every training curve you plot starts with either a `linspace` or `arange` over the epoch numbers, the loss on the other axis. The decision boundary plots in the classification demos use `linspace` to build a lattice over the feature space, which we will come back to in Part 2 when we meet `np.meshgrid`.

    It is also how you sweep a hyperparameter. If you want ten learning rates between 0.001 and 0.1, `np.linspace` gives you ten evenly spaced ones and `np.logspace` gives you ten spaced evenly in orders of magnitude, which for learning rates is usually the one you want.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Drag the slider and watch what stays fixed. Both endpoints are always there and the points are always evenly spaced — the only thing changing is how many of them there are, which is exactly the difference between `linspace` and `arange`.
    """)
    return


@app.cell
def _(mo):
    sample_count = mo.ui.slider(
        3, 50, value=5, label="how many samples", show_value=True
    )
    sample_count
    return (sample_count,)


@app.cell
def _(np, plt, sample_count):
    _positions = np.linspace(0, 1, sample_count.value)

    _fig, _ax = plt.subplots(figsize=(7, 1.6))
    _ax.scatter(_positions, np.zeros_like(_positions), s=60)
    _ax.set(xlim=(-0.05, 1.05), yticks=[], xlabel="input value")
    _ax.set_title(f"np.linspace(0, 1, {sample_count.value})", loc="left")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [np.zeros](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html), [np.ones](https://numpy.org/doc/stable/reference/generated/numpy.ones.html) and [np.full](https://numpy.org/doc/stable/reference/generated/numpy.full.html)

    Allocate an array of a known shape up front, filled with a constant.

    ```python
    np.zeros(shape, dtype=float, order='C')
    np.ones(shape, dtype=float, order='C')
    np.full(shape, fill_value, dtype=None, order='C')
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `shape` | required | an int, or a tuple of ints for more dimensions |
    | `fill_value` | required for `full` | the constant to fill with |
    | `dtype` | `float` for zeros/ones, inferred for `full` | element type |

    Note the defaults differ. `np.zeros(5)` gives you `float64`; `np.full(5, 255)` looks at the 255 and gives you `int64`. If the dtype matters you need to explicitly declare it.
    """)
    return


@app.cell
def _(np):
    print(np.zeros(5), np.zeros(5).dtype)
    print(np.zeros((2, 3)))  # note the tuple for 2D
    print(np.full(5, 255), np.full(5, 255).dtype)
    print(np.full((2, 3), 0.5, dtype=np.float32))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### np.zeros_like

    [`np.zeros_like`](https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html) takes an existing array and gives you a new one with the same shape and dtype, filled with zeros. There is a `ones_like` and a `full_like` too.

    ```python
    np.zeros_like(a, dtype=None, order='K', subok=True, shape=None)
    ```

    This matters more than it looks. When we write backpropagation by hand in `Neuron/nn_from_scratch.py` we need a gradient buffer for every weight array, and it has to match that array exactly. Writing `np.zeros_like(w)` keeps the two in step; writing `np.zeros((2, 3))` means going back and editing it the moment the layer size changes.
    """)
    return


@app.cell
def _(np):
    weights = np.array([[0.1, -0.4, 0.7], [0.2, 0.5, -0.3]], dtype=np.float32)
    grad = np.zeros_like(weights)
    print("weights", weights.shape, weights.dtype)
    print("grad   ", grad.shape, grad.dtype)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### In machine learning

    Pre-allocating is the habit that separates array code from list code. If you find yourself writing `results = []` and appending inside a loop over your data, stop and ask whether you can allocate the whole thing first and fill it by index, this will _always_ be faster and, more usefully, it forces you to work out the shape before you start rather than discovering it went wrong at the end.

    Filled arrays are also how you make test data. An image of a constant colour, a batch of identical inputs, a target vector of all ones etc.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [astype](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html)

    Everything so far has set the dtype when the array was made. `astype` is how you change it afterwards, and it is a method on the array rather than a function in `np`.

    ```python
    a.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `dtype` | required | the type to convert to |
    | `casting` | `'unsafe'` | how fussy to be about losing information; `'safe'` raises instead |
    | `copy` | `True` | always return a new array, even when the dtype already matches |

    The default of `casting='unsafe'` is worth reading twice. NumPy will throw away whatever does not fit and not tell you, and the two ways it does that are both worth seeing once.
    """)
    return


@app.cell
def _(np):
    # float to int truncates towards zero. it does not round.
    print(
        np.array([1.9, 2.5, -1.9]).astype(np.int32),
        " - note 2.5 became 2, and -1.9 became -1",
    )

    # and an out of range value wraps rather than clamping
    print(np.array([300, -20]).astype(np.uint8), " - 300 wrapped round to 44")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### In machine learning

    This is the line between how an image is *stored* and how a network *wants* it. A photograph is 8-bit integers, 0 - 255, because that is three times smaller on disk than floats and no screen can show the difference. A network wants small floats centred near zero, because that is the range its weights and gradients are scaled for.

    So every image pipeline in this repository has a version of the cell below in it. The thing to notice is that it is two separate jobs: `astype` changes the type, and the division changes the range. Do them the wrong way round — divide first, cast second — and the cast truncates every fraction you just made, so the whole image collapses to nought and one. The shapes are right, nothing raises, and the loss does not move.
    """)
    return


@app.cell
def _(np):
    stored_pixels = np.array([[0, 64, 128, 255]], dtype=np.uint8)
    normalised_pixels = stored_pixels.astype(np.float32) / 255.0

    print("as stored ", stored_pixels, stored_pixels.dtype)
    print("as fed in ", normalised_pixels, normalised_pixels.dtype)
    print()
    print(
        "the wrong way round:",
        (stored_pixels / 255.0).astype(np.uint8),
        " - truncated back to nothing",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [np.fromfile](https://numpy.org/doc/stable/reference/generated/numpy.fromfile.html)

    Reads raw binary straight into an array. No parsing, no header handling, it treats the file as a flat sequence of values of the dtype you name.

    ```python
    np.fromfile(file, dtype=float, count=-1, sep='', offset=0)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `file` | required | an open file object or a path |
    | `dtype` | `float` | how to interpret the bytes — you almost always need to set this |
    | `count` | `-1` | how many items to read; `-1` means the rest of the file |
    | `offset` | `0` | bytes to skip first |

    The `dtype` default of `float` is rarely what you want and gives you garbage silently rather than an error, so treat it as a required argument.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We use this in `MNIST/ReadDigitsTraining.ipynb` to read the MNIST IDX files, which are a short header followed by a solid block of unsigned bytes:

    ```python
    images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    ```

    The file itself has no idea it contains images — it is 47,040,000 bytes in a row. The `dtype` says how wide each value is and the `reshape` says how to carve them into pictures. Below I have written a tiny file in the same shape so you can see the round trip without downloading anything.
    """)
    return


@app.cell
def _(np):
    import tempfile
    from pathlib import Path

    # write three 4x4 "images" of uint8 as raw bytes
    fake_images = np.arange(3 * 4 * 4, dtype=np.uint8)
    tmp_path = Path(tempfile.gettempdir()) / "fake_mnist.raw"
    fake_images.tofile(tmp_path)
    print("wrote", tmp_path.stat().st_size, "bytes")

    # read it back and carve it into images
    loaded_images = np.fromfile(tmp_path, dtype=np.uint8).reshape(3, 4, 4)
    print("loaded shape", loaded_images.shape)
    print(loaded_images[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Try changing the dtype on the read to `np.uint16` and see what happens. You will not get an error — you will get half as many values, each made of two bytes glued together, and an unhelpful reshape failure. That is worth seeing once, because it is exactly the failure mode you get with a real dataset when you misread the format.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Make a 1D array of the numbers 10 to 100 inclusive, in steps of 10, two different ways — once with `arange` and once with `linspace`. Which one did you have to think harder about?
    2. Make a 3x3 array of `float32` zeros, then a second array with the same shape and dtype without repeating the numbers.
    3. Write down how many values you expect from `np.arange(0.1, 0.4, 0.1)`, then run it. Now get the sequence you meant, using `linspace`.
    4. Create a 28x28 array filled with the value 128 as `uint8`, which is what a mid-grey MNIST digit would look like. Now make the `float32` version scaled to the range 0 to 1 that a network would actually want.
    5. Take `np.array([0.9, 1.1, 255.6])` and cast it to `uint8`. Work out what you expect first, then check. Two of the three will surprise you.

    Part 2 picks up from here with reshaping, adding axes and broadcasting.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

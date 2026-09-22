#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NumPy for Machine Learning, Part 2: shapes, views and broadcasting

    Looked at how we can create arrays, this notebook is about changing their shape and about what NumPy does when you combine two arrays whose shapes do not match.

    Reshaping is you telling NumPy what shape you want, broadcasting is NumPy deciding what shape it needs. Almost every error message you will see this term is one of the two going wrong, and the fix is nearly always to print the shapes.

    Another thing to concider when doing a reshape or broadcast of two arrays is do they share memory or not? Get that wrong and you will change something you did not mean to change, a long way from where the bugs finally appear.

    Full documentation on [array manipulation](https://numpy.org/doc/stable/reference/routines.array-manipulation.html) and [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) is on the NumPy site.
    """)
    return


@app.cell
def _():
    import numpy as np

    print("numpy", np.__version__)
    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The data does not move

    "Data not moving" is the core concept we need to understand in this notebook.

    An array is a flat block of memory plus a note saying how to read it. The note is the shape. When you reshape an array, NumPy does not rearrange a single byte, it writes a new note. That is why reshaping is effectively free however large the array, and it is why a reshape can only ever produce a shape with the same total number of elements.

    Run the cell below and look at the two arrays. They share memory, so changing one changes the other.
    """)
    return


@app.cell
def _(np):
    flat = np.arange(12)
    grid = flat.reshape(3, 4)

    print("flat", flat.shape)
    print(grid)

    grid[0, 0] = 99
    print("changing the reshaped array changed the original:", flat[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [Indexing and slicing](https://numpy.org/doc/stable/user/basics.indexing.html)

    Before the shape functions, the thing you will type most often. Indexing an array looks like indexing a list and then keeps going.

    ```python
    a[2]            # one row
    a[2, 3]         # one element - note the comma, not a[2][3]
    a[:, 1]         # every row, column 1
    a[:2]           # the first two rows
    a[::2]          # every other row
    a[a > 0]        # every element where the condition holds
    ```

    `a[2][3]` works and is slower, because it builds the whole of row 2 as an intermediate array and then indexes that. Use the comma.

    Two things to remember are, a slice keeps the axis it slices and a plain integer removes it. `a[:, 1]` on a `(3, 4)` array gives you `(3,)`, not `(3, 1)`. And a slice is a view onto the original, following the same rule as reshape, so writing to a slice writes to the array it came from.
    """)
    return


@app.cell
def _(np):
    samples = np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0], [3.0, 30.0, 300.0]])

    print("one element     ", samples[1, 2])
    print("row 1           ", samples[1], samples[1].shape)
    print(
        "column 1        ",
        samples[:, 1],
        samples[:, 1].shape,
        "  the axis is gone",
    )
    print("column 1 kept   ", samples[:, 1:2].shape, "  a slice keeps it")
    print("first two rows  ", samples[:2].shape)
    return (samples,)


@app.cell
def _(samples):
    # a slice is a view, so this writes back through to samples
    first_two = samples[:2]
    first_two[0, 0] = -1
    print("samples[0, 0] is now", samples[0, 0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Boolean masks

    Index with an array of booleans and you get back every element where the mask is `True`. This is the one form of indexing that does **not** give you a view because the elements it picks are scattered through memory so, in this case, you always get a copy.

    The mask usually comes from a comparison, and comparisons are elementwise, so they produce an array rather than a single `True` or `False`. That is also why `if some_array > 0:` raises a `ValueError` rather than doing anything useful.
    """)
    return


@app.cell
def _(np):
    scores = np.array([-1.2, 0.4, 2.7, -0.3, 1.1])

    print("the mask itself  ", scores > 0)
    print("the values       ", scores[scores > 0])
    print("how many         ", (scores > 0).sum(), " - True counts as 1")
    print(
        "what fraction    ",
        (scores > 0).mean(),
        " - which is how accuracy is computed",
    )

    try:
        if scores > 0:
            pass
    except ValueError as e:
        print()
        print("ValueError:", e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### In machine learning

    `(predictions == labels).mean()` is the accuracy of a whole batch in one line, and it is a mask reduction, the comparison makes the boolean array and the mean counts the trues. `Utils/functions.py` writes the PyTorch version as `.sum().item() / N`, which is the same sum over the same count with the division spelled out.

    Masks are also how you pull a class out of a dataset to look at it  `images[labels == 3]` gives you every three in MNIST — and how you ignore padding in a batch of sequences that are not all the same length.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html)

    Reinterprets the same data under a new shape, this is used a lot in machine learning.

    ```python
    a.reshape(shape, order='C')          # the method, what you will mostly write
    np.reshape(a, shape, order='C')      # the function, identical result
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `shape` | required | an int or tuple; the product must equal the number of elements |
    | `order` | `'C'` | `'C'` fills the last axis fastest (row by row), `'F'` the first |

    The one piece of syntax worth memorising is `-1`. You may put it in exactly one position and NumPy works that dimension out from the total.
    """)
    return


@app.cell
def _(np):
    batch = np.arange(24)

    print(batch.reshape(4, 6).shape)
    print(batch.reshape(2, 3, 4).shape)
    print(batch.reshape(4, -1).shape)  # NumPy works out the 6
    print(batch.reshape(-1, 4).shape)  # or the 6 the other way round
    return (batch,)


@app.cell
def _(batch):
    # a mismatched total is an error, and the message tells you both numbers
    try:
        batch.reshape(5, 5)
    except ValueError as e:
        print("ValueError:", e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### In machine learning

    The `-1` form is how you flatten a batch of images without hard-coding the image size. A batch of 64 MNIST digits arrives as `(64, 1, 28, 28)`; `images.reshape(64, -1)` gives you `(64, 784)`, which is what a fully connected layer takes. Write `reshape(64, 784)` instead and the code breaks the day you change dataset.

    PyTorch has the same method with the same name, plus `view` which does the same thing but refuses if the data is not laid out contiguously. If you are unsure which to use, `reshape` is the safe one.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [ravel](https://numpy.org/doc/stable/reference/generated/numpy.ravel.html) and [flatten](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html)

    Both collapse an array to one dimension. They differ in one respect that matters.

    ```python
    np.ravel(a, order='C')      # a view where it can manage one
    a.flatten(order='C')        # always a copy
    ```

    `ravel` hands back a view onto the same memory when the layout allows it, so it is free but writing to the result writes to the original. `flatten` always copies, so it costs memory but is safe to modify. If you are only reading, use `ravel`.
    """)
    return


@app.cell
def _(np):
    square = np.arange(6).reshape(2, 3)

    ravelled = square.ravel()
    flattened = square.flatten()

    ravelled[0] = 100  # writes through to square
    flattened[1] = 200  # writes only to its own copy

    print("the ravel  ", ravelled)
    print("the flatten", flattened)
    print("the original")
    print(square)
    print("100 reached the original, 200 did not")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### [copy](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.copy.html), views and assignment

    Basic slices and transposes create views, new array objects that share the original data. Reshaping also produces a view where possible, but may need to copy the data if the memory layout requires it. When we need a separate array with its own data, we can explicitly request one using `a.copy()`. For numerical arrays, we can then change the copy without affecting the original. See NumPy’s guide to [copies and views](https://numpy.org/doc/stable/user/basics.copies.html).

    Assignment behaves differently. Writing `b = a` gives the existing array another name. It creates neither a copy nor a view, so `b is a` returns `True`. Changing an element through `b` therefore changes the array we also access through `a`. This is ordinary Python behaviour, just as it is with lists.

    We need to keep these three cases separate: assignment gives us another name for the same array, a view gives us another array object sharing the data, and a copy gives us separate data.
    """)
    return


@app.cell
def _(np):
    original = np.array([10.0, 20.0, 30.0])

    alias = original  # another name for the same array
    duplicate = original.copy()  # a genuinely separate array

    alias[0] = -1
    duplicate[1] = -2

    print("original  ", original)
    print("alias is original?    ", alias is original)
    print("duplicate is original?", duplicate is original)
    print()
    print("-1 from the alias reached the original, -2 from the copy did not")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If you ever need to know which you have got, `arr.base` tells you — it is `None` for an array that owns its data and points at the parent for a view.
    """)
    return


@app.cell
def _(np):
    owner = np.arange(6)
    view = owner.reshape(2, 3)

    print("owner.base", owner.base)
    print("view.base is owner?", view.base is owner)
    print("owner.copy().base", owner.copy().base)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adding an axis: [np.newaxis](https://numpy.org/doc/stable/reference/constants.html#numpy.newaxis) and [np.expand_dims](https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html)

    Sometimes the data is right and only the number of dimensions is wrong. A hundred numbers might need to be a hundred samples of one feature each — `(100,)` becoming `(100, 1)`.

    ```python
    a[:, np.newaxis]              # np.newaxis is just None, both work
    np.expand_dims(a, axis)       # the function form
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `axis` | required for `expand_dims` | where to insert the new length-1 axis |

    Where you put the axis is the whole decision. `a[:, np.newaxis]` gives a column, `a[np.newaxis, :]` gives a row, and they are not interchangeable.
    """)
    return


@app.cell
def _(np):
    line = np.arange(4)
    print("original       ", line.shape)
    print("column         ", line[:, np.newaxis].shape)
    print("row            ", line[np.newaxis, :].shape)
    print("expand_dims 1  ", np.expand_dims(line, 1).shape)

    print()
    print(line[:, np.newaxis])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### In machine learning

    This is `unsqueeze` in PyTorch, and it comes up constantly because every PyTorch layer assumes a leading batch dimension — even when the batch is one item. Pass a single `(3, 224, 224)` image to a model and it will complain; pass `(1, 3, 224, 224)` and it is happy.

    `LinearModel/LinearModel.py` does the NumPy-shaped version of this at line 13, turning a flat range into a column of single-feature samples before it goes anywhere near a layer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [squeeze](https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html)

    `np.squeeze` removes axes of length one from an array. It changes the shape without copying the underlying data.

    ```python
    np.squeeze(a, axis=None)
    ```

    | Parameter | Default | What it does |
    | --------- | ------- | ------------ |
    | `axis` | `None` | An axis or tuple of axes to remove. `None` removes all axes of length one. |

    We need to be careful with the default when working with batches. A single greyscale image might have shape `(1, 1, 28, 28)`, representing batch, channel, height and width. Both the batch and channel axes have length one, so calling `np.squeeze(a)` removes both:

    ```python
    a = np.zeros((1, 1, 28, 28))

    print(np.squeeze(a).shape)          # (28, 28)
    print(np.squeeze(a, axis=1).shape)  # (1, 28, 28)
    ```

    If we only want to remove the channel axis, `axis=1` preserves the batch dimension. Losing that dimension can cause errors later, or allow broadcasting to produce an unintended result.

    Specify the axis when its meaning matters. This also checks our assumption: `np.squeeze(a, axis=1)` raises a `ValueError` if the channel axis does not have length one.
    """)
    return


@app.cell
def _(np):
    one_image = np.zeros((1, 1, 28, 28))  # batch, channels, height, width

    print("as it arrives          ", one_image.shape)
    print(
        "squeeze(axis=1)        ",
        np.squeeze(one_image, axis=1).shape,
        " - channel gone, batch kept",
    )
    print(
        "squeeze()              ",
        np.squeeze(one_image).shape,
        " - the batch went too",
    )

    try:
        np.squeeze(one_image, axis=2)  # height is 28, not 1
    except ValueError as e:
        print()
        print("ValueError:", e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### In machine learning

    A model with one output per sample gives you `(n, 1)` and you want `(n,)` to compare against the labels. This is the mismatch from the broadcasting section at the end of this notebook, and `squeeze` is one of the two ways to fix it, `ravel` being the other. PyTorch spells this one `squeeze` as well, with the same `dim` argument.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [transpose](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html)

    Swaps the axes. For a 2D array `a.T` is all you need.

    ```python
    a.T                          # shorthand, reverses the axis order
    np.transpose(a, axes=None)   # axes=None reverses; otherwise give a permutation
    ```

    Like reshape, this moves no data, it just rewrites the note about how to read it, which is why a transpose is instant even on a large array.

    Reshape and transpose are not interchangeable, and this is worth being clear about because both will take you from `(2, 3)` to `(3, 2)` without complaint. Reshape keeps the reading order and re-cuts it; transpose keeps each value's coordinates and swaps the axes. Only one of them is what you meant, and the shapes will not tell you which.
    """)
    return


@app.cell
def _(np):
    before = np.array([[1, 2, 3], [4, 5, 6]])

    print("reshaped to (3, 2)")
    print(before.reshape(3, 2))
    print()
    print("transposed to (3, 2)")
    print(before.T)
    print()
    print(
        "same shape, different arrays:",
        np.array_equal(before.reshape(3, 2), before.T),
    )
    return


@app.cell
def _(np):
    mat = np.arange(6).reshape(2, 3)
    print(mat, mat.shape)
    print()
    print(mat.T, mat.T.shape)

    # for more than two dimensions, name the permutation you want
    volume = np.zeros((8, 3, 32, 32))  # batch, channels, height, width
    print()
    print("channels last:", np.transpose(volume, (0, 2, 3, 1)).shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### In machine learning

    Two places you will meet it. In backpropagation, the gradient flowing backwards through a matrix multiply needs the transpose of the weights,  `nn_from_scratch.py` is full of lines like `da1 = dz2 @ W2.T`. And when you display a tensor, PyTorch and torchvision keep images as `(channels, height, width)` while Matplotlib wants `(height, width, channels)`, so a transpose sits between the model and the picture.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [np.meshgrid](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html) and [np.c_](https://numpy.org/doc/stable/reference/generated/numpy.c_.html)

    We use these together to prepare coordinates for decision boundary plots. `np.meshgrid` builds a grid of sample positions, and `np.c_` combines the flattened coordinates into an array we can pass to a classifier.

    ```python
    np.meshgrid(*xi, copy=True, sparse=False, indexing="xy")
    np.c_[a, b]
    ```

    `np.c_` uses square brackets because it is an indexing object. For two 1D arrays of equal length, it places their values in two columns.

    | Parameter | Default | What it does |
    | --------- | ------- | ------------ |
    | `*xi` | Required | One 1D coordinate array per dimension. |
    | `copy` | `True` | Copy the coordinate data; `False` returns views where possible. |
    | `sparse` | `False` | Use compact coordinate arrays intended for broadcasting when `True`. |
    | `indexing` | `"xy"` | Cartesian indexing; `"ij"` uses matrix indexing. |

    For a 2D grid, we supply the x and y values separately. With the defaults, `meshgrid` returns two arrays of shape `(len(y), len(x))`: one holding each position’s x coordinate, the other its y coordinate.

    ```python
    x = np.array([0, 1, 2])
    y = np.array([10, 20])

    xx, yy = np.meshgrid(x, y)

    print(xx)
    # [[0 1 2]
    #  [0 1 2]]

    print(yy)
    # [[10 10 10]
    #  [20 20 20]]
    ```

    We then flatten both grids with `ravel()` and combine them:

    ```python
    points = np.c_[xx.ravel(), yy.ravel()]

    print(points)
    # [[ 0 10]
    #  [ 1 10]
    #  [ 2 10]
    #  [ 0 20]
    #  [ 1 20]
    #  [ 2 20]]
    ```

    The result has shape `(6, 2)`, with one `(x, y)` pair per row. For a classifier with two input features, we can predict a class at each point, then reshape those predictions to `xx.shape` for plotting.
    """)
    return


@app.cell
def _(np):
    # a tiny 3x4 lattice so the numbers are readable
    xs = np.linspace(0, 1, 4)
    ys = np.linspace(10, 12, 3)
    xx, yy = np.meshgrid(xs, ys)

    print("xx")
    print(xx)
    print("yy")
    print(yy)
    print("both are", xx.shape)
    return xx, yy


@app.cell
def _(np, xx, yy):
    # ravel each to a flat list of coordinates, then stack them into columns
    points = np.c_[xx.ravel(), yy.ravel()]
    print(points.shape)
    print(points[:5])
    return (points,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In machine learning, we can draw a decision boundary by asking a model to classify points across the two features shown in a plot. We then colour the regions according to its predictions.

    The process has five steps:

    1. Use `meshgrid` to create a grid covering both feature ranges.
    2. Flatten the coordinate arrays with `ravel()` and combine them using `np.c_`. This gives an array of shape `(n_points, 2)`, with one point per row.
    3. Pass these points to the model as a batch.
    4. Reshape the predictions back to the original grid shape.
    5. Use a contour plot to display the predicted regions and their boundaries.

    The reshape connects the model’s output to the plot. Assuming the model returns one prediction per point, those predictions arrive in the same order as the input rows. Reshaping them to the grid shape places each prediction at its corresponding position.

    The example below uses a simple function in place of a model so we can follow the shapes through each step.
    """)
    return


@app.cell
def _(np, points, xx):
    # pretend this is a trained model: anything above the diagonal is class 1
    def fake_model(p):
        return (p[:, 1] - 10.0 > p[:, 0]).astype(np.float32)

    predictions = fake_model(points)
    print("predictions come back flat:", predictions.shape)

    zz = predictions.reshape(xx.shape)
    print("folded back onto the lattice:", zz.shape)
    print(zz)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)

    Broadcasting allows NumPy to perform element-wise operations on arrays with different shapes. We use it whenever we add a scalar to an array, apply a separate scale to each feature, or subtract a mean from every sample.

    NumPy compares the shapes from right to left. Two dimensions are compatible when they are equal or one is `1`. Missing dimensions on the left are treated as `1`. If any pair is incompatible, NumPy raises a `ValueError`.

    ```text
    (3, 4) +    (4,) → (3, 4)   one value per column, reused across rows
    (3, 4) +  (3, 1) → (3, 4)   one value per row, reused across columns
    (3, 4) +    (3,) → error    the rightmost dimensions, 4 and 3, conflict
    ```

    For example, we can add the same four values to every row of a `(3, 4)` array:
    """)
    return


@app.cell
def _(np):
    a = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ]
    )
    offsets = np.array([10, 20, 30, 40])

    print("adding a (4,) row vector:")
    print(a + offsets)
    return (a,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To add a different value to each row, we need a column with shape `(3, 1)`. `np.newaxis` adds the length-one dimension, and NumPy then reuses each row offset across its four columns.
    """)
    return


@app.cell
def _(a, np):
    row_offsets = np.array([100, 200, 300])

    print("adding a (3, 1) column vector:")
    print(a + row_offsets[:, np.newaxis])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A flat `(3,)` array is the case that fails, because the rightmost dimensions, 4 and 3, conflict:
    """)
    return


@app.cell
def _(a, np):
    try:
        _ = a + np.array([1, 2, 3])
    except ValueError as e:
        print("ValueError:", e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Broadcasting does not build repeated copies of the inputs. However, the operation normally allocates its result, which can be much larger than either input.

    In machine learning, a common mistake is subtracting targets shaped `(n,)` from predictions shaped `(n, 1)`:
    """)
    return


@app.cell
def _(np):
    probabilities = np.array([[0.2], [0.8], [0.6]])
    targets = np.array([0.0, 1.0, 1.0])

    errors = probabilities - targets
    print("errors.shape:", errors.shape)
    return probabilities, targets


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This compares every prediction with every target. NumPy accepts the shapes, but we wanted one error per sample. We can make that intention explicit by giving the targets the same shape:
    """)
    return


@app.cell
def _(np, probabilities, targets):
    matched_errors = probabilities - targets[:, np.newaxis]
    print("matched_errors.shape:", matched_errors.shape)
    print(matched_errors)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Check the expected output shape as well as the input shapes. An operation running without an error does not guarantee that it performed the calculation we intended.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### In machine learning

    Broadcasting is doing the work in `z = X @ w + b`. `X @ w` is one value per sample, and `b` is a single number, so broadcasting adds the bias to every sample without you writing a loop or making a copy of it. Every bias term in every layer works this way.

    It is also how per-channel normalisation works: subtract a mean of shape `(3,)` from an image of shape `(H, W, 3)` and each channel gets its own value.

    The reason to learn the rule rather than rely on it is the case where broadcasting succeeds and you did not want it to. Compare a column of predictions `(n, 1)` against a flat row of labels `(n,)` and NumPy will happily broadcast them into an `(n, n)` array comparing everything with everything. No error, no warning, and an accuracy figure that means nothing. This is the exact bug that `view_as` guards against in the PyTorch demos.
    """)
    return


@app.cell
def _(np):
    preds = np.array([[1], [0], [1], [1]])  # (4, 1)
    labels = np.array([1, 0, 1, 0])  # (4,)

    broadcast_compare = preds == labels
    print("shape of the comparison:", broadcast_compare.shape, "- we wanted (4,)")
    print(
        "accuracy comes out as",
        broadcast_compare.mean(),
        "which is meaningless",
    )

    matched_compare = preds.ravel() == labels
    print()
    print(
        "with the shapes matched:",
        matched_compare.shape,
        "accuracy",
        matched_compare.mean(),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Take `np.arange(24)` and make it into every 2D shape you can. How many are there?
    2. A batch of 8 colour images at 64x64 has shape `(8, 3, 64, 64)`. Flatten it for a fully connected layer without writing the number 12288 anywhere.
    3. You have predictions of shape `(50, 1)` and labels of shape `(50,)`. Write the comparison two ways — one that broadcasts by accident and one that does not — and print both shapes.
    4. Build a 5x5 lattice over the range -1 to 1 in both directions, turn it into a list of points, and count how many of them fall inside the unit circle.
    5. Start from `np.zeros((1, 1, 4, 4))` and get to `(1, 4, 4)` two different ways. Then get to `(4, 4)` and say which axis you lost.
    6. Make a `(3, 4)` array, take a slice of it, change one value in the slice and print the original. Now do the same with a boolean mask instead of a slice. Why do the two behave differently?

    Part 3 moves on to what you do with the arrays once the shapes are right.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PyTorch for Machine Learning, Part 1: tensors and devices

    These five notebooks are the PyTorch half of the NumPy set in `NumPyForML/`. Same approach: I scanned every demo in this repository and counted what was actually called, so what is covered and the order it comes in follow what you will meet rather than what a reference manual lists first.

    The five parts are:

    1. Tensors and devices (this notebook)
    2. Autograd and the training loop
    3. Building models with `torch.nn`
    4. Evaluation and inference
    5. Data pipelines and saving models

    If you have worked through the NumPy notebooks, most of this one will feel like revision with different names, which is the point. I have put the NumPy equivalent beside each function where there is one. The two genuinely new ideas are the device and the gradient, and the gradient is Part 2.

    Full documentation is at [pytorch.org/docs](https://pytorch.org/docs/stable/index.html).
    """)
    return


@app.cell
def _():
    import numpy as np
    import torch

    print("torch", torch.__version__)
    print("numpy", np.__version__)
    return np, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What a tensor adds

    A [tensor](https://pytorch.org/docs/stable/tensors.html) is a NumPy array with two things bolted on:

    - it can live on a GPU rather than in main memory.
    - it can record the operations performed on it, so gradients can be computed backwards through them.

    Everything else, the shape, the dtype, the indexing, the broadcasting rules are the same as NumPy. That is not an accident; the API was deliberately made to look familiar.

    The two additions are why we bother. The GPU is what makes training finish quickly, and the gradient recording is what removes the need to derive backpropagation by hand, which we did in `Neuron/nn_from_scratch.py`.

    PyTorch’s gradient tracking records how a calculation depends on its inputs, so we can work out how changing those inputs would affect the result. The system responsible for this is called autograd.
    When training a model, we calculate a loss: a number measuring how far the prediction is from the expected answer. We then need to know how each weight affects that loss. A gradient gives us this information: the rate at which the loss changes as a weight changes. The optimiser uses these gradients to adjust the weights. We will investigate this in more detail later, once we have understood the basics.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [torch.tensor](https://pytorch.org/docs/stable/generated/torch.tensor.html)

    Builds a tensor from data you already have, copying it.

    ```python
    torch.tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `data` | required | list, tuple, NumPy array or scalar |
    | `dtype` | `None` | element type; inferred from the data if not given |
    | `device` | `None` | where to put it; `None` means CPU |
    | `requires_grad` | `False` | track operations on this tensor for gradients |

    One convention to understand early, because everything downstream assumes it is that a 2D tensor of data, **rows are samples and columns are features**. A batch of 32 samples with 4 features each is `(32, 4)` and never `(4, 32)`. Floats go in as the data, integers come back out as class labels, which is also why the dtype of a label tensor is `int64` and not something you should convert.

    `Lecture7/PyTorchAndTensors.py` walks up the ranks with this, starting at `torch.tensor(5)`, and that is the right way in, rank, shape and dtype are the three things you must be able to read off any tensor.
    """)
    return


@app.cell
def _(torch):
    scalar = torch.tensor(5)
    vector = torch.tensor([1, 2, 3])
    matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])

    for name, t in [
        ("scalar", scalar),
        ("vector", vector),
        ("matrix", matrix),
    ]:
        print(f"{name:7} ndim={t.ndim}  shape={tuple(t.shape)}  dtype={t.dtype}")
    return (scalar,)


@app.cell
def _(scalar, torch):
    # .item() pulls a python number out of a single-element tensor
    print(scalar.item(), type(scalar.item()))

    # and the dtype is inferred the same way numpy does it
    print(torch.tensor([1, 2, 3]).dtype)  # int64
    print(torch.tensor([1, 2, 3.0]).dtype)  # one float promotes the lot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## dtypes

    The real difference between NumPy and PyTorch is that **NumPy defaults floats to `float64`. PyTorch defaults to `float32`.**

    PyTorch made that choice because `float32` is what GPUs prefer for speed and the extra precision of double buys a network nothing. The consequence is that data arriving from NumPy or scikit-learn is `float64` and every PyTorch layer wants `float32`, so something has to cast this to the correct GPU type. If you forget, the error surfaces somewhere that has nothing obviously to do with dtypes.
    """)
    return


@app.cell
def _(np, torch):
    from_numpy_default = torch.from_numpy(np.array([1.0, 2.0, 3.0]))
    from_torch_default = torch.tensor([1.0, 2.0, 3.0])

    print("came from numpy:", from_numpy_default.dtype)
    print("made in torch:  ", from_torch_default.dtype)
    return


@app.cell
def _(np, torch):
    # what it looks like when it goes wrong
    layer = torch.nn.Linear(3, 1)  # its weights are float32
    bad_input = torch.from_numpy(np.random.rand(4, 3))  # float64

    try:
        layer(bad_input)
    except RuntimeError as e:
        print("RuntimeError:", str(e).split("\n")[0])

    print()
    print("the fix, either way round:")
    print(layer(bad_input.float()).shape)
    print(layer(torch.from_numpy(np.random.rand(4, 3)).type(torch.float32)).shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the error carefully. The exact wording moves about between PyTorch versions, you will see "got Double and Float", or "expected scalar type Float but found Double"  but either way it is talking about dtypes in the C names rather than the Python ones. **Double is `float64` and Float is `float32`.** This is worth knowing because nothing in your code says "double" anywhere, so the message reads as though it is about something else entirely.

    Three ways to cast, all equivalent:

    ```python
    t.float()              # shorthand for float32
    t.to(torch.float32)    # the general form, also used for devices
    t.type(torch.float32)  # what the older demos here use
    ```

    `Classification/BinaryClassification.ipynb:20` does it at the point of conversion, which is the correct place, cast once on the way in rather than scattering `.float()` through the model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [torch.from_numpy](https://pytorch.org/docs/stable/generated/torch.from_numpy.html) and [.numpy()](https://pytorch.org/docs/stable/generated/torch.Tensor.numpy.html)

    We can convert between NumPy arrays and PyTorch tensors using these two functions:

    ```python
    tensor = torch.from_numpy(arr)  # NumPy array to tensor
    arr = tensor.numpy()           # compatible CPU tensor to NumPy array
    ```

    Both share the underlying memory, so changing a value in one also changes it in the other. This avoids the time and extra memory needed to copy the data, which is particularly useful when working with large arrays. We can use NumPy and PyTorch operations on the same data without keeping two separate copies.

    We do need to remember that they are connected: modifying the array also modifies the tensor, and vice versa. Use `torch.tensor(arr)` when we want a separate copy.

    By default, `.numpy()` requires a CPU tensor that does not require gradients and has a supported data type and layout, with no conjugate or negative bit set. Using `.numpy(force=True)` handles detaching and moving to the CPU, but the result may no longer share memory with the original tensor.
    """)
    return


@app.cell
def _(np, torch):
    shared_array = np.array([1.0, 2.0, 3.0])

    shared_tensor = torch.from_numpy(shared_array)  # shares the memory
    copied_tensor = torch.tensor(shared_array)  # takes a copy

    shared_array[0] = 99.0  # change the array after making both

    print("the array  ", shared_array)
    print("from_numpy ", shared_tensor.numpy(), " <- followed the change")
    print("tensor()   ", copied_tensor.numpy(), " <- did not")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Going the other way has a restriction. `.numpy()` only works on a CPU tensor that is not part of a gradient graph, so in real code you will see the full chain:

    ```python
    zz = test_preds.reshape(xx.shape).detach().cpu().numpy()
    ```

    That is `Classification/BinaryClassification.ipynb:132`. Read it right to left: detach from the graph, move to the CPU, hand the buffer to NumPy. Each step is there because the one after it refuses otherwise. We will come back to `detach` in Part 2 once the graph exists.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Making tensors from nothing

    The same family as NumPy, with the same names.

    ```python
    torch.zeros(*size, dtype=None, device=None, requires_grad=False)
    torch.ones(*size, ...)
    torch.full(size, fill_value, ...)
    torch.empty(*size, ...)        # uninitialised, whatever was in the memory
    torch.arange(start=0, end, step=1, ...)
    torch.linspace(start, end, steps, ...)
    torch.randn(*size, ...)        # standard normal
    torch.rand(*size, ...)         # uniform [0, 1)
    torch.randint(low, high, size, ...)
    ```

    Two differences from NumPy worth flagging. `torch.linspace` requires `steps` — there is no default of 50. And the size arguments are loose: `torch.zeros(2, 3)` and `torch.zeros((2, 3))` both work, where NumPy insists on the tuple.
    """)
    return


@app.cell
def _(torch):
    print(
        "zeros   ",
        torch.zeros(2, 3).shape,
        "and",
        torch.zeros((2, 3)).shape,
        "both fine",
    )
    print("arange  ", torch.arange(0, 10, 2))
    print("linspace", torch.linspace(0, 1, 5))
    print("randn   ", torch.randn(3).round(decimals=3))
    print("full    ", torch.full((2, 2), 0.5))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### [torch.manual_seed](https://pytorch.org/docs/stable/generated/torch.manual_seed.html)

    ```python
    torch.manual_seed(seed)
    ```

    Seeds the global generator so weight initialisation and shuffling repeat. Every demo here that trains anything calls `torch.manual_seed(42)`, and for the reason set out in NumPy Part 4: without it you cannot tell whether a change you made moved the loss or whether you got a luckier initialisation.

    The thing to remember is that **seeding NumPy does not seed PyTorch**. They have separate generators. If your pipeline uses both, and most do, because the data handling is NumPy and the model is PyTorch, you will need to seed both.
    """)
    return


@app.cell
def _(np, torch):
    torch.manual_seed(42)
    first_run = torch.randn(3)

    torch.manual_seed(42)
    second_run = torch.randn(3)

    print("same seed:", torch.allclose(first_run, second_run))

    np.random.seed(0)  # does nothing to the line below
    print("torch still moves on:", torch.randn(1).item() != torch.randn(1).item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Shapes

    All the NumPy shape operations are here under names that are mostly the same.

    | What you want | NumPy | PyTorch |
    | --- | --- | --- |
    | new shape | `a.reshape(2, 3)` | `t.reshape(2, 3)` or `t.view(2, 3)` |
    | flatten | `a.ravel()` | `t.flatten()` |
    | add an axis | `a[:, None]` | `t.unsqueeze(1)` |
    | drop length-1 axes | `a.squeeze()` | `t.squeeze()` |
    | swap axes | `a.transpose(0, 2, 1)` | `t.permute(0, 2, 1)` |

    ```python
    t.reshape(*shape)              # works whatever the memory layout
    t.view(*shape)                 # no-copy only, raises if it cannot
    t.unsqueeze(dim)               # insert a length-1 axis at dim
    t.squeeze(dim=None)            # remove length-1 axes; all of them if dim is None
    t.permute(*dims)               # reorder axes, a full permutation
    ```

    `reshape` and `view` do the same job. `view` refuses if the tensor is not laid out contiguously, after a `permute`, for instance. While `reshape` quietly copies instead. If you are not sure, use `reshape`.
    """)
    return


@app.cell
def _(torch):
    block = torch.arange(24)

    print("reshape       ", tuple(block.reshape(2, 3, 4).shape))
    print("with a -1     ", tuple(block.reshape(2, -1).shape))
    print("flatten       ", tuple(block.reshape(2, 3, 4).flatten().shape))
    return (block,)


@app.cell
def _(block):
    # view fails where reshape succeeds
    awkward = block.reshape(4, 6).t()  # transposing breaks the contiguous layout

    try:
        awkward.view(24)
    except RuntimeError as e:
        print("RuntimeError:", str(e).split("\n")[0])

    print("reshape copes:", tuple(awkward.reshape(24).shape))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why `unsqueeze` comes up so often

    Every PyTorch layer assumes a leading batch dimension, even when the batch is one item. That single design decision is behind most of the shape juggling you will do.

    Pass one 28x28 image to a model and it complains. Give it a batch of one  `(1, 1, 28, 28)`  and it is happy. `squeeze` is the reverse, usually to drop that axis again before displaying the result.
    """)
    return


@app.cell
def _(torch):
    one_image = torch.rand(28, 28)
    print("a single image        ", tuple(one_image.shape))

    batched = one_image.unsqueeze(0).unsqueeze(0)  # add channel, then batch
    print("as a batch of one     ", tuple(batched.shape))
    print("back to something plottable", tuple(batched.squeeze().shape))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [torch.device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) and [.to()](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html)

    This is the part with no NumPy equivalent at all.

    ```python
    torch.device(type)             # "cpu", "cuda", "cuda:0", "mps"
    t.to(device)                   # returns a tensor on that device
    model.to(device)               # moves the model in place
    ```

    A device says where a tensor's memory lives. The rule PyTorch enforces is that **everything in an operation must be on the same device**, and it will raise rather than move anything for you.

    `.to()` is the most-called function in the whole repository. Note the asymmetry, which is a common source of confusion: for a tensor `.to()` returns a moved copy and you must assign it, while for a model it moves the parameters in place and the return value is the same model.
    """)
    return


@app.cell
def _(torch):
    # this is Utils/TorchUtils.py, which every demo here uses
    def get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():  # mac metal backend
            return torch.device("mps")
        else:
            return torch.device("cpu")

    device = get_device()
    print("cuda available:", torch.cuda.is_available())
    print("mps available: ", torch.backends.mps.is_available())
    print("using:         ", device)
    return (device,)


@app.cell
def _(device, torch):
    data_on_device = torch.randn(4, 3).to(device)
    model_on_device = torch.nn.Linear(3, 1).to(device)

    print("tensor is on", data_on_device.device)
    print("model weight is on", next(model_on_device.parameters()).device)
    print("so this works:", model_on_device(data_on_device).shape)

    print()
    print("note the asymmetry:")
    cpu_tensor = torch.randn(2)
    cpu_tensor.to(device)  # thrown away - the original is unchanged
    print(
        "  tensor.to() without assigning:",
        cpu_tensor.device,
        "- still where it was",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the labs this will pick `cuda`, on a Mac it will pick `mps`, and on anything else `cpu`. Because the choice is wrapped in `Utils/TorchUtils.py` rather than written out in each notebook, the same code runs unchanged in all three places.

    If you get a device mismatch error, the message names both devices and the fix is nearly always a missing `.to(device)` on the data rather than on the model. The model gets moved once at construction; the data has to be moved every batch.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try it

    Shape, dtype and memory are the three things you read off a tensor, and they are easier to believe when you can watch them move. Drag the batch size and see the leading axis change while everything after it stays fixed — that is the batch dimension doing its job.

    The byte counts are the argument for `float32` restated as a number: the same batch in `float64` is exactly twice the memory, for precision a network never uses.
    """)
    return


@app.cell
def _(mo):
    batch_slider = mo.ui.slider(1, 64, value=8, label="Batch size", show_value=True)
    batch_slider
    return (batch_slider,)


@app.cell
def _(batch_slider, torch):
    mnist_batch = torch.rand(batch_slider.value, 1, 28, 28)
    flat_batch = mnist_batch.flatten(start_dim=1)

    print("batch of images   ", tuple(mnist_batch.shape))
    print("after flatten(1)  ", tuple(flat_batch.shape), "<- batch axis survives")
    print("elements          ", f"{mnist_batch.numel():,}")
    print()
    print("as float32        ", f"{mnist_batch.numel() * 4 / 1024:8.1f} KiB")
    print("as float64        ", f"{mnist_batch.numel() * 8 / 1024:8.1f} KiB")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Make a tensor from `np.linspace(0, 1, 10)` and check its dtype. Now make it `float32` three different ways.
    2. Take a `(3, 224, 224)` image tensor and get it ready for a model expecting a batch. Now take the model's `(1, 1000)` output and get a plain Python integer for the predicted class.
    3. Create a tensor with `torch.from_numpy`, modify the original array, and confirm the tensor changed. Then do the same with `torch.tensor` and confirm it did not.
    4. Seed with 42, draw two tensors, reseed with 42, draw again. Now do it seeding NumPy instead and explain what happens.
    5. Build a tensor and a `nn.Linear` on different devices (if you have a GPU) and read the error message properly. Which one does it say is where?
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

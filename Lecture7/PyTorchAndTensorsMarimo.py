#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="true")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # PyTorch and Tensors

    We can define tensors with different shapes. In general we have

    | Rank | Math Entity | Python Example |
    |------|-------------|----------------|
    | 0    | Scalar      | `torch.tensor(1)` |
    | 1    | Vector      | `torch.tensor([1, 2, 3])` |
    | 2    | Matrix      | `torch.tensor([[1, 2], [3, 4]])` |
    | 3    | 3-Tensor    | `torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])` |
    | n    | n-Tensor    | `torch.randn(2, 3, 4, 5)` |

    The following code will generate a simple tensor.
    """
    )
    return


@app.cell
def _():
    import torch
    import matplotlib.pyplot as plt
    import numpy as np

    scalar = torch.tensor(5)
    print(scalar)  # tensor(5)
    print(scalar.shape)  # torch.Size([])
    return np, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 1D Tensors

    A 1D tensor is basically a vector, we can produce one like this.
    """
    )
    return


@app.cell
def _(torch):
    vector = torch.tensor([1, 2, 3])
    print(vector)
    print(vector.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2D Tensors

    A 2d tensor is a matrix, we can produce one like this.
    """
    )
    return


@app.cell
def _(torch):
    matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(matrix)
    # tensor([[1, 2, 3],
    #         [4, 5, 6]])
    print(matrix.shape)  # torch.Size([2, 3])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3 Tensor

    This tensor has 3 dimensions, note how when printing the output is truncated.
    """
    )
    return


@app.cell
def _(torch):
    tensor3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(tensor3d)
    print(tensor3d.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## N Tensor

    An N tensor is a tensor with N dimensions. These are typically created for existing data or functions as it is quite complex to generate higher order ones easily.
    """
    )
    return


@app.cell
def _(torch):
    tensor = torch.rand(2, 3, 4, 5)
    print(tensor)
    print(tensor.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Generating Tensors

    There are a number of ways to generate tensors and pytorch comes with a number of functions to help us generate them.

    ### torch.tensor(data)

    This method will generate a tensor from data passed in which could be a standard python container (list,tuple) or numpy array as follows.
    """
    )
    return


@app.cell
def _(np, torch):
    from_tuple = torch.tensor(((1, 2, 3), (4, 5, 6)))
    print(f"{from_tuple} {from_tuple.shape}")
    from_list = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f"{from_list} {from_list.shape}")
    data = np.random.rand(2, 2, 3)
    from_array = torch.tensor(data)
    print(f"{from_array} {from_array.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## from similar values

    As with numpy we can create arrays as zeros, ones of full values
    """
    )
    return


@app.cell
def _(torch):
    tensor_1 = torch.zeros(3, 3)
    print(f"{tensor_1} {tensor_1.shape}")
    tensor_1 = torch.ones(3, 3)
    print(f"{tensor_1} {tensor_1.shape}")
    tensor_1 = torch.full((3, 3), 7)
    print(f"{tensor_1} {tensor_1.shape}")
    tensor_1 = torch.empty(2, 2)
    print(f"{tensor_1} {tensor_1.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## ranges

    As with numpy there are a number of range based functions to generate tensors, most of them are of the format start,stop, step
    """
    )
    return


@app.cell
def _(torch):
    tensor_2 = torch.arange(0, 10, 2)
    print(f"{tensor_2} {tensor_2.shape}")
    tensor_2 = torch.linspace(0, 1, steps=5)
    print(f"{tensor_2} {tensor_2.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Identity and Diagonal

    We can generate identity and diagonal matrices as follows, not there are other methods to set diagonals as well.
    """
    )
    return


@app.cell
def _(torch):
    tensor_3 = torch.eye(4)
    print(f"{tensor_3} {tensor_3.shape}")
    tensor_3 = torch.diag(torch.tensor([1, 2, 3, 4]))
    print(f"{tensor_3} {tensor_3.shape}")
    tensor_3 = torch.zeros(4, 4)
    tensor_3.fill_diagonal_(8)
    print(f"{tensor_3} {tensor_3.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Random Tensors

    We can generate random tensors as follows, note that the random number generation is based on the current random seed. The two core functions are the rand and randn functions which give us random values between 0 and 1 and random values from a normal distribution respectively.

    We also have integer functions as well as random permutations, which can be useful for shuffling data.
    """
    )
    return


@app.cell
def _(torch):
    tensor_4 = torch.rand(3, 2)
    print(f"{tensor_4} {tensor_4.shape}")
    tensor_4 = torch.randn(3, 2)
    print(f"{tensor_4} {tensor_4.shape}")
    tensor_4 = torch.randint(0, 10, (2, 3))
    print(f"{tensor_4} {tensor_4.shape}")
    tensor_4 = torch.randperm(5)
    print(f"{tensor_4} {tensor_4.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We can generate normal distributions based around a mean and standard deviation as follows."""
    )
    return


@app.cell
def _(torch):
    tensor_5 = torch.normal(mean=0, std=1, size=(2, 2))
    print(f"{tensor_5} {tensor_5.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""There are more functions available in the documentation which can be found [here](https://pytorch.org/docs/stable/torch.html#creation-ops)"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Device

    The following function will determine which device we have available. If we have properly installed PyTorch and setup for CUDA it should show ```cuda``` as the device. On mac you may get mps for the metal device (note not all things work with this), else we will get CPU, which is slower but should work fine.
    """
    )
    return


@app.cell
def _(torch):
    def get_device() -> torch.device:
        """
        Returns the appropriate device for the current environment.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():  # mac metal backend
            return torch.device("mps")
        else:
            return torch.device("cpu")

    device = get_device()
    print(device)
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""When we create tensors we can specify the device we want to use, this is useful for training on GPUs."""
    )
    return


@app.cell
def _(device, torch):
    tensor_6 = torch.rand(3, 3, device=device)
    print(f"{tensor_6} {tensor_6.shape} {tensor_6.device}")
    return (tensor_6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""If we want to move a tensor to a different device we can use the to function as follows."""
    )
    return


@app.cell
def _(tensor_6):
    cpu_tensor = tensor_6.to("cpu")
    print(f"{cpu_tensor} {cpu_tensor.shape} {cpu_tensor.device}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    It is our job to manage the device we are using, if we try to use a tensor on a device that is not available we will get an error. Typically we will generate data on the CPU side (i.e. loading / processing it) and then move it to the GPU for training.

    Later on when we have a trained model we can move it back to the CPU for inference.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Saving and Loading Tensors

    We can save and load tensors and groups of tensors to a file using the .save and .load functions. This will save data in a binary format by default.
    """
    )
    return


@app.cell
def _(torch):
    tensor_7 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    torch.save(tensor_7, "tensor.pt")
    new_tensor = torch.load("tensor.pt")
    print(f"{new_tensor} {new_tensor.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""You can also save multiple tensors or other objects by placing them in a dictionary and then saving the dictionary as follows."""
    )
    return


@app.cell
def _(torch):
    # Create multiple tensors
    tensor1 = torch.tensor([1.0, 2.0, 3.0])
    tensor2 = torch.tensor([[4.0, 5.0], [6.0, 7.0]])

    # Save the tensors in a dictionary
    torch.save({"tensor1": tensor1, "tensor2": tensor2}, "tensors.pt")

    loaded_data = torch.load("tensors.pt")

    t1 = loaded_data["tensor1"]
    t2 = loaded_data["tensor2"]

    print(f"{t1} {t1.shape}")
    print(f"{t2} {t2.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Loading Images

    We can load images as tensors using the PIL library and then convert them to tensors. As this is quite a common operation PyTorch provides a function to do this for us in the torchvision library.
    """
    )
    return


@app.cell
def _():
    from PIL import Image
    import torchvision.transforms as transforms

    image = Image.open("banana.png")
    transform = transforms.ToTensor()
    _tensor_image = transform(image)
    print(f"{_tensor_image.shape}")
    return Image, image, transforms


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can now convert this back to an image and display it as follows.""")
    return


@app.cell
def _(image, np, plt):
    _numpy_image = np.array(image)
    plt.imshow(_numpy_image)
    plt.axis("off")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""It is quite common to convert an image into a tensor and then process it (such as normalizing it) before passing it to a model for training or inference."""
    )
    return


@app.cell
def _(Image, plt, transforms):
    image_1 = Image.open("banana.png")
    image_1 = image_1.convert("RGB")
    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.3, 0.3, 0.3]),
        ]
    )
    _tensor_image = normalize(image_1)
    _numpy_image = _tensor_image.permute(1, 2, 0).numpy()
    plt.imshow(_numpy_image)
    plt.axis("off")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We will look at more advanced image processing options in a later lectures / lab."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Other Tensor Data Formats

    We can load tensors from other formats including numpy, pandas and csv files, as well as things like audio and video. These are typically done using the appropriate libraries and then converting the data to tensors such as torch.audio or torch.video

    In a future lecture we will look at how we can load batches of data for training using the DataLoader class.
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

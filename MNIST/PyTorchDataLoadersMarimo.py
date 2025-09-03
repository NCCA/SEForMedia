#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Overview

    In the previous examples we have loaded the data from source and processed it ourselves. This helps us to get a deeper understanding of how the whole process works, and how we would go about training our own models.

    It is however, quite common to download datasets directly using the [`datasets`](https://pytorch.org/vision/stable/datasets.html#) library. This is a library that provides a simple way to download and load datasets for processing, it also includes many common datasets that are used in the research community and can be used to extend existing models or to train new models.

    In this example we will show how this works be using same process we used for the manual MNIST dataset, but this time we will use the `datasets` library to download the data for us.
    """
    )
    return


@app.cell
def _():
    import sys
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.utils.data import DataLoader

    # Visualization tools
    import torchvision
    import torchvision.transforms.v2 as transforms

    sys.path.append("../")
    import Utils

    device = Utils.get_device()
    return Adam, DataLoader, Utils, device, nn, torch, torchvision, transforms


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We will attempt to download the data into the /transfer folder if in the labs, else we will place it locally. In this case we will put it into a folder called mnist_data."""
    )
    return


@app.cell
def _(Utils):
    DATASET_LOCATION = ""
    if Utils.in_lab():
        DATASET_LOCATION = "/transfer/mnist_data/"
    else:
        DATASET_LOCATION = "./mnist_data/"

    print(f"Dataset location: {DATASET_LOCATION}")
    return (DATASET_LOCATION,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""we can now use the dataloaders to download the datasets to the above location."""
    )
    return


@app.cell
def _(DATASET_LOCATION, torchvision):
    train_set = torchvision.datasets.MNIST(DATASET_LOCATION, train=True, download=True)
    valid_set = torchvision.datasets.MNIST(DATASET_LOCATION, train=False, download=True)
    return train_set, valid_set


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""You will notice that the data is downloaded to the location specified and the data is loaded in the same way as before. However the loader returns a class called `torch.utils.data.DataLoader` which is a class that provides an iterator over the dataset. This is useful as it allows us to iterate over the dataset in a for loop, and also provides a way to shuffle the data and load it in batches."""
    )
    return


@app.cell
def _(train_set, valid_set):
    print(type(train_set))
    print(train_set)

    print(valid_set)
    return


@app.cell
def _(train_set):
    x_0, y_0 = train_set[0]
    print(type(x_0), type(y_0), x_0.size, y_0)
    x_0
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""You will notice that the data is stored as a PIL image and an integer. This is unlike the data we loaded in the previous demo which was the raw bytes. Basically the data loader class has done some pre-processing for us, and has loaded the data in a format that is ready to be used by the model. This is a common feature of the `datasets` library, and is one of the reasons why it is so popular."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Transforms

    The data loader class also allows us to apply transformations to the data. This is useful as it allows us to apply pre-processing to the data before it is loaded into the model. This can be useful for normalizing the data, or for augmenting the data to increase the size of the dataset.
    """
    )
    return


@app.cell
def _(torch, train_set, transforms, valid_set):
    trans = transforms.Compose(
        [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
    )
    train_set.transform = trans
    valid_set.transform = trans
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""As before we need to generate a full dataloader to batch our data. However is is now much simpler as we don't need to write our own class to do it as the data is already in the correct format."""
    )
    return


@app.cell
def _(DataLoader, train_set, valid_set):
    batch_size = 32

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    type(valid_loader.dataset[0])
    print(valid_loader.dataset[0][1])
    return train_loader, valid_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Building the Model

    We will use the same model we used in the [previous notebook](ReadDigitsTraining.ipynb) to train the model.
    """
    )
    return


@app.cell
def _(device, nn, torch):
    n_classes = 10
    input_size = 28 * 28
    layers = [
        nn.Flatten(),
        nn.Linear(input_size, 512),  # Input
        nn.ReLU(),  # Activation for input
        nn.Linear(512, 512),  # Hidden
        nn.ReLU(),  # Activation for hidden
        nn.Linear(512, n_classes),  # Output
    ]
    model = nn.Sequential(*layers)
    model.to(device)
    next(model.parameters()).device
    model = torch.compile(model)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Loss and Optimizer

    Next we can create our loss function and optimizer. We will use the same loss function and optimizer as before.

    We can also generate our loss calculation in a similar way.
    """
    )
    return


@app.cell
def _(Adam, model, nn, train_loader, valid_loader):
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())
    train_N = len(train_loader.dataset)
    valid_N = len(valid_loader.dataset)

    def get_batch_accuracy(output, y, N):
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(y.view_as(pred)).sum().item()
        return correct / N

    return get_batch_accuracy, loss_function, optimizer, train_N, valid_N


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Train Function""")
    return


@app.cell
def _(
    device,
    get_batch_accuracy,
    loss_function,
    model,
    optimizer,
    train_N,
    train_loader,
):
    def train():
        loss = 0
        accuracy = 0
        # put model into training mode
        model.train()
        # send data to the device
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            optimizer.zero_grad()
            batch_loss = loss_function(output, y)
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()
            accuracy += get_batch_accuracy(output, y, train_N)
        print("Train - Loss: {:.4f} Accuracy: {:.4f}".format(loss, accuracy))

    return (train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Validate Function""")
    return


@app.cell
def _(
    device,
    get_batch_accuracy,
    loss_function,
    model,
    torch,
    valid_N,
    valid_loader,
):
    def validate():
        loss = 0
        accuracy = 0
        # put model into evaluation mode
        model.eval()
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)

                loss += loss_function(output, y).item()
                accuracy += get_batch_accuracy(output, y, valid_N)
        print("Valid - Loss: {:.4f} Accuracy: {:.4f}".format(loss, accuracy))

    return (validate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Training loop""")
    return


@app.cell
def _(train, validate):
    epochs = 5

    for epoch in range(epochs):
        print("Epoch: {}".format(epoch))
        train()
        validate()
    return


@app.cell
def _(device, model, train_set):
    prediction = model(train_set[0][0].to(device).unsqueeze(0))
    print(prediction.argmax(dim=1, keepdim=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Conclusion

    As you can see the processes are very similar, just the model loading and prep are a little simpler.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

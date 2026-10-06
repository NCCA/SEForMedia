#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Overview

    In the previous examples we have loaded the data from source and processed it ourselves. This helps us to get a deeper understanding of how the whole process works, and how we would go about training our own models.

    It is however, quite common to download datasets directly using the [`datasets`](https://pytorch.org/vision/stable/datasets.html#) library. This is a library that provides a simple way to download and load datasets for processing, it also includes many common datasets that are used in the research community and can be used to extend existing models or to train new models.

    In this example we will show how this works be using same process we used for the manual MNIST dataset, but this time we will use the `datasets` library to download the data for us.
    """)
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
    mo.md(r"""
    We will attempt to download the data into the /transfer folder if in the labs, else we will place it locally. In this case we will put it into a folder called mnist_data.
    """)
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
    mo.md(r"""
    we can now use the dataloaders to download the datasets to the above location.
    """)
    return


@app.cell
def _(DATASET_LOCATION, torchvision):
    train_set = torchvision.datasets.MNIST(DATASET_LOCATION, train=True, download=True)
    valid_set = torchvision.datasets.MNIST(DATASET_LOCATION, train=False, download=True)
    return train_set, valid_set


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You will notice that the data is downloaded to the location specified and the data is loaded in the same way as before. However the loader returns a class called `torch.utils.data.DataLoader` which is a class that provides an iterator over the dataset. This is useful as it allows us to iterate over the dataset in a for loop, and also provides a way to shuffle the data and load it in batches.
    """)
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
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You will notice that the data is stored as a PIL image and an integer. This is unlike the data we loaded in the previous demo which was the raw bytes. Basically the data loader class has done some pre-processing for us, and has loaded the data in a format that is ready to be used by the model. This is a common feature of the `datasets` library, and is one of the reasons why it is so popular.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Transforms

    The data loader class also allows us to apply transformations to the data. This is useful as it allows us to apply pre-processing to the data before it is loaded into the model. This can be useful for normalizing the data, or for augmenting the data to increase the size of the dataset.
    """)
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
    mo.md(r"""
    As before we need to generate a full dataloader to batch our data. However is is now much simpler as we don't need to write our own class to do it as the data is already in the correct format.
    """)
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
    mo.md(r"""
    ## Building the Model

    We will use the same model we used in the [previous notebook](ReadDigitsTraining.ipynb) to train the model.
    """)
    return


@app.cell
def _(nn):
    n_classes = 10
    input_size = 28 * 28

    def build_model() -> nn.Sequential:
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),  # Input
            nn.ReLU(),  # Activation for input
            nn.Linear(512, 512),  # Hidden
            nn.ReLU(),  # Activation for hidden
            nn.Linear(512, n_classes),  # Output
        )

    build_model()
    return (build_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The model is wrapped in a `build_model` function so that each time we press **Train** below we get a fresh, untrained model rather than carrying on from the last run.

    # Loss and Optimizer

    Next we can create our loss function. We will use the same loss function and optimizer as before, but the optimizer needs the parameters of the model it is updating, so it is created inside the training cell along with the model.
    """)
    return


@app.cell
def _(nn):
    loss_function = nn.CrossEntropyLoss()
    return (loss_function,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training

    In the [previous notebook](ReadDigitsTrainingMarimo.py) we wrote our own `train_epoch` and `evaluate` functions. As they are the same for most of the models we are going to build, I have moved them into the `Utils` package (see [Utils/training.py](../Utils/training.py)) and we can use them from there. Each returns the mean loss and the accuracy for one pass over the data.

    Choose the settings and press **Train**. Submitting again starts a fresh model.
    """)
    return


@app.cell
def _(mo):
    training_settings = mo.ui.dictionary(
        {
            "epochs": mo.ui.number(start=1, stop=100, value=5, label="Epochs"),
            "learning_rate": mo.ui.number(
                start=0.0001,
                stop=0.1,
                step=0.0001,
                value=0.001,
                label="Learning rate",
            ),
        }
    ).form(submit_button_label="Train")
    training_settings
    return (training_settings,)


@app.cell
def _(
    Adam,
    Utils,
    build_model,
    device,
    loss_function,
    mo,
    torch,
    train_loader,
    training_settings,
    valid_loader,
):
    mo.stop(training_settings.value is None, mo.md("Press Train above to begin."))

    torch.manual_seed(42)
    model = build_model().to(device)
    _model_compiled = torch.compile(model)
    _optimizer = Adam(model.parameters(), lr=training_settings.value["learning_rate"])
    history = []
    for _epoch in mo.status.progress_bar(
        range(training_settings.value["epochs"]), title="Training"
    ):
        _train_loss, _train_accuracy = Utils.train_epoch(
            _model_compiled, train_loader, loss_function, _optimizer, device
        )
        _valid_loss, _valid_accuracy = Utils.evaluate(
            _model_compiled, valid_loader, loss_function, device
        )
        history.append((_train_loss, _valid_loss, _train_accuracy, _valid_accuracy))
        print(
            f"Epoch {_epoch + 1}: train loss {_train_loss:.3f}, validation loss {_valid_loss:.3f}, validation accuracy {_valid_accuracy:.1%}"
        )
    mo.md(f"Training finished on **{device}**.")
    return history, model


@app.cell
def _(history):
    import matplotlib.pyplot as plt

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3), layout="constrained")
    _epochs = range(1, len(history) + 1)
    for _index, _name in enumerate(["Loss", "Accuracy"]):
        for _offset, _label in enumerate(["Training", "Validation"]):
            _axes[_index].plot(
                _epochs,
                [_row[2 * _index + _offset] for _row in history],
                label=_label,
            )
        _axes[_index].set(xlabel="Epoch", ylabel=_name)
        _axes[_index].legend()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(device, model, torch, train_set):
    model.eval()
    with torch.inference_mode():
        prediction = model(train_set[0][0].to(device).unsqueeze(0))
    print(f"predicted {prediction.argmax(dim=1).item()}, label {train_set[0][1]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Conclusion

    As you can see the processes are very similar, just the model loading and prep are a little simpler.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multi-class Classification

    In the [previous notebook](BinaryClassificationMarimo.py) we built a binary classifier. Here we will choose one of four mutually exclusive classes for each point. This is multi-class classification. Multi-label classification is a different task, where several labels can apply to the same example.

    ## Getting started

    We are going to start by importing our base libraries and setting the random seed for reproducibility.
    """)
    return


@app.cell
def _():
    from sklearn.datasets import make_blobs
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.optim as optim

    return make_blobs, nn, np, optim, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data generation

    We will use the  ```make_blobs``` function from the ```sklearn.datasets``` module to generate our data.
    """)
    return


@app.cell
def _(make_blobs):
    # Make 1000 samples
    n_samples = 1000

    CLASSES = 4
    FEATURES = 2
    SEED = 12345

    # 1. Create multi-class data
    inputs_array, targets_array = make_blobs(
        n_samples=n_samples,
        n_features=FEATURES,
        centers=CLASSES,
        cluster_std=1.2,
        random_state=SEED,
    )
    return CLASSES, FEATURES, inputs_array, targets_array


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As in the [linear regression notebook](../LinearModel/LinearModelMarimo.py), I use `inputs` and `targets` where books and library examples often use `X` and `y`. These names help us follow what each value does whilst reading the code.

    The inputs are the information we give the model. Here each row is one point and its two features are the horizontal and vertical coordinates. The targets are the class labels we want it to predict, one label for each point. A target is not the vertical coordinate; both coordinates are input features.

    In this example the targets are class IDs from 0 to 3. Each point belongs to one class, so each row of inputs has one target.

    We start with NumPy arrays called `inputs_array` and `targets_array`, then convert them to tensors called `inputs` and `targets`. The names describe the same roles in both forms.
    """)
    return


@app.cell
def _(inputs_array, plt, targets_array):
    plt.figure(figsize=(4, 4))

    plt.scatter(
        x=inputs_array[:, 0],
        y=inputs_array[:, 1],
        c=targets_array,
        cmap=plt.cm.plasma,
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Preprocessing

    We will convert the data into tensors and use 80% for training and 20% for validation. The training examples update the model parameters; the validation examples let us check progress and compare settings without updating parameters on those examples.

    This introductory example has no separate final test set. If we use the validation results to choose settings or how long to train, we need another reserved set for a final assessment. See the [glossary](../GLOSSARY.md#training-validation-and-testing).
    """)
    return


@app.cell
def _(inputs_array, targets_array, torch):
    inputs = torch.from_numpy(inputs_array).type(torch.float)
    targets = torch.from_numpy(targets_array).type(torch.float)
    train_split = int(0.8 * len(inputs))
    inputs_train = inputs[:train_split]
    inputs_val = inputs[train_split:]
    targets_train = targets[:train_split]
    targets_val = targets[train_split:]
    print(
        inputs_train.shape,
        inputs_val.shape,
        targets_train.shape,
        targets_val.shape,
    )
    return (
        inputs,
        inputs_train,
        inputs_val,
        targets,
        targets_train,
        targets_val,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    again we will use the gpu if avaiable.
    """)
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

    # Make device agnostic code
    device = get_device()
    print(device)
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Building a Model

    We are going to setup our model in a similar way as before, however this time we have more than one output. To make  it easier to write our forward pass we are going to use the ```torch.nn.Sequential``` class which will call the forward method of each module in the order they are passed to the constructor.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will still inherit from the nn.Module but change the shape of our model to include a hidden layer with a ReLU activation function.
    """)
    return


@app.cell
def _(CLASSES, FEATURES, device, nn):
    # Build model with non-linear activation function

    class Classify(nn.Module):
        def __init__(self, input_features=2, output_features=1, hidden_size=8):
            super().__init__()
            self.linear_layer = nn.Sequential(
                nn.Linear(in_features=input_features, out_features=hidden_size),
                nn.ReLU(),
                nn.Linear(in_features=hidden_size, out_features=hidden_size),
                nn.ReLU(),
                nn.Linear(in_features=hidden_size, out_features=output_features),
            )

        def forward(self, inputs):
            return self.linear_layer(inputs)

    model = Classify(input_features=FEATURES, output_features=CLASSES).to(device)
    print(model.parameters)
    model
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loss Function and Optimizer

    Our model returns one raw score (logit) per class. We pass these logits directly to [`CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), with integer class IDs as targets. It combines log-softmax and negative log likelihood, so we do not apply softmax before calculating the loss.

    For a prediction, `argmax(dim=1)` selects the class with the highest logit. Softmax is useful when we want the class probabilities; it gives the same winning class. A multi-label model would instead make independent binary decisions, commonly using `BCEWithLogitsLoss`.
    """)
    return


@app.cell
def _(model, nn, optim):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.1)
    return loss_fn, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can now look at the basic output of the untrained model.
    """)
    return


@app.cell
def _(device, inputs_train, model, torch):
    logits = model(inputs_train.to(device))
    print(logits.shape)
    probabilities = torch.softmax(logits, dim=1)
    print(logits[:5])
    print(probabilities[:5])
    return (probabilities,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The softmax function is used to convert the output of the model into a probability distribution, this should sum to 1. We can then find the class with the highest probability using the argmax function to determine the predicted class.
    """)
    return


@app.cell
def _(probabilities, torch):
    print(probabilities[0])
    print(torch.argmax(probabilities[0]))
    return


@app.cell
def _(torch):
    def accuracy(targets, preds):
        correct = torch.eq(targets, preds).sum().item()
        acc = (correct / len(preds)) * 100
        return acc

    return (accuracy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training the Model

    We pass `inputs_train_device` through the model and compare its logits with `targets_train_device` to calculate the loss. The targets are the known answers; `preds` are the class labels chosen by the model, which we compare with those answers to calculate accuracy. Moving tensors to a device changes where we calculate, not what the data means.

    We keep the same pairing for validation: `inputs_val_device` contains the points and `targets_val_device` contains their expected classes. Those examples do not update the model parameters.

    We will now train the model using the training data. We will use the same training loop as in the previous lab but using the new data sets, we will now also copy the data to the device to help speed up the training process.
    """)
    return


@app.cell
def _(
    accuracy,
    device,
    inputs_train,
    inputs_val,
    loss_fn,
    model,
    optimizer,
    targets_train,
    targets_val,
    torch,
):
    torch.manual_seed(1234)

    epochs = 100

    # copy to device
    inputs_train_device = inputs_train.to(device)

    # cross entropy expects integer class IDs on every device
    targets_train_device = targets_train.type(torch.LongTensor)
    targets_train_device = targets_train_device.to(device)

    inputs_val_device = inputs_val.to(device)
    targets_val_device = targets_val.type(torch.LongTensor)
    targets_val_device = targets_val_device.to(device)

    for epoch in range(epochs):
        ### Training
        model.train()

        # 1. Forward pass
        train_logits = model(inputs_train_device)
        # turn logits -> pred probs -> pred labls
        preds = torch.softmax(train_logits, dim=1).argmax(dim=1)

        # 2. Calculate loss/accuracy
        loss = loss_fn(train_logits, targets_train_device)
        acc = accuracy(targets=targets_train_device, preds=preds)

        # reset the optimizer to zero
        optimizer.zero_grad()
        # calculate the gradients
        loss.backward()
        # update the weights
        optimizer.step()

        # Validation
        model.eval()
        with torch.inference_mode():
            # 1. Forward pass
            val_logits = model(inputs_val_device)
            val_pred = torch.softmax(val_logits, dim=1).argmax(dim=1)
            # 2. calculate loss/accuracy
            val_loss = loss_fn(val_logits, targets_val_device)
            val_acc = accuracy(targets=targets_val_device, preds=val_pred)

        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Validation loss: {val_loss:.5f}, Validation acc: {val_acc:.2f}%"
            )
    return (inputs_val_device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is much better accuracy around 80% which is much better than the previous model. Let's make some predictions and see how well the model is doing.
    """)
    return


@app.cell
def _(inputs_val_device, model, targets_val, torch):
    model.eval()
    with torch.inference_mode():
        predictions = model(inputs_val_device).argmax(dim=1)
    (predictions[:10], targets_val[:10])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can use the same code we used in the previous lab to plot the decision boundary.
    """)
    return


@app.cell
def _(device, inputs, model, np, plt, targets, torch):
    x_min, x_max = (inputs[:, 0].min() - 0.1, inputs[:, 0].max() + 0.1)
    y_min, y_max = (inputs[:, 1].min() - 0.1, inputs[:, 1].max() + 0.1)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    plt.figure(figsize=(4, 4))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.from_numpy(grid).type(torch.float).to(device)
    model.eval()
    with torch.inference_mode():
        _grid_logits = model(grid_tensor).squeeze()
        grid_preds = torch.softmax(_grid_logits, dim=1).argmax(dim=1)
        zz = grid_preds.reshape(xx.shape).detach().cpu().numpy()
        plt.contourf(xx, yy, zz, cmap=plt.cm.plasma, alpha=0.2)
        plt.scatter(
            x=inputs[:, 0],
            y=inputs[:, 1],
            c=targets,
            s=10,
            cmap=plt.cm.coolwarm,
        )
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title("Decision Boundary")
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This works well. Try re-running the model with the ReLU activation function removed and see how this affects the decision boundary.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

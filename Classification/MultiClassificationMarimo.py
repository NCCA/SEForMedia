#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Multi Label Classification

    In the [previous notebook](BinaryClassification.ipynb) we build a model to do binary classification. In this notebook we are going to build a model to do multi label classification.

    ## Getting started

    We are going to start by importing our base libraries and setting the random seed for reproducibility.
    """
    )
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
    mo.md(
        r"""
    ## Data generation

    We will use the  ```make_blobs``` function from the ```sklearn.datasets``` module to generate our data.
    """
    )
    return


@app.cell
def _(make_blobs):
    # Make 1000 samples
    n_samples = 1000

    CLASSES = 4
    FEATURES = 2
    SEED = 12345

    # 1. Create multi-class data
    x, y = make_blobs(
        n_samples=n_samples,
        n_features=FEATURES,
        centers=CLASSES,
        cluster_std=1.2,
        random_state=SEED,
    )
    return CLASSES, FEATURES, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's see the dataset""")
    return


@app.cell
def _(plt, x, y):
    plt.figure(figsize=(4, 4))

    plt.scatter(x=x[:, 0], y=x[:, 1], c=y, cmap=plt.cm.plasma)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Data Preprocessing

    We now need to convert our data into tensors and generate our test / train split. We will use the typical 80:20 split for this.
    """
    )
    return


@app.cell
def _(torch, x, y):
    X = torch.from_numpy(x).type(torch.float)
    y_1 = torch.from_numpy(y).type(torch.float)
    train_split = int(0.8 * len(X))
    X_train = X[:train_split]
    X_test = X[train_split:]
    y_train = y_1[:train_split]
    y_test = y_1[train_split:]
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X, X_test, X_train, y_1, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Building a Model

    We are going to setup our model in a similar way as before, however this time we have more than one output. To make  it easier to write our forward pass we are going to use the ```torch.nn.Sequential``` class which will call the forward method of each module in the order they are passed to the constructor.
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

    # Make device agnostic code
    device = get_device()
    print(device)
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We will still inherit from the nn.Module but change the shape of our model to include a hidden layer with a ReLU activation function."""
    )
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

        def forward(self, x):
            return self.linear_layer(x)

    model = Classify(input_features=FEATURES, output_features=CLASSES).to(device)
    print(model.parameters)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Loss Function and Optimizer

    This time as we are dealing with more than one output we need to use the CrossEntropyLoss function. This is a combination of the softmax activation function and the negative log likelihood loss function.
    """
    )
    return


@app.cell
def _(model, nn, optim):
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.SGD(params=model.parameters(), lr=0.1)
    return loss_fn, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can now look at the basic output of the untrained model.""")
    return


@app.cell
def _(X_train, device, model, torch):
    logits = model(X_train.to(device))
    print(logits.shape)
    probabilities = torch.softmax(logits, dim=1)
    print(logits[:5])
    print(probabilities[:5])
    return (probabilities,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""The softmax function is used to convert the output of the model into a probability distribution, this should sum to 1. We can then find the class with the highest probability using the argmax function to determine the predicted class."""
    )
    return


@app.cell
def _(probabilities, torch):
    print(probabilities[0])
    print(torch.argmax(probabilities[0]))
    return


@app.cell
def _(torch):
    def accuracy(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100
        return acc

    return (accuracy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Training the Model

    We will now train the model using the training data. We will use the same training loop as in the previous lab but using the new data sets, we will now also copy the data to the device to help speed up the training process.
    """
    )
    return


@app.cell
def _(
    X_test,
    X_train,
    accuracy,
    device,
    loss_fn,
    model,
    optimizer,
    torch,
    y_test,
    y_train,
):
    torch.manual_seed(1234)

    epochs = 100

    # copy to device
    X_train_device = X_train.to(device)

    # Note we need to convert here as the cuda model doesn't work on floats
    y_train_device = y_train.type(torch.LongTensor)
    y_train_device = y_train_device.to(device)

    X_test_device = X_test.to(device)
    y_test_device = y_test.type(torch.LongTensor)
    y_test_device = y_test_device.to(device)

    for epoch in range(epochs):
        ### Training
        model.train()

        # 1. Forward pass
        y_logits = model(X_train_device)
        # turn logits -> pred probs -> pred labls
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

        # 2. Calculate loss/accuracy
        loss = loss_fn(y_logits, y_train_device)
        acc = accuracy(y_true=y_train_device, y_pred=y_pred)

        # reset the optimizer to zero
        optimizer.zero_grad()
        # calculate the gradients
        loss.backward()
        # update the weights
        optimizer.step()

        # Testing
        model.eval()
        with torch.inference_mode():
            # 1. Forward pass
            test_logits = model(X_test_device)
            test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
            # 2. calculate loss/accuracy
            test_loss = loss_fn(test_logits, y_test_device)
            test_acc = accuracy(y_true=y_test_device, y_pred=test_pred)

        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%"
            )
    return (X_test_device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""This is much better accuracy around 80% which is much better than the previous model. Let's make some predictions and see how well the model is doing."""
    )
    return


@app.cell
def _(X_test_device, model, torch, y_1):
    model.eval()
    with torch.inference_mode():
        y_preds = torch.round(torch.sigmoid(model(X_test_device))).squeeze()
    (y_preds[:10], y_1[:10])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We can use the same code we used in the previous lab to plot the decision boundary."""
    )
    return


@app.cell
def _(X, device, model, np, plt, torch, y_1):
    x_min, x_max = (X[:, 0].min() - 0.1, X[:, 0].max() + 0.1)
    y_min, y_max = (X[:, 1].min() - 0.1, X[:, 1].max() + 0.1)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    plt.figure(figsize=(4, 4))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.from_numpy(grid).type(torch.float).to(device)
    model.eval()
    with torch.inference_mode():
        _test_logits = model(grid_tensor).squeeze()
        test_preds = torch.softmax(_test_logits, dim=1).argmax(dim=1)
        zz = test_preds.reshape(xx.shape).detach().cpu().numpy()
        plt.contourf(xx, yy, zz, cmap=plt.cm.plasma, alpha=0.2)
        plt.scatter(x=X[:, 0], y=X[:, 1], c=y_1, s=10, cmap=plt.cm.coolwarm)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title("Decision Boundary")
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""This works well. Try re-running the model with the ReLU activation function removed and see how this affects the decision boundary."""
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

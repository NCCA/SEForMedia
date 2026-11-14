#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Classification

    In the [previous notebook](SimpleClassificationMarimo.py) we build a simple Linear model to try and classify a simple circle dataset. We discovered that it could not do this well. We will now try to build a more complex model to see if we can improve the performance.

    ## Getting started

    We are going to start by importing our base libraries and setting the random seed for reproducibility.
    """)
    return


@app.cell
def _():
    from sklearn.datasets import make_circles
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.optim as optim

    return make_circles, nn, np, optim, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data generation

    We will use the  ```make_circles``` function from the ```sklearn.datasets``` module to generate our data. This function generates a large circle containing a smaller circle in 2D. A simple toy dataset to visualize clustering and classification algorithms.
    """)
    return


@app.cell
def _(make_circles):
    # Make 1000 samples
    n_samples = 1000

    # Create circles
    inputs_array, targets_array = make_circles(
        n_samples,
        noise=0.03,
        random_state=42,  # a little bit of noise to the dots
    )  # keep random state so we get the same values
    return inputs_array, targets_array


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As in the [linear regression notebook](../LinearModel/LinearModelMarimo.py), I use `inputs` and `targets` where books and library examples often use `X` and `y`. These names help us follow what each value does whilst reading the code.

    The inputs are the information we give the model. Here each row is one point and its two features are the horizontal and vertical coordinates. The targets are the class labels we want it to predict, one label for each point. A target is not the vertical coordinate; both coordinates are input features.

    Here each target is 0 or 1, identifying which circle the point belongs to. We use floating point targets for `BCEWithLogitsLoss`, even though the labels describe two classes.

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
    Whilst we are setting things up we may as well get the device we are using to we can train on a gpu.
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

    In the previous example we tried to use a line to fit data that is not linear. We will now try to use a non linear activation function to see if we can fit our data better.

    PyTorch has a number of [non linear activation](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) functions we can use.

    In this example we will use the [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) function defined in  [`torch.nn.ReLU()`](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will still inherit from the nn.Module but change the shape of our model to include a hidden layer with a ReLU activation function.
    """)
    return


@app.cell
def _(device, nn):
    # Build model with non-linear activation function
    class BinaryClassify(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear(in_features=2, out_features=10)
            self.layer_2 = nn.Linear(in_features=10, out_features=10)
            self.layer_3 = nn.Linear(in_features=10, out_features=1)
            self.relu = nn.ReLU()  # non-linear activation function

        def forward(self, inputs):
            # This is a more complex model with 3 layers and a
            # non-linear activation function between each layer
            return self.layer_3(
                self.relu(self.layer_2(self.relu(self.layer_1(inputs))))
            )

    model = BinaryClassify().to(device)
    print(model.parameters)
    # print(model.state_dict())
    model
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As you can now see this is a far more complex model than the previous one. If you uncomment the .state_dict() line you can see the weights and biases of the model and there are many more than before.

    We will use this model (basically forward is written for us now) this is really easy and for simple tasks ideal. However for more complex models we will need to use the more manual method and define our own forward function.

    We can now see what our untrained model does.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loss Function and Optimizer

    As we are dealing with a binary classification problem we can choose a suitable function from the [PyTorch documentation](https://pytorch.org/docs/stable/nn.html#loss-functions).

    PyTorch has two
    1. [`torch.nn.BCELoss()`](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)
    2. [`torch.nn.BCEWithLogitsLoss()`](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)

    They are both similar however the BCEWithLogitsLoss is more numerically stable and has a built in sigmoid function.

    We will use the Stochastic Gradient Descent (SGD) optimizer to train our model as in the previous lab.
    """)
    return


@app.cell
def _(model, nn, optim):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.1)
    return loss_fn, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we need to see how close our function is to the actual labels we can generate a function to do this this is know as an evaluation function and is basically the opposite of the loss function but it can sometimes be more useful to see how well the model is doing.
    """)
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
    torch.manual_seed(42)
    epochs = 1000
    inputs_train_device, targets_train_device = (
        inputs_train.to(device),
        targets_train.to(device),
    )
    inputs_val_device, targets_val_device = (
        inputs_val.to(device),
        targets_val.to(device),
    )
    for epoch in range(epochs):
        model.train()
        train_logits = model(inputs_train_device).squeeze()
        preds = torch.round(torch.sigmoid(train_logits))
        loss = loss_fn(train_logits, targets_train_device)
        acc = accuracy(targets=targets_train_device, preds=preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            _val_logits = model(inputs_val_device).squeeze()
            val_pred = torch.round(torch.sigmoid(_val_logits))
            val_loss = loss_fn(_val_logits, targets_val_device)
            val_acc = accuracy(targets=targets_val_device, preds=val_pred)
        if epoch % 100 == 0:
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
        predictions = torch.round(torch.sigmoid(model(inputs_val_device))).squeeze()
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
        grid_preds = torch.round(torch.sigmoid(_grid_logits))
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
    This is much better, you can see for the most part the model has correctly classified the data. With a little more training and a few tweaks we could probably get this to >90% accuracy.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

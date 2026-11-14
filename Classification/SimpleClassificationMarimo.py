#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Classification

    In this demo we are going to build a simple network to do binary classification, we will generate the example data ourselves using the [sklearn](https://scikit-learn.org/stable/index.html) library then we will train the network to classify the data.

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

    print(inputs_array.shape, targets_array.shape)
    print(type(inputs_array), type(targets_array))
    print(inputs_array[:10], targets_array[:10])
    return inputs_array, targets_array


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As in the [linear regression notebook](../LinearModel/LinearModelMarimo.py), I use `inputs` and `targets` where books and library examples often use `X` and `y`. These names help us follow what each value does whilst reading the code.

    The inputs are the information we give the model. Here each row is one point and its two features are the horizontal and vertical coordinates. The targets are the class labels we want it to predict, one label for each point. A target is not the vertical coordinate; both coordinates are input features.

    Here each target is 0 or 1, identifying which circle the point belongs to. We use floating point targets for `BCEWithLogitsLoss`, even though the labels describe two classes.

    We start with NumPy arrays called `inputs_array` and `targets_array`, then convert them to tensors called `inputs` and `targets`. The names describe the same roles in both forms.

    We can plot this to see the data in full.
    """)
    return


@app.cell
def _(inputs_array, plt, targets_array):
    plt.figure(figsize=(4, 4))

    plt.scatter(
        x=inputs_array[:, 0], y=inputs_array[:, 1], c=targets_array, cmap=plt.cm.plasma
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You will notice we are using a colourmap here for more details on colourmaps see the [matplotlib documentation](https://matplotlib.org/stable/tutorials/colors/colormaps.html) and to see all the colourmaps use ```list(matplotlib.colormaps)``` in your python console.
    """)
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
    print(inputs_train.shape, inputs_val.shape, targets_train.shape, targets_val.shape)
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
    ## Building a Model

    We are going to build a simple model and in this case we will also use the GPU if it is available. To do this we will use the get_device function outlined in the lab. This will make the code work across different devices.
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
    We are going to use the same linear model we used in the previous lab. We will use the ```nn.Linear``` class to create a linear layer rather than constructing it ourselves. We will still inherit from nn.Module and use the super() function to initialise the parent class.

    We will then define the forward function to pass the data through the model.
    """)
    return


@app.cell
def _(device, nn):
    class BinaryClassify(nn.Module):
        def __init__(self):
            super().__init__()
            # layer 1 has 2 inputs and 5 outputs (hidden layer)
            self.layer_1 = nn.Linear(in_features=2, out_features=5)
            # layer 2 has 5 inputs and 1 output (is it label 0 or 1)
            self.layer_2 = nn.Linear(in_features=5, out_features=1)

        # as we have nn.Linear layers, we can use them directly in forward as they
        # are callable objects able to perform the forward pass
        def forward(self, inputs):
            return self.layer_2(self.layer_1(inputs))

    model = BinaryClassify().to(device)
    print(model.parameters)
    print(model.state_dict())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You will notice that we are taking an input with 2 features and expanding it to 5 features. This is a common technique in neural networks to allow the model to learn more complex patterns (this does not always work and can lead to overfitting). This is a hyperparameter and can be tuned to improve the model.

    It is important that the next layer has the same number of features as the previous layer has output. This is why the next layer has 5 input features.

    ## Using nn.Sequential

    This mode of building a model is fine for simple models but can become cumbersome for more complex models. PyTorch provides the ```nn.Sequential``` class to allow us to build models more easily. We can pass the layers as arguments to the ```nn.Sequential``` class and it will build the model for us.

    The model above can be built using ```nn.Sequential``` as follows:
    """)
    return


@app.cell
def _(device, nn):
    model_1 = nn.Sequential(
        nn.Linear(in_features=2, out_features=5),
        nn.Linear(in_features=5, out_features=1),
    ).to(device)
    print(model_1.parameters)
    print(model_1.state_dict())
    # Note if we return the model marimo will show us the contents as well a nice feature.
    model_1
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will use this model (basically forward is written for us now) this is really easy and for simple tasks ideal. However for more complex models we will need to use the more manual method and define our own forward function.

    We can now see what our untrained model does.
    """)
    return


@app.cell
def _(device, inputs_val, model_1, targets_val):
    prediction = model_1(inputs_val.to(device))
    print(f"Length of predictions: {len(prediction)}, Shape: {prediction.shape}")
    print(
        f"Length of validation samples: {len(targets_val)}, Shape: {targets_val.shape}"
    )
    print(f"\nFirst 10 predictions:\n{prediction[:10]}")
    print(f"\nFirst 10 validation labels:\n{targets_val[:10]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    note the output is in the form of a tensor (and has negative values) not the binary output we want. We will need to convert this to a binary output but we will train our model first.

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
def _(model_1, nn, optim):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(params=model_1.parameters(), lr=0.1)
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
    model_1,
    optimizer,
    targets_train,
    targets_val,
    torch,
):
    torch.manual_seed(42)
    epochs = 100
    inputs_train_device, targets_train_device = (
        inputs_train.to(device),
        targets_train.to(device),
    )
    inputs_val_device, targets_val_device = (
        inputs_val.to(device),
        targets_val.to(device),
    )
    for epoch in range(epochs):
        model_1.train()
        train_logits = model_1(inputs_train_device).squeeze()
        preds = torch.round(torch.sigmoid(train_logits))
        loss = loss_fn(train_logits, targets_train_device)
        acc = accuracy(targets=targets_train_device, preds=preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model_1.eval()
        with torch.inference_mode():
            _val_logits = model_1(inputs_val_device).squeeze()
            val_pred = torch.round(torch.sigmoid(_val_logits))
            val_loss = loss_fn(_val_logits, targets_val_device)
            val_acc = accuracy(targets=targets_val_device, preds=val_pred)
        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Validation loss: {val_loss:.5f}, Validation acc: {val_acc:.2f}%"
            )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As you can see from this result the model is not very good but what has happened?
    ## Plotting the Decision Boundary

    We can plot the decision boundary by generating a meshgrid of points and passing them through the model. We can then plot the result using the contourf function from matplotlib.

    The code below does the following.

    1. `x_min, x_max = inputs[:, 0].min() - 0.1, inputs[:, 0].max() + 0.1` calculates the minimum and maximum values for the first feature (column) of the dataset `inputs`, adding a small margin of 0.1 to both ends. This ensures that the grid will extend slightly beyond the range of the data.

    2. `y_min, y_max = inputs[:, 1].min() - 0.1, inputs[:, 1].max() + 0.1` this line calculates the minimum and maximum values for the second feature of the dataset `inputs`, also adding a margin of 0.1.

    3. `xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))`  This line creates a meshgrid using `numpy`. It generates two 2D arrays (`xx` and `yy`) that represent all combinations of horizontal and vertical coordinates within the specified ranges. The `np.linspace` function creates 100 evenly spaced values between `x_min` and `x_max`, and between `y_min` and `y_max`.

    This will then allow us to plot into this grid to get the values we need.
    """)
    return


@app.cell
def _(device, inputs, model_1, np, plt, targets, torch):
    x_min, x_max = (inputs[:, 0].min() - 0.1, inputs[:, 0].max() + 0.1)
    y_min, y_max = (inputs[:, 1].min() - 0.1, inputs[:, 1].max() + 0.1)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    plt.figure(figsize=(4, 4))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.from_numpy(grid).type(torch.float).to(device)
    model_1.eval()
    with torch.inference_mode():
        _grid_logits = model_1(grid_tensor).squeeze()
        grid_preds = torch.round(torch.sigmoid(_grid_logits))
        zz = grid_preds.reshape(xx.shape).detach().cpu().numpy()
        plt.contourf(xx, yy, zz, cmap=plt.cm.coolwarm, alpha=0.2)
        plt.scatter(
            x=inputs[:, 0], y=inputs[:, 1], c=targets, s=10, cmap=plt.cm.coolwarm
        )
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title("Decision Boundary")
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we can see using the linear model to classify non-linear data doesn't work that well and at best we can cut the data in half hence the 50% accuracy.

    This is know as underfitting as it can't (at present) learn any patterns from this data.

    # Conclusion

    We have built a simple neural network to classify data generated by sklearn. We have seen that a linear model is not suitable for this data and we will need to use a more complex model to classify this data. We have also seen how to use the nn.Sequential class to build models more easily.

    In the next lab we will build a more complex model to classify this data.

    # References

    1. https://scikit-learn.org/stable/index.html
    2. https://matplotlib.org/stable/tutorials/colors/colormaps.html
    3. https://pytorch.org/docs/stable/nn.html#loss-functions
    4. https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    5. https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    6. https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
    7. https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    8. https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    9. https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Classification

    In this demo we are going to build a simple network to do binary classification, we will generate the test data ourselves using the [sklearn](https://scikit-learn.org/stable/index.html) library then we will train the network to classify the data.

    ## Getting started

    We are going to start by importing our base libraries and setting the random seed for reproducibility.
    """
    )
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
    mo.md(
        r"""
    ## Data generation

    We will use the  ```make_circles``` function from the ```sklearn.datasets``` module to generate our data. This function generates a large circle containing a smaller circle in 2D. A simple toy dataset to visualize clustering and classification algorithms.
    """
    )
    return


@app.cell
def _(make_circles):
    # Make 1000 samples
    n_samples = 1000

    # Create circles
    x, y = make_circles(
        n_samples,
        noise=0.03,
        random_state=42,  # a little bit of noise to the dots
    )  # keep random state so we get the same values

    print(x.shape, y.shape)
    print(type(x), type(y))
    print(x[:10], y[:10])
    return x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    You can see that the data has two components, the first one is the data itself as x,y co-ordinates and the second one is the labels. We will use the data to train our network and the labels to evaluate it.

    We can plot this to see the data in full.
    """
    )
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
        r"""You will notice we are using a colourmap here for more details on colourmaps see the [matplotlib documentation](https://matplotlib.org/stable/tutorials/colors/colormaps.html) and to see all the colourmaps use ```list(matplotlib.colormaps)``` in your python console."""
    )
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

    We are going to build a simple model and in this case we will also use the GPU if it is available. To do this we will use the get_device function outlined in the lab. This will make the code work across different devices.
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
        r"""
    We are going to use the same linear model we used in the previous lab. We will use the ```nn.Linear``` class to create a linear layer rather than constructing it ourselves. We will still inherit from nn.Module and use the super() function to initialise the parent class.

    We will then define the forward function to pass the data through the model.
    """
    )
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
        def forward(self, x):
            return self.layer_2(self.layer_1(x))

    model = BinaryClassify().to(device)
    print(model.parameters)
    print(model.state_dict())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    You will notice that we are taking an input with 2 features and expanding it to 5 features. This is a common technique in neural networks to allow the model to learn more complex patterns (this does not always work and can lead to overfitting). This is a hyperparameter and can be tuned to improve the model.

    It is important that the next layer has the same number of features as the previous layer has output. This is why the next layer has 5 input features.

    ## Using nn.Sequential

    This mode of building a model is fine for simple models but can become cumbersome for more complex models. PyTorch provides the ```nn.Sequential``` class to allow us to build models more easily. We can pass the layers as arguments to the ```nn.Sequential``` class and it will build the model for us.

    The model above can be built using ```nn.Sequential``` as follows:
    """
    )
    return


@app.cell
def _(device, nn):
    model_1 = nn.Sequential(
        nn.Linear(in_features=2, out_features=5),
        nn.Linear(in_features=5, out_features=1),
    ).to(device)
    print(model_1.parameters)
    print(model_1.state_dict())
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We will use this model (basically forward is written for us now) this is really easy and for simple tasks ideal. However for more complex models we will need to use the more manual method and define our own forward function.

    We can now see what our untrained model does.
    """
    )
    return


@app.cell
def _(X_test, device, model_1, y_test):
    prediction = model_1(X_test.to(device))
    print(f"Length of predictions: {len(prediction)}, Shape: {prediction.shape}")
    print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
    print(f"\nFirst 10 predictions:\n{prediction[:10]}")
    print(f"\nFirst 10 test labels:\n{y_test[:10]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    note the output is in the form of a tensor (and has negative values) not the binary output we want. We will need to convert this to a binary output but we will train our model first.

    ## Loss Function and Optimizer

    As we are dealing with a binary classification problem we can choose a suitable function from the [PyTorch documentation](https://pytorch.org/docs/stable/nn.html#loss-functions).

    PyTorch has two
    1. [`torch.nn.BCELoss()`](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)
    2. [`torch.nn.BCEWithLogitsLoss()`](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)

    They are both similar however the BCEWithLogitsLoss is more numerically stable and has a built in sigmoid function.

    We will use the Stochastic Gradient Descent (SGD) optimizer to train our model as in the previous lab.
    """
    )
    return


@app.cell
def _(model_1, nn, optim):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(params=model_1.parameters(), lr=0.1)
    return loss_fn, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Next we need to see how close our function is to the actual labels we can generate a function to do this this is know as an evaluation function and is basically the opposite of the loss function but it can sometimes be more useful to see how well the model is doing."""
    )
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
    model_1,
    optimizer,
    torch,
    y_test,
    y_train,
):
    torch.manual_seed(42)
    epochs = 100
    X_train_1, y_train_1 = (X_train.to(device), y_train.to(device))
    X_test_1, y_test_1 = (X_test.to(device), y_test.to(device))
    for epoch in range(epochs):
        model_1.train()
        y_logits = model_1(X_train_1).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))
        loss = loss_fn(y_logits, y_train_1)
        acc = accuracy(y_true=y_train_1, y_pred=y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model_1.eval()
        with torch.inference_mode():
            _test_logits = model_1(X_test_1).squeeze()
            test_pred = torch.round(torch.sigmoid(_test_logits))
            test_loss = loss_fn(_test_logits, y_test_1)
            test_acc = accuracy(y_true=y_test_1, y_pred=test_pred)
        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%"
            )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    As you can see from this result the model is not very good but what has happened?
    ## Plotting the Decision Boundary

    We can plot the decision boundary by generating a meshgrid of points and passing them through the model. We can then plot the result using the contourf function from matplotlib.

    The code below does the following.

    1. `x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1` calculates the minimum and maximum values for the first feature (column) of the dataset `X`, adding a small margin of 0.1 to both ends. This ensures that the grid will extend slightly beyond the range of the data.

    2. `y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1` this line calculates the minimum and maximum values for the second feature of the dataset `X`, also adding a margin of 0.1.

    3. `xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))`  This line creates a meshgrid using `numpy`. It generates two 2D arrays (`xx` and `yy`) that represent all combinations of x and y values within the specified ranges. The `np.linspace` function creates 100 evenly spaced values between `x_min` and `x_max`, and between `y_min` and `y_max`.

    This will then allow us to plot into this grid to get the values we need.
    """
    )
    return


@app.cell
def _(X, device, model_1, np, plt, torch, y_1):
    x_min, x_max = (X[:, 0].min() - 0.1, X[:, 0].max() + 0.1)
    y_min, y_max = (X[:, 1].min() - 0.1, X[:, 1].max() + 0.1)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    plt.figure(figsize=(4, 4))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.from_numpy(grid).type(torch.float).to(device)
    model_1.eval()
    with torch.inference_mode():
        _test_logits = model_1(grid_tensor).squeeze()
        test_preds = torch.round(torch.sigmoid(_test_logits))
        zz = test_preds.reshape(xx.shape).detach().cpu().numpy()
        plt.contourf(xx, yy, zz, cmap=plt.cm.coolwarm, alpha=0.2)
        plt.scatter(x=X[:, 0], y=X[:, 1], c=y_1, s=10, cmap=plt.cm.coolwarm)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title("Decision Boundary")
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

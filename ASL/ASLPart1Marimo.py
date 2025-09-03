#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full", app_title="ASL Part 1")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Introduction

    In this notebook we are going to download training data for the American Sign Language from Kaggle. This data will be processed and used to train a Convolutional Neural Network (CNN) to classify the images.

    The [American Sign Language alphabet](http://www.asl.gs/) contains 26 letters. Two of those letters (j and z) require movement, so they are not included in the training dataset.  The data set is unusual as it is contained in a CSV file so we will need to do some processing of the data.

    ## Kaggle

    We will download the data set from [Kaggle](http://www.kaggle.com) which contains a number of different data sets and examples we can look at.

    As usual we will test to see if we are in the lab and download the data set to the /transfer else locally.
    """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import pathlib
    import sys
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.utils.data import Dataset, DataLoader
    import zipfile
    import string

    # Visualization tools
    import pandas as pd

    sys.path.append("../")
    import Utils

    device = Utils.get_device()

    DATASET_LOCATION = ""
    if Utils.in_lab():
        DATASET_LOCATION = "/transfer/mnist_asl/"
    else:
        DATASET_LOCATION = "./mnist_asl/"

    print(f"Dataset location: {DATASET_LOCATION}")
    pathlib.Path(DATASET_LOCATION).mkdir(parents=True, exist_ok=True)
    return (
        Adam,
        DATASET_LOCATION,
        DataLoader,
        Dataset,
        Utils,
        device,
        nn,
        pathlib,
        pd,
        plt,
        string,
        torch,
        zipfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can use the Utils class to download the data set then unzip it.""")
    return


@app.cell
def _(DATASET_LOCATION, Utils, pathlib, zipfile):
    URL = "https://www.kaggle.com/api/v1/datasets/download/nadaemad2002/asl-mnist"
    desitnation = DATASET_LOCATION + "/asl-mnist.zip"
    if not pathlib.Path(desitnation).exists():
        Utils.download(URL, desitnation)
        # now we need to unzip the file
        with zipfile.ZipFile(desitnation, "r") as zip_ref:
            zip_ref.extractall(DATASET_LOCATION)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Data

    Lets have a look at what we have downloaded, we can get a list of the files we have downloaded by using pathlib as follows.
    """
    )
    return


@app.cell
def _(DATASET_LOCATION, pathlib):
    for file in pathlib.Path(DATASET_LOCATION).glob("*"):
        print(file)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    There are two image files showing the the signs and two CSV files for the actual test and train data.
    ![](./mnist_asl/american_sign_language.PNG)
    ![](./mnist_asl/amer_sign3.png)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Processing the data with pandas

    As the data is in csv format the easiest approach to processing it is to use the pandas library. We can read the csv file into a pandas dataframe and then process the data as required.
    """
    )
    return


@app.cell
def _(DATASET_LOCATION, pd):
    train_df = pd.read_csv(f"{DATASET_LOCATION}sign_mnist_train.csv")
    valid_df = pd.read_csv(f"{DATASET_LOCATION}sign_mnist_test.csv")
    return train_df, valid_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We can now look at the data format and begin to process it into a more useful format for training the CNN."""
    )
    return


@app.cell
def _(train_df):
    train_df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Each row is an image which has a `label` column followed by the pixel data (28 * 28 = 784 columns). The pixel data is in the range 0 to 255. We need to extract the label and pixel data and reshape the pixel data into a 28 x 28 image. We can get the labels by using the pandas pop method and the pixel data by using the pandas iloc method."""
    )
    return


@app.cell
def _(train_df, valid_df):
    y_train = train_df.pop("label")
    y_valid = valid_df.pop("label")
    # look at the data
    print(y_train.value_counts(sort=True, ascending=True).sort_index())
    print(y_train.agg(["min", "max"]))
    return y_train, y_valid


@app.cell
def _(train_df, valid_df):
    x_train = train_df.values
    x_valid = valid_df.values
    print(x_train.shape, x_valid.shape)
    print(type(x_train), type(x_valid))
    return (x_train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""If we look at the data we can see what it contains. We can use some of the pandas functions to find the min() and max() as well as sort and tabulate the data."""
    )
    return


@app.cell
def _(y_train, y_valid):
    print(y_train.agg(["min", "max"]))
    print(y_valid.agg(["min", "max"]))

    print(y_train.value_counts(sort=True, ascending=True).sort_index())
    print(y_valid.value_counts(sort=True, ascending=True).sort_index())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    As you can see the data ranges from 0-24 so whilst the J and the Z are missing the J value index (9) is still in the dataset, so we need to specify for 25 classes not 24.

    ## Visualizing the data

    We can take a look at the data by writing a simple function to turn the nparrays into a 28 x 28 image and then display the image. Note that the labels are the numbers 0 to 24 and the letters are the alphabet minus j and z. We need to convert the labels to the letters for display purposes.
    """
    )
    return


@app.cell
def _(plt, string, x_train, y_train):
    def plot_image(images, labels, num_images, image_index):
        image = images.reshape(28, 28)
        label = labels
        plt.subplot(1, num_images, image_index + 1)
        plt.title(label, fontdict={"fontsize": 30})
        plt.axis("off")
        plt.imshow(image, cmap="gray")
        plt.show()

    alphabet = string.ascii_letters[:25]

    num_images = 12
    plt.figure(figsize=(10, 10))
    for x in range(num_images):
        row = x_train[x]
        label = y_train[x]
        plot_image(row, alphabet[label], num_images, x)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We can now normalize our image data so it is in the range 0 to 1. We can do this by dividing the pixel data by 255 (as they are nparrays this will be element wise division)."""
    )
    return


@app.cell
def _(train_df, valid_df):
    x_train_1 = train_df.values / 255
    x_valid_1 = valid_df.values / 255
    return x_train_1, x_valid_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Generating a Dataset

    As we did in the previous example we can create a dataset class to represent our data. Again this will be a simple image / label dataset.
    """
    )
    return


@app.cell
def _(Dataset, device, torch):
    class ASLDataSet(Dataset):
        def __init__(self, x_df, y_df):
            print(device)
            self.xs = torch.tensor(x_df).float().to(device)
            self.ys = torch.tensor(y_df).to(device)

        def __getitem__(self, idx):
            x = self.xs[idx]
            y = self.ys[idx]
            return x, y

        def __len__(self):
            return len(self.xs)

    return (ASLDataSet,)


@app.cell
def _(ASLDataSet, DataLoader, x_train_1, x_valid_1, y_train, y_valid):
    BATCH_SIZE = 32
    train_data = ASLDataSet(x_train_1, y_train)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    train_N = len(train_loader.dataset)
    valid_data = ASLDataSet(x_valid_1, y_valid)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)
    valid_N = len(valid_loader.dataset)
    print(f"Train size: {train_N}, Valid size: {valid_N}")
    return train_N, train_loader, valid_N, valid_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Building a Model

    In this example we will use a similar approach to the Digits example Where we used a Linear model.


    * Has a flatten layer.
    * Has a dense input layer. This layer should contain 512 neurons amd use the `relu` activation function
    * Has a second dense layer with 512 neurons which uses the `relu` activation function
    * Has a dense output layer with neurons equal to the number of classes

    We will define a few variables to get started:

    The size of the images is 28 x 28 and there are 24 classes (the alphabet minus j and z).
    """
    )
    return


@app.cell
def _():
    input_size = 28 * 28
    n_classes = 25
    return input_size, n_classes


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can add the layers to the model using the nn.Sequential class. See if you can think of what we should do next.

    <details>

    <summary>Solution</summary>

    ```python
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_size, 512),  # Input
        nn.ReLU(),  # Activation for input
        nn.Linear(512, 512),  # Hidden
        nn.ReLU(),  # Activation for hidden
        nn.Linear(512, n_classes)  # Output
    )
    ```

    </details>
    """
    )
    return


@app.cell
def _(input_size, n_classes, nn):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_size, 512),  # Input
        nn.ReLU(),  # Activation for input
        nn.Linear(512, 512),  # Hidden
        nn.ReLU(),  # Activation for hidden
        nn.Linear(512, n_classes),  # Output
    )
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can compile and send the model to the device.""")
    return


@app.cell
def _(device, model, torch):
    model_compiled = torch.compile(model.to(device))
    model_compiled.to(device)
    return (model_compiled,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Since categorizing these ASL images is similar to categorizing MNIST's handwritten digits, we will use the same `loss_function` ([Categorical CrossEntropy](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)) and `optimizer` ([Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)) as we used in the last example."""
    )
    return


@app.cell
def _(Adam, model_compiled, nn):
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model_compiled.parameters())
    return loss_function, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Training

    The data in the DataLoader is already on the GPU so we can now just process it in the same way as we did in the previous example.

    The steps are the same as before.


    1. Get an `output` prediction from the model
    2. Set the gradient to zero with the `optimizer`'s [zero_grad](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html) function
    3. Calculate the loss with our `loss_function`
    4. Compute the gradient with [backward](https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html)
    5. Update our model parameters with the `optimizer`'s [step](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html) function.
    6. Update the `loss` and `accuracy` totals
    """
    )
    return


@app.cell
def _(loss_function, model_compiled, optimizer, train_N, train_loader):
    def get_batch_accuracy(output, y, N):
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(y.view_as(pred)).sum().item()
        return correct / N

    train_accuracy = []
    train_loss = []

    def train():
        loss = 0
        accuracy = 0
        model_compiled.train()
        for x, y in train_loader:
            output = model_compiled(x)
            optimizer.zero_grad()
            batch_loss = loss_function(output, y)
            batch_loss.backward()
            optimizer.step()
            loss = loss + batch_loss.item()
            accuracy = accuracy + get_batch_accuracy(output, y, train_N)
        train_accuracy.append(accuracy)
        train_loss.append(loss)
        print("Train - Loss: {:.4f} Accuracy: {:.4f}".format(loss, accuracy))

    return get_batch_accuracy, train, train_accuracy, train_loss


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Validate Function

    The core part of the validate process is to set the model to evaluation mode with the `model.eval()` function. This will turn off dropout and batch normalization. We then loop through the validation data and calculate the loss and accuracy in the same way as we did for the training data.

    More details of [model.evaluate](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval) can be found here.
    """
    )
    return


@app.cell
def _(
    get_batch_accuracy,
    loss_function,
    model_compiled,
    torch,
    valid_N,
    valid_loader,
):
    valid_accuracy = []
    valid_loss = []

    def validate():
        loss = 0
        accuracy = 0
        model_compiled.eval()
        with torch.no_grad():
            for x, y in valid_loader:
                output = model_compiled(x)
                loss = loss + loss_function(output, y).item()
                accuracy = accuracy + get_batch_accuracy(output, y, valid_N)
        valid_accuracy.append(accuracy)
        valid_loss.append(loss)
        print("Valid - Loss: {:.4f} Accuracy: {:.4f}".format(loss, accuracy))

    return valid_accuracy, valid_loss, validate


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Training

    Finally we can train the model as before.
    """
    )
    return


@app.cell
def _(train, validate):
    epochs = 10

    for epoch in range(epochs):
        print("Epoch: {}".format(epoch))
        train()
        validate()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Discussion

    Our models doesn't seem to be getting very good results. We can see that the training accuracy seems to a fairly high level, but the validation accuracy was not as high. This is a sign of overfitting, which means that is it guessing against a learnt data set and never generalizing to new data.

    As we have accumulated the data when training, we can plot this out to see how things went.
    """
    )
    return


@app.cell
def _(plt, train_accuracy, train_loss, valid_accuracy, valid_loss):
    # Lets plot the data
    plt.figure(figsize=(5, 2))
    plt.plot(train_accuracy, label="Train")
    plt.plot(valid_accuracy, label="Valid")
    plt.xlabel("Epoch")
    plt.title("Accuracy")
    plt.legend()
    plt.show()

    plt.figure(figsize=(5, 2))
    plt.plot(train_loss, label="Train")
    plt.plot(valid_loss, label="Valid")
    plt.xlabel("Epoch")
    plt.title("Loss")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We should see that the lines are diverging, which is a sign of overfitting. We need to look at a different model to make this work better. In particular we are going to build a CNN model to see if this can improve the accuracy."""
    )
    return


@app.cell
def _(model, torch):
    # tidy up before the next session
    import gc

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

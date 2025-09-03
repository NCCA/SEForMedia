#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full", app_title="ASL CNN Data Augmentation")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # ASL CNN Data Augmentation

    In the [previous notebook](ASLPart2CNN.ipynb), we built a CNN model to classify the ASL dataset. In this notebook, we will look at how we can use data augmentation to improve the performance of our model.

    The validation accuracy is still lagging behind the training accuracy, which is a sign of overfitting and the model getting confused.

    One method we can use to improve the model's performance is data augmentation. Data augmentation is a technique used to artificially increase the size of the training dataset by applying various transformations to the existing images. This helps the model generalize better and reduces overfitting.

    The increase in size gives the model more images to learn from while training. The increase in variance helps the model ignore unimportant features and select only the features that are truly important in classification, allowing it to generalize better.

    We will start with the exact same data and model as in the previous notebook and then apply data augmentation to see if it improves the model's performance as part of the training process.

    ## Loading data

    The following code was outlined in the previous two examples.
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
    from PIL import Image

    # Visualization tools
    import torchvision.transforms.v2 as transforms
    import torchvision.transforms.functional as F
    import pandas as pd
    import string

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
        F,
        Image,
        Utils,
        device,
        nn,
        pd,
        plt,
        string,
        torch,
        transforms,
    )


@app.cell
def _(DATASET_LOCATION, pd):
    train_df = pd.read_csv(f"{DATASET_LOCATION}sign_mnist_train.csv")
    valid_df = pd.read_csv(f"{DATASET_LOCATION}sign_mnist_test.csv")
    return train_df, valid_df


@app.cell
def _(Dataset, device, torch):
    BATCH_SIZE = 32
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    IMAGE_CHANNELS = 1

    class ASLImages(Dataset):
        def __init__(self, base_df):
            x_df = base_df.copy()
            y_df = x_df.pop("label")
            x_df = x_df.values / 255  # Normalize values from 0 to 1
            x_df = x_df.reshape(-1, IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT)
            # send to device for processing
            self.xs = torch.tensor(x_df).float().to(device)
            self.ys = torch.tensor(y_df).to(device)

        def __getitem__(self, idx):
            x = self.xs[idx]
            y = self.ys[idx]
            return x, y

        def __len__(self):
            return len(self.xs)

    return ASLImages, BATCH_SIZE, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH


@app.cell
def _(ASLImages, BATCH_SIZE, DataLoader, train_df, valid_df):
    train_data = ASLImages(train_df)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    train_N = len(train_loader.dataset)

    valid_data = ASLImages(valid_df)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)
    valid_N = len(valid_loader.dataset)
    return train_N, train_loader, valid_N, valid_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Building a Model

    We going to use exactly the same model as in the previous notebook, however this model had a lot of repetition in the code, especially with the Convolution layers. We can re-factor this to be our own custom class and then add this as a layer in the base Sequential model, to do this we need to look at this code and see what parameters are changing and what we can make more generic.

    ```
    nn.Conv2d(IMAGE_CHANNELS, 25, kernel_size, stride=1, padding=1),  # 25 x 28 x 28
    nn.BatchNorm2d(25),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),  # 25 x 14 x 14
    # Second convolution
    nn.Conv2d(25, 50, kernel_size, stride=1, padding=1),  # 50 x 14 x 14
    nn.BatchNorm2d(50),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.MaxPool2d(2, stride=2),  # 50 x 7 x 7
    # Third convolution
    nn.Conv2d(50, 75, kernel_size, stride=1, padding=1),  # 75 x 7 x 7
    nn.BatchNorm2d(75),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),  # 75 x 3 x 3
    ```

    We can see that the only thing that is changing is the number of input channels and the number of output channels and the use  of dropouts. We can make this more generic by passing in these parameters as arguments to the class and then using them to create the layers.
    """
    )
    return


@app.cell
def _(nn):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch, dropout_p):
            kernel_size = 3
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.MaxPool2d(2, stride=2),
            )

        def forward(self, x):
            return self.model(x)

    return (ConvBlock,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""The forward method just needs to be updated to use the class variables that we have created, so it will evaluate the same as before."""
    )
    return


@app.cell
def _(Adam, ConvBlock, IMAGE_CHANNELS, device, nn, torch):
    flattened_img_size = 75 * 3 * 3
    N_CLASSES = 25
    # Input 1 x 28 x 28
    model = nn.Sequential(
        ConvBlock(IMAGE_CHANNELS, 25, 0),  # 25 x 14 x 14
        ConvBlock(25, 50, 0.2),  # 50 x 7 x 7
        ConvBlock(50, 75, 0),  # 75 x 3 x 3
        # Flatten to Dense Layers
        nn.Flatten(),
        nn.Linear(flattened_img_size, 512),
        nn.Dropout(0.3),
        nn.ReLU(),
        nn.Linear(512, N_CLASSES),
    )

    if device == "cuda":
        model = torch.compile(model.to(device))
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())
    return loss_function, model, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # [Torchvision Transforms](https://pytorch.org/vision/0.9/transforms.html)

    We have used these before for simple transforms such as scaling, we will now look in more depth at some other transforms we can apply to augment our data and provide variance to the input data.

    We will start by extracting an image from the data to process and see what the results are.
    """
    )
    return


@app.cell
def _(IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, torch, train_df):
    row_0 = train_df.head(1)
    _ = row_0.pop("label")
    # normalize the values to 0-1
    x_0 = row_0.values / 255
    # convert to an image
    x_0 = x_0.reshape(IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT)
    # transform to a tensor
    x_0 = torch.tensor(x_0)
    x_0.shape
    return (x_0,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can define a simple image plot functions to display the images.""")
    return


@app.cell
def _(plt, x_0):
    def plot_image(images, label, num_images=1, image_index=0):
        image = images.reshape(28, 28)
        plt.subplot(1, num_images, image_index + 1)
        plt.title(label)
        plt.axis("off")
        plt.imshow(image, cmap="gray")

    plt.figure(figsize=(1, 1))
    plot_image(x_0, "Base Image")
    plt.show()

    return (plot_image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    [RandomResizeCrop](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.RandomResizedCrop)

    This transform will apply both a crop and a resize to the image, the crop will be random and the resize will be to the size specified. It needs to know the aspect ratio of the image to be able to crop it correctly, however in our case it is 1:1 as the image is square.
    """
    )
    return


@app.cell
def _(IMAGE_HEIGHT, IMAGE_WIDTH, plot_image, plt, transforms, x_0):
    def plot_multiple(num_images, transform, data, fig_size=(6, 6)):
        plt.figure(figsize=fig_size)
        for i in range(num_images):
            new_x_0 = transform(x_0)
            plot_image(new_x_0, i, num_images, i)
        return plt

    trans = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                (IMAGE_WIDTH, IMAGE_HEIGHT), scale=(0.7, 1), ratio=(1, 1)
            )
        ]
    )
    # as this returns a plt object we can call show directly
    plot_multiple(8, trans, x_0).show()

    return (plot_multiple,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    You can see the results are very subtle, but the image has changed enough to provide variance to the input data.

    ## [RandomHorizontalFlip](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.RandomHorizontalFlip)

    We can also randomly flip our images both horizontally and vertically, this will depend upon the data and what we are trying to do. In our case we are only going to flip the images horizontally, as this is the only way that the ASL data can be flipped and still be valid. (Note ASL is typically done with the dominant hand so both left or right handed is fine).
    """
    )
    return


@app.cell
def _(plot_multiple, transforms, x_0):
    _trans = transforms.Compose([transforms.RandomHorizontalFlip()])
    plot_multiple(8, _trans, x_0).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## [RandomRotation](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.RandomRotation)

    We can also rotate the images by a random amount, but like the flipping we must be careful with this as the ASL data is very specific and we don't want to rotate the images too much as it will make the data invalid. We will limit the rotation to 20 degrees in either direction.
    """
    )
    return


@app.cell
def _(plot_multiple, transforms, x_0):
    _trans = transforms.Compose([transforms.RandomRotation(20)])
    plot_multiple(8, _trans, x_0).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Notice how this has added borders to the image, this is because the image has been rotated and the corners are now empty and set to an empty (black) pixel value.

    ## [ColorJitter](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.ColorJitter)

    The `ColorJitter` transform has 4 arguments:
        - brightness
        - contrast
        - saturation
        - hue

    The saturation and hue apply to color images, so we will only use the first 2 for this example as we are using grayscale images.
    """
    )
    return


@app.cell
def _(plot_multiple, transforms, x_0):
    brightness = 0.3
    contrast = 0.5
    _trans = transforms.Compose(
        [transforms.ColorJitter(brightness=brightness, contrast=contrast)]
    )
    plot_multiple(8, _trans, x_0).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## [Compose](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.Compose)

    It is possible to combine all of these transforms into a single transform using the `Compose` class, this will apply all of the transforms in the order that they are passed in.
    """
    )
    return


@app.cell
def _(IMAGE_HEIGHT, IMAGE_WIDTH, plot_multiple, transforms, x_0):
    _random_transforms = transforms.Compose(
        [
            transforms.RandomRotation(5),
            transforms.RandomResizedCrop(
                (IMAGE_WIDTH, IMAGE_HEIGHT), scale=(0.9, 1), ratio=(1, 1)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.5),
        ]
    )
    for _ in range(4):
        plot_multiple(8, _random_transforms, x_0).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Training

    We will now train our model as before, however now we can pass in the transformation to our model, this will then apply it to each image before it is passed into the model.

    As we are re-using a lot of code now, I have moved the get_batch_accuracy function into the Utils module and will use that instead. Note later we will also move the train / validate functions here too.
    """
    )
    return


@app.cell
def _(
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    Utils,
    device,
    loss_function,
    model,
    optimizer,
    train_N,
    train_loader,
    transforms,
):
    train_accuracy = []
    train_loss = []
    if device == "cuda" or device == "cpu":
        _random_transforms = transforms.Compose(
            [
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(
                    (IMAGE_WIDTH, IMAGE_HEIGHT), scale=(0.9, 1), ratio=(1, 1)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.5),
            ]
        )
    else:
        _random_transforms = transforms.Compose(
            [
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.5),
            ]
        )

    def train():
        loss = 0
        accuracy = 0
        model.train()
        for x, y in train_loader:
            output = model(_random_transforms(x))
            optimizer.zero_grad()
            batch_loss = loss_function(output, y)
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
            accuracy += Utils.get_batch_accuracy(output, y, train_N)
        train_accuracy.append(accuracy)
        train_loss.append(loss)
        print("Train - Loss: {:.4f} Accuracy: {:.4f}".format(loss, accuracy))

    return train, train_accuracy, train_loss


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""The validation remains the same as before and we don't add any transformations to the validation data, as we want to see how the model performs on the original data."""
    )
    return


@app.cell
def _(Utils, loss_function, model, torch, valid_N, valid_loader):
    valid_accuracy = []
    valid_loss = []

    def validate():
        loss = 0
        accuracy = 0

        model.eval()
        with torch.no_grad():
            for x, y in valid_loader:
                output = model(x)
                loss += loss_function(output, y).item()
                accuracy += Utils.get_batch_accuracy(output, y, valid_N)
        valid_accuracy.append(accuracy)
        valid_loss.append(loss)
        print("Valid - Loss: {:.4f} Accuracy: {:.4f}".format(loss, accuracy))

    return valid_accuracy, valid_loss, validate


@app.cell
def _(train, validate):
    epochs = 20

    for epoch in range(epochs):
        print("Epoch: {}".format(epoch))
        train()
        validate()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can plot our results as before and see how the model performs.""")
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
        r"""
    The results are much better, and it is not showing the signs of overfitting we had before. The validation accuracy is now much closer to the training accuracy and the model is performing much better.

    The training accuracy may be lower, and that's ok. Compared to before, the model is being exposed to a much larger variety of data.

    ## Testing the model.

    We can now test the model as before and see how it performs on the test data.
    """
    )
    return


@app.cell
def _(model, torch):
    # Save the model
    torch.save(model.state_dict(), "asl_model.pth")
    # Also save the full model
    torch.save(model, "asl_model_full.pth")
    return


@app.cell
def _(model, plt, string, torch, valid_loader):
    model.eval()
    _alphabet = string.ascii_letters[:25]
    with torch.no_grad():
        x, y = next(iter(valid_loader))
        output = model(x)
        _pred = output.argmax(dim=1, keepdim=True)
        num_images = 10
        plt.figure(figsize=(20, 20))
        for _i in range(num_images):
            plt.subplot(1, num_images, _i + 1)
            plt.title(
                f" {_alphabet[y[_i].item()]} {_alphabet[_pred[_i].item()]}",
                fontdict={"fontsize": 30},
            )
            plt.axis("off")
            plt.imshow(x[_i].cpu().numpy().reshape(28, 28), cmap="gray")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # More Testing

    At present all the data we have used is from the same dataset, we can now test the model on some new data that it has not seen before.  The dataset we downloaded has the following image

    ![](mnist_asl/amer_sign3.png)

    We can partition this into new test data and see how the model performs on this data.
    """
    )
    return


@app.cell
def _(DATASET_LOCATION, F, Image, plt, torch, transforms):
    image = Image.open(DATASET_LOCATION + "/amer_sign3.png")
    sub_images = []
    image_width = image.width / 6
    image_height = image.height / 4
    for _i in range(4):
        for j in range(6):
            left = j * image_width
            upper = _i * image_height
            sub_images.append(F.crop(image, upper, left, image_width, image_height))
    preprocess_trans = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.ToTensor(),
        ]
    )
    tensor_images = [preprocess_trans(img) for img in sub_images]
    plt.figure(figsize=(4, 4))
    for _i, img in enumerate(tensor_images):
        plt.subplot(4, 6, _i + 1)
        plt.axis("off")
        plt.imshow(F.to_pil_image(img), cmap="gray")
    plt.show()
    return (tensor_images,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""<!-- Now lets feed this into our model and see how it does. The signs are in alphabetical order, so the first sign is A, the second is B and so on. -->"""
    )
    return


@app.cell
def _(device, model, tensor_images, torch):
    _alphabet = "abcdefghijklmnopqrstuvwxy"
    with torch.no_grad():
        for _i in range(23):
            _pred = model(tensor_images[_i].unsqueeze(0).to(device))
            print(f"{_alphabet[_pred.argmax().item()]} ", sep="", end="")
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

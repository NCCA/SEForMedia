#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full", app_title="Transfer Learning")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Transfer Learning

    The process of transfer learning involves taking a pre-trained model and adapting the model to a new, different data set. In this notebook, we will demonstrate how to use transfer learning to train a model to perform image classification on a data set that is different from the data set on which the pre-trained model was trained.


    Transfer learning is really useful when we have a small dataset to train against, and the pre-trained model has been trained on a larger dataset because a small dataset will memorize the data quickly and not work on the new data.

    In the previous notebook, we trained a model on the vgg16 model or animal images, we will use the same model to train on the new images.
    """
    )
    return


@app.cell
def _():
    # from imports import *

    import pathlib
    import sys

    import matplotlib.pyplot as plt
    import torch
    import torchvision.transforms.v2 as transforms
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    import torch.nn as nn
    import torch.optim as optim

    sys.path.append("../")
    import Utils

    device = Utils.get_device()

    DATASET_LOCATION = ""
    if Utils.in_lab():
        DATASET_LOCATION = "/transfer/dog_door/"
    else:
        DATASET_LOCATION = "./dog_door/"

    print(f"Dataset location: {DATASET_LOCATION}")
    pathlib.Path(DATASET_LOCATION).mkdir(parents=True, exist_ok=True)
    return (
        DATASET_LOCATION,
        DataLoader,
        Dataset,
        Image,
        Utils,
        device,
        nn,
        optim,
        pathlib,
        plt,
        torch,
        transforms,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Dataset download

    In this example (based on the nVidia deep learning course) we are going to download a dataset of a specific dog (Bo the president dog) and a cat. The dataset is available at the following link: https://www.kaggle.com/api/v1/datasets/download/thomaschxu/doggydata

    We will then train our model to classify if it is the specific dog or something else.
    """
    )
    return


@app.cell
def _(DATASET_LOCATION, Utils, pathlib):
    url = "https://www.kaggle.com/api/v1/datasets/download/thomaschxu/doggydata"

    desitnation = DATASET_LOCATION + "doggydata.zip"
    if not pathlib.Path(desitnation).exists():
        Utils.download(url, desitnation)
        Utils.unzip_file(desitnation, DATASET_LOCATION)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # VGG16 Model

    we are going to use the vgg16 model which has a 1000 categories, we can now add the new trainable layers to the pre-trained model.

    They will take the features from the pre-trained layers and turn them into predictions on the new dataset. We will add two layers to the model.  Then, we'll add a `Linear` layer connecting all `1000` of VGG16's outputs to `1` neuron to predict if we have the correct dog or not.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We will now download the pre-trained model and as before and do the setup"""
    )
    return


@app.cell
def _(device):
    from torchvision.models import vgg16
    from torchvision.models import VGG16_Weights

    # load the VGG16 network *pre-trained* on the ImageNet dataset
    weights = VGG16_Weights.DEFAULT
    vgg_model = vgg16(weights=weights)
    vgg_model.to(device)
    return vgg_model, weights


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""No to create the new layer for our model.""")
    return


@app.cell
def _(device, nn, vgg_model):
    N_CLASSES = 1
    dog_model = nn.Sequential(vgg_model, nn.Linear(1000, N_CLASSES))

    dog_model.to(device)
    return (dog_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We can verify that the VGG layers are frozen,by looping through the model parameters and checking the `requires_grad` attribute."""
    )
    return


@app.cell
def _(dog_model):
    for idx, param in enumerate(dog_model.parameters()):
        print(idx, param.requires_grad)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""for now we do not want to train the VGG layers, we will freeze them.""")
    return


@app.cell
def _(vgg_model):
    vgg_model.requires_grad_(False)
    print("VGG16 Frozen")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""As we are now classifying only two classes, we will use the binary cross entropy loss, we will use the Adam optimizer and load our model to the device."""
    )
    return


@app.cell
def _(device, dog_model, nn, optim):
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(dog_model.parameters())
    dog_model_1 = dog_model.to(device)
    return dog_model_1, loss_function, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""The vgg model has been trained on the ImageNet dataset, this data has a specific format which we need to use. We can get the transforms from the model."""
    )
    return


@app.cell
def _(weights):
    pre_trans = weights.transforms()
    return (pre_trans,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We can read in the data (which is in JPG format and infer the labels from the folder structure) and then apply the transforms to the data. The data lives in two folders one for valid and one for train. Within this set we have a folder called bo and one called not_bo.
    We can see this here
    """
    )
    return


@app.cell
def _(DATASET_LOCATION, pathlib):
    directory = pathlib.Path(DATASET_LOCATION + "/data")
    folders = [item for item in directory.rglob("*") if item.is_dir()]
    for f in folders:
        print(f)
    return


@app.cell
def _(Dataset, Image, device, pre_trans, torch):
    import glob

    DATA_LABELS = ["bo", "not_bo"]

    class MyDataset(Dataset):
        def __init__(self, data_dir):
            self.imgs = []
            self.labels = []

            for l_idx, label in enumerate(DATA_LABELS):
                data_paths = glob.glob(data_dir + label + "/*.jpg", recursive=True)
                for path in data_paths:
                    img = Image.open(path)
                    self.imgs.append(pre_trans(img).to(device))
                    self.labels.append(torch.tensor(l_idx).to(device).float())

        def __getitem__(self, idx):
            img = self.imgs[idx]
            label = self.labels[idx]
            return img, label

        def __len__(self):
            return len(self.imgs)

    return (MyDataset,)


@app.cell
def _(DATASET_LOCATION, DataLoader, MyDataset):
    batch_size = 32

    train_path = DATASET_LOCATION + "data/presidential_doggy_door/train/"
    train_data = MyDataset(train_path)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    train_N = len(train_loader.dataset)

    valid_path = DATASET_LOCATION + "data/presidential_doggy_door/valid/"
    valid_data = MyDataset(valid_path)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    valid_N = len(valid_loader.dataset)
    return train_N, train_loader, valid_N, valid_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""we can also add colour jitter to our transforms now as we have colour input images."""
    )
    return


@app.cell
def _(device, transforms):
    IMG_WIDTH, IMG_HEIGHT = (224, 224)

    random_trans = transforms.Compose(
        [
            transforms.RandomRotation(25),
            *(
                [
                    transforms.RandomResizedCrop(
                        (IMG_WIDTH, IMG_HEIGHT), scale=(0.8, 1), ratio=(1, 1)
                    )
                ]
                if device == "cuda"
                else []
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
        ]
    )
    return (random_trans,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""as we are using BinaryCrossEntropyLoss we need to modify our batch_accuracy function to take into account the fact that we are only predicting one class."""
    )
    return


@app.cell
def _(device, torch):
    def get_batch_accuracy(output, y, N):
        zero_tensor = torch.tensor([0]).to(device)
        pred = torch.gt(output, zero_tensor)
        correct = pred.eq(y.view_as(pred)).sum().item()
        return correct / N

    return (get_batch_accuracy,)


@app.cell
def _(
    get_batch_accuracy,
    loss_function,
    optimizer,
    random_trans,
    torch,
    train_N,
    train_loader,
):
    def train(model):
        loss = 0
        accuracy = 0
        model.train()
        for x, y in train_loader:
            output = torch.squeeze(model(random_trans(x)))
            optimizer.zero_grad()
            batch_loss = loss_function(output, y)
            batch_loss.backward()
            optimizer.step()
            loss = loss + batch_loss.item()
            accuracy = accuracy + get_batch_accuracy(output, y, train_N)
        print("Train - Loss: {:.4f} Accuracy: {:.4f}".format(loss, accuracy))

    return (train,)


@app.cell
def _(get_batch_accuracy, loss_function, torch, valid_N, valid_loader):
    def validate(model):
        loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for x, y in valid_loader:
                output = torch.squeeze(model(x))
                loss = loss + loss_function(output, y.float()).item()
                accuracy = accuracy + get_batch_accuracy(output, y, valid_N)
        print("Valid - Loss: {:.4f} Accuracy: {:.4f}".format(loss, accuracy))

    return (validate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Now for the training. We will train the model for 10 epochs and then save the model."""
    )
    return


@app.cell
def _(dog_model_1, train, validate):
    epochs = 10
    for epoch in range(epochs):
        print("Epoch: {}".format(epoch))
        train(dog_model_1)
        validate(dog_model_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can now save the model and test it.""")
    return


@app.cell
def _(dog_model_1, torch):
    torch.save(dog_model_1.state_dict(), "dog_model.pth")
    return


@app.cell
def _(Image, device, dog_model_1, plt, pre_trans):
    import matplotlib.image as mpimg

    def show_image(image_path):
        image = mpimg.imread(image_path)
        plt.imshow(image)
        plt.show()

    def make_prediction(file_path):
        show_image(file_path)
        image = Image.open(file_path)
        image = pre_trans(image).to(device)
        image = image.unsqueeze(0)
        output = dog_model_1(image)
        prediction = output.item()
        return prediction

    return (make_prediction,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We can test against some of the data we used to train the model and see how well it performs."""
    )
    return


@app.cell
def _(DATASET_LOCATION, make_prediction):
    make_prediction(
        DATASET_LOCATION + "data/presidential_doggy_door/valid/bo/bo_20.jpg"
    )
    return


@app.cell
def _(DATASET_LOCATION, make_prediction):
    make_prediction(
        DATASET_LOCATION + "data/presidential_doggy_door/valid/not_bo/121.jpg"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""A Negative number means that the model is predicting the correct class so we can see that the model is working well."""
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

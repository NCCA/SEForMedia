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
    mo.md(
        r"""Now to create the new layer for our model. I have put this in a `build_dog_model` function so that each time we press **Train** we get a new, untrained final layer on top of the same pre-trained VGG16. The final [Flatten](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html) turns the `[batch, 1]` output into `[batch]`, which is the shape `BCEWithLogitsLoss` expects for our labels (previously I used `torch.squeeze` but that also removes the batch dimension if the last batch only has one image in it)."""
    )
    return


@app.cell
def _(device, nn, vgg_model):
    N_CLASSES = 1

    def build_dog_model() -> nn.Sequential:
        return nn.Sequential(
            vgg_model,
            nn.Linear(1000, N_CLASSES),
            # [batch, 1] -> [batch] so the output matches the shape of the labels
            nn.Flatten(start_dim=0),
        ).to(device)

    dog_model = build_dog_model()
    dog_model
    return build_dog_model, dog_model


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
        r"""As we are now classifying only two classes, we will use the binary cross entropy loss. The Adam optimizer is created with the model when we train."""
    )
    return


@app.cell
def _(nn):
    loss_function = nn.BCEWithLogitsLoss()
    return (loss_function,)


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
                if device.type == "cuda"
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
        r"""
    ## Training

    I use `Utils.train_epoch` and `Utils.evaluate` (see [Utils/training.py](../Utils/training.py)) as in the ASL notebooks, passing the random transforms in as the `transform` so they are only applied to the training data.

    As we are using `BCEWithLogitsLoss` there is only one output per image rather than one per class, so taking the `argmax` to get the prediction won't work. Instead `Utils.binary_predict` says the image is class 1 (`not_bo`) if the output is above 0, which is the same as the sigmoid of the output being above 0.5. We pass this in as the `predict` parameter.

    I keep the weights from the epoch with the lowest validation loss. Only the new final layer is trained as VGG16 is frozen, so each epoch is fairly quick even though the model is large. Choose the settings and press **Train**, each press starts with a new final layer.
    """
    )
    return


@app.cell
def _(mo):
    training_settings = mo.ui.dictionary(
        {
            "epochs": mo.ui.number(start=1, stop=100, value=10, label="Epochs"),
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
    Utils,
    build_dog_model,
    device,
    loss_function,
    mo,
    optim,
    random_trans,
    torch,
    train_loader,
    training_settings,
    valid_loader,
):
    mo.stop(training_settings.value is None, mo.md("Press Train above to begin."))

    torch.manual_seed(42)
    model = build_dog_model()
    _optimizer = optim.Adam(
        model.parameters(), lr=training_settings.value["learning_rate"]
    )
    history = []
    _best_loss = float("inf")
    best_epoch = 0
    _best_weights = None
    for _epoch in mo.status.progress_bar(
        range(training_settings.value["epochs"]), title="Training"
    ):
        _train_loss, _train_accuracy = Utils.train_epoch(
            model,
            train_loader,
            loss_function,
            _optimizer,
            device,
            transform=random_trans,
            predict=Utils.binary_predict,
        )
        _valid_loss, _valid_accuracy = Utils.evaluate(
            model, valid_loader, loss_function, device, predict=Utils.binary_predict
        )
        history.append((_train_loss, _valid_loss, _train_accuracy, _valid_accuracy))
        if _valid_loss < _best_loss:
            _best_loss = _valid_loss
            best_epoch = _epoch + 1
            _best_weights = Utils.copy_weights(model)
        print(
            f"Epoch {_epoch + 1}: train loss {_train_loss:.3f}, validation loss {_valid_loss:.3f}, validation accuracy {_valid_accuracy:.1%}"
        )
    if _best_weights is not None:
        model.load_state_dict(_best_weights)
        _summary = f"Training finished on **{device}**. Restored weights from epoch **{best_epoch}**."
    else:
        _summary = f"Training finished on **{device}**, but the validation loss never improved (it is probably NaN). Try a lower learning rate."
    mo.md(_summary)
    return history, model


@app.cell
def _(history, plt):
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We can now save the model and test it. Saving overwrites `dog_model.pth` so it only happens when you press the button."""
    )
    return


@app.cell
def _(mo):
    save_btn = mo.ui.run_button(label="Save model")
    save_btn
    return (save_btn,)


@app.cell
def _(mo, model, save_btn, torch):
    mo.stop(not save_btn.value)
    torch.save(model.state_dict(), "dog_model.pth")
    mo.md("Saved `dog_model.pth`.")
    return


@app.cell
def _(Image, device, model, plt, pre_trans, torch):
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
        model.eval()
        with torch.inference_mode():
            output = model(image)
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
        r"""The labels are 0 for `bo` and 1 for `not_bo`, so a negative number means the model thinks it is Bo and a positive number means it is not. If the first image gives a negative number and the second a positive one the model is working well."""
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

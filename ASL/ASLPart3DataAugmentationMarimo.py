#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full", app_title="ASL CNN Data Augmentation")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ASL CNN Data Augmentation

    In the [previous notebook](ASLPart2CNNMarimo.py), we built a CNN model to classify the ASL dataset. In this notebook, we will look at how we can use data augmentation to improve the performance of our model.

    The validation accuracy is still lagging behind the training accuracy, which is a sign of overfitting and the model getting confused.

    One method we can use to improve the model's performance is data augmentation. Data augmentation is a technique used to artificially increase the size of the training dataset by applying various transformations to the existing images. This helps the model generalize better and reduces overfitting.

    The increase in size gives the model more images to learn from while training. The increase in variance helps the model ignore unimportant features and select only the features that are truly important in classification, allowing it to generalize better.

    We will start with the exact same data and model as in the previous notebook and then apply data augmentation to see if it improves the model's performance as part of the training process.

    ## Loading data

    The following code was outlined in the previous two examples.
    """)
    return


@app.cell
def _():
    import pathlib
    import sys

    import matplotlib.pyplot as plt
    import pandas as pd
    import torch
    import torch.nn as nn
    import torchvision.transforms.functional as F

    # Visualization tools
    import torchvision.transforms.v2 as transforms
    from PIL import Image
    from torch.optim import Adam
    from torch.utils.data import DataLoader, Dataset

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

    valid_data = ASLImages(valid_df)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)
    return train_loader, valid_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
    """)
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
    mo.md(r"""
    The forward method just needs to be updated to use the class variables that we have created, so it will evaluate the same as before.
    """)
    return


@app.cell
def _(ConvBlock, IMAGE_CHANNELS, nn):
    flattened_img_size = 75 * 3 * 3
    N_CLASSES = 25

    def build_model() -> nn.Sequential:
        # Input 1 x 28 x 28
        return nn.Sequential(
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

    loss_function = nn.CrossEntropyLoss()
    build_model()
    return build_model, loss_function


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # [Torchvision Transforms](https://pytorch.org/vision/0.9/transforms.html)

    We have used these before for simple transforms such as scaling, we will now look in more depth at some other transforms we can apply to augment our data and provide variance to the input data.

    We will start by extracting an image from the data to process and see what the results are.
    """)
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
    mo.md(r"""
    We can define a simple image plot functions to display the images.
    """)
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
    mo.md(r"""
    [RandomResizeCrop](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.RandomResizedCrop)

    This transform will apply both a crop and a resize to the image, the crop will be random and the resize will be to the size specified. It needs to know the aspect ratio of the image to be able to crop it correctly, however in our case it is 1:1 as the image is square.
    """)
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
    mo.md(r"""
    You can see the results are very subtle, but the image has changed enough to provide variance to the input data.

    ## [RandomHorizontalFlip](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.RandomHorizontalFlip)

    We can also randomly flip our images both horizontally and vertically, this will depend upon the data and what we are trying to do. In our case we are only going to flip the images horizontally, as this is the only way that the ASL data can be flipped and still be valid. (Note ASL is typically done with the dominant hand so both left or right handed is fine).
    """)
    return


@app.cell
def _(plot_multiple, transforms, x_0):
    _trans = transforms.Compose([transforms.RandomHorizontalFlip()])
    plot_multiple(8, _trans, x_0).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [RandomRotation](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.RandomRotation)

    We can also rotate the images by a random amount, but like the flipping we must be careful with this as the ASL data is very specific and we don't want to rotate the images too much as it will make the data invalid. We will limit the rotation to 20 degrees in either direction.
    """)
    return


@app.cell
def _(plot_multiple, transforms, x_0):
    _trans = transforms.Compose([transforms.RandomRotation(20)])
    plot_multiple(8, _trans, x_0).show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice how this has added borders to the image, this is because the image has been rotated and the corners are now empty and set to an empty (black) pixel value.

    ## [ColorJitter](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.ColorJitter)

    The `ColorJitter` transform has 4 arguments:
        - brightness
        - contrast
        - saturation
        - hue

    The saturation and hue apply to color images, so we will only use the first 2 for this example as we are using grayscale images.
    """)
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
    mo.md(r"""
    ## [Compose](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.Compose)

    It is possible to combine all of these transforms into a single transform using the `Compose` class, this will apply all of the transforms in the order that they are passed in.
    """)
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
    mo.md(r"""
    ## Training

    We will now train our model as before, however now we can pass in the transformation to our model, this will then apply it to each image before it is passed into the model.

    As we are re-using a lot of code now I have moved the training and evaluation functions into the Utils module (see [Utils/training.py](../Utils/training.py)), they are the same ones we used in the last notebook. `Utils.train_epoch` has an optional `transform` parameter which is applied to each batch before it goes into the model. `RandomResizedCrop` doesn't work on the mac (MPS) backend so it is left out there.

    The validation remains the same as before and we don't add any transformations to the validation data, as we want to see how the model performs on the original data.

    Augmentation makes each epoch harder for the model, so it needs more of them, I have set the default to 20. Choose the settings and press **Train**, each press starts a fresh model.
    """)
    return


@app.cell
def _(IMAGE_HEIGHT, IMAGE_WIDTH, device, transforms):
    random_transforms = transforms.Compose(
        [
            transforms.RandomRotation(30),
            *(
                [
                    transforms.RandomResizedCrop(
                        (IMAGE_WIDTH, IMAGE_HEIGHT), scale=(0.9, 1), ratio=(1, 1)
                    )
                ]
                if device.type != "mps"
                else []
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.5),
        ]
    )
    return (random_transforms,)


@app.cell
def _(mo):
    training_settings = mo.ui.dictionary(
        {
            "epochs": mo.ui.number(start=1, stop=100, value=20, label="Epochs"),
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
    random_transforms,
    torch,
    train_loader,
    training_settings,
    valid_loader,
):
    mo.stop(training_settings.value is None, mo.md("Press Train above to begin."))

    torch.manual_seed(42)
    model = build_model().to(device)
    _model_compiled = torch.compile(model) if device.type == "cuda" else model
    _optimizer = Adam(model.parameters(), lr=training_settings.value["learning_rate"])
    history = []
    _best_loss = float("inf")
    best_epoch = 0
    _best_weights = None
    for _epoch in mo.status.progress_bar(
        range(training_settings.value["epochs"]), title="Training"
    ):
        _train_loss, _train_accuracy = Utils.train_epoch(
            _model_compiled,
            train_loader,
            loss_function,
            _optimizer,
            device,
            transform=random_transforms,
        )
        _valid_loss, _valid_accuracy = Utils.evaluate(
            _model_compiled, valid_loader, loss_function, device
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can plot our results as before and see how the model performs.
    """)
    return


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
    mo.md(r"""
    The results are much better, and it is not showing the signs of overfitting we had before. The validation accuracy is now much closer to the training accuracy and the model is performing much better.

    The training accuracy may be lower, and that's ok. Compared to before, the model is being exposed to a much larger variety of data.

    ## Saving the model

    We save both the `state_dict` and the full model, the [real time capture demo](RealTimeCapture/) loads `asl_model_full.pth`. Saving overwrites the previous files, so it only happens when you press the button. The same button exports the ONNX version at the end of this notebook.
    """)
    return


@app.cell
def _(mo):
    save_btn = mo.ui.run_button(label="Save model")
    save_btn
    return (save_btn,)


@app.cell
def _(mo, model, save_btn, torch):
    mo.stop(not save_btn.value, mo.md("Press Save model to save the trained model."))
    # Save the model
    torch.save(model.state_dict(), "asl_model.pth")
    # Also save the full model
    torch.save(model, "asl_model_full.pth")
    mo.md("Saved `asl_model.pth` and `asl_model_full.pth`.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Testing the model.

    We can now test the model as before and see how it performs on the test data.
    """)
    return


@app.cell
def _(model, plt, torch, valid_loader):
    model.eval()
    alphabet = "abcdefghijklmnopqrstuvwxy"

    with torch.no_grad():
        x, y = next(iter(valid_loader))
        output = model(x)
        _pred = output.argmax(dim=1, keepdim=True)
        num_images = 10
        plt.figure(figsize=(20, 20))
        for _i in range(num_images):
            plt.subplot(1, num_images, _i + 1)
            plt.title(
                f" {alphabet[y[_i].item()]} {alphabet[_pred[_i].item()]}",
                fontdict={"fontsize": 30},
            )
            plt.axis("off")
            plt.imshow(x[_i].cpu().numpy().reshape(28, 28), cmap="gray")
    return (alphabet,)


@app.cell(hide_code=True)
def _(mo):
    _image = mo.image("public/amer_sign2.png")
    mo.md(
        rf"""
    # More Testing

    At present all the data we have used is from the same dataset, we can now test the model on some new data that it has not seen before.  The dataset we downloaded has the following image

    {_image}


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
            # Define the preprocessing transformations
    preprocess_trans = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.Resize((28, 28)),  # Resize to 28x28
            transforms.ToImage(),  # Ensure input is treated as an image
            transforms.ToDtype(
                torch.float32, scale=True
            ),  # Convert to float32 and scale to [0,1]
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
    mo.md(r"""
    <!-- Now lets feed this into our model and see how it does. The signs are in alphabetical order, so the first sign is A, the second is B and so on. -->
    """)
    return


@app.cell
def _(alphabet, device, model, tensor_images, torch):
    model.eval()
    with torch.inference_mode():
        for _i in range(23):
            _pred = model(tensor_images[_i].unsqueeze(0).to(device))
            print(f"{alphabet[_pred.argmax().item()]} ", sep="", end="")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Saving to ONXX

    The ONXX ( Open Neural Network Exchange.) format is a way of exchanging models in an open format. Torch allows us to export using this as well as it's own format. We need to ensure the onxx tools are installed (```uv add onnx onnxruntime onnxscript```) in our own projects.

    We need to ensure everything is on the same device, so in the case I copy the model to the cpu before saving. I use a [deepcopy](https://docs.python.org/3/library/copy.html#copy.deepcopy) as `model.to("cpu")` would move the trained model itself and break the cells above if they re-run. As this overwrites `asl_model.onnx` it only runs when you press **Save model** above.
    """)
    return


@app.cell
def _(IMAGE_CHANNELS, mo, model, save_btn, torch):
    import copy

    mo.stop(not save_btn.value)
    # Create a dummy input with the correct shape
    dummy_input = torch.randn(1, IMAGE_CHANNELS, 28, 28)
    onxx_model = copy.deepcopy(model).to("cpu").eval()
    # Export the model
    exported_model = torch.onnx.export(
        onxx_model,
        dummy_input,
        "asl_model.onnx",
        dynamo=True,
        input_names=["input"],
        output_names=["output"],
    )
    print(exported_model)
    return


@app.cell
def _(mo):
    _image = mo.image("public/asl_model.onnx.png")
    mo.md(
        rf"""
    This will save the file to disk, and you can see the output of the model if you print it. We can now use a tool like https://netron.app/ to load this back in and visualize the model, this will also allow us to generate an image which is useful for write ups etc.
    {_image}

    """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full", app_title="MNIST Digits")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # MNIST Digits

    In this notebook we are going to train a neural network to recognize handwritten digits. This is the  "Hello World" of deep learning: training a deep learning model to correctly classify hand-written digits.

    In the previous [notebook](TheMNISTDataSet.ipynb) we downloaded the MNIST dataset, which is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.  We will re-use this data (downloaded either to your local hard drive or /transfer) to train a neural network to recognize the digits.

    We will start by importing the necessary libraries, including our Utils module.
    """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import struct
    import sys
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.utils.data import Dataset, DataLoader

    # Visualization tools
    import torchvision.transforms.v2 as transforms

    sys.path.append("../")
    import Utils

    print(f"{Utils.in_lab()=}")
    return (
        Adam,
        DataLoader,
        Dataset,
        Utils,
        nn,
        np,
        plt,
        struct,
        torch,
        transforms,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## GPU Support

    In this notebook we will use the GPU to train our model, we can use the function from our Utils module to check if the GPU is available and set this as the device to use for our data.
    """
    )
    return


@app.cell
def _(Utils):
    device = Utils.get_device()
    print(device)
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Image classification

    The approach we are going to take with this example is to load a set of know images and their labels, and train a neural network to learn the relationship between the images and their labels.

    We will use a neural network and a trial and error system to begin to recognize the patterns in the images. The images are small (28x28 pixels) and the neural network will learn to recognize the patterns in the images that are associated with the digits.

    We have a set of 60,000 images to train the network and a separate set of 10,000 images to test the network and on each step of the training process we will check the accuracy of the network on the test set.

    In the previous notebook we created two functions for loading the data and labels so we will use these functions to load the data and labels for the training and test sets.
    """
    )
    return


@app.cell
def _(np, struct):
    def load_mnist_labels(filename: str) -> np.ndarray:
        with open(filename, "rb") as f:
            magic, num = struct.unpack(">II", f.read(8))
            labels = np.fromfile(f, dtype=np.uint8)
            if len(labels) != num:
                raise ValueError(f"Expected {num} labels, but got {len(labels)}")
        return labels

    def load_mnist_images(filename: str) -> np.ndarray:
        with open(filename, "rb") as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
            if len(images) != num:
                raise ValueError(f"Expected {num} images, but got {len(images)}")
        return images

    return load_mnist_images, load_mnist_labels


@app.cell
def _(Utils, load_mnist_images, load_mnist_labels):
    DATASET_LOCATION = ""
    if Utils.in_lab():
        DATASET_LOCATION = "/transfer/MNIST/"
    else:
        DATASET_LOCATION = "./MNIST/"

    train_labels = load_mnist_labels(DATASET_LOCATION + "train-labels-idx1-ubyte")
    test_labels = load_mnist_labels(DATASET_LOCATION + "t10k-labels-idx1-ubyte")

    print(len(train_labels), len(test_labels))
    print(train_labels[0], test_labels[0])

    # We can now load the images from both the datasets.
    train_images = load_mnist_images(DATASET_LOCATION + "train-images-idx3-ubyte")
    test_images = load_mnist_images(DATASET_LOCATION + "t10k-images-idx3-ubyte")
    return test_images, test_labels, train_images, train_labels


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""To see the images we can define a simple function to display the images. We will use the matplotlib library to display the images."""
    )
    return


@app.cell
def _(np, plt, train_images, train_labels):
    def display_image(image: np.array, label: str) -> None:
        plt.figure(figsize=(1, 1))
        plt.title(f"Label : {label}")
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.show()

    # We can now display the first image from the training dataset.

    display_image(train_images[0], train_labels[0])
    print(type(train_images[0]))
    print(train_images[0].shape)
    print(train_images[0].dtype)
    return (display_image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    If we look at the data it is stored in a numpy array of 28,28 and a single unsigned char data type. We need to transform this data into the correct type for machine learning. In particular we need to convert the data into a Tensor of type float32, then we need to batch the data into a DataLoader.

    We can use the torchvision library to transform our data as follows.
    """
    )
    return


@app.cell
def _(torch, train_images, transforms):
    trans = transforms.Compose(
        [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
    )
    tensor = trans(train_images[0])
    print(tensor.shape)
    print(tensor.dtype)
    print(tensor.min(), tensor.max())
    print(tensor.device)
    return (tensor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""By default the data for this tensor is processed on the CPU, we can convert it to run on the GPU by using the .to(device) method."""
    )
    return


@app.cell
def _(device, tensor):
    tensor.to(device).device
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Preparing the Data for Training

    Earlier, we created a `trans` variable to convert our ndarray to a tensor. [Transforms](https://pytorch.org/vision/stable/transforms.html) are a group of torchvision functions that can be used to transform a dataset.

    At present our train_images and test_images are numpy arrays. We need to convert them to tensors. We can do this using the `trans` variable we created earlier.
    """
    )
    return


@app.cell
def _(device, test_images, test_labels, torch, train_images, train_labels):
    train_images_tensor = torch.tensor(train_images, dtype=torch.float32).to(device)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.uint8).to(device)

    test_images_tensor = torch.tensor(test_images, dtype=torch.float32).to(device)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.uint8).to(device)
    return (
        test_images_tensor,
        test_labels_tensor,
        train_images_tensor,
        train_labels_tensor,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Dataloaders

    We can use the DataLoader class from the torch.utils.data module to create a DataLoader for our training and test data, you can think of this as batching images into smaller groups for training.

    First we need to create a custom class to hold our data, we can do this by creating a subclass of the Dataset class from the torch.utils.data module. We need to implement the __len__ and __getitem__ methods to return the length of the dataset and the data and label for a given index.
    """
    )
    return


@app.cell
def _(Dataset):
    # Custom dataset class
    class DigitsDataset(Dataset):
        def __init__(self, images_tensor, labels_tensor):
            self.images_tensor = images_tensor
            self.labels_tensor = labels_tensor

        def __len__(self):
            return len(self.labels_tensor)

        def __getitem__(self, idx):
            image = self.images_tensor[idx]
            label = self.labels_tensor[idx]
            return image, label

    return (DigitsDataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We could show our models the entire dataset at once. Not only does this take a lot of computational resources, but [research shows](https://arxiv.org/pdf/1804.07612) using a smaller batch of data is more efficient for model training.

    For example, if our `batch_size` is 32, we will train our model by shuffling the deck and drawing 32 cards. We do not need to shuffle for validation as the model is not learning, but we will still use a `batch_size` to prevent memory errors.

    The batch size is something the model developer decides, and the best value will depend on the problem being solved. Research shows 32 or 64 is sufficient for many machine learning problems and is the default in some machine learning frameworks, so we will use 32 here.
    """
    )
    return


@app.cell
def _(
    DataLoader,
    DigitsDataset,
    test_images_tensor,
    test_labels_tensor,
    train_images_tensor,
    train_labels_tensor,
):
    batch_size = 32

    train_data = DigitsDataset(train_images_tensor, train_labels_tensor)
    valid_data = DigitsDataset(test_images_tensor, test_labels_tensor)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    return train_loader, valid_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Creating a Model

    Neural networks are composed of layers where each layer performs a mathematical operation on the data it receives before passing it to the next layer. To start, we will create a "Hello World" level model made from 4 components:

    1. A [Flatten](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html) used to convert n-dimensional data into a vector.
    2. An input layer, the first layer of neurons
    3. A hidden layer, another layer of neurons "hidden" between the input and output
    4. An output layer, the last set of neurons which returns the final prediction from the model

    We will use a variable called layers to store the layers of our model.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Flatten the image

    The first thing we need to do is convert the image from a 28x28  array into a flat 1d tensor. We saw the images had 3 dimensions: `C x H x W`. To flatten an image means to combine all of these images into 1 dimension. Let's say we have a tensor like the one below.
    """
    )
    return


@app.cell
def _(nn, torch):
    test_matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(test_matrix)
    print(nn.Flatten()(test_matrix))
    return (test_matrix,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""You will notice nothing happened, this is because neural networks expect to receive a batch of data. Currently, the Flatten layer sees three vectors as opposed to one 2d matrix. To fix this, we can "batch" our data by adding an extra pair of brackets. Since `test_matrix` is now a tensor, we can do that with the shorthand below. `None` adds a new dimension where `:` selects all the data in a tensor."""
    )
    return


@app.cell
def _(nn, test_matrix):
    batch_test_matrix = test_matrix[None, :]
    print(batch_test_matrix)
    print(nn.Flatten()(batch_test_matrix))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The Input Layer

    The input layer is the first layer of neurons in the neural network. It is responsible for receiving the input data and passing it to the next layer.

    This layer will be *densely connected*, meaning that each neuron in it, and its weights, will affect every neuron in the next layer.

    In order to create these weights, Pytorch needs to know the size of our inputs and how many neurons we want to create.
    Since we've flattened our images, the size of our inputs is the number of channels, number of pixels vertically, and number of pixels horizontally multiplied together.
    """
    )
    return


@app.cell
def _():
    input_size = 1 * 28 * 28
    print(input_size)
    return (input_size,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Choosing the correct number of neurons is what puts the "science" in "data science" as it is a matter of capturing the statistical complexity of the dataset. For now, we will use `512` neurons. Try playing around with this value later to see how it affects training and to start developing a sense for what this number means.

    We will learn more about activation functions later, but for now, we will use the [relu](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html) activation function, which in short, will help our network to learn how to make more sophisticated guesses about data than if it were required to make guesses based on some strictly linear function.
    """
    )
    return


@app.cell
def _(input_size, nn):
    layers_1 = [nn.Flatten(), nn.Linear(input_size, 512), nn.ReLU()]
    layers_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The hidden layer

    A hidden layer is a layer of neurons between the input and output layers. It is called "hidden" because it is not directly exposed to the input data and the output predictions. We will cover why we combine multiple layers in another lecture, but for now, we will add a hidden layer to our model.

    As with the previous layers, the shape of the data is important. [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) needs to know the shape of the data being passed to it. Each neuron in the previous layer will compute one number, so the number of inputs into the hidden layer is the same as the number of neurons in the previous later.
    """
    )
    return


@app.cell
def _(input_size, nn):
    layers_2 = [
        nn.Flatten(),
        nn.Linear(input_size, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
    ]
    layers_2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The Output Layer

    The output layer is the final layer of neurons in the neural network. It is responsible for producing the output of the model. This will be a tensor of length 10, where each element represents the probability of the input image being a particular digit.

    We will not assign the `relu` function to the output layer. Instead, we will apply a `loss function` covered in the next section.
    """
    )
    return


@app.cell
def _(input_size, nn):
    n_classes = 10
    layers_3 = [
        nn.Flatten(),
        nn.Linear(input_size, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, n_classes),
    ]
    layers_3
    return (layers_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Building the Model

    A [Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) model expects a sequence of arguments, not a list, so we can use the [* operator](https://docs.python.org/3/reference/expressions.html#expression-lists) to unpack our list of layers into a sequence. We can print the model to verify these layers loaded correctly.

    Each layer in `layers_3` is an object holding its own weights, so if we built two models from the same list they would share the weights. I want a fresh, untrained model each time we press **Train** below, so I have wrapped the same layers in a `build_model` function which makes new ones each time it is called.
    """
    )
    return


@app.cell
def _(input_size, nn):
    def build_model(n_classes: int = 10) -> nn.Sequential:
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes),
        )

    build_model()
    return (build_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Training the Model

    Now that we have prepared training and validation data, and a model, it's time to train our model with our training data, and verify it with its validation data.

    This is called fitting and we need to add two functions to help us with this process.

    ## Loss and Optimization

    The loss function measures the difference between the model's prediction and the target. The optimizer updates the model's parameters to reduce the loss. In this example we will use  a loss function called [CrossEntropy](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) which is designed to grade if a model predicted the correct category from a group of categories.
    """
    )
    return


@app.cell
def _(nn):
    loss_function = nn.CrossEntropyLoss()
    return (loss_function,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""The optimizer is used to update the model's weights based on the data it sees and the loss function. We will use the [Adam](https://pytorch.org/docs/stable/optim.html) optimizer, which is a popular optimizer in deep learning. It needs to know the models parameters so it can update them, so we create it in the training cell below at the same time as the model."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Accuracy

    When training a model, it is important to know how well it is performing. We can calculate the accuracy of the model by comparing the model's prediction to the actual target. The simplest way to do this is to compare the model's prediction to the target and calculate the percentage of correct predictions.

    We need to generate our own accuracy functions as these are typically dependent on the problem being solved.

    We need to compare the number of correct classifications compared to the total number of predictions made. Since we're showing data to the model in batches, our accuracy can be calculated along with these batches.

    We typically use N as a postfix to denote the number of samples in a dataset. We can use this to calculate the accuracy of our model.
    """
    )
    return


@app.cell
def _(train_loader, valid_loader):
    train_N = len(train_loader.dataset)
    valid_N = len(valid_loader.dataset)
    return train_N, valid_N


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We can then accumulate the accuracy for each batch and divide by the total number of samples to get the overall accuracy."""
    )
    return


@app.function
def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The Training Loop

    We now generate a function that will train the model for a single epoch. This is similar to the approch we used in the the Linear example but we have added a few more steps to calculate the accuracy of the model.

    The loss function gives us the mean loss for each batch, so I multiply it by the number of images in the batch and divide by `train_N` at the end. This gives the mean loss per image for the whole epoch, which we can compare with the validation loss even though the two datasets are different sizes.

    The function takes the model and optimizer as parameters and returns the loss and accuracy, rather than printing them, so we can keep a history and plot it.
    """
    )
    return


@app.cell
def _(device, loss_function, torch, train_N, train_loader):
    def train(
        model: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) -> tuple[float, float]:
        loss = 0
        accuracy = 0
        model.train()
        for x, y in train_loader:
            x, y = (x.to(device), y.to(device))
            output = model(x)
            optimizer.zero_grad()
            batch_loss = loss_function(output, y)
            batch_loss.backward()
            optimizer.step()
            loss = loss + batch_loss.item() * len(y)
            accuracy = accuracy + get_batch_accuracy(output, y, train_N)
        return loss / train_N, accuracy

    return (train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Validation

    Once the model has done a training step we need to see how close we are to the correct answer. We can do this by running the model on the validation data and calculating the loss and accuracy of the model on the validation data. Again we will use a function to do this.

    [torch.inference_mode](https://pytorch.org/docs/stable/generated/torch.inference_mode.html) turns off gradient tracking as we are not going to update the weights here.
    """
    )
    return


@app.cell
def _(device, loss_function, torch, valid_N, valid_loader):
    def validate(model: torch.nn.Module) -> tuple[float, float]:
        loss = 0
        accuracy = 0
        model.eval()
        with torch.inference_mode():
            for x, y in valid_loader:
                x, y = (x.to(device), y.to(device))
                output = model(x)
                loss = loss + loss_function(output, y).item() * len(y)
                accuracy = accuracy + get_batch_accuracy(output, y, valid_N)
        return loss / valid_N, accuracy

    return (validate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The Training loop

    We can now create a training loop that will train the model for a number of epochs. An `epoch` is one complete pass through the entire dataset.

    PyTorch 2.0 introduced the ability to [compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) the model using `torch.compile` which can give faster performance. The compiled model shares its weights with the original, so we train the compiled one and keep `model` for saving later (the names in the compiled model's `state_dict` are changed).

    Choose the number of epochs and the learning rate and press **Train**. Each time you press it a fresh model is built, so you can try different settings and compare. I would start with 5 epochs to see how it learns.
    """
    )
    return


@app.cell
def _(mo):
    training_settings = mo.ui.dictionary(
        {
            "epochs": mo.ui.number(start=1, stop=100, value=5, label="Epochs"),
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
    build_model,
    device,
    mo,
    torch,
    train,
    training_settings,
    validate,
):
    mo.stop(training_settings.value is None, mo.md("Press Train above to begin."))

    torch.manual_seed(42)
    model = build_model().to(device)
    model_compiled = torch.compile(model)
    _optimizer = Adam(model.parameters(), lr=training_settings.value["learning_rate"])
    history = []
    for _epoch in mo.status.progress_bar(
        range(training_settings.value["epochs"]), title="Training"
    ):
        _train_loss, _train_accuracy = train(model_compiled, _optimizer)
        _valid_loss, _valid_accuracy = validate(model_compiled)
        history.append((_train_loss, _valid_loss, _train_accuracy, _valid_accuracy))
        print(
            f"Epoch {_epoch + 1}: train loss {_train_loss:.3f}, validation loss {_valid_loss:.3f}, validation accuracy {_valid_accuracy:.1%}"
        )
    mo.md(f"Training finished on **{device}**.")
    return history, model, model_compiled


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""As we kept the history we can plot the loss and accuracy for each epoch. If the validation loss starts going up whilst the training loss keeps going down the model is overfitting."""
    )
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
    mo.md(
        r"""We can see that we are quite close (nearly 100%) so we can try and test our model on some existing data."""
    )
    return


@app.cell
def _(device, model_compiled, test_images_tensor, torch):
    model_compiled.eval()
    with torch.inference_mode():
        prediction = model_compiled(test_images_tensor[0].to(device).unsqueeze(0))
    prediction
    return (prediction,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    There should be ten numbers, each corresponding to a different output neuron. Thanks to how the data is structured, the index of each number matches the corresponding handwritten number. The 0th index is a prediction for a handwritten 0, the 1st index is a prediction for a handwritten 1, and so on.

    We can use the `argmax` function to find the index of the highest value.
    """
    )
    return


@app.cell
def _(display_image, prediction, test_images):
    print(prediction.argmax(dim=1, keepdim=True))
    display_image(test_images[0], prediction.argmax(dim=1, keepdim=True).item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This seems to have worked, we should now save the model so we can use it later.

    ## Saving the Model

    We can save the model using the ```torch.save``` function. We can save the model to a file called `mnist_model.pth` in the current directory. Note we save the original uncompiled model for simplicity as when compiled names get changed. We can always re-compile later if required.

    To prove the save worked I build a brand new model with `build_model`, load the saved weights into it and make the same prediction. Saving overwrites any previous file, so it only happens when you press the button.
    """
    )
    return


@app.cell
def _(mo):
    save_btn = mo.ui.run_button(label="Save model")
    save_btn
    return (save_btn,)


@app.cell
def _(
    build_model,
    device,
    display_image,
    mo,
    model,
    save_btn,
    test_images,
    test_images_tensor,
    torch,
):
    mo.stop(not save_btn.value, mo.md("Press Save model to save and reload."))
    torch.save(model.state_dict(), "mnist_model.pth")
    model2 = build_model()
    model2.load_state_dict(torch.load("mnist_model.pth"))
    model2.to(device)
    model2.eval()
    with torch.inference_mode():
        prediction_1 = model2(test_images_tensor[0].to(device).unsqueeze(0))
    torch.save(test_images_tensor[0], "test_image.pth")
    display_image(test_images[0], prediction_1.argmax(dim=1, keepdim=True).item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""In the above case we saved only the model state dictionary. We can also save the entire model including the architecture and the state dictionary. This is a little more rigid as it requires the model to be defined in the same way when it is loaded, however we don't also need to re-create the model architecture when we load the model."""
    )
    return


@app.cell
def _(
    device,
    display_image,
    mo,
    model,
    save_btn,
    test_images,
    test_images_tensor,
    torch,
):
    mo.stop(not save_btn.value)
    torch.save(model, "mnist_model_full.pth")
    model3 = torch.load("mnist_model_full.pth", weights_only=False)
    model3.to(device)
    model3.eval()
    with torch.inference_mode():
        prediction_2 = model3(test_images_tensor[0].to(device).unsqueeze(0))
    display_image(test_images[0], prediction_2.argmax(dim=1, keepdim=True).item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We can now use this model to classify new images of digits. We will demonstrate this in a stand alone Qt applications."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Conclusion

    In this notebook we have trained a neural network to recognize handwritten digits. We have used the MNIST dataset to train the model and have used a simple neural network with a single hidden layer to classify the images. We have trained the model for 5 epochs and have achieved an accuracy of nearly 100%. We have saved the model so we can use it later.

    This is basically the process we will use for all of our machine learning models. We will load the data, create a model, train the model and then save the model. We can then use the model to classify new data.
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

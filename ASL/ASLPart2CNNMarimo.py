#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.15.2"
app = marimo.App(
    width="full",
    app_title="ASL Processing Part 2 Convolutional Neural Network",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # ASL Processing Part 2

    In the [previous notebook](./ASLPart1.ipynb), we have seen how to preprocess the data and train a model, the model began to overfit after 10 epochs. In this notebook, we will see how to use data augmentation to improve the model's performance.

    We will use the same data set as before, if the data set is not present run the first notebook to download the data set.
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
        device,
        nn,
        pd,
        plt,
        string,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We can load our data set  using pandas as before, however this time we will want to format it into a different shape (28x28 pixels) so we can run image processing on it. This is because most image processing algorithms are designed to work with images, and not flattened arrays."""
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
        r"""
    ## Data processing

    To demonstrate the data processing, we will use the first image in the data set as a sample and re-shape it to 28x28 pixels.
    """
    )
    return


@app.cell
def _(train_df):
    sample_df = train_df.head().copy()  # Grab the top 5 rows
    sample_df.pop("label")
    sample_x = sample_df.values
    print(sample_x.shape)
    return (sample_x,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In this format we don't have pixel proximity locations which convulational neural networks use to learn patterns. We will use the `reshape` function from NumPy to convert the image to 28x28 pixels.


    Note that for the first convolution layer of our model, we need to have not only the height and width of the image, but also the number of color channels (1 in our case as greyscale images).

    That means that we need to convert the current shape `(5, 784)` to `(5, 1, 28, 28)`. With [NumPy](https://numpy.org/doc/stable/index.html) arrays, we can pass a `-1` for any dimension we wish to remain the same.

    Which is 5 image of 1 channel   with 28x28 pixels.
    """
    )
    return


@app.cell
def _(plt, sample_x, string, train_df):
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    IMAGE_CHANNELS = 1
    sample_x_1 = sample_x.reshape(-1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    print(sample_x_1.shape)

    def plot_image(plt, index, images, labels, num_images):
        image = images.reshape(28, 28)
        label = labels
        plt.subplot(1, num_images, index + 1)
        plt.title(label, fontdict={"fontsize": 30})
        plt.axis("off")
        plt.imshow(image, cmap="gray")

    plt.figure(figsize=(10, 10))
    alphabet = string.ascii_letters[:25]
    for i in range(len(sample_x_1)):
        plot_image(plt, i, sample_x_1[i], alphabet[train_df["label"][i]], 5)
    plt.show()
    return IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, alphabet


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Creating a Dataset

    We can create our own data set class using the same method as outlined above, we will now generate the class to do this, and then use the DataLoader class to load the data in batches.

    As the data is in a dataframe we can set the df.copy method to copy the data into a new dataframe, this will allow us to manipulate the data without changing the original data.
    """
    )
    return


@app.cell
def _(Dataset, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, device, torch):
    class ASLImages(Dataset):
        def __init__(self, base_df):
            x_df = base_df.copy()
            y_df = x_df.pop("label")
            x_df = x_df.values / 255
            x_df = x_df.reshape(-1, IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT)
            self.xs = torch.tensor(x_df).float().to(device)
            self.ys = torch.tensor(y_df).to(device)

        def __getitem__(self, idx):
            _x = self.xs[idx]
            _y = self.ys[idx]
            return (_x, _y)

        def __len__(self):
            return len(self.xs)

    return (ASLImages,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We can now build our Dataloaders using the DataLoader class from PyTorch. Remember to set the train data to shuffle so the model does not learn the order of the data. We don't need to do this for the validation data as we are not training on it."""
    )
    return


@app.cell
def _(ASLImages, DataLoader, train_df, valid_df):
    BATCH_SIZE = 32

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
    ## Creating a Convolution model

    A Convolutional Neural Network (CNN) is a type of deep learning model specifically designed to process and analyze visual data, like images and videos. CNNs are highly effective at tasks like image classification, object detection, and facial recognition because they can learn spatial hierarchies and patterns in visual data.

    They are made up of many layers but in general follow this pattern:

    1. Convolutional Layers:
    	- These layers use filters (small matrices) that slide over the input image to detect patterns or features, such as edges, textures, or shapes.
    	- Each filter detects a specific feature in a local region of the image, producing a “feature map” that highlights the presence and position of that feature.

    2.	Activation Function (e.g., ReLU):
    	- After each convolution, an activation function like ReLU (Rectified Linear Unit) is applied, which introduces non-linearity to help the network learn complex patterns.

    3.	Pooling Layers:
    	- Pooling layers reduce the spatial size (width and height) of feature maps, which helps lower computation and reduces the risk of overfitting.
    	- Max pooling is the most common type of pooling, which selects the highest value in a region, making the feature map smaller while keeping key information.
    4.	Fully Connected Layers:
    	- Near the end of the CNN, fully connected (FC) layers combine all learned features from previous layers to make a final prediction.
    	- These layers “flatten” the output of the final convolutional layers and pass them to a typical neural network layer to classify the image or detect objects.
    5.	Output Layer:
    	- The final layer generates the prediction, often using a softmax function for classification tasks, which provides probabilities for each class.

    The overall structure of our CNN is going to be as follows :

    1. input layer
    2. convolutional layer
    3. Max pooling layer (with ReLU activation)
    4. Convolutional layer
    5. Dropout layer
    6. max pooling layer
    7. convolutional layer
    8. max pooling layer
    9. Flatten to Dense layer
    10. Dense layer reduction
    11. output. Linear layer

    We can build this as follows with our Sequential model:
    """
    )
    return


@app.cell
def _(IMAGE_CHANNELS, nn):
    n_classes = 25
    kernel_size = 3
    flattened_img_size = 75 * 3 * 3

    model = nn.Sequential(
        # First convolution
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
        # Flatten to Dense
        nn.Flatten(),
        nn.Linear(flattened_img_size, 512),
        nn.Dropout(0.3),
        nn.ReLU(),
        nn.Linear(512, n_classes),
    )
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Conv2D

    These are our 2D convolutional layers. Small kernels will go over the input image and detect features that are important for classification. Earlier convolutions in the model will detect simple features such as lines. Later convolutions will detect more complex features.

    In this first example

    ```Python
    nn.Conv2d(IMG_CHS, 25, kernel_size, stride=1, padding=1)
    ```

    25 refers to the number of filters that will be learned. Even though `kernel_size = 3`, PyTorch will assume we want 3 x 3 filters. Stride refer to the step size that the filter will take as it passes over the image. Padding refers to whether the output image that's created from the filter will match the size of the input image.

    ## Batch Normalization

    Batch normalization scales the values in the hidden layers to improve training. It can also help with the [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem).

    ## MaxPool2D

    Max pooling takes an image and essentially shrinks it to a lower resolution. It does this to help the model be robust to translation (objects moving side to side), and also makes our model faster as it has less data to process as it goes through the network.

    ## Dropout

    As we saw in our last example we had overfitting of our data, Dropout is a technique for preventing overfitting.

    It will randomly turn on and off neurons in the network. This will help the network to learn more robust features and not rely on a single neuron to make a decision leading to less overfitting.

    ## Flatten

    This layer will take the output of the last convolutional layer and flatten it into a 1D tensor. This will allow us to pass it to a fully connected layer. The output is called a feature vector and will be connected to the final classification layer.

    ## Linear

    This is our final classification layer. It will take the feature vector and output a prediction for each class. We will use the softmax activation function to convert the output to a probability.

    We have seen dense linear layers before in our earlier models. Our first dense layer (512 units) takes the feature vector as input and learns which features will contribute to a particular classification. The second dense layer (24 units) is the final classification layer that outputs our prediction.

    ## The final model

    We can print out the different layers of the model to see the structure of the model. Note that some of the CNN elements do not work compiled on a mac so we need to take this into account as shown below.
    """
    )
    return


@app.cell
def _(device, model, torch):
    if device == "cuda":
        model_compiled = torch.compile(model.to(device))
    else:
        model_compiled = model.to(device)
    model_compiled.to(device)

    print(model_compiled)
    return (model_compiled,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""You will notice as we traverse the next layers the size of the image is reduced, this is due to the max pooling layers reducing the size of the image. It is important that the size of the input and the prevous layer output match, otherwise the model will not work."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Training the model

    Whilst the model is very different the overall processes we are going to use for everything else are the same as before.

    First we need to define the loss and optimazation functions.
    """
    )
    return


@app.cell
def _(Adam, model_compiled, nn):
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model_compiled.parameters())
    return loss_function, optimizer


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


@app.cell
def _(
    plt,
    train,
    train_accuracy,
    train_loss,
    valid_accuracy,
    valid_loss,
    validate,
):
    epochs = 10

    for epoch in range(epochs):
        print("Epoch: {}".format(epoch))
        train()
        validate()

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
        r"""You will notice that this is much improved from before, however there are some jumps in the validation accuracy. Let's see how well it performs on the test data."""
    )
    return


@app.cell
def _(alphabet, model_compiled, plt, torch, valid_loader):
    model_compiled.eval()
    with torch.no_grad():
        x, y = next(iter(valid_loader))
        output = model_compiled(x)
        pred = output.argmax(dim=1, keepdim=True)
        num_images = 10
        plt.figure(figsize=(10, 10))

        for _i in range(num_images):
            plt.subplot(1, num_images, _i + 1)
            plt.title(
                f" {alphabet[y[_i].item()]} {alphabet[pred[_i].item()]}",
                fontdict={"fontsize": 30},
            )
            plt.axis("off")
            plt.imshow(x[_i].cpu().numpy().reshape(28, 28), cmap="gray")
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""![]({DATASET_LOCATION}/mnist_asl/american_sign_language.PNG)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""As you can see from visual inspection it is close but not 100% accurate.  We will improve on this model in the next notebook."""
    )
    return


@app.cell
def _(model_compiled, torch, train_loader):
    model_compiled.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for _x, _y in train_loader:
            _output = model_compiled(_x)
            _pred = _output.argmax(dim=1, keepdim=True)
            correct = correct + _pred.eq(_y.view_as(_pred)).sum().item()
            total = total + _y.size(0)
    print(f"Accuracy: {correct / total}")
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

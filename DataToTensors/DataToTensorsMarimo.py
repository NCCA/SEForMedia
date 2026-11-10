#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # From media data to tensors

    We can represent an image, a sound recording or an animated character as numbers. The harder part is deciding how to organise these numbers into tensors our models can use, and keeping track of what each axis represents.

    In this notebook I will start with a single sample, identify its axes and then show how we combine samples into a batch. We will use the same approach for images, video, sound, point clouds, meshes and motion capture.

    For each example we will decide what belongs in the input `X`, what we want the model to predict as the target `y`, and what we need to keep alongside the data as metadata.

    All examples use small amounts of generated data and run on the CPU, so there are no assets to download or models to train. Change the controls to see how they affect the labelled shapes and plots.

    This notebook follows on from [NumPy shapes](../NumPyForML/NumPyForMLPart2Shapes.py) and [PyTorch tensors](../PyTorchForML/PyTorchForMLPart1Tensors.py). We will build on these ideas in the later examples in `TorchVisionForML/`, `TorchAudioForML/`, `TorchCodecForML/` and `PointCloudsAndMeshNet/`.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    print("marimo", mo.__version__)
    print("PyTorch", torch.__version__)
    return mo, np, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Start with the question

    Before we convert any data, we need to decide what we want the model to do. I find it useful to write this as a sentence: **given this information, predict this answer** (this is our [problem domain](https://developers.google.com/machine-learning/problem-framing/ml-framing)). We can use the same source file for several different tasks, each with different inputs and targets.

    | Task | Input available when making a prediction (`X`) | Target used during training (`y`) |
    | --- | --- | --- |
    | Image classification | An image | One object class |
    | Image segmentation | An image | One class for each pixel |
    | Audio event detection | A recording | Events present in each time interval |
    | Audio denoising | A noisy recording | The corresponding clean recording |
    | Video action recognition | A video clip | One action class |
    | Mocap forecasting | Earlier poses | Later poses |
    | Mesh deformation | A rest mesh and control values | Deformed vertex positions |
    | Audio-driven animation | Speech and an initial pose | A sequence of facial or body poses |

    The task determines the shape of the target, which may have more dimensions than the input. An [autoencoder](https://en.wikipedia.org/wiki/Autoencoder) uses the input data as the target it learns to reconstruct. In [self supervised learning](https://en.wikipedia.org/wiki/Self-supervised_learning) we construct targets from the data itself. For [supervised training](https://en.wikipedia.org/wiki/Supervised_learning) we need both inputs and targets when making predictions (inference), we usually only supply the inputs and any necessary metadata.

    For image classification we keep the image and its label together as an `(image, label)` pair. If we included the known class in the model input, we would be giving away the answer. The label is not a fourth colour channel.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Samples, features and batches

    Before we look at the different types of media, we need a few terms to describe how we organise the data.

    A **sample** is one example for our task. This might be one image, a five-second audio clip or a sequence of character poses. We decide what counts as a sample based on what we want the model to predict. A single frame and a complete video clip could each be a sample for different tasks.

    A **feature** is a measurement or value we give the model. For a point in 3D space, the features might be its X, Y and Z coordinates. For a weather observation, they might be temperature, pressure and humidity. The **feature axis** groups these values for each item.

    For example, if one sample contains 100 points, each with three coordinates, we can represent it as:

    ```python
    points.shape  # (100, 3): (points, coordinates)
    ```

    The first axis selects a point. The second selects one of its coordinates:

    ```python
    points[7]     # XYZ coordinates of point 7
    points[7, 0]  # X coordinate of point 7
    ```

    The coordinates in a row all describe the same point. If we add RGB colour values, we could use a shape of `(100, 6)`, with each row containing `(x, y, z, r, g, b)`. We would also need to record the coordinate units and colour range.

    A **batch** is a group of samples we process together. The **batch axis** selects a sample from that group. In these examples I will normally put it first. A batch of eight point clouds, each containing 100 points, would have the shape:

    ```python
    point_batch.shape  # (8, 100, 3): (batch, points, coordinates)
    ```

    We can now select a whole sample, a point within it or one coordinate:

    ```python
    point_batch[2]        # point cloud 2: shape (100, 3)
    point_batch[2, 7]     # point 7 in point cloud 2: shape (3,)
    point_batch[2, 7, 0]  # its X coordinate: a single value
    ```

    Other axes describe the structure *within* a sample. A pose sequence might have shape `(60, 24, 3)`: 60 time steps, 24 joints and three coordinates per joint. Those 60 time steps belong to one sample. A batch of eight such sequences would have shape `(8, 60, 24, 3)`.

    | Term | Meaning | Example |
    | --- | --- | --- |
    | Sample | One example for the task | One sequence of character poses |
    | Batch axis | Selects an example in a batch | Eight pose sequences |
    | Time axis | Selects a time step within a sample | 60 frames per sequence |
    | Joint axis | Selects a joint within a pose | 24 joints per frame |
    | Feature axis | Selects a measurement for an item | XYZ coordinates per joint |

    Not every input has one axis called “features”. An image usually has separate height, width and colour-channel axes. Together these hold the information the model receives. We keep the axes separate because their structure matters: neighbouring pixels, successive audio samples and connected joints have relationships we may want the model to use.

    The target also needs to match the samples in the batch. For image classification, a batch of eight images might have input shape `(8, 3, 64, 64)` and target shape `(8,)`, containing one class label per image. For pose forecasting, each target might contain several future poses. The batch axis still selects corresponding examples in `X` and `y`; the remaining axes depend on the task.

    ## 2. Name axes before choosing their order

    A tensor is a rectangular collection of values of one dtype. Its **rank** is the number of axes; its **shape** gives the length of each axis. Rank here does not mean matrix rank.

    Hover over a symbol for an example of how we use it in a shape. Each symbol describes the length of an axis. Later in the notebook, hover over a symbolic shape to see its axis meanings again.

    | Symbol | Meaning in this notebook |
    | --- | --- |
    | <span data-tooltip="Number of independent samples processed together. Four RGB images have shape (4, 3, H, W), so B = 4."><code>B</code></span> | Independent samples in a batch |
    | <span data-tooltip="Number of measurements per observation. Mass and speed give F = 2 and a batch shape of (B, 2)."><code>F</code></span> | Features measured at the same observation |
    | <span data-tooltip="Number of channels. RGB has C = 3; stereo audio has C = 2. Channels are separate from output classes."><code>C</code></span> | Channels, such as RGB or stereo |
    | <span data-tooltip="Number of rows in an image or grid. An RGB image 480 pixels high has shape (3, 480, W)."><code>H</code></span> | Height of a spatial grid |
    | <span data-tooltip="Number of columns in an image or grid. An RGB image 640 pixels wide has shape (3, H, 640)."><code>W</code></span> | Width of a spatial grid |
    | <span data-tooltip="Number of depth slices in a volume. One occupancy grid has shape (1, D, H, W)."><code>D</code></span> | Depth of a spatial grid |
    | <span data-tooltip="Number of ordered frames or time steps. A 12-frame video clip can have shape (12, C, H, W)."><code>T</code></span> | Ordered frames or time steps |
    | <span data-tooltip="Number of waveform samples per channel. One second of stereo audio at 8 kHz has shape (2, 8000)."><code>S</code></span> | Audio samples in time |
    | <span data-tooltip="Number of frequency bins. A one-sided STFT of real audio with n_fft = 256 has Q = 129."><code>Q</code></span> | Frequency bins in a spectrogram |
    | <span data-tooltip="Number of time windows in a spectrogram. Shape (C, Q, L) separates channels, frequency bins and window positions."><code>L</code></span> | Spectrogram time frames |
    | <span data-tooltip="Number of points in a point cloud. XYZ positions have shape (N, 3); points with nine attributes have shape (N, 9)."><code>N</code></span> | Points in a point cloud |
    | <span data-tooltip="Number of mesh vertices. Vertex positions have shape (V, 3). A cube can have V = 8 corner vertices."><code>V</code></span> | Mesh vertices |
    | <span data-tooltip="Number of triangle faces. Face indices have shape (M, 3). A triangulated cube can have V = 8 and M = 12; M avoids reusing F, which means features here."><code>M</code></span> | Triangle faces in a mesh |
    | <span data-tooltip="Number of joints in the recorded skeleton. A position-based motion clip has shape (T, J, 3), with a fixed joint order."><code>J</code></span> | Joints in a skeleton |
    | <span data-tooltip="Number of rotation components per joint. Euler angles use R = 3; quaternions use R = 4. Record the rotation convention as metadata."><code>R</code></span> | Rotation components per joint |
    | <span data-tooltip="Number of output classes. Five-class image classification produces logits shaped (B, 5), while class-ID targets have shape (B,)."><code>K</code></span> | Classes or output categories |
    | <span data-tooltip="Number of future time steps to predict. Mocap forecasting can output (B, Tout, J, 3) from an input of (B, T, J, 3)."><code>Tout</code></span> | Forecast time steps |

    The letters describe what each axis means; they are not stored automatically in an ordinary tensor. A shape of `(32, 3)` could represent 32 points with XYZ coordinates, 32 samples with three measurements each, or 32 audio samples with left, right and reference channels. We need to record what the axes represent, as the shape alone cannot tell us.

    I try to use the following rules when organising the data:

    1. Define a single sample and its target before adding a batch axis.
    2. Give repeated structures their own axes, such as time steps, image rows and columns, joints or points.
    3. Group measurements of the same item along a feature axis, making sure they refer to the same point, time step or other entity.
    4. Choose an axis order the next operation expects, and document it beside the tensor.
    5. Decide the data type, units, value range and how to handle missing values.
    6. To form a batch, use samples with matching shapes or decide how to handle differences using padding, sampling, bucketing or a structured representation.
    7. Check some individual values as well as the shape. An incorrect layout can still have the expected shape.

    In these examples we will normally put the batch axis first. Other APIs and layers may expect a different order, so we need to check what the next operation accepts.
    """)
    return


@app.cell
def _(torch):
    for _name, _tensor in [
        ("one value", torch.tensor(4.0)),
        ("one XYZ point", torch.tensor([1.0, 2.0, 3.0])),
        ("two XYZ points", torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
    ]:
        print(
            f"{_name:18} rank={_tensor.ndim}, shape={tuple(_tensor.shape)}, dtype={_tensor.dtype}"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Convert values, units and categories

    Converting data into a tensor does not change what the values represent. In this example each row describes one object, with its mass in kilograms, speed in metres per second and surface material.

    We can group the mass and speed into a tensor with shape <span data-tooltip="B = independent samples in the batch; F = features per observation."><code>(B, F)</code></span>, where each row is one object and each column is a measurement. We need to keep track of the units, as the tensor does not store them for us.

    The surface material is a category. We might assign an integer ID to each material, but material 2 is not twice as metallic as material 1. We can keep these IDs in a separate integer tensor for an embedding layer, or use a [one hot representation](https://en.wikipedia.org/wiki/One-hot) if we only have a small number of materials.

    [`torch.from_numpy`](https://docs.pytorch.org/docs/stable/generated/torch.from_numpy.html) shares memory with a compatible NumPy array, whilst [`torch.tensor`](https://docs.pytorch.org/docs/stable/generated/torch.tensor.html) copies the values. Neither decides how we should scale the data. Converting mass and speed to `float32` changes their data type; it does not normalise them or convert their units.

    We calculate any scaling statistics, such as the mean and standard deviation, using the training data. We also build the category vocabulary from the training data, then reuse both when processing validation, test and inference data. We need to decide how to handle an unfamiliar material, for example by assigning it an unknown ID. Using the test data to make these decisions would leak information from the test set into our preprocessing.
    """)
    return


@app.cell
def _(np, torch):
    training_measurements = np.array(
        [[2.0, 10.0], [4.0, 20.0], [6.0, 30.0]], dtype=np.float32
    )
    measurements = torch.from_numpy(training_measurements)
    _feature_mean = measurements.mean(dim=0, keepdim=True)
    _feature_std = measurements.std(dim=0, correction=0, keepdim=True).clamp_min(1e-6)
    scaled_measurements = (measurements - _feature_mean) / _feature_std
    material_ids = torch.tensor([0, 2, 1], dtype=torch.int64)
    material_one_hot = torch.nn.functional.one_hot(material_ids, num_classes=3).float()
    object_features = torch.cat((scaled_measurements, material_one_hot), dim=-1)
    print("continuous measurements (B, F):", tuple(measurements.shape))
    print("category IDs (B,):", tuple(material_ids.shape))
    print("combined features (B, 2 + 3):", tuple(object_features.shape))
    print(object_features)
    print(
        "shares the NumPy buffer:",
        measurements.data_ptr() == training_measurements.ctypes.data,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Images: height and width are different axes

    An RGB image has three colour channels. In PyTorch image models we commonly use <span data-tooltip="C = channels, such as RGB or stereo; H = height in rows; W = width in columns."><code>(C, H, W)</code></span>, so a single image is <span data-tooltip="H = height in rows; W = width in columns; 3 = three colour channels: red, green and blue."><code>(3, H, W)</code></span>. Writing <span data-tooltip="W = width in columns; H = height in rows; 3 = three colour channels: red, green and blue."><code>(3, W, H)</code></span> is valid only when that is the documented convention of our pipeline. I use unequal height and width below so that an accidental swap is visible.

    Many image readers and plotting functions use <span data-tooltip="H = height in rows; W = width in columns; C = channels, such as RGB or stereo."><code>(H, W, C)</code></span>. We change axis order with [`permute`](https://docs.pytorch.org/docs/stable/generated/torch.permute.html), or NumPy `transpose`, rather than `reshape`.

    `image[c, row, column]` is one channel value; `image[:, row, column]` is one RGB pixel. RGB versus BGR, colour space, alpha and bit depth also matter. Dividing an 8-bit image by 255 gives values in `[0, 1]`; it does not linearise sRGB or apply the mean/std normalisation expected by a particular model.
    """)
    return


@app.cell
def _(np, torch):
    _height, _width = 3, 5
    image_hwc = np.zeros((_height, _width, 3), dtype=np.uint8)
    image_hwc[..., 0] = np.arange(_width) * 40
    image_hwc[..., 1] = np.arange(_height)[:, None] * 80
    image_hwc[..., 2] = 64
    image_chw = torch.from_numpy(image_hwc).permute(2, 0, 1).float() / 255.0
    _wrong = torch.from_numpy(image_hwc).reshape(3, _height, _width).float() / 255.0
    print("reader: (H, W, C)", image_hwc.shape)
    print("model:  (C, H, W)", tuple(image_chw.shape))
    print("same shape after reshape?", _wrong.shape == image_chw.shape)
    print("correct RGB at row 1, column 2:", image_chw[:, 1, 2])
    print("reshape RGB at the same indices:", _wrong[:, 1, 2])
    assert not torch.equal(_wrong, image_chw)
    return (image_chw,)


@app.cell
def _(image_chw, plt):
    _image_figure, _axes = plt.subplots(1, 2, figsize=(9, 3))
    _axes[0].imshow(image_chw.permute(1, 2, 0).numpy())
    _axes[0].set_title("RGB image: display as (H, W, C)")
    _axes[1].imshow(image_chw[0].numpy(), cmap="Reds", vmin=0, vmax=1)
    _axes[1].set_title("Red channel: (H, W)")
    for _axis in _axes:
        _axis.set_xlabel("Column / width")
        _axis.set_ylabel("Row / height")
    _image_figure.tight_layout()
    plt.close(_image_figure)
    _image_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Logical axes and memory layout

    `permute` changes how indices refer to the underlying storage; it returns a view. The result may be non-contiguous. [`contiguous()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html) can create storage in the requested memory format when needed. `reshape` preserves element order and may return a view or a copy; it does not know which axis represents colour.

    PyTorch's `channels_last` memory format is a separate concept: a four-dimensional tensor can retain logical shape <span data-tooltip="B = independent samples in the batch; C = channels, such as RGB or stereo; H = height in rows; W = width in columns."><code>(B, C, H, W)</code></span> whilst storing neighbouring channel values together. That is different from permuting the logical axes to <span data-tooltip="B = independent samples in the batch; H = height in rows; W = width in columns; C = channels, such as RGB or stereo."><code>(B, H, W, C)</code></span>.
    """)
    return


@app.cell
def _(image_chw, torch):
    _single_batch = image_chw.unsqueeze(0)
    _channels_last_storage = _single_batch.contiguous(memory_format=torch.channels_last)
    print("logical axes (B, C, H, W):", tuple(_channels_last_storage.shape))
    print("ordinary contiguous strides:", _single_batch.contiguous().stride())
    print("channels-last storage strides:", _channels_last_storage.stride())
    print("values unchanged:", torch.equal(_single_batch, _channels_last_storage))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Stack samples, concatenate features, preserve time

    We often need to combine tensors, either to make a batch or to bring related measurements together. [`stack`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html) creates a new axis, whilst [`cat`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html) joins tensors along an existing axis. We need to choose the operation and axis based on what the data represents.

    | Intention | Operation | Example |
    | --- | --- | --- |
    | Make a batch of equal-sized images | `torch.stack(images, dim=0)` | A list of <span data-tooltip="C = colour channels; H = image height in rows; W = image width in columns."><code>(C,H,W)</code></span> images → <span data-tooltip="B = images in the batch; C = colour channels; H = image height; W = image width."><code>(B,C,H,W)</code></span> |
    | Add aligned depth to RGB | `torch.cat((rgb, depth), dim=1)` | <span data-tooltip="B = samples in the batch; 3 = RGB colour channels; H = image height; W = image width."><code>(B,3,H,W)</code></span> and <span data-tooltip="B = samples in the batch; 1 = one depth channel; H = image height; W = image width."><code>(B,1,H,W)</code></span> → <span data-tooltip="B = samples in the batch; 4 = three RGB channels and one depth channel; H = image height; W = image width."><code>(B,4,H,W)</code></span> |
    | Join consecutive sequence chunks | `torch.cat(chunks, dim=1)` | <span data-tooltip="B = sequences in the batch; T1 = time steps in the first chunk; F = features at each time step."><code>(B,T1,F)</code></span> and <span data-tooltip="B = sequences in the batch; T2 = time steps in the second chunk; F = features at each time step."><code>(B,T2,F)</code></span> → <span data-tooltip="B = sequences in the batch; T1+T2 = combined number of time steps; F = features at each time step."><code>(B,T1+T2,F)</code></span> |
    | Add a batch axis to one image | `image.unsqueeze(0)` | <span data-tooltip="C = colour channels; H = image height; W = image width."><code>(C,H,W)</code></span> → <span data-tooltip="1 = one image in the batch; C = colour channels; H = image height; W = image width."><code>(1,C,H,W)</code></span> |
    | Flatten image values for a dense layer | `images.flatten(start_dim=1)` | <span data-tooltip="B = images in the batch; C = colour channels; H = image height; W = image width."><code>(B,C,H,W)</code></span> → <span data-tooltip="B = images in the batch; C*H*W = all channel and pixel values arranged along one feature axis."><code>(B,C*H*W)</code></span> |

    To stack samples into a batch, all samples must have the same shape. To concatenate tensors, their shapes must match along every axis except the one we are extending. These checks tell us whether the operation is possible, but we still need to check whether it makes sense.

    For example, we can add depth to RGB along the channel axis if each depth value corresponds to the same pixel as its RGB values. We also need to record the depth units and how each channel is scaled. If some depth values are missing, we should keep a validity mask alongside the data.

    When joining sequence chunks, we need to check that they belong to the same recording, appear in the correct order and use compatible sampling rates. For a batch, each row must refer to the same sequence across the chunks.

    Adding a batch axis with `unsqueeze` gives us a batch containing one sample; it does not create additional samples. Flattening keeps the batch axis but combines the remaining axes into one. The values are still there, but the height, width and channel axes are no longer separate. In each case, the resulting shape only makes sense if we know what the axes represent.
    """)
    return


@app.cell
def _(image_chw, torch):
    image_batch = torch.stack((image_chw, image_chw.flip(-1)), dim=0)
    _depth = torch.full((2, 1, 3, 5), 2.0)  # metres, aligned to the RGB pixels
    rgb_depth = torch.cat((image_batch, _depth), dim=1)
    print("two images (B, C, H, W):", tuple(image_batch.shape))
    print("RGB plus depth:", tuple(rgb_depth.shape))
    print("dense-layer input:", tuple(image_batch.flatten(start_dim=1).shape))
    try:
        torch.stack((image_chw, image_chw[:, :2, :]), dim=0)
    except RuntimeError as _error:
        print("Different image sizes:", str(_error).split("\n")[0])
    return (image_batch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Derive outputs from the task and the loss

    For a batch of RGB images, `X` might always be <span data-tooltip="B = independent samples in the batch; H = height in rows; W = width in columns; 3 = three colour channels: red, green and blue."><code>(B, 3, H, W)</code></span>. The output depends on the question. Below we use ordinary PyTorch conventions; <span data-tooltip="K = output classes."><code>K</code></span> is the number of classes.

    | Task | Model output | Target | Typical loss |
    | --- | --- | --- | --- |
    | One class per image | Float logits <span data-tooltip="B = independent samples in the batch; K = output classes."><code>(B,K)</code></span> | Integer class IDs <span data-tooltip="B = independent samples in the batch."><code>(B,)</code></span> | `CrossEntropyLoss` |
    | Several independent labels | Float logits <span data-tooltip="B = independent samples in the batch; K = output classes."><code>(B,K)</code></span> | Float indicators <span data-tooltip="B = independent samples in the batch; K = output classes."><code>(B,K)</code></span> | `BCEWithLogitsLoss` |
    | Binary classification with one logit | Float logits <span data-tooltip="B = independent samples in the batch; 1 = a length-one axis; its role is given by the surrounding example."><code>(B,1)</code></span> | Float 0/1 values <span data-tooltip="B = independent samples in the batch; 1 = a length-one axis; its role is given by the surrounding example."><code>(B,1)</code></span> | `BCEWithLogitsLoss` |
    | One real number per image | Float values <span data-tooltip="B = independent samples in the batch; 1 = a length-one axis; its role is given by the surrounding example."><code>(B,1)</code></span> | Float values <span data-tooltip="B = independent samples in the batch; 1 = a length-one axis; its role is given by the surrounding example."><code>(B,1)</code></span> | `MSELoss` or `L1Loss` |
    | Pixel classification | Float logits <span data-tooltip="B = independent samples in the batch; K = output classes; H = height in rows; W = width in columns."><code>(B,K,H,W)</code></span> | Integer IDs <span data-tooltip="B = independent samples in the batch; H = height in rows; W = width in columns."><code>(B,H,W)</code></span> | `CrossEntropyLoss` |
    | Depth prediction | Float depth <span data-tooltip="B = independent samples in the batch; H = height in rows; W = width in columns; 1 = a length-one axis; its role is given by the surrounding example."><code>(B,1,H,W)</code></span> | Float depth and validity mask | Masked regression loss |
    | Object detection | Model-specific boxes and class scores | Variable-length boxes and labels per image | Model-specific matching and losses |

    A [logit](https://en.wikipedia.org/wiki/Logit) is an unrestricted score before a probability conversion. Feed logits directly to these classification losses. For single-class predictions, `argmax(dim=1)` chooses the class. For independent labels, apply `sigmoid` and choose thresholds.

    The class dimension belongs in the prediction even when the target uses a single integer ID. A one-hot target is not required for the class-index form of cross entropy. PyTorch also supports probability targets, but that is a different contract.

    See the official [cross entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) and [binary cross entropy with logits](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) documentation for their accepted shapes and dtypes.
    """)
    return


@app.cell
def _(image_batch, torch):
    _class_logits = torch.tensor([[2.0, -1.0, 0.0], [-1.0, 0.0, 2.0]])
    class_targets = torch.tensor([0, 2], dtype=torch.int64)
    _class_loss = torch.nn.functional.cross_entropy(_class_logits, class_targets)
    _segmentation_logits = torch.zeros(2, 3, 3, 5)
    _segmentation_targets = torch.zeros(2, 3, 5, dtype=torch.int64)
    _segmentation_loss = torch.nn.functional.cross_entropy(
        _segmentation_logits, _segmentation_targets
    )
    print("X:", tuple(image_batch.shape), "y:", tuple(class_targets.shape))
    print("class predictions:", _class_logits.argmax(dim=1).tolist())
    print("classification loss:", _class_loss.item())
    print(
        "segmentation prediction / target:",
        tuple(_segmentation_logits.shape),
        tuple(_segmentation_targets.shape),
    )
    print("segmentation loss:", _segmentation_loss.item())
    return (class_targets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A silent broadcasting error

    We need to check shapes as well as data types. Suppose our model produces one prediction per sample with shape <span data-tooltip="B = samples in the batch; 1 = one predicted value per sample."><code>(B, 1)</code></span>, but our targets have shape <span data-tooltip="B = samples in the batch, with one target value per sample."><code>(B,)</code></span>.

    Subtracting these tensors produces a result with shape <span data-tooltip="B = samples in the batch; the two axes pair every prediction with every target."><code>(B, B)</code></span>. Broadcasting allows the operation to run, but we have compared every prediction with every target instead of matching each prediction to its own target.

    For this regression task we require the predictions and targets to have exactly the same shape. We can check this before calculating the loss:

    ```python
    assert predictions.shape == targets.shape
    ```

    If we need to remove a trailing axis of length one, we can use `predictions.squeeze(-1)` to change `(B, 1)` into `(B,)`. Avoid calling `squeeze()` without specifying an axis here: when the batch contains one sample, it would also remove the batch axis.
    """)
    return


@app.cell
def _(torch):
    _predicted = torch.tensor([[1.0], [2.0], [3.0]])
    _target = torch.tensor([1.0, 2.0, 3.0])
    print("accidental errors:", tuple((_predicted - _target).shape))
    _target_column = _target.unsqueeze(-1)
    assert _predicted.shape == _target_column.shape
    print("intended errors:", tuple((_predicted - _target_column).shape))
    print(
        "intended MSE:",
        torch.nn.functional.mse_loss(_predicted, _target_column).item(),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Explore an image batch

    Change the batch size, image dimensions and number of channels. The controls below calculate shapes and input storage, they do not allocate a large tensor. Compare how adding samples differs from adding channels.

    A float32 value occupies four bytes. Input storage is <span data-tooltip="B = independent samples in the batch; C = channels, such as RGB or stereo; H = height in rows; W = width in columns; 4 = three RGB channels plus one aligned depth channel."><code>B * C * H * W * 4</code></span>; model weights, gradients, activations, optimiser state and temporary buffers are additional costs.
    """)
    return


@app.cell
def _(mo):
    batch_control = mo.ui.slider(1, 32, value=4, label="Batch size")
    height_control = mo.ui.slider(16, 256, step=16, value=64, label="Height")
    width_control = mo.ui.slider(16, 256, step=16, value=96, label="Width")
    channel_control = mo.ui.dropdown(
        {"Greyscale": 1, "RGB": 3, "RGB + depth": 4},
        value="RGB",
        label="Channels",
    )
    mo.hstack(
        [batch_control, height_control, width_control, channel_control],
        wrap=True,
    )
    return batch_control, channel_control, height_control, width_control


@app.cell
def _(batch_control, channel_control, height_control, mo, width_control):
    _shape = (
        batch_control.value,
        channel_control.value,
        height_control.value,
        width_control.value,
    )
    _elements = 1
    for _length in _shape:
        _elements *= _length
    mo.md(f"""
    `X` with axes <span data-tooltip="B = independent samples in the batch; C = channels, such as RGB or stereo; H = height in rows; W = width in columns."><code>(B, C, H, W)</code></span> has shape `{_shape}` and **{_elements:,} values**.

    At float32 this is **{_elements * 4 / 1024**2:.3f} MiB** for the input alone.

    For a five-class classifier: logits `({_shape[0]}, 5)`, targets `({_shape[0]},)`.
    For five-class segmentation: logits `({_shape[0]}, 5, {_shape[2]}, {_shape[3]})`, targets `({_shape[0]}, {_shape[2]}, {_shape[3]})`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Video: time is an ordered axis

    For video we need to preserve the order of the frames within each clip. One decoded RGB clip might have shape <span data-tooltip="T = frames in the clip; H = image height; W = image width; C = colour channels."><code>(T, H, W, C)</code></span>. We can move the channel axis and stack clips with matching shapes to produce <span data-tooltip="B = clips in the batch; T = frames per clip; C = colour channels; H = image height; W = image width."><code>(B, T, C, H, W)</code></span>.

    The required order depends on the next operation. A PyTorch [`Conv3d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html) layer expects <span data-tooltip="B = samples in the batch; C = channels; D = depth slices; H = height; W = width."><code>(B, C, D, H, W)</code></span>. For video we use the depth axis for time, giving us <span data-tooltip="B = clips in the batch; C = colour channels; T = frames per clip; H = image height; W = image width."><code>(B, C, T, H, W)</code></span>. We change between these layouts by permuting the axes.

    We can also combine the batch and time axes to process every frame with a 2D image encoder. This changes <span data-tooltip="B = clips in the batch; T = frames per clip; C = colour channels; H = image height; W = image width."><code>(B, T, C, H, W)</code></span> into <span data-tooltip="B*T = all frames from all clips treated as individual images; C = colour channels; H = image height; W = image width."><code>(B*T, C, H, W)</code></span>. The encoder processes each frame independently, so it cannot use movement between frames. We restore the batch and time axes in its output before passing the features to a temporal model.

    We also need to keep the frame rate and timestamps alongside the tensor. Twelve frames sampled at 12 fps cover a different amount of time from twelve frames sampled at 60 fps, even though the tensors have the same shape. For constant-frame-rate video, keeping every second frame halves the effective frame rate. For variable-frame-rate video we need the timestamps to know when each frame occurs.

    The target shape depends on whether we want one answer for the whole clip or one answer per frame:

    | Task | Model output | Target |
    | --- | --- | --- |
    | Clip classification | <span data-tooltip="B = clips in the batch; K = possible classes."><code>(B, K)</code></span> logits | <span data-tooltip="B = clips in the batch, with one integer class label per clip."><code>(B,)</code></span> |
    | Frame classification | <span data-tooltip="B = clips in the batch; T = frames per clip; K = possible classes."><code>(B, T, K)</code></span> logits | <span data-tooltip="B = clips in the batch; T = frames per clip, with one integer class label per frame."><code>(B, T)</code></span> |

    For frame classification, `CrossEntropyLoss` with integer class targets expects the class axis second. We can permute the logits to <span data-tooltip="B = clips in the batch; K = possible classes; T = frames per clip."><code>(B, K, T)</code></span> and keep the targets as <span data-tooltip="B = clips in the batch; T = frames per clip."><code>(B, T)</code></span>.

    Alternatively, we can flatten batch and time together, giving logits of shape <span data-tooltip="B*T = all frames in the batch; K = possible classes."><code>(B*T, K)</code></span> and targets of shape <span data-tooltip="B*T = all frames in the batch, with one integer class label per frame."><code>(B*T,).</code></span> In either case, each prediction must remain paired with the target for the same frame.
    """)
    return


@app.cell
def _(plt, torch):
    video_btchw = torch.zeros(2, 6, 3, 8, 12)
    for _frame in range(6):
        video_btchw[:, _frame, 0, 2:5, _frame : _frame + 3] = 1.0
    video_bcthw = video_btchw.permute(0, 2, 1, 3, 4)
    _frame_inputs = video_btchw.reshape(2 * 6, 3, 8, 12)
    _frame_features = _frame_inputs.mean(dim=(-2, -1)).reshape(2, 6, 3)
    _video_figure, _axes = plt.subplots(1, 3, figsize=(9, 3))
    for _axis, _frame in zip(_axes, [0, 2, 5]):
        _axis.imshow(video_btchw[0, _frame].permute(1, 2, 0).numpy())
        _axis.set_title(f"Frame {_frame}")
        _axis.set_xlabel("Column")
        _axis.set_ylabel("Row")
    _video_figure.tight_layout()
    plt.close(_video_figure)
    print("sequence layout (B,T,C,H,W):", tuple(video_btchw.shape))
    print("Conv3d layout (B,C,T,H,W):", tuple(video_bcthw.shape))
    print("per-frame features (B,T,F):", tuple(_frame_features.shape))
    _video_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Sound: samples, channels and frequency bins

    A waveform stores amplitude values sampled at regular intervals over time. Here we represent one stereo recording as <span data-tooltip="C = audio channels, two for stereo; S = waveform samples per channel."><code>(C, S)</code></span>. A batch of recordings with matching shapes becomes <span data-tooltip="B = recordings in the batch; C = audio channels; S = waveform samples per channel."><code>(B, C, S)</code></span>.

    The word *sample* has two meanings here. One training sample might be a complete recording, whilst one audio sample is an amplitude value at a particular time in one channel. The channel axis identifies channels such as left and right; it does not identify sound classes.

    We also need to record the sample rate, measured in samples per second per channel. This tells us how the waveform indices relate to time:

    ```python
    duration_seconds = S / sample_rate
    ```

    To examine how the frequency content changes over time, we can use a short-time Fourier transform ([STFT](https://en.wikipedia.org/wiki/Short-time_Fourier_transform)). This divides the waveform into overlapping windows and calculates a frequency representation for each window.

    With `n_fft=256`, a one-sided STFT of a real waveform produces `Q = 256 // 2 + 1 = 129` frequency bins. If we use `center=False`, no additional padding and `hop_length=64`, a waveform containing at least 256 samples produces:

    ```python
    L = 1 + (S - 256) // 64
    ```

    Here `L` is the number of analysis frames. For each audio channel, the result has shape <span data-tooltip="Q = frequency bins; L = analysis frames at successive positions in time."><code>(Q, L)</code></span>. The hop length determines the spacing between frames. Changing the padding or centring settings can change how many frames we get.

    The STFT contains complex values, from which we can calculate magnitude and phase. We often use magnitude or power as model inputs, but these do not retain the original phase. If we want to reconstruct the waveform, we need to retain or estimate that phase information. A mel transform further changes the frequency representation by combining frequency bins using a filter bank; it is not just a reshape.

    The model output depends on the task:

    | Task | Example output shape | Meaning |
    | --- | --- | --- |
    | Audio classification | <span data-tooltip="B = recordings in the batch; K = possible sound classes."><code>(B, K)</code></span> | One set of class logits per recording |
    | Audio denoising | <span data-tooltip="B = recordings in the batch; C = audio channels; S = waveform samples per channel."><code>(B, C, S)</code></span> | A predicted clean waveform for each recording |
    | Source separation | <span data-tooltip="B = recordings in the batch; number_of_sources = sources to separate; C = audio channels per source; S = waveform samples per channel."><code>(B, number_of_sources, C, S)</code></span> | A separate waveform for each source |

    For source separation, the source and channel axes have different roles. A source might be a voice or an instrument, and each source could itself have left and right audio channels.

    See [`torch.stft`](https://docs.pytorch.org/docs/stable/generated/torch.stft.html) and the waveform and feature lessons in `TorchAudioForML/`.
    """)
    return


@app.cell
def _(plt, torch):
    sample_rate = 8000
    _audio_time = torch.arange(2000, dtype=torch.float32) / sample_rate
    waveform = torch.stack(
        (
            0.5 * torch.sin(2 * torch.pi * 440 * _audio_time),
            0.5 * torch.sin(2 * torch.pi * 660 * _audio_time),
        )
    )
    _stft = torch.stft(
        waveform,
        n_fft=256,
        hop_length=64,
        window=torch.hann_window(256),
        center=False,
        return_complex=True,
    )
    spectrogram = _stft.abs()
    print(
        "waveform (C,S):",
        tuple(waveform.shape),
        "duration:",
        waveform.shape[-1] / sample_rate,
        "seconds",
    )
    print("complex STFT (C,Q,L):", tuple(_stft.shape), _stft.dtype)
    print("magnitude (C,Q,L):", tuple(spectrogram.shape), spectrogram.dtype)
    print("spectrum bins and frames:", 256 // 2 + 1, 1 + (2000 - 256) // 64)
    _audio_figure, _axes = plt.subplots(1, 2, figsize=(10, 3))
    _axes[0].plot(_audio_time[:100], waveform[0, :100], label="Left: 440 Hz")
    _axes[0].plot(
        _audio_time[:100], waveform[1, :100], label="Right: 660 Hz", alpha=0.7
    )
    _axes[0].set(xlabel="Time (seconds)", ylabel="Amplitude")
    _axes[0].legend()
    _times = (torch.arange(spectrogram.shape[-1]) * 64 + 128) / sample_rate
    _frequencies = torch.arange(129) * sample_rate / 256
    _plot = _axes[1].pcolormesh(
        _times, _frequencies, torch.log1p(spectrogram[0]), shading="auto"
    )
    _axes[1].set(
        xlabel="Window centre (seconds)",
        ylabel="Frequency (Hz)",
        title="Left-channel log(1 + magnitude)",
    )
    _audio_figure.colorbar(_plot, ax=_axes[1], label="log(1 + magnitude)")
    _audio_figure.tight_layout()
    plt.close(_audio_figure)
    _audio_figure
    return (waveform,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Variable data lengths

    Recordings have different durations, point clouds contain different numbers of points, and sentences contain different numbers of words. To combine these into a dense batch, we need to decide how to handle the different lengths.

    | Approach | Useful when | What we need to consider |
    | --- | --- | --- |
    | Crop or sample | The task can use fixed windows or a fixed number of points | We leave out some of the original data |
    | Resample | We need a common temporal or spatial sampling grid | Interpolation changes values; reducing the temporal sampling rate needs filtering to avoid aliasing |
    | Pad and mask | We want to keep every observation in each sample | We must exclude padding from calculations such as pooling and loss |
    | Bucket similar lengths | Sample lengths vary substantially | Grouping similar lengths reduces the amount of padding |
    | Keep lists, or pack values with offsets | The model supports variable-sized inputs | Operations must keep track of where each sample starts and ends |

    In this example each sequence has shape <span data-tooltip="T = time steps in one sequence; F = features measured at each time step."><code>(T, F)</code></span>. The features might be readings from several sensors. We pad the sequences to the longest length in the batch, giving us <span data-tooltip="B = sequences in the batch; T_max = longest sequence length in this batch; F = features per time step."><code>(B, T_max, F)</code></span>.

    We keep the original lengths and create a validity mask with shape <span data-tooltip="B = sequences in the batch; T_max = padded sequence length; each entry indicates whether that time step contains a real observation."><code>(B, T_max)</code></span>. Here `valid=True` means the position contains a real observation, whilst `False` means padding.

    Mask conventions vary between APIs. A Transformer key-padding mask uses `True` to mean that a position should be ignored, so we would pass `~valid` for that convention. We need to check what the receiving operation expects.

    Zero may be a valid sensor reading, so we cannot identify padding by looking for zero values. We construct the mask from the original sequence lengths. If a real time step has a missing sensor reading, we need a separate feature-level mask to record which measurements are available.
    """)
    return


@app.cell
def _(torch):
    _sequences = [
        torch.tensor([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]]),
        torch.tensor([[2.0, 20.0]]),
    ]
    sequence_lengths = torch.tensor([len(_sequence) for _sequence in _sequences])
    padded_sequences = torch.nn.utils.rnn.pad_sequence(_sequences, batch_first=True)
    valid_steps = (
        torch.arange(padded_sequences.shape[1])[None, :] < sequence_lengths[:, None]
    )
    _naive_mean = padded_sequences.mean(dim=1)
    masked_mean = (padded_sequences * valid_steps.unsqueeze(-1)).sum(
        dim=1
    ) / sequence_lengths[:, None]
    print("padded (B,Tmax,F):", tuple(padded_sequences.shape))
    print("valid (B,Tmax):", valid_steps)
    print("mean including padding:", _naive_mean)
    print("mean of real observations:", masked_mean)
    print("Transformer key-padding mask:", ~valid_steps)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Point clouds: points and their attributes

    A point cloud contains positions and may also include attributes such as normals and colours. For one cloud, we can store the positions as <span data-tooltip="N = points in the cloud; 3 = X, Y and Z coordinates per point."><code>(N, 3)</code></span>. Each row describes one point.

    If we also have a normal and RGB colour for each point, we can combine these along the feature axis:

    | Attribute | Shape | Values per point |
    | --- | --- | --- |
    | Position | <span data-tooltip="N = points; 3 = X, Y and Z position components."><code>(N, 3)</code></span> | X, Y, Z |
    | Normal | <span data-tooltip="N = points; 3 = X, Y and Z normal components."><code>(N, 3)</code></span> | Normal X, Y, Z |
    | Colour | <span data-tooltip="N = points; 3 = red, green and blue colour components."><code>(N, 3)</code></span> | Red, green, blue |
    | Combined features | <span data-tooltip="N = points; 9 = three position, three normal and three colour values per point."><code>(N, 9)</code></span> | Position, normal and colour |

    This only works if the rows correspond: row 7 in each tensor must describe the same point. A batch of clouds with the same number of points has shape <span data-tooltip="B = point clouds in the batch; N = points per cloud; 9 = position, normal and colour features per point."><code>(B, N, 9)</code></span>. Some models expect <span data-tooltip="B = point clouds in the batch; 9 = position, normal and colour features; N = points per cloud."><code>(B, 9, N)</code></span>, so we need to check the required axis order.

    The order of points in a cloud usually has no physical meaning. We can reorder them, but we must apply the same permutation to their normals, colours and any per-point targets. For object classification, reordering the points should leave the prediction unchanged. For point segmentation, the output should follow the new point order. We need a model designed to respect these properties; concatenating the attributes does not provide them automatically.

    We also need to record the coordinate system, units, up axis, handedness and whether positions are in world or local space. Positions, normals and colours have different meanings, so we should choose how to process each attribute separately. Normals represent directions and do not receive translation. With non-uniform scaling, we transform them using the inverse transpose of the transform’s linear part and then normalise them.

    The output shape depends on what we want to predict:

    | Task | Prediction shape | Target |
    | --- | --- | --- |
    | Object classification | <span data-tooltip="B = point clouds in the batch; K = object classes."><code>(B, K)</code></span> logits | <span data-tooltip="B = point clouds in the batch, with one integer class label per cloud."><code>(B,)</code></span> |
    | Point segmentation | <span data-tooltip="B = point clouds in the batch; K = point classes; N = points per cloud."><code>(B, K, N)</code></span> logits | <span data-tooltip="B = point clouds in the batch; N = points per cloud, with one integer class label per point."><code>(B, N)</code></span> |
    | Scene flow | <span data-tooltip="B = point clouds in the batch; N = points per cloud; 3 = X, Y and Z displacement components."><code>(B, N, 3)</code></span> displacements | A displacement vector for each point |

    For scene flow, the predicted vector describes how a point moves between observations. We need to preserve its association with the original point, just as we do with colours, normals and segmentation labels.
    """)
    return


@app.cell
def _(torch):
    point_positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    _normals = torch.tensor([[0.0, 0.0, 1.0]]).expand(4, 3)
    _colours = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
    )
    point_features = torch.cat((point_positions, _normals, _colours), dim=-1)
    _point_classes = torch.tensor([0, 0, 1, 1])
    _permutation = torch.tensor([2, 0, 3, 1])
    print("one cloud (N,F):", tuple(point_features.shape))
    print("one cloud in a batch:", tuple(point_features.unsqueeze(0).shape))
    print("XYZ, normal, RGB for point zero:", point_features[0])
    print("permuted positions:", point_features[_permutation, :3])
    print("matching permuted labels:", _point_classes[_permutation])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Meshes: geometry and topology are separate

    A triangle mesh describes both where its vertices are and how they connect. We normally store these in separate tensors:

    | Data | Shape | Meaning |
    | --- | --- | --- |
    | Vertex positions | <span data-tooltip="V = vertices in the mesh; 3 = X, Y and Z coordinates per vertex."><code>(V, 3)</code></span> | Floating-point coordinates |
    | Face indices | <span data-tooltip="M = triangle faces; 3 = vertex indices per triangle."><code>(M, 3)</code></span> | Integer indices into the vertex array |

    A face containing `[0, 1, 2]` connects vertices from rows 0, 1 and 2 of the position tensor. The positions describe geometry, whilst the indices describe topology. We keep them separate because coordinates are measurements and indices are references to other rows.

    Mesh attributes need a little more care than point-cloud attributes. Normals, colours and UV coordinates can belong to vertices, faces or individual face corners. At a UV seam, one vertex position may have a different UV coordinate for each face that uses it. We therefore cannot always concatenate UVs onto the position rows. We need to record which part of the mesh each attribute describes (its *domain*).

    To batch meshes, we also need to know whether their vertices correspond. Two meshes can have the same number of vertices without row 7 representing the same part of each object. For a character dataset with matching vertex identities and shared topology, we can store positions as <span data-tooltip="B = meshes in the batch; V = corresponding vertices per mesh; 3 = X, Y and Z coordinates."><code>(B, V, 3)</code></span> and reuse one face array.

    For meshes with different topology, we can keep separate tensors in lists or pack their vertices and faces into larger tensors. When packing, we add an offset to each mesh’s face indices so they refer to the correct rows in the combined vertex array. We also keep track of which vertices and faces belong to each mesh.

    Padding is another option, but we need masks for both vertices and faces. Padded faces must be excluded before their indices are used to calculate connectivity or gather vertex data.

    In the example below, the second mesh has a different number of vertices. When we append its vertices to the first mesh’s vertex array, we add the first mesh’s vertex count to every face index in the second mesh.
    """)
    return


@app.cell
def _(torch):
    mesh_vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    mesh_faces = torch.tensor(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=torch.int64
    )
    _other_vertices = torch.tensor([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    _other_faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    packed_vertices = torch.cat((mesh_vertices, _other_vertices), dim=0)
    packed_faces = torch.cat((mesh_faces, _other_faces + len(mesh_vertices)), dim=0)
    vertex_offsets = torch.tensor([0, 4, 7])
    _face_offsets = torch.tensor([0, 4, 5])
    print("packed positions:", tuple(packed_vertices.shape))
    print("packed faces:", packed_faces)
    print(
        "vertex boundaries:",
        vertex_offsets.tolist(),
        "face boundaries:",
        _face_offsets.tolist(),
    )
    print("second mesh triangle positions:", packed_vertices[packed_faces[-1]])
    assert packed_faces.min() >= 0 and packed_faces.max() < len(packed_vertices)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Other 3D representations

    Meshes and point clouds are two ways of representing 3D data. We can also use voxel grids, distance fields, particles, graphs or several camera views of the same object. I would choose the representation based on the structure the model needs for its task.

    Each choice affects what we retain. A point cloud stores positions without explicit surface connectivity. A voxel grid divides a region of space into cells, so we need to choose its resolution and bounding box. For a field sampled at arbitrary positions, we need to keep the query coordinates alongside the values.

    | Representation | Example input for one sample | Typical target or output | Information to retain |
    | --- | --- | --- | --- |
    | Voxel occupancy | <span data-tooltip="1 = one occupancy channel; D = grid depth; H = grid height; W = grid width."><code>(1, D, H, W)</code></span> grid indicating occupied cells | An object class or predicted occupancy grid | Voxel size, grid origin and mapping from grid axes to world coordinates |
    | Signed distance field on a grid | <span data-tooltip="1 = one signed-distance channel; D = grid depth; H = grid height; W = grid width."><code>(1, D, H, W)</code></span> grid of signed distances to a surface | Reconstructed or predicted distances | Grid-to-world mapping, distance units and which sign means inside |
    | Implicit field queries | Query coordinates <span data-tooltip="N = query positions; 3 = X, Y and Z coordinates per query."><code>(N, 3)</code></span>, optionally with a code identifying the shape | Distance or density values <span data-tooltip="N = query positions; 1 = one predicted scalar value per query."><code>(N, 1)</code></span> | Shape identity, coordinate frame and scale |
    | Particle simulation | <span data-tooltip="N = particles; F = features per particle, such as position, velocity and mass."><code>(N, F)</code></span> attributes | Next positions or velocities <span data-tooltip="N = particles; 3 = X, Y and Z components per particle."><code>(N, 3)</code></span> | Particle identities, time step and neighbourhood information |
    | Graph | Node attributes <span data-tooltip="N = graph nodes; F = features per node."><code>(N, F)</code></span> and edge indices <span data-tooltip="2 = source and destination index rows; E = edges in the graph."><code>(2, E)</code></span> | Predictions for the whole graph, individual nodes or edges | Connectivity, any edge attributes and sample boundaries when batching |
    | Multi-view object images | <span data-tooltip="number_of_views = camera views of one object; C = colour channels; H = image height; W = image width."><code>(number_of_views, C, H, W)</code></span> | An object class, reconstructed geometry or a new camera view | Camera intrinsics, camera poses and which views are valid |

    The axes have different meanings in each representation. In a grid, neighbouring indices identify neighbouring cells in space. In a point cloud, neighbouring rows do not necessarily identify nearby points. Flattening either representation into a vector does not remove these differences; we still need to understand the structure when choosing how the model processes the data.
    """)
    return


@app.cell
def _(torch):
    _grid_axis = torch.linspace(-1, 1, 5)
    _z, _y, _x = torch.meshgrid(_grid_axis, _grid_axis, _grid_axis, indexing="ij")
    signed_distance = (torch.sqrt(_x**2 + _y**2 + _z**2) - 0.7).unsqueeze(0)
    _query_xyz = torch.stack((_x, _y, _z), dim=-1).reshape(-1, 3)
    _query_targets = signed_distance[0].reshape(-1, 1)
    print("SDF grid (C,D,H,W):", tuple(signed_distance.shape))
    print(
        "same grid as query positions:",
        tuple(_query_xyz.shape),
        "targets:",
        tuple(_query_targets.shape),
    )
    print("centre distance:", signed_distance[0, 2, 2, 2].item())
    print("grid axis order z,y,x; query feature order x,y,z")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13. Motion capture and animated geometry

    For motion capture we need to preserve both the order of the frames and the identity of each joint. A clip containing joint positions can have shape <span data-tooltip="T = frames in the clip; J = joints in a fixed order; 3 = X, Y and Z coordinates per joint."><code>(T, J, 3)</code></span>. A batch of clips with matching shapes becomes <span data-tooltip="B = clips in the batch; T = frames per clip; J = joints in a fixed order; 3 = X, Y and Z coordinates per joint."><code>(B, T, J, 3)</code></span>.

    If the model expects one feature vector per frame, we can flatten the joint and coordinate axes to give <span data-tooltip="B = clips in the batch; T = frames per clip; J*3 = all joint XYZ coordinates arranged along one feature axis."><code>(B, T, J*3)</code></span>. Every clip must use the same joint order, so each feature refers to the same joint and coordinate across the dataset.

    Positions alone do not tell us how the joints connect. We also need joint names, parent indices, frame rate, units and the coordinate convention. Another approach is to store local joint rotations alongside the root translation:

    | Data | Shape | Meaning |
    | --- | --- | --- |
    | Joint rotations | <span data-tooltip="B = clips in the batch; T = frames per clip; J = joints; R = rotation components per joint."><code>(B, T, J, R)</code></span> | A rotation for each joint at each frame |
    | Root translation | <span data-tooltip="B = clips in the batch; T = frames per clip; 3 = X, Y and Z root-position components."><code>(B, T, 3)</code></span> | The root position at each frame |

    Here `R` depends on how we represent rotations. We might use three Euler angles or four quaternion components. For Euler angles we need the rotation order and angle units. For quaternions we need the component order, such as `(w, x, y, z)` or `(x, y, z, w)`. We also need to record local versus world space, the rest pose and any joint offsets required to reconstruct the skeleton.

    Rotation values have constraints. Quaternion outputs need to be normalised to unit length, with near-zero predictions handled explicitly. We also need to account for `q` and `-q` representing the same rotation when comparing predictions with targets. Producing the expected number of values does not guarantee a valid rotation.

    For motion forecasting, we use earlier frames as `X` and later frames as `y`. We should split the data into training, validation and test groups before creating overlapping windows. The grouping depends on what we want to evaluate: new recordings, sessions or actors. Otherwise, almost identical sections of motion may appear in both training and test data. If we calculate input velocities or accelerations, those calculations must only use information available at prediction time.

    We can apply the same approach to animated geometry. Mesh positions can have shape <span data-tooltip="B = animation clips in the batch; T = frames per clip; V = vertices with fixed identities; 3 = X, Y and Z coordinates per vertex."><code>(B, T, V, 3)</code></span>, provided the topology and vertex identities remain fixed across frames and samples. We keep the face indices alongside the positions.

    For blendshape animation, we might instead store weights with shape <span data-tooltip="B = animation clips in the batch; T = frames per clip; number_of_shapes = blendshapes in a fixed, named order."><code>(B, T, number_of_shapes)</code></span>. These weights only make sense with the corresponding base mesh, blendshape definitions and shape order.
    """)
    return


@app.cell
def _(plt, torch):
    _mocap_time = torch.arange(8, dtype=torch.float32) / 30.0
    _rest_pose = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 1.5, 0.0]])
    joint_parents = torch.tensor([-1, 0, 1])
    _moving_pose = _rest_pose[None, :, :].repeat(8, 1, 1)
    _moving_pose[:, :, 0] += _mocap_time[:, None]
    _moving_pose[:, 2, 1] += 0.1 * torch.sin(2 * torch.pi * 3 * _mocap_time)
    mocap = torch.stack((_moving_pose, _moving_pose + torch.tensor([0.0, 0.0, 1.0])))
    past_poses = mocap[:, :5]
    future_poses = mocap[:, 5:]
    print("clip batch (B,T,J,XYZ):", tuple(mocap.shape))
    print("forecast X:", tuple(past_poses.shape), "y:", tuple(future_poses.shape))
    print(
        "sequence-model X (B,T,J*3):",
        tuple(past_poses.flatten(start_dim=2).shape),
    )
    _motion_figure, _axes = plt.subplots(1, 2, figsize=(8, 3))
    for _axis, _frame in zip(_axes, [0, 7]):
        _pose = mocap[0, _frame]
        for _joint, _parent in enumerate(joint_parents.tolist()):
            if _parent >= 0:
                _axis.plot(
                    _pose[[_parent, _joint], 0],
                    _pose[[_parent, _joint], 1],
                    "o-",
                )
        _axis.set(
            xlabel="World X (metres)",
            ylabel="World Y (metres)",
            title=f"Frame {_frame}",
            xlim=(-0.2, 1.0),
            ylim=(-0.2, 1.8),
        )
        _axis.set_aspect("equal")
    _motion_figure.tight_layout()
    plt.close(_motion_figure)
    _motion_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14. Combine modalities by identity and time

    One training sample can contain several types of data. Suppose we take a 0.25-second section of a recording containing video, audio and motion capture:

    | Modality | Sampling rate | Data in the interval |
    | --- | --- | --- |
    | Video | 12 fps | 3 frames |
    | Audio | 8 kHz | 2000 audio samples per channel |
    | Motion capture | 60 Hz | 15 poses |

    These lengths differ because each modality samples time at a different rate. We can keep them in a dictionary of tensors and metadata, with each entry describing part of the same take.

    When we form a batch, the corresponding entries in each modality must refer to the same take and time interval. We also need timestamps that use a common clock, or a known mapping between clocks. Matching durations are not enough to establish synchronisation: recordings can start at different times, drift apart or contain dropped frames.

    There are several stages at which we can combine the data:

    | Approach | How we combine the modalities | Example |
    | --- | --- | --- |
    | Early fusion | Align measurements or input features, then concatenate them | Add registered depth as another channel alongside RGB |
    | Intermediate fusion | Encode each modality separately, then align or pool the representations and combine them | Combine video, audio and mocap embeddings for the same interval |
    | Late fusion | Combine predictions from separate models | Combine audio and video class scores using a shared class vocabulary |

    For late fusion, we also need to consider how the scores are calibrated. Two models may use the same classes without producing scores that are directly comparable.

    Resampling can put measurements onto a common time grid, but the method depends on what the values represent. We should not linearly interpolate integer class IDs. Rotations need interpolation suited to their representation, and waveform downsampling needs filtering to avoid aliasing.

    In the example below I use fixed toy encoders that average each modality over time or space. This lets us inspect the shapes and see how the resulting features combine. Averaging discards temporal order, so these encoders cannot describe how an action develops over time. They are not trained recognition models; for a temporal task we would choose a model that retains the relevant sequence structure.
    """)
    return


@app.cell
def _(torch, waveform):
    multimedia_sample = {
        "video": torch.zeros(3, 3, 8, 12),  # T,C,H,W
        "audio": waveform.clone(),  # C,S
        "pose": torch.zeros(15, 3, 3),  # T,J,XYZ
        "metadata": {
            "sample_id": "take_001",
            "fps": 12,
            "sample_rate": 8000,
            "mocap_rate": 60,
            "start_time_seconds": 0.0,
        },
    }
    print(
        "one take:",
        {k: tuple(v.shape) for k, v in multimedia_sample.items() if torch.is_tensor(v)},
    )
    print("metadata:", multimedia_sample["metadata"])
    _video_embedding = multimedia_sample["video"].mean(dim=(0, 2, 3)).unsqueeze(0)
    _audio_embedding = (
        multimedia_sample["audio"].square().mean(dim=-1).sqrt().unsqueeze(0)
    )
    _pose_embedding = multimedia_sample["pose"].mean(dim=0).flatten().unsqueeze(0)
    fused_embedding = torch.cat(
        (_video_embedding, _audio_embedding, _pose_embedding), dim=-1
    )
    print(
        "video / audio / pose embeddings:",
        tuple(_video_embedding.shape),
        tuple(_audio_embedding.shape),
        tuple(_pose_embedding.shape),
    )
    print("combined embedding (B,3+2+9):", tuple(fused_embedding.shape))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Unequal clocks: align a derived feature

    Here each video frame uses the audio feature whose timestamp is nearest its frame time. This is one explicit alignment policy, not something we infer from the array lengths. For a causal model, we would instead use only information available by the prediction time and avoid future-looking windows.

    Real data may also have a modality missing for an entire take. Keep a modality-valid mask and train the model to handle missing inputs; padding with zeros alone does not say whether silence was recorded or audio was absent.
    """)
    return


@app.cell
def _(torch, waveform):
    video_timestamps = torch.arange(3, dtype=torch.float32) / 12
    _audio_window_centres = (torch.arange(25, dtype=torch.float32) * 80 + 40) / 8000
    _audio_window_features = waveform.reshape(2, 25, 80).square().mean(dim=-1).sqrt().T
    _nearest = (
        (video_timestamps[:, None] - _audio_window_centres[None, :]).abs().argmin(dim=1)
    )
    aligned_audio_features = _audio_window_features[_nearest]
    print("video times (seconds):", video_timestamps)
    print("chosen audio window centres:", _audio_window_centres[_nearest])
    print(
        "aligned audio features (Tvideo,C):",
        tuple(aligned_audio_features.shape),
    )
    print(
        "first frame uses a window centred after time zero: suitable here for offline processing"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 15. Write the sample contract down

    A useful description includes more than `shape=(...)`. We need enough information to interpret the values at inference and to rebuild the preprocessing. This can be integrated into unit tests for classes / data processors.

    | Field | Example |
    | --- | --- |
    | Sample identity | One take, identified by recording and time interval |
    | Task | Predict action class from video and audio |
    | Logical axes | Video <span data-tooltip="B = independent samples in the batch; C = channels, such as RGB or stereo; T = ordered frames or time steps; H = height in rows; W = width in columns."><code>B,C,T,H,W</code></span>; audio <span data-tooltip="B = independent samples in the batch; C = channels, such as RGB or stereo; S = waveform samples per channel."><code>B,C,S</code></span> |
    | Dtypes | Inputs float32; class IDs int64; masks bool |
    | Values and units | RGB `[0,1]`; audio amplitude; poses in metres |
    | Timing | FPS, sample rate, timestamps, offset and time origin |
    | Coordinate conventions | Joint order, up axis, handedness, local/world space |
    | Missingness | Lengths, validity masks, absent-modality flags |
    | Target | One class per take; logits <span data-tooltip="B = independent samples in the batch; K = output classes."><code>B,K</code></span>; targets <span data-tooltip="B = independent samples in the batch."><code>B</code></span> |
    | Preprocessing | Crop/resize, resampling, training-set scaling statistics |
    | Split policy | Separate source takes, subjects or sessions before windowing |

    A practical pipeline is: decode the source, verify identity and timing, transform values into the chosen representation, attach the target and metadata, then collate samples into a batch. A `Dataset` usually describes one item and a `DataLoader` collates several items. Custom collation handles variable lengths or topology.

    Keep a class-name-to-ID mapping with the model. At inference, ID 2 is only useful if we know which class it means.
    """)
    return


@app.cell
def _(class_targets, image_batch, torch):
    image_training_batch = {
        "inputs": {"image": image_batch},
        "targets": {"class_id": class_targets},
        "metadata": {
            "sample_ids": ["image_001", "image_002"],
            "axes": "B,C,H,W",
            "colour_space": "sRGB",
            "range": [0.0, 1.0],
            "class_names": ["chair", "table", "lamp"],
        },
    }
    _x = image_training_batch["inputs"]["image"]
    _y = image_training_batch["targets"]["class_id"]
    assert _x.ndim == 4 and _x.shape[1] == 3
    assert _x.dtype == torch.float32 and torch.isfinite(_x).all()
    assert _x.min() >= 0 and _x.max() <= 1
    assert _y.shape == (_x.shape[0],) and _y.dtype == torch.int64
    assert _y.min() >= 0 and _y.max() < len(
        image_training_batch["metadata"]["class_names"]
    )
    print("Image batch agrees with its documented input and target contract.")
    print("Targets are separate:", image_training_batch["targets"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 16. A shape reference

    I use the following layouts as starting points. The model or operation may expect a different axis order, so we still need to check its requirements and document our choice. Here `K` is the number of classes, `T` is the input sequence length and `Tout` is the number of future time steps we want to predict.

    | Data and task | Batched inputs | Model output | Training target |
    | --- | --- | --- | --- |
    | Tabular classification | <span data-tooltip="B = samples in the batch; F = numerical features per sample."><code>(B, F)</code></span>, plus any category IDs | <span data-tooltip="B = samples in the batch; K = classes."><code>(B, K)</code></span> logits | <span data-tooltip="B = samples in the batch, with one integer class label per sample."><code>(B,)</code></span> class IDs |
    | RGB image classification | <span data-tooltip="B = images in the batch; 3 = RGB channels; H = image height; W = image width."><code>(B, 3, H, W)</code></span> | <span data-tooltip="B = images in the batch; K = classes."><code>(B, K)</code></span> logits | <span data-tooltip="B = images in the batch, with one integer class label per image."><code>(B,)</code></span> class IDs |
    | Semantic segmentation | <span data-tooltip="B = images in the batch; 3 = RGB channels; H = image height; W = image width."><code>(B, 3, H, W)</code></span> | <span data-tooltip="B = images in the batch; K = classes; H = image height; W = image width."><code>(B, K, H, W)</code></span> logits | <span data-tooltip="B = images in the batch; H = image height; W = image width, with one integer class label per pixel."><code>(B, H, W)</code></span> class IDs |
    | Video clip classification | <span data-tooltip="B = clips in the batch; C = colour channels; T = frames per clip; H = image height; W = image width."><code>(B, C, T, H, W)</code></span> | <span data-tooltip="B = clips in the batch; K = classes."><code>(B, K)</code></span> logits | <span data-tooltip="B = clips in the batch, with one integer class label per clip."><code>(B,)</code></span> class IDs |
    | Audio waveform classification | <span data-tooltip="B = recordings in the batch; C = audio channels; S = waveform samples per channel."><code>(B, C, S)</code></span> | <span data-tooltip="B = recordings in the batch; K = classes."><code>(B, K)</code></span> logits | <span data-tooltip="B = recordings in the batch, with one integer class label per recording."><code>(B,)</code></span> class IDs |
    | Spectrogram classification | <span data-tooltip="B = recordings in the batch; C = audio channels; Q = frequency bins; L = analysis frames."><code>(B, C, Q, L)</code></span> | <span data-tooltip="B = recordings in the batch; K = classes."><code>(B, K)</code></span> logits | <span data-tooltip="B = recordings in the batch, with one integer class label per recording."><code>(B,)</code></span> class IDs |
    | Audio denoising | <span data-tooltip="B = recordings in the batch; C = audio channels; S = waveform samples per channel."><code>(B, C, S)</code></span> noisy waveforms | <span data-tooltip="B = recordings in the batch; C = audio channels; S = waveform samples per channel."><code>(B, C, S)</code></span> clean estimates | <span data-tooltip="B = recordings in the batch; C = audio channels; S = waveform samples per channel."><code>(B, C, S)</code></span> clean waveforms |
    | Point segmentation | <span data-tooltip="B = point clouds in the batch; N = points per cloud; F = features per point."><code>(B, N, F)</code></span> | <span data-tooltip="B = point clouds in the batch; K = classes; N = points per cloud."><code>(B, K, N)</code></span> logits | <span data-tooltip="B = point clouds in the batch; N = points per cloud, with one integer class label per point."><code>(B, N)</code></span> class IDs |
    | Shared-topology mesh deformation | <span data-tooltip="B = meshes in the batch; V = corresponding vertices per mesh; 3 = XYZ coordinates."><code>(B, V, 3)</code></span> rest positions, plus faces and controls | <span data-tooltip="B = meshes in the batch; V = corresponding vertices per mesh; 3 = predicted XYZ coordinates."><code>(B, V, 3)</code></span> deformed positions | <span data-tooltip="B = meshes in the batch; V = corresponding vertices per mesh; 3 = target XYZ coordinates."><code>(B, V, 3)</code></span> deformed positions |
    | Mocap forecasting | <span data-tooltip="B = clips in the batch; T = input frames; J = joints in a fixed order; 3 = XYZ coordinates per joint."><code>(B, T, J, 3)</code></span>, plus skeleton hierarchy | <span data-tooltip="B = clips in the batch; Tout = future frames to predict; J = joints; 3 = XYZ coordinates per joint."><code>(B, Tout, J, 3)</code></span> future positions | <span data-tooltip="B = clips in the batch; Tout = future frames; J = joints; 3 = XYZ coordinates per joint."><code>(B, Tout, J, 3)</code></span> future positions |
    | Voxel occupancy prediction | <span data-tooltip="B = grids in the batch; C = input feature channels; D = grid depth; H = grid height; W = grid width."><code>(B, C, D, H, W)</code></span> | <span data-tooltip="B = grids in the batch; 1 = one occupancy logit per voxel; D = grid depth; H = grid height; W = grid width."><code>(B, 1, D, H, W)</code></span> logits | <span data-tooltip="B = grids in the batch; 1 = one occupancy value per voxel; D = grid depth; H = grid height; W = grid width."><code>(B, 1, D, H, W)</code></span> floating-point occupancy values |
    | Multimodal classification | A dictionary of modalities aligned by sample identity and time | <span data-tooltip="B = multimodal samples in the batch; K = classes."><code>(B, K)</code></span> logits | <span data-tooltip="B = multimodal samples in the batch, with one integer class label per sample."><code>(B,)</code></span> class IDs |

    The multiclass examples use integer class targets with the class-index form of cross entropy. Binary voxel occupancy can use binary cross entropy with logits, with floating-point targets of the same shape as the output. The regression examples use floating-point targets.

    These shapes describe the numerical arrays, but we may also need masks, sequence lengths, mesh topology, skeleton definitions or camera calibration. We keep that information alongside the tensors so we can interpret and process them correctly.

    Before passing data into a model, I would check three things: what does each axis index, where did the values come from, and will the required input information be available when we ask the trained model to make a prediction?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Read one RGB pixel from the generated image in both HWC and CHW layouts. Write the index expressions before running them. Repeat with a non-square image of your own.
    2. Make a batch of greyscale images and separate class labels. Change the task to segmentation. What changes in the inputs, model output and target?
    3. Add depth and a depth-validity mask to the image batch. Explain which channels have units and how you would scale them.
    4. Start with <span data-tooltip="B = independent samples in the batch; T = ordered frames or time steps; C = channels, such as RGB or stereo; H = height in rows; W = width in columns."><code>(B,T,C,H,W)</code></span> video. Run a tiny 2D encoder on individual frames, restore <span data-tooltip="B = independent samples in the batch; T = ordered frames or time steps; F = features per observation."><code>(B,T,F)</code></span>, then describe what a temporal classifier must add.
    5. Double the audio sample rate whilst keeping duration fixed. Predict the waveform shape. Then distinguish resampling the waveform from merely changing its sample-rate metadata.
    6. Pad two unequal motion clips. Compute a per-coordinate regression loss that excludes padded frames and justify the denominator. Add a missing-joint mask.
    7. Give the second mesh its own vertex labels. Pack both meshes, preserve sample boundaries and demonstrate that each face still refers to the intended vertices.
    8. Design a sample contract for speech-driven facial animation: inputs, outputs, blendshape order, clocks, masks, units and train/test split. Which measurements would leak future information?
    9. Take a real dataset you intend to use. Write its input/output shapes and the meaning of each axis before implementing its `Dataset`. Identify everything that must remain as metadata.

    The next step is to implement one of these contracts in a dataset and inspect a real batch before choosing a network. The existing [PyTorch data pipeline lesson](../PyTorchForML/PyTorchForMLPart5DataAndSaving.py) follows that path.
    """)
    return


if __name__ == "__main__":
    app.run()

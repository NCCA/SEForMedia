#!/usr/bin/env uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # torchvision for Machine Learning, Part 4: datasets and pre-trained models

    We will finish by loading images from class folders and preparing a model for transfer learning. This follows the approach used in the `PreTrainedModels/` demos.

    I use `weights=None` in the runnable cells so we can inspect the architecture without downloading weights. These models are randomly initialised: they show the structure and tensor shapes, but do not give useful predictions. The code snippets show where to request pre-trained weights when we want to train a new classifier.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    from torchvision import datasets
    from torchvision.transforms import v2

    return datasets, nn, plt, torch, v2


@app.cell
def _(plt, torch, v2):
    def image_grid(
        panels: list[tuple[str, torch.Tensor]], columns: int = 3
    ) -> plt.Figure:
        """Show labelled CHW images with a fixed display range."""
        rows = (len(panels) + columns - 1) // columns
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(3.6 * columns, 3.3 * rows),
            squeeze=False,
            layout="constrained",
        )
        for axis in axes.flat:
            axis.axis("off")
        for axis, (title, tensor) in zip(axes.flat, panels):
            pixels = tensor.detach().cpu()
            if pixels.dtype != torch.uint8:
                pixels = pixels.clamp(0, 1)
            axis.imshow(pixels.permute(1, 2, 0), interpolation="nearest")
            axis.set_title(title, fontsize=10)
        plt.close(figure)
        return figure

    def as_tensor(image) -> torch.Tensor:
        """Convert a PIL sample for plotting without changing its pixel range."""
        return v2.functional.to_image(image)

    return as_tensor, image_grid


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [ImageFolder](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html)

    ```python
    datasets.ImageFolder(root, transform=None, target_transform=None,
                         loader=default_loader, is_valid_file=None, allow_empty=False)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `root` | required | a directory with one subdirectory per class |
    | `transform` | `None` | applied to each image |
    | `target_transform` | `None` | applied to each label |
    | `loader` | `default_loader` | PIL by default; swap for `read_image` if you prefer tensors |

    `ImageFolder` gives us a dataset when the images are arranged in one directory per class:

    ```
    train/
        cats/
            img001.jpg
            img002.jpg
        dogs/
            img001.jpg
    ```

    It sorts the directory names and records the label mapping in `class_to_idx`. We need that same mapping when interpreting predictions, so save it with the model.

    The next cell creates three folders in a different order from their names. Check the indices assigned by `ImageFolder`, then compare the samples with and without a transform.
    """)
    return


@app.cell
def _(as_tensor, datasets, image_grid, mo, plt):
    import tempfile
    from pathlib import Path

    from PIL import Image, ImageDraw

    dataset_directory = tempfile.TemporaryDirectory(prefix="torchvision-datasets-")
    root = Path(dataset_directory.name) / "train"
    # these drawings are stand-ins for photographs, not a useful training dataset
    for _cls, _n_files, _colour in [
        ("dogs", 4, "#dfa65a"),
        ("cats", 3, "#86bed3"),
        ("axolotls", 2, "#e99bbb"),
    ]:
        (root / _cls).mkdir(parents=True)
        for _i in range(_n_files):
            _image = Image.new("RGB", (96, 72), "#edf1f5")
            _draw = ImageDraw.Draw(_image)
            _x = 46 + 2 * _i
            if _cls == "cats":
                _draw.polygon([(_x - 24, 34), (_x - 23, 6), (_x - 5, 22)], fill=_colour)
                _draw.polygon([(_x + 24, 34), (_x + 23, 6), (_x + 5, 22)], fill=_colour)
            elif _cls == "dogs":
                _draw.ellipse((_x - 34, 18, _x - 12, 59), fill="#926237")
                _draw.ellipse((_x + 12, 18, _x + 34, 59), fill="#926237")
            else:
                for _y in [22, 34, 46]:
                    _draw.line([(_x - 18, 36), (_x - 37, _y)], fill="#cc5a86", width=4)
                    _draw.line([(_x + 18, 36), (_x + 37, _y)], fill="#cc5a86", width=4)
            _draw.ellipse((_x - 24, 16, _x + 24, 61), fill=_colour)
            for _eye in [_x - 10, _x + 10]:
                _draw.ellipse((_eye - 2, 31, _eye + 2, 35), fill="#253047")
            _draw.arc((_x - 8, 35, _x + 8, 47), 0, 180, fill="#253047", width=2)
            _image.save(root / _cls / f"{_i}.png")

    folder = datasets.ImageFolder(root)
    _counts = [folder.targets.count(_label) for _label in range(len(folder.classes))]
    _figure, _axis = plt.subplots(figsize=(8, 2.5), layout="constrained")
    _bars = _axis.barh(folder.classes, _counts, color=["#cc5a86", "#559bb8", "#be843b"])
    _axis.bar_label(_bars, padding=4)
    _axis.set_xlim(0, max(_counts) + 1)
    _axis.set_xticks(range(max(_counts) + 1))
    _axis.set_xlabel("Images in each class")
    _axis.invert_yaxis()
    plt.close(_figure)
    mo.vstack(
        [
            mo.md(
                "These small drawings stand in for photographs. The folders were created as dogs, cats, axolotls; ImageFolder assigns indices alphabetically."
            ),
            mo.ui.table(
                [
                    {
                        "Folder": _name,
                        "Class index": folder.class_to_idx[_name],
                        "Images": _count,
                    }
                    for _name, _count in zip(folder.classes, _counts)
                ],
                selection=None,
            ),
            image_grid(
                [
                    (
                        f"{folder.classes[_label]} | label {_label} | sample {_i}",
                        as_tensor(_image),
                    )
                    for _i, (_image, _label) in enumerate(folder)
                ]
            ),
            _figure,
        ]
    )
    return dataset_directory, folder, root


@app.cell
def _(datasets, root, torch, v2):
    with_transform = datasets.ImageFolder(
        root,
        transform=v2.Compose(
            [v2.Resize((32, 32)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        ),
    )
    return (with_transform,)


@app.cell(hide_code=True)
def _(folder, mo):
    sample_index = mo.ui.slider(
        0, len(folder) - 1, value=0, label="Dataset sample", show_value=True
    )
    sample_index
    return (sample_index,)


@app.cell
def _(as_tensor, folder, image_grid, mo, sample_index, with_transform):
    _index = sample_index.value
    _original, _label = folder[_index]
    _transformed, _ = with_transform[_index]
    mo.vstack(
        [
            mo.md(
                f"`{folder.classes[_label]}/{_index - folder.targets.index(_label)}.png` → class **{_label}** → **{folder.classes[_label]}**"
            ),
            image_grid(
                [
                    (
                        f"PIL image: {_original.width} × {_original.height}\nuint8 pixels, range 0–255",
                        as_tensor(_original),
                    ),
                    (
                        f"Tensor: {tuple(_transformed.shape)}\n{_transformed.dtype}, range {_transformed.min():.2f}–{_transformed.max():.2f}",
                        _transformed,
                    ),
                ],
                columns=2,
            ),
            mo.md(
                "The transform resizes the image and converts its pixels. The class index stays the same. Save `class_to_idx` with the model."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    batch_size = mo.ui.slider(1, 9, value=4, label="Batch size", show_value=True)
    shuffle_samples = mo.ui.checkbox(value=True, label="Shuffle samples")
    mo.vstack(
        [
            mo.md(
                "### From samples to a batch\n\nA DataLoader stacks images into NCHW order and keeps their labels aligned. Toggle shuffle to compare the first batch with the folder order. We use a fixed seed so the shuffled order is repeatable."
            ),
            mo.hstack([batch_size, shuffle_samples]),
        ]
    )
    return batch_size, shuffle_samples


@app.cell
def _(batch_size, image_grid, mo, shuffle_samples, torch, with_transform):
    from torch.utils.data import DataLoader

    _loader = DataLoader(
        with_transform,
        batch_size=batch_size.value,
        shuffle=shuffle_samples.value,
        generator=torch.Generator().manual_seed(42),
    )
    batch_images, batch_labels = next(iter(_loader))
    mo.vstack(
        [
            mo.md(
                f"Images: **{tuple(batch_images.shape)}** (N, C, H, W) · labels: **{tuple(batch_labels.shape)}**"
            ),
            image_grid(
                [
                    (
                        f"Batch slot {_i} → label {int(_label)}\n{with_transform.classes[int(_label)]}",
                        _image,
                    )
                    for _i, (_image, _label) in enumerate(
                        zip(batch_images, batch_labels)
                    )
                ]
            ),
        ]
    )
    return batch_images, batch_labels


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Built-in datasets

    ```python
    datasets.MNIST(root, train=True, transform=None, download=False)
    datasets.CIFAR10(root, train=True, transform=None, download=False)
    ```

    For these datasets, `download=True` fetches the files if a valid local copy is missing. In a lab without network access we need to prepare the data first and point `root` at it.

    The examples in `MNIST/` show both reading the IDX files with NumPy and using `datasets.MNIST`. I like to look at the file layout first so we can see what the dataset loader is doing for us.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choosing weights

    ```python
    from torchvision.models import vgg16, VGG16_Weights

    weights = VGG16_Weights.DEFAULT
    model = vgg16(weights=weights)  # downloads the weights if they are not cached
    model = vgg16(weights=None)    # randomly initialised parameters
    ```

    The weights enum identifies a set of trained parameters. Older examples may use `pretrained=True`; the weights API lets us select a particular version instead.

    `DEFAULT` is an alias and may change between torchvision releases. For an experiment we need to reproduce, record the library version and the explicit weights member, such as `VGG16_Weights.IMAGENET1K_V1`.

    The next cell reads the enum's metadata without loading the weights.
    """)
    return


@app.cell
def _():
    from torchvision.models import VGG16_Weights

    print("DEFAULT resolves to:", VGG16_Weights.DEFAULT)
    print("available for VGG16:", [w.name for w in VGG16_Weights])
    print()
    print("the file it would fetch:")
    print("  ", VGG16_Weights.DEFAULT.url)
    print(f"   {VGG16_Weights.DEFAULT.meta['num_params']:,} parameters")
    return (VGG16_Weights,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Preprocessing from the weights

    ```python
    pre_trans = weights.transforms()
    ```

    The weights object provides the preprocessing for inference. This includes the resize, crop, type conversion and normalisation expected by that set of weights. I use this preset so the preprocessing stays tied to the model we have selected.

    The next cells inspect the VGG16 preset and show its resize and crop on a generated colour pattern. Creating the preset does not download the model weights.
    """)
    return


@app.cell
def _(VGG16_Weights):
    preset = VGG16_Weights.DEFAULT.transforms()
    print(preset)
    print()
    print("the preset includes resizing, centre cropping and normalisation.")
    print("compare its settings with the pipeline from Part 2.")
    print("here the weights object supplies the settings.")
    return (preset,)


@app.cell
def _(image_grid, mo, plt, preset, torch):
    from matplotlib.patches import Rectangle
    from torchvision.transforms import functional as preset_functional

    # gradients and coloured blocks make the crop easier to follow than noise
    fake_photo = torch.zeros(3, 240, 400, dtype=torch.uint8)
    fake_photo[0] = torch.linspace(0, 255, 400).to(torch.uint8)[None, :]
    fake_photo[2] = torch.linspace(0, 255, 240).to(torch.uint8)[:, None]
    fake_photo[1, 30:105, 25:105] = 240
    fake_photo[:, 130:205, 285:365] = 230
    _resized = preset_functional.resize(
        fake_photo,
        preset.resize_size,
        interpolation=preset.interpolation,
        antialias=preset.antialias,
    )
    _cropped = preset_functional.center_crop(_resized, preset.crop_size)
    ready = preset(fake_photo)
    restored = ready * torch.tensor(preset.std).view(3, 1, 1) + torch.tensor(
        preset.mean
    ).view(3, 1, 1)
    _height, _width = _resized.shape[-2:]
    _crop_height, _crop_width = ready.shape[-2:]
    _figure, _axis = plt.subplots(figsize=(7, 4), layout="constrained")
    _axis.imshow(_resized.permute(1, 2, 0))
    _axis.add_patch(
        Rectangle(
            (
                round((_width - _crop_width) / 2) - 0.5,
                round((_height - _crop_height) / 2) - 0.5,
            ),
            _crop_width,
            _crop_height,
            fill=False,
            edgecolor="yellow",
            linewidth=2,
        )
    )
    _axis.set_title(f"Resized: {_width} × {_height}; yellow box is the centre crop")
    _axis.axis("off")
    plt.close(_figure)
    mo.vstack(
        [
            _figure,
            image_grid(
                [
                    ("Source: 400 × 240", fake_photo),
                    ("Centre crop: 224 × 224", _cropped),
                    ("Preset output: normalisation undone", restored),
                ]
            ),
            mo.md(
                f"The model receives `{tuple(ready.shape)}` floats, from **{ready.min():.2f} to {ready.max():.2f}**. We undo normalisation for the preview; displaying the normalised tensor directly would distort its colours."
            ),
        ]
    )
    return fake_photo, ready, restored


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Class names and metadata

    `weights.meta` includes the original class names and information about the trained model. For the ImageNet classifier we can use the predicted index to look up a name in `categories`.

    Once we replace the classifier for our own dataset, we need our own class mapping instead.
    """)
    return


@app.cell
def _(VGG16_Weights, mo):
    meta = VGG16_Weights.DEFAULT.meta
    category_index = mo.ui.slider(
        0,
        len(meta["categories"]) - 1,
        value=207,
        label="ImageNet class index",
        show_value=True,
    )
    mo.vstack(
        [
            mo.md(
                f"The original classifier has **{len(meta['categories'])} classes**. Choose an index to decode it using the weights metadata."
            ),
            category_index,
        ]
    )
    return category_index, meta


@app.cell(hide_code=True)
def _(category_index, folder, meta, mo):
    mo.vstack(
        [
            mo.md(
                f"ImageNet index **{category_index.value}** → **{meta['categories'][category_index.value]}**"
            ),
            mo.md(
                "After replacing the classifier for our folders, the output columns have a different meaning:"
            ),
            mo.ui.table(
                [
                    {"New output column": _index, "Our class": _name}
                    for _name, _index in folder.class_to_idx.items()
                ],
                selection=None,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Transfer learning

    A model trained on a large image dataset can provide useful features for another task. We can keep those layers and train a new classifier on our own classes. How well this works depends on the images and the task.

    For the example below we will freeze the existing parameters, then add a trainable output layer. This is one approach to transfer learning; we can also fine-tune some of the existing layers later.

    First we build VGG16 with `weights=None` and inspect its parts. Remember that this version has no learned features to transfer.
    """)
    return


@app.cell
def _(mo, plt, torch):
    from torchvision.models import vgg16

    with torch.random.fork_rng():
        torch.manual_seed(42)
        vgg = vgg16(weights=None)
    _names = []
    _counts = []
    for _name, _part in vgg.named_children():
        _names.append(_name)
        _counts.append(sum(_parameter.numel() for _parameter in _part.parameters()))
    _figure, _axis = plt.subplots(figsize=(9, 3), layout="constrained")
    _bars = _axis.barh(
        _names,
        [_count / 1e6 for _count in _counts],
        color=["#559bb8", "#999999", "#be843b"],
    )
    _axis.bar_label(_bars, labels=[f"{_count:,}" for _count in _counts], padding=5)
    _axis.set_xlim(0, max(_counts) / 1e6 * 1.25)
    _axis.set_xlabel("Parameters (millions)")
    _axis.invert_yaxis()
    plt.close(_figure)
    mo.vstack(
        [
            mo.md(
                "**Input** `N × 3 × 224 × 224` → **features** → **avgpool** `N × 512 × 7 × 7` → **flatten** `N × 25088` → **classifier** `N × 1000`"
            ),
            _figure,
            mo.md(
                "Most VGG16 parameters are in the classifier. Average pooling has no learned parameters. This model uses `weights=None`; its features are randomly initialised."
            ),
        ]
    )
    return vgg, vgg16


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Freezing with [requires_grad_](https://pytorch.org/docs/stable/generated/torch.nn.Module.requires_grad_.html)

    ```python
    model.requires_grad_(False)
    ```

    This sets `requires_grad=False` on the module's parameters. We call it before training so autograd does not accumulate gradients for them. The trailing underscore indicates that the method changes the module in place.

    The next chart compares trainable and frozen parameters before and after the call. Freezing parameters is separate from `eval()`, which changes the behaviour of layers such as dropout. Use evaluation mode when checking predictions.
    """)
    return


@app.cell
def _(nn, plt, vgg):
    def count(model: nn.Module) -> tuple[int, int]:
        """Count trainable and frozen parameters separately."""
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        frozen = sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        )
        return trainable, frozen

    # restore the starting state so rerunning this cell gives the same comparison
    vgg.requires_grad_(True)
    _before = count(vgg)
    frozen_vgg = vgg.requires_grad_(False)
    _after = count(frozen_vgg)
    _figure, _axis = plt.subplots(figsize=(9, 3), layout="constrained")
    _trainable = [_before[0] / 1e6, _after[0] / 1e6]
    _frozen = [_before[1] / 1e6, _after[1] / 1e6]
    _axis.barh(
        ["As loaded", "After freezing"], _trainable, color="#be843b", label="Trainable"
    )
    _axis.barh(
        ["As loaded", "After freezing"],
        _frozen,
        left=_trainable,
        color="#559bb8",
        label="Frozen",
    )
    _axis.set_xlabel("Parameters (millions)")
    _axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2)
    _axis.invert_yaxis()
    plt.close(_figure)
    _figure
    return count, frozen_vgg


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Adding a classifier

    The transfer learning demo adds a layer after VGG's existing output:

    ```python
    dog_model = nn.Sequential(vgg_model, nn.Linear(1000, N_CLASSES))
    ```

    That layer receives the 1000 class scores. We can instead replace VGG's final layer:

    ```python
    vgg.classifier[6] = nn.Linear(4096, N_CLASSES)
    ```

    The replacement receives the 4096 features from the previous layer. We will build both versions, count their trainable parameters and check their output shapes. Comparing their accuracy would require pre-trained weights and a training run on the same dataset.
    """)
    return


@app.cell
def _(count, folder, frozen_vgg, mo, nn, plt, torch, vgg16):
    N_CLASSES = len(folder.classes)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        stacked = nn.Sequential(frozen_vgg, nn.Linear(1000, N_CLASSES))
        replaced = vgg16(weights=None)
        replaced.requires_grad_(False)
        replaced.classifier[6] = nn.Linear(4096, N_CLASSES)
    stacked.eval()
    replaced.eval()

    _figure, _axes = plt.subplots(2, 1, figsize=(12, 4.5), layout="constrained")
    for _axis, _name, _stages in [
        (
            _axes[0],
            "Stack a layer",
            [
                "Frozen VGG16",
                "1000 scores",
                f"Trainable Linear\n1000 → {N_CLASSES}",
                f"{N_CLASSES} class scores",
            ],
        ),
        (
            _axes[1],
            "Replace the head",
            [
                "Frozen VGG16\nup to classifier[5]",
                "4096 features",
                f"Trainable Linear\n4096 → {N_CLASSES}",
                f"{N_CLASSES} class scores",
            ],
        ),
    ]:
        _axis.set_xlim(-0.6, 3.6)
        _axis.set_ylim(-0.5, 0.5)
        _axis.axis("off")
        _axis.set_title(_name, loc="left")
        for _i, _stage in enumerate(_stages):
            _axis.text(
                _i,
                0,
                _stage,
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.7",
                    "facecolor": "#f4d9b4" if _i == 2 else "#dceaf0",
                    "edgecolor": "#607080",
                },
            )
            if _i < 3:
                _axis.annotate(
                    "",
                    xy=(_i + 0.65, 0),
                    xytext=(_i + 0.35, 0),
                    arrowprops={"arrowstyle": "->"},
                )
    plt.close(_figure)
    _sample = torch.zeros(2, 3, 224, 224)
    with torch.inference_mode():
        stacked_output = stacked(_sample)
        replaced_output = replaced(_sample)
    mo.vstack(
        [
            _figure,
            mo.ui.table(
                [
                    {
                        "Approach": _name,
                        "Trainable parameters": count(_model)[0],
                        "Frozen parameters": count(_model)[1],
                        "Output shape": str(tuple(_output.shape)),
                    }
                    for _name, _model, _output in [
                        ("Stacked", stacked, stacked_output),
                        ("Replaced", replaced, replaced_output),
                    ]
                ],
                selection=None,
            ),
            mo.md(
                f"Both produce one score per class: **{' · '.join(folder.classes)}**. The values are untrained scores, not useful predictions. Both models are in evaluation mode for the shape check."
            ),
        ]
    )
    return N_CLASSES, replaced, replaced_output, stacked, stacked_output


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We add the new `nn.Linear` after freezing the existing model. Its parameters are trainable by default. If we freeze the whole model after adding it, we freeze the new layer too.

    We can pass just the trainable parameters to the optimiser:

    ```python
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    ```

    This makes our choice explicit. Passing all the parameters also works here: the frozen parameters have no gradients, and [Adam skips parameters whose gradient is None](https://github.com/pytorch/pytorch/blob/main/torch/optim/adam.py). It does not allocate per-parameter state for them. If we change what is frozen during training, we also need to consider existing gradients and optimiser state.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preparing a training run

    ```python
    from torch import nn, optim
    from torchvision.models import vgg16, VGG16_Weights
    from torchvision import datasets
    from torch.utils.data import DataLoader

    weights = VGG16_Weights.DEFAULT
    model = vgg16(weights=weights)
    model.requires_grad_(False)                        # freeze the backbone
    model.classifier[6] = nn.Linear(4096, N_CLASSES)   # new, trainable head
    model = model.to(device)

    train_data = datasets.ImageFolder("data/train", transform=weights.transforms())
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    loss_fn = nn.CrossEntropyLoss()
    ```

    Set `N_CLASSES` and `device` for your task, then use the training loop from the PyTorch notebooks. This snippet downloads the weights if needed; the runnable examples above do not.

    I have used the weights' preprocessing as a starting point. We can add suitable training augmentation after checking it on the dataset. Keep the class mapping with the saved model so we can interpret its output later.

    ## Exercises

    1. Create class folders named `1`, `2` and `10`. Inspect `class_to_idx` and explain the ordering.
    2. Compare the preprocessing presets for VGG16 and ResNet50. What would you need to check when changing the model?
    3. Freeze a model and replace its final layer. Print `requires_grad` for each parameter, then repeat the operations in the other order.
    4. With a small model, compare Adam's state after a training step when it receives all parameters and when it receives only trainable parameters. Freeze some parameters before constructing each optimiser and inspect which ones acquire state.
    5. Consider transfer learning for the ASL images. What preprocessing would be needed, and how would you compare the result with a model trained from scratch?

    We can now use these pieces in the image classification and transfer learning demos.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

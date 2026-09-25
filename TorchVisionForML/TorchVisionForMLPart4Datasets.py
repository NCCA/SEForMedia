#!/usr/bin/env uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App()


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
    import torch
    from torch import nn
    from torchvision import datasets
    from torchvision.transforms import v2

    torch.manual_seed(42)
    return datasets, nn, torch, v2


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
def _(datasets, torch):
    import tempfile
    from pathlib import Path

    from torchvision.io import write_png

    root = Path(tempfile.mkdtemp()) / "train"
    # deliberately created in a non-alphabetical order
    for _cls, _n_files in [("dogs", 4), ("cats", 3), ("axolotls", 2)]:
        (root / _cls).mkdir(parents=True)
        for _i in range(_n_files):
            write_png(
                torch.randint(0, 255, (3, 24, 24), dtype=torch.uint8),
                str(root / _cls / f"{_i}.png"),
            )

    folder = datasets.ImageFolder(root)

    print("created in this order: dogs, cats, axolotls")
    print("class_to_idx         :", folder.class_to_idx)
    print()
    print("total samples:", len(folder))
    print("labels in order:", [label for _, label in folder.samples])
    print()
    print("index 0 is axolotls, not dogs - it sorted them")
    return folder, root


@app.cell
def _(datasets, folder, root, torch, v2):
    with_transform = datasets.ImageFolder(
        root,
        transform=v2.Compose(
            [v2.Resize((16, 16)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        ),
    )

    _img, _label = with_transform[0]
    print(
        "with a transform:",
        tuple(_img.shape),
        _img.dtype,
        f"{_img.min():.2f} to {_img.max():.2f}",
    )
    print("label            ", _label, "->", with_transform.classes[_label])
    print()
    print("without one, you get a PIL image:", type(folder[0][0]).__name__)
    print()
    print("save this mapping with the model so we can name its predictions:")
    print("  ", with_transform.class_to_idx)
    return


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

    The next cells print the VGG16 preset and apply it to a generated image. Creating the preset does not download the model weights.
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
def _(preset, torch):
    fake_photo = torch.randint(0, 255, (3, 480, 640), dtype=torch.uint8)
    ready = preset(fake_photo)

    print("a 480x640 uint8 image through the preset:")
    print(
        "  ", tuple(ready.shape), ready.dtype, f"{ready.min():.2f} to {ready.max():.2f}"
    )
    print()
    print("it handled resize, crop, float conversion and normalisation in one call")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Class names and metadata

    `weights.meta` includes the original class names and information about the trained model. For the ImageNet classifier we can use the predicted index to look up a name in `categories`.

    Once we replace the classifier for our own dataset, we need our own class mapping instead.
    """)
    return


@app.cell
def _(VGG16_Weights):
    meta = VGG16_Weights.DEFAULT.meta

    print("keys available:", sorted(meta.keys()))
    print()
    print("categories:", len(meta["categories"]))
    print("  first five:", meta["categories"][:5])
    print("  index 207 is:", meta["categories"][207])
    print()
    print("reported accuracy:", meta["_metrics"])
    print()
    print("so decoding a prediction is just:")
    print("  meta['categories'][logits.argmax(dim=1).item()]")
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
def _():
    from torchvision.models import vgg16

    vgg = vgg16(
        weights=None
    )  # use weights=VGG16_Weights.DEFAULT to load trained parameters

    print("VGG16 has three top-level parts:")
    for part_name, part in vgg.named_children():
        n = sum(p.numel() for p in part.parameters())
        print(f"  {part_name:12} {n:>12,} parameters")
    print()
    print("the classifier, which is the bit we replace:")
    print(vgg.classifier)
    return vgg, vgg16


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Freezing with [requires_grad_](https://pytorch.org/docs/stable/generated/torch.nn.Module.requires_grad_.html)

    ```python
    model.requires_grad_(False)
    ```

    This sets `requires_grad=False` on the module's parameters. We call it before training so autograd does not accumulate gradients for them. The trailing underscore indicates that the method changes the module in place.

    The next cell counts trainable and frozen parameters before and after the call. Freezing parameters is separate from `eval()`, which changes the behaviour of layers such as dropout. Use evaluation mode when checking predictions.
    """)
    return


@app.cell
def _(vgg):
    def count(m):
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in m.parameters() if not p.requires_grad)
        return trainable, frozen

    print(f"{'':16} {'trainable':>14} {'frozen':>14}")
    _t, _f = count(vgg)
    print(f"{'as loaded':16} {_t:>14,} {_f:>14,}")

    vgg.requires_grad_(False)
    _t, _f = count(vgg)
    print(f"{'after freezing':16} {_t:>14,} {_f:>14,}")
    return (count,)


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
def _(count, nn, torch, vgg, vgg16):
    N_CLASSES = 7

    # the repository's approach: stack on top of the 1000-class output
    stacked = nn.Sequential(vgg, nn.Linear(1000, N_CLASSES))

    # alternatively, replace the final layer to use its input features
    replaced = vgg16(weights=None)
    replaced.requires_grad_(False)
    replaced.classifier[6] = nn.Linear(
        4096, N_CLASSES
    )  # a new layer is trainable by default

    for _label, _model in [("stacked on top", stacked), ("head replaced ", replaced)]:
        _t, _f = count(_model)
        print(f"{_label}  trainable {_t:>9,}  frozen {_f:>12,}")

    print()
    print("what the new head actually receives:")
    print("  stacked : 1000 outputs (ImageNet class scores with trained weights)")
    print("  replaced: 4096 features from the layer before")
    print()
    sample = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        print(
            "both give the right output shape:",
            tuple(stacked(sample).shape),
            tuple(replaced(sample).shape),
        )
    return


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

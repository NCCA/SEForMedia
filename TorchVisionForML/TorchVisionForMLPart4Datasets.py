#!/usr/bin/env uv run marimo edit

import marimo

__generated_with = "0.14.17"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # torchvision for Machine Learning, Part 4: datasets and pre-trained models

    The last of the four. Two halves: getting a folder of images into a `Dataset` without writing one yourself, and using a model somebody else already trained.

    The second half is what `PreTrainedModels/` in this repository is about, and it is the most practical thing in the whole unit. Training VGG16 from scratch on ImageNet takes a few GPU-weeks. Downloading those weights and fitting a new head onto them takes a few minutes, and for most problems you will meet it works better than anything you could train from scratch on the data you have.

    Nothing here downloads weights — I build the architectures with `weights=None` so it runs offline. The parts that need the real weights are marked.
    """
    )
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
    mo.md(
        r"""
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

    If your data is laid out as one folder per class, this replaces the whole `Dataset` subclass from `PyTorchForML` Part 5:

    ```
    train/
        cats/  img001.jpg  img002.jpg ...
        dogs/  img001.jpg ...
    ```

    **The class ordering is alphabetical, not the order you think of them in.** `ImageFolder` sorts the directory names and assigns indices in that order, which it records in `class_to_idx`. That mapping is how prediction index 3 becomes a class name, and it has to be the same at training and at inference time — so save it alongside the model, or read it from the same folder structure both times. Getting it wrong gives you a model that appears to work and names everything incorrectly.
    """
    )
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
    print("save this with the model, or predictions are meaningless:")
    print("  ", with_transform.class_to_idx)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### The built-in datasets

    ```python
    datasets.MNIST(root, train=True, transform=None, download=False)
    datasets.CIFAR10(root, train=True, transform=None, download=False)
    datasets.FashionMNIST(...)
    ```

    torchvision ships loaders for the standard research datasets. `download=True` fetches and unpacks on first use and is a no-op afterwards, which is why it is safe to leave on.

    `download=True` needs the network, so the first run in a lab with no outbound access fails — point `root` at a shared copy instead. `Utils/functions.py` in this repository has `in_lab()` and a `download` helper for exactly that reason.

    The MNIST material here does it both ways, and the order is worth noticing. `ReadDigitsTraining` reads the raw IDX files by hand with `np.fromfile`, pulling the header apart and reshaping the byte stream into images. `TheMNISTDataSet` then uses `datasets.MNIST(..., download=True)` and gets the same data in one line.

    That sequence is the right way round. Do the convenience loader first and the byte layout never gets looked at; do it second and it is obviously a convenience rather than magic. Which is the general argument for the raw version of anything in a teaching context.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The weights API

    ```python
    from torchvision.models import vgg16, VGG16_Weights

    weights = VGG16_Weights.DEFAULT          # or .IMAGENET1K_V1 to pin a version
    model = vgg16(weights=weights)           # downloads on first use
    model = vgg16(weights=None)              # the architecture, randomly initialised
    ```

    `PreTrainedModelsPart1Marimo.py:75` does exactly this. Two things worth knowing about the API.

    First, it replaced an older one. You will find plenty of code and tutorials written as `vgg16(pretrained=True)`. That is deprecated — it worked when each architecture had exactly one set of weights, and broke down once torchvision started shipping improved retrainings of the same architectures.

    Second, `DEFAULT` is not a fixed thing. It means "the best currently available weights for this architecture", so it can change when you upgrade torchvision, and your results change with it. For teaching that is a feature — students get the good weights without choosing. For anything you need to reproduce in six months, pin the version explicitly.
    """
    )
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
    mo.md(
        r"""
    ### weights.transforms() — the part worth the price of admission

    ```python
    pre_trans = weights.transforms()
    ```

    **The weights carry their own preprocessing.** This is the single most useful thing in the torchvision models API and the easiest to miss.

    A pre-trained network is only valid on inputs prepared exactly as its training data was — the same resize, the same crop, the same normalisation constants. Get the normalisation wrong and the model still runs and still produces confident-looking predictions; they are just worse, quietly, with nothing to indicate why.

    Rather than expecting you to look those up, the weights object builds the correct pipeline for you. `PreTrainedModelsPart1Marimo.py:110` uses it, and every ImageNet model you load should.

    Note it also means you do not need to remember `[0.485, 0.456, 0.406]` at all — which is the honest answer to why that magic number appears in so many tutorials without explanation.
    """
    )
    return


@app.cell
def _(VGG16_Weights):
    preset = VGG16_Weights.DEFAULT.transforms()
    print(preset)
    print()
    print("that is Resize(256) + CenterCrop(224) + the ImageNet normalisation,")
    print("which is precisely the recipe we wrote out by hand in Part 2 -")
    print("except this one is guaranteed to match the weights.")
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
    mo.md(
        r"""
    ### weights.meta

    The weights also carry their metadata, including the class names — so you can turn an output index into a label without shipping a separate file of 1000 strings.
    """
    )
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
    mo.md(
        r"""
    ## Transfer learning

    The idea: a network trained on ImageNet has learned, in its early layers, generic visual features — edges, textures, repeated patterns — that are useful for almost any image task. Only the last layers are specific to "which of these 1000 ImageNet classes is this". So keep the early layers, replace the end, and train only the new part on your data.

    Two things have to happen:

    1. **Replace the head** so the output has your number of classes
    2. **Freeze the rest** so training does not destroy the features you came for

    Let me build the architecture first — `weights=None` so this runs offline, but the shapes are identical to the real thing.
    """
    )
    return


@app.cell
def _():
    from torchvision.models import vgg16

    vgg = vgg16(weights=None)  # weights=VGG16_Weights.DEFAULT for the real thing

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
    mo.md(
        r"""
    ### Freezing with [requires_grad_](https://pytorch.org/docs/stable/generated/torch.nn.Module.requires_grad_.html)

    ```python
    model.requires_grad_(False)
    ```

    Sets `requires_grad = False` on every parameter in the module tree. Those tensors stop being tracked by autograd, get no gradients, and so cannot be changed by the optimiser. Note the trailing underscore — it is an in-place operation, like `zero_()`.

    `TransferLearningMarimo.py` calls `vgg_model.requires_grad_(False)` before training. The saving is real: on VGG16 it takes the trainable parameter count from 138 million to whatever your new head has.
    """
    )
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
    mo.md(
        r"""
    ### Two ways to attach a head

    This repository takes the simpler route. `TransferLearningMarimo.py` keeps the whole VGG, including its 1000-class ImageNet classifier, and stacks a new layer on the end:

    ```python
    dog_model = nn.Sequential(vgg_model, nn.Linear(1000, N_CLASSES))
    ```

    The more usual approach replaces VGG's final layer instead, so the new head sees the 4096-dimensional features rather than the 1000 class scores:

    ```python
    vgg.classifier[6] = nn.Linear(4096, N_CLASSES)
    ```

    Both work and the first is easier to explain, which is a fair reason to teach it. But it is worth being clear with students about what the difference costs. In the stacked version, everything your new layer knows about the image has been squeezed through "how much does this look like each of 1000 ImageNet categories". If your classes are ImageNet-like — breeds of dog, say, which the demo is — that is a decent summary and it works fine. If they are not — X-rays, circuit boards, hand signs — a lot of relevant information has already been thrown away, and the replacement approach will do better.
    """
    )
    return


@app.cell
def _(count, nn, torch, vgg, vgg16):
    N_CLASSES = 7

    # the repository's approach: stack on top of the 1000-class output
    stacked = nn.Sequential(vgg, nn.Linear(1000, N_CLASSES))

    # the usual approach: replace VGG's final layer
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
    print("  stacked : 1000 ImageNet class scores")
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
    mo.md(
        r"""
    One detail in that cell worth pointing out. A newly created `nn.Linear` has `requires_grad=True`, so assigning it *after* freezing gives you a frozen backbone and a trainable head with no extra work. Do it in the other order — create the head, then call `requires_grad_(False)` on the whole model — and you freeze your new layer too, so nothing trains at all. The loss sits perfectly flat, which at least makes it easy to spot.

    It is also worth passing only the trainable parameters to the optimiser:

    ```python
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    ```

    `TransferLearningMarimo.py` passes `dog_model.parameters()` — all of them — which works, because frozen parameters have no gradient and Adam leaves them alone. But the filtered version says what you mean, and it avoids Adam allocating optimiser state for 138 million parameters it will never update.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The whole thing

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

    Then the training loop from `PyTorchForML` Part 2, unchanged.

    That is eleven lines for a model that would otherwise need weeks of GPU time and a million labelled images. It is the most useful thing in this unit, and the reason it works is worth restating: the features are the valuable part, and somebody else already paid for them.

    ## Exercises

    1. Build an `ImageFolder` with classes named `1`, `2` and `10`. What does `class_to_idx` give you, and why is it not what you wanted?
    2. Compare `VGG16_Weights.DEFAULT.transforms()` with `ResNet50_Weights.DEFAULT.transforms()`. Are they the same? What would happen if you used one with the other's model?
    3. Freeze a model, replace the head, and print `requires_grad` for every parameter to confirm exactly what will train. Now do the two operations in the wrong order and diff the output.
    4. Take the stacked and replaced models above and count the parameters your optimiser would allocate state for, with and without the `filter`. At what model size does that matter?
    5. For the ASL dataset — hand signs, greyscale, 28x28 — would you expect transfer learning from ImageNet to help? Argue both sides, then check what `ASLPart2CNN` achieves training from scratch.

    That is the torchvision set. Between `NumPyForML/`, `PyTorchForML/` and these four you have every function used in the machine learning demos in this repository.
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

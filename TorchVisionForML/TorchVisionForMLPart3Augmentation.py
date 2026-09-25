#!/usr/bin/env uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # torchvision for Machine Learning, Part 3: augmentation

    Data augmentation gives us variations of the training images. We can change the position, brightness or orientation of a subject without collecting another photograph.

    The choice depends on what we are trying to recognise. A transformation is useful only if the resulting image still supports its label. I will use a small asymmetric pattern to show what the transforms do, then combine them into a training pipeline.

    For these examples we will keep validation preprocessing fixed so we can compare runs on the same inputs.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import torch
    from torchvision.transforms import v2

    return plt, torch, v2


@app.cell
def _(plt, torch):
    def image_grid(
        panels: list[tuple[str, torch.Tensor]], columns: int = 4
    ) -> plt.Figure:
        """Show CHW images at their original aspect ratio with a shared display range."""
        rows = (len(panels) + columns - 1) // columns
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(3 * columns, 3.1 * rows),
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

    letter_f = torch.zeros(3, 32, 32, dtype=torch.uint8)
    letter_f[:, 6:26, 8:12] = 255
    letter_f[:, 6:10, 8:22] = 255
    letter_f[:, 14:18, 8:18] = 255

    # colour gradients let us see hue and saturation as well as brightness
    colour_image = torch.zeros(3, 64, 64, dtype=torch.uint8)
    colour_image[0] = torch.linspace(30, 230, 64).to(torch.uint8)[None, :]
    colour_image[2] = torch.linspace(30, 230, 64).to(torch.uint8)[:, None]
    colour_image[1, 8:28, 8:28] = 220
    colour_image[:, 38:54, 36:56] = 210
    image_grid(
        [
            ("Asymmetric F: position and orientation", letter_f),
            ("Colour pattern: gradients and grey patch", colour_image),
        ],
        columns=2,
    )
    return colour_image, image_grid, letter_f


@app.cell(hide_code=True)
def _(mo):
    sample_seed = mo.ui.slider(0, 100, value=42, label="Random seed", show_value=True)
    flip_probability = mo.ui.slider(
        0, 1, step=0.1, value=0.5, label="Flip probability", show_value=True
    )
    rotation_angle = mo.ui.slider(
        0, 60, value=25, label="Rotation angle (degrees)", show_value=True
    )
    mo.vstack(
        [
            mo.md(
                "Change the seed to see another set of samples. The same seed repeats the same examples. The other controls update the flip and rotation comparisons below."
            ),
            mo.hstack([sample_seed, flip_probability, rotation_angle]),
        ]
    )
    return flip_probability, rotation_angle, sample_seed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [RandomHorizontalFlip](https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomHorizontalFlip.html)

    ```python
    v2.RandomHorizontalFlip(p=0.5)
    v2.RandomVerticalFlip(p=0.5)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `p` | `0.5` | probability of flipping; `p=1.0` always flips |

    `p` is the probability of applying the flip. With `p=0.5`, some calls leave the image unchanged. To check the direction of the operation, I use `p=1.0` below and compare the position of the pattern before and after flipping.
    """)
    return


@app.cell
def _(flip_probability, image_grid, letter_f, mo, sample_seed, torch, v2):
    with torch.random.fork_rng():
        torch.manual_seed(sample_seed.value)
        _samples = [
            v2.RandomHorizontalFlip(p=flip_probability.value)(letter_f)
            for _ in range(8)
        ]
    _changed = [not torch.equal(letter_f, _sample) for _sample in _samples]
    mo.vstack(
        [
            mo.md(
                f"**{sum(_changed)} of 8 samples flipped** with `p={flip_probability.value:.1f}`. A small sample need not match the probability exactly."
            ),
            image_grid(
                [
                    (f"Sample {_i + 1}: {'flipped' if _flip else 'unchanged'}", _sample)
                    for _i, (_sample, _flip) in enumerate(zip(_samples, _changed))
                ]
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Does the label still fit?

    Before adding a flip, look at what it does to the subject. A mirrored cat still supports the label "cat". Mirroring text can change or remove the character we wanted to recognise.

    The next cell mirrors our F-shaped pattern. The result is no longer an ordinary F, but a training pipeline would keep the original label. Whether that is useful depends on the task: recognising printed letters and recognising shapes under reflection are different problems.

    For the hand signs in `ASL/`, consider which variations the dataset represents and whether each transformed sign remains recognisable. We need to make that judgement for the actual classes and images.
    """)
    return


@app.cell
def _(image_grid, letter_f, v2):
    image_grid(
        [
            ("Original: label F", letter_f),
            ("Horizontal flip: still an F?", v2.RandomHorizontalFlip(p=1)(letter_f)),
            ("Vertical flip: still an F?", v2.RandomVerticalFlip(p=1)(letter_f)),
        ],
        columns=3,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [RandomRotation](https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomRotation.html)

    ```python
    v2.RandomRotation(degrees, interpolation='nearest', expand=False, center=None, fill=0)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `degrees` | required | a number `d` means the range `(-d, +d)`; a tuple gives it explicitly |
    | `interpolation` | `'nearest'` | `'bilinear'` is smoother but slower |
    | `expand` | `False` | grow the canvas so nothing is cut off |
    | `fill` | `0` | what to put in the corners the rotation leaves empty |

    With `expand=False`, the output keeps the same size and rotated content can be clipped at the edges. `expand=True` allows a larger canvas, so we may need to resize or pad the outputs before batching them.

    `fill` sets the value in the exposed corners. Our pattern has a black background, so zero fits. For a white background use 255 with `uint8` data, or 1.0 with floating point data in 0–1. The next cell compares the effect on a centred pattern and an image filled to the edges.
    """)
    return


@app.cell
def _(image_grid, letter_f, rotation_angle, torch, v2):
    _angle = rotation_angle.value
    _full = torch.full_like(letter_f, 255)
    _panels = []
    for _name, _source in [("Centred F", letter_f), ("Full frame", _full)]:
        _panels.append((f"{_name}: original", _source))
        for _expand, _fill in [(False, 0), (True, 0), (False, 120)]:
            # fix the angle so only the canvas and fill change between panels
            _rotated = v2.RandomRotation((_angle, _angle), expand=_expand, fill=_fill)(
                _source
            )
            _panels.append(
                (
                    f"{_angle}° | expand={_expand}, fill={_fill}\n{_rotated.shape[-1]} × {_rotated.shape[-2]} pixels",
                    _rotated,
                )
            )
    image_grid(_panels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [ColorJitter](https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.ColorJitter.html)

    ```python
    v2.ColorJitter(brightness=None, contrast=None, saturation=None, hue=None)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `brightness` | `None` | a float `b` gives a factor drawn from `[max(0, 1-b), 1+b]` |
    | `contrast` | `None` | same form |
    | `saturation` | `None` | same form |
    | `hue` | `None` | a float `h` shifts hue within `[-h, +h]`, and must be ≤ 0.5 |

    We need to set at least one argument to change the image. `ColorJitter()` on its own leaves the image unchanged.

    Each row below changes one property of the same colour pattern. Compare the gradients, green square and grey patch across repeated calls. If colour distinguishes our classes, a hue shift may produce an image that no longer supports its label.
    """)
    return


@app.cell
def _(colour_image, image_grid, sample_seed, torch, v2):
    _panels = []
    with torch.random.fork_rng():
        torch.manual_seed(sample_seed.value)
        for _name, _transform in [
            ("Brightness", v2.ColorJitter(brightness=0.6)),
            ("Contrast", v2.ColorJitter(contrast=0.6)),
            ("Saturation", v2.ColorJitter(saturation=0.9)),
            ("Hue", v2.ColorJitter(hue=0.5)),
        ]:
            _panels.append((f"{_name}: original", colour_image))
            _panels.extend(
                (f"{_name}: sample {_i + 1}", _transform(colour_image))
                for _i in range(3)
            )
    image_grid(_panels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [RandomResizedCrop](https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomResizedCrop.html)

    ```python
    v2.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.333),
                         interpolation='bilinear', antialias=True)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `size` | required | output size, so the batch shape stays fixed |
    | `scale` | `(0.08, 1.0)` | fraction of the original area the crop covers |
    | `ratio` | `(0.75, 1.333)` | aspect ratio range of the crop before resizing |

    This transform chooses a region and resizes it to the output size. `scale` describes the fraction of the original area used for the crop, not the width or height.

    The default allows a crop covering only 8% of the image. For a small subject this can remove much of what we wanted to recognise. Below we compare the default with `scale=(0.7, 1.0)`. The orange boxes show the actual sampled regions beside their resized outputs. I would inspect the results on the training images before choosing either range.
    """)
    return


@app.cell
def _(image_grid, letter_f, plt, sample_seed, torch, v2):
    from matplotlib.patches import Rectangle

    _figure, _axes = plt.subplots(2, 8, figsize=(16, 5), layout="constrained")
    with torch.random.fork_rng():
        torch.manual_seed(sample_seed.value)
        for _row, _scale in enumerate([(0.08, 1.0), (0.7, 1.0)]):
            for _sample in range(4):
                _top, _left, _height, _width = v2.RandomResizedCrop.get_params(
                    letter_f, _scale, (0.75, 4 / 3)
                )
                _crop = v2.functional.resized_crop(
                    letter_f, _top, _left, _height, _width, [32, 32], antialias=True
                )
                _source_axis, _crop_axis = _axes[_row, 2 * _sample : 2 * _sample + 2]
                _source_axis.imshow(letter_f.permute(1, 2, 0))
                _source_axis.add_patch(
                    Rectangle(
                        (_left - 0.5, _top - 0.5),
                        _width,
                        _height,
                        fill=False,
                        edgecolor="orange",
                        linewidth=2,
                    )
                )
                _source_axis.set_title(
                    f"Area: {_height * _width / 1024:.0%}", fontsize=10
                )
                _crop_axis.imshow(_crop.permute(1, 2, 0))
                _crop_axis.set_title("Resized to 32 × 32", fontsize=9)
            _axes[_row, 0].set_ylabel(f"scale={_scale}")
    for _axis in _axes.flat:
        _axis.set_xticks([])
        _axis.set_yticks([])
    plt.close(_figure)
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choosing when to apply a transform

    ```python
    v2.RandomApply([transform, ...], p=0.5)  # apply the list with probability p
    v2.RandomChoice([transform, ...])       # choose one transform
    v2.RandomOrder([transform, ...])        # apply all in a random order
    ```

    These let us control how transforms are combined. For example, we can use `RandomApply` to apply a group of colour changes to only some training samples. The next grid shows which copies of our pattern change.
    """)
    return


@app.cell
def _(image_grid, letter_f, mo, sample_seed, torch, v2):
    with torch.random.fork_rng():
        torch.manual_seed(sample_seed.value)
        _occasional = v2.RandomApply([v2.RandomRotation(45)], p=0.2)
        _samples = [_occasional(letter_f) for _ in range(12)]
    mo.vstack(
        [
            mo.md(
                "`RandomApply(p=0.2)`: most copies stay unchanged. Here we label visible pixel changes; applying a very small rotation can still leave the pixels unchanged."
            ),
            image_grid(
                [
                    (
                        "Changed"
                        if not torch.equal(letter_f, _sample)
                        else "Unchanged",
                        _sample,
                    )
                    for _sample in _samples
                ],
                columns=6,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training and validation pipelines

    We can now put the transforms together. Both pipelines finish with the same conversion and normalisation. The training pipeline also adds random variations; the validation pipeline uses fixed preprocessing.

    This is a demonstration of the mechanics. The flips and crops still need checking against the labels of any dataset we use.
    """)
    return


@app.cell
def _(torch, v2):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_transform = v2.Compose(
        [
            v2.RandomResizedCrop(224, scale=(0.7, 1.0)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(20),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    valid_transform = v2.Compose(
        [
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    print("train pipeline:", len(train_transform.transforms), "steps")
    print("valid pipeline:", len(valid_transform.transforms), "steps")
    print()
    print("both pipelines use the same final conversion and normalisation")
    print("this keeps the input type and scale consistent for the model")
    return IMAGENET_MEAN, IMAGENET_STD, train_transform, valid_transform


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Keeping validation inputs fixed

    If we apply random augmentation during validation, the inputs can change between evaluations. A change in the score may then come from the augmentation as well as the model.

    The next cell passes the same colour image through each pipeline four times and displays the outputs. For our usual validation run we use the fixed pipeline. Testing on transformed inputs can be a separate experiment when we want to measure a particular behaviour.
    """)
    return


@app.cell
def _(
    IMAGENET_MEAN,
    IMAGENET_STD,
    colour_image,
    image_grid,
    mo,
    sample_seed,
    torch,
    train_transform,
    valid_transform,
):
    with torch.random.fork_rng():
        torch.manual_seed(sample_seed.value)
        train_outputs = torch.stack([train_transform(colour_image) for _ in range(4)])
        valid_outputs = torch.stack([valid_transform(colour_image) for _ in range(4)])
    _mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    _std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    # undo normalisation for display; the model still receives normalised tensors
    _panels = [
        (f"Training: run {_i + 1}", _sample * _std + _mean)
        for _i, _sample in enumerate(train_outputs)
    ]
    _panels += [
        (f"Validation: run {_i + 1}", _sample * _std + _mean)
        for _i, _sample in enumerate(valid_outputs)
    ]
    mo.vstack(
        [
            image_grid(_panels),
            mo.md(
                f"Validation outputs identical: **{torch.equal(valid_outputs[0].expand_as(valid_outputs), valid_outputs)}**. Normalisation is undone only for these previews."
            ),
        ]
    )
    return train_outputs, valid_outputs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checking the result

    I would inspect augmented samples before starting a training run. The grid below puts several versions of the same image together so we can check that each still supports the label.

    The captions show foreground pixel area divided by the original foreground area. Resizing can enlarge the remaining strokes, so a ratio above one does not mean more of the letter survived. This can help spot nearly empty crops, but it does not tell us whether the image is still recognisable. Likewise, variation between outputs tells us that something changed, not whether the change was useful.

    To test whether augmentation helps the model, compare training runs with and without it using the same validation data. Measure the extra processing time too.
    """)
    return


@app.cell
def _(image_grid, letter_f, sample_seed, torch, v2):
    _panels = []
    with torch.random.fork_rng():
        torch.manual_seed(sample_seed.value)
        for _name, _transform in [
            ("Flip", v2.RandomHorizontalFlip(p=0.5)),
            ("Rotate ±20°", v2.RandomRotation(20)),
            ("Crop: default", v2.RandomResizedCrop(32)),
            ("Crop: scale 0.7–1.0", v2.RandomResizedCrop(32, scale=(0.7, 1.0))),
        ]:
            _panels.append((f"{_name}: original", letter_f))
            for _i in range(3):
                _sample = _transform(letter_f)
                _ratio = (_sample[0] > 128).sum().item() / (
                    letter_f[0] > 128
                ).sum().item()
                _panels.append((f"Sample {_i + 1} | ink area ×{_ratio:.2f}", _sample))
    image_grid(_panels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Build two augmentation pipelines for MNIST: one you expect to preserve the digit labels and one you expect to cause problems. Explain your choices and inspect their outputs.
    2. Apply `RandomResizedCrop(28)` to a digit 1000 times. Count crops with no foreground pixels, then repeat with a narrower `scale` range.
    3. Apply `ColorJitter(hue=0.5)` repeatedly to a red patch. Inspect the colours produced and discuss what this would mean for a red-versus-green classifier.
    4. Move `Normalize` before the augmentation in the training pipeline. Which transforms accept the result? Do they still behave as intended?
    5. Time the training and validation pipelines on the same images. Then measure data loading and model execution separately to find where the training run spends its time.

    In Part 4 we will load datasets and prepare a pre-trained model for a new set of classes.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

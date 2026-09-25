#!/usr/bin/env uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App()


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
    import torch
    from torchvision.transforms import v2

    torch.manual_seed(42)
    return torch, v2


@app.cell
def _(torch):
    # use an asymmetric pattern so we can check flips and rotations
    def make_letter_f() -> torch.Tensor:
        img = torch.zeros(3, 32, 32, dtype=torch.uint8)
        img[:, 6:26, 8:12] = 255  # the upright
        img[:, 6:10, 8:22] = 255  # the top bar
        img[:, 14:18, 8:18] = 255  # the middle bar
        return img

    letter_f = make_letter_f()

    def ink_centre(img):
        """
        Find the mean position of bright pixels in the first channel.

        Parameters
        ----------
        img : torch.Tensor
            CHW image with foreground values above 128.

        Returns
        -------
        tuple of float
            Mean row and column, rounded to one decimal place.
        """
        mask = img[0] > 128
        ys, xs = torch.nonzero(mask, as_tuple=True)
        return round(ys.float().mean().item(), 1), round(xs.float().mean().item(), 1)

    print("test pattern", tuple(letter_f.shape))
    print("ink centre (row, col):", ink_centre(letter_f))
    print(
        "ink covers", f"{(letter_f[0] > 128).float().mean().item():.1%}", "of the image"
    )
    return ink_centre, letter_f


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
def _(ink_centre, letter_f, v2):
    flipper = v2.RandomHorizontalFlip(p=0.5)

    flipped_count = 0
    original_centre = ink_centre(letter_f)
    for _ in range(1000):
        if ink_centre(flipper(letter_f)) != original_centre:
            flipped_count += 1

    print(
        f"out of 1000 calls, {flipped_count} actually flipped ({flipped_count / 10:.1f}%)"
    )
    print()
    print("original       ink centre", original_centre)
    print(
        "always flipped ink centre",
        ink_centre(v2.RandomHorizontalFlip(p=1.0)(letter_f)),
    )
    print("the row is unchanged; the column is reflected about the image centre")
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
def _(letter_f, torch, v2):
    mirrored = v2.RandomHorizontalFlip(p=1.0)(letter_f)

    print("the upright stroke, by column, in the original and the mirror:")
    print(
        "  original",
        (letter_f[0, 6:26, :] > 128)
        .float()
        .mean(dim=0)
        .round(decimals=1)[:16]
        .tolist(),
    )
    print(
        "  mirrored",
        (mirrored[0, 6:26, :] > 128)
        .float()
        .mean(dim=0)
        .round(decimals=1)[:16]
        .tolist(),
    )
    print()
    print("identical images?", torch.equal(letter_f, mirrored))
    print("but we would be handing both to the model with the same label")
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
def _(letter_f, torch, v2):
    rotator = v2.RandomRotation(degrees=20)

    print("shape is preserved with expand=False:", tuple(rotator(letter_f).shape))
    print(
        "with expand=True it grows:           ",
        tuple(v2.RandomRotation(25, expand=True)(letter_f).shape),
    )
    print("  resize or pad differing output sizes before stacking them into a batch")
    print()

    corner_dark = v2.RandomRotation(30, fill=0)(letter_f)
    corner_light = v2.RandomRotation(30, fill=255)(letter_f)
    print("top-left corner pixel after a rotation:")
    print("  fill=0   ", corner_dark[:, 0, 0].tolist())
    print("  fill=255 ", corner_light[:, 0, 0].tolist())
    print()
    print("how much content does a 20-degree rotation push off the edges?")

    _full = torch.full((3, 32, 32), 255, dtype=torch.uint8)  # subject fills the frame
    for _name, _src in [
        ("small centred subject", letter_f),
        ("subject fills the frame", _full),
    ]:
        _before = (_src[0] > 128).sum().item()
        _after = torch.tensor(
            [(v2.RandomRotation(20)(_src)[0] > 128).sum().item() for _ in range(50)]
        ).float()
        print(
            f"  {_name:24} {_before:4} -> {_after.mean():6.0f} px  ({_after.mean() / _before:.0%} kept)"
        )

    print()
    print("compare the centred pattern with the full-frame image:")
    print("the available margin affects how much content is clipped.")
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

    We need to set at least one argument to change the image. The first check below confirms that `ColorJitter()` on its own leaves our pattern unchanged.

    I then vary brightness and contrast over several calls. For hue we use a coloured patch, as the white pattern would not show the change. If colour distinguishes our classes, a hue shift may produce an image that no longer supports its label.
    """)
    return


@app.cell
def _(letter_f, torch, v2):
    print("ColorJitter() with no arguments:")
    print("  changes anything?", not torch.equal(v2.ColorJitter()(letter_f), letter_f))
    print()

    jitter = v2.ColorJitter(brightness=0.4, contrast=0.3)
    means = torch.tensor([jitter(letter_f).float().mean().item() for _ in range(200)])
    print("brightness=0.4, contrast=0.3 over 200 runs:")
    print(f"  original mean pixel {letter_f.float().mean():.1f}")
    print(
        f"  augmented mean      {means.mean():.1f}  (sd {means.std():.1f}, range {means.min():.1f} to {means.max():.1f})"
    )
    print()
    # hue only means something on coloured data, so test it on colour
    red_patch = torch.zeros(3, 8, 8, dtype=torch.uint8)
    red_patch[0] = 220
    green_patch = torch.zeros(3, 8, 8, dtype=torch.uint8)
    green_patch[1] = 220

    print("hue=0.5 on actual colour:")
    print(
        "  a red pixel  ",
        red_patch[:, 0, 0].tolist(),
        "->",
        v2.ColorJitter(hue=0.5)(red_patch)[:, 0, 0].tolist(),
    )
    print(
        "  a green pixel",
        green_patch[:, 0, 0].tolist(),
        "->",
        v2.ColorJitter(hue=0.5)(green_patch)[:, 0, 0].tolist(),
    )
    print()
    print("  check whether these colours still support the original class labels")
    print(
        "  (white and grey are unaffected, which is why the test pattern above shows nothing)"
    )
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

    The default allows a crop covering only 8% of the image. For a small subject this can remove much of what we wanted to recognise. Below we compare the default with `scale=(0.7, 1.0)` and measure how much of our pattern remains. I would inspect the results on the training images before choosing either range.
    """)
    return


@app.cell
def _(letter_f, torch, v2):
    def ink_fraction(img):
        return (img[0] > 128).float().mean().item()

    default_crop = v2.RandomResizedCrop(size=32)
    gentle_crop = v2.RandomResizedCrop(size=32, scale=(0.7, 1.0))

    _orig = ink_fraction(letter_f)
    _default = torch.tensor([ink_fraction(default_crop(letter_f)) for _ in range(300)])
    _gentle = torch.tensor([ink_fraction(gentle_crop(letter_f)) for _ in range(300)])

    print(f"ink covering the frame, original: {_orig:.1%}")
    print()
    print(
        f"scale=(0.08, 1.0)  the default: mean {_default.mean():.1%}, range {_default.min():.1%} to {_default.max():.1%}"
    )
    print(
        f"scale=(0.7, 1.0)   narrower    : mean {_gentle.mean():.1%}, range {_gentle.min():.1%} to {_gentle.max():.1%}"
    )
    print()
    _empty = (_default < 0.01).float().mean().item()
    print(f"crops that came back essentially blank with the default: {_empty:.1%}")
    print("these crops have little foreground left, but would keep the original label")
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

    These let us control how transforms are combined. For example, we can use `RandomApply` to apply a group of colour changes to only some training samples. The next cell measures how often `RandomApply` changes our pattern.
    """)
    return


@app.cell
def _(ink_centre, letter_f, v2):
    occasional = v2.RandomApply([v2.RandomRotation(45)], p=0.2)

    changed = sum(
        1
        for _ in range(500)
        if ink_centre(occasional(letter_f)) != ink_centre(letter_f)
    )
    print(
        f"RandomApply(p=0.2) changed the image {changed}/500 times ({changed / 5:.0f}%)"
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
    return train_transform, valid_transform


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Keeping validation inputs fixed

    If we apply random augmentation during validation, the inputs can change between evaluations. A change in the score may then come from the augmentation as well as the model.

    The next cell passes the same image through each pipeline twenty times and compares the outputs. For our usual validation run we use the fixed pipeline. Testing on transformed inputs can be a separate experiment when we want to measure a particular behaviour.
    """)
    return


@app.cell
def _(letter_f, torch, train_transform, valid_transform):
    train_outputs = torch.stack([train_transform(letter_f) for _ in range(20)])
    valid_outputs = torch.stack([valid_transform(letter_f) for _ in range(20)])

    print("running the same image through each pipeline 20 times:")
    print(f"  train: all 20 identical? {bool((train_outputs.std(dim=0) < 1e-6).all())}")
    print(
        f"         spread across runs, mean sd per pixel {train_outputs.std(dim=0).mean():.4f}"
    )
    print(f"  valid: all 20 identical? {bool((valid_outputs.std(dim=0) < 1e-6).all())}")
    print(
        f"         spread across runs, mean sd per pixel {valid_outputs.std(dim=0).mean():.4f}"
    )
    print()
    print("random augmentation can also introduce variation in validation inputs")
    print("which can change the score between evaluations")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checking the result

    I would inspect augmented samples before starting a training run. Use `make_grid` from Part 1 to put several versions of the same image together, then check that each still supports the label.

    Here we measure the area of the remaining pattern. This can help spot nearly empty crops, but it does not tell us whether the image is still recognisable. Likewise, variation between outputs tells us that something changed, not whether the change was useful.

    To test whether augmentation helps the model, compare training runs with and without it using the same validation data. Measure the extra processing time too.
    """)
    return


@app.cell
def _(letter_f, torch, v2):
    def ink_ratio(pipeline, source, trials=200):
        """
        Measure foreground area after repeated applications of a transform.

        Parameters
        ----------
        pipeline : callable
            Transform to apply to each copy of the source.
        source : torch.Tensor
            CHW image with foreground values above 128 in the first channel.
        trials : int
            Number of transformed samples to measure.

        Returns
        -------
        torch.Tensor
            Foreground pixel count for each result, divided by the source count.
        """
        base = (source[0] > 128).float().sum()
        return torch.tensor(
            [
                ((pipeline(source)[0] > 128).float().sum() / base).item()
                for _ in range(trials)
            ]
        )

    print(f"{'transform':24} {'mean':>6} {'worst':>6}   (ink area vs the original)")
    for name, pipe in [
        ("flip only", v2.RandomHorizontalFlip(p=0.5)),
        ("rotate 20", v2.RandomRotation(20)),
        ("rotate 90", v2.RandomRotation(90)),
        ("crop, default scale", v2.RandomResizedCrop(32)),
        ("crop, scale 0.7-1.0", v2.RandomResizedCrop(32, scale=(0.7, 1.0))),
    ]:
        s = ink_ratio(pipe, letter_f)
        print(f"{name:24} {s.mean():6.2f} {s.min():6.2f}")

    print()
    print("a ratio above 1 means the output contains more foreground pixels.")
    print("resizing a crop can enlarge the remaining part of the pattern.")
    print("a minimum near zero indicates a sample with very little foreground.")
    print("inspect the images as well as these counts.")
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

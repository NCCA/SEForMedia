#!/usr/bin/env uv run marimo edit

import marimo

__generated_with = "0.14.17"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # torchvision for Machine Learning, Part 3: augmentation

    Data augmentation applies a random transformation to each training image every time it is seen, so the model never gets the same input twice. It enlarges the effective dataset without collecting more data, and it teaches the model to ignore things it should ignore — where the subject sits in the frame, how bright the room was, which way round the camera was held.

    It is also the part of a pipeline most often applied without thinking. There are two rules, and the second needs judgement rather than a lookup:

    1. **Training set only.** Augmenting the validation set makes the score meaningless.
    2. **An augmentation is only valid if it preserves the label.**

    The ASL demo in this repository makes a proper job of the second, which is why I keep pointing at it. This notebook is mostly about how to check both rather than assume them.
    """
    )
    return


@app.cell
def _():
    import torch
    from torchvision.transforms import v2

    torch.manual_seed(42)
    return torch, v2


@app.cell
def _(torch):
    # an asymmetric test pattern - a crude "F", which has no symmetry at all,
    # so any flip or rotation is immediately visible in the numbers
    def make_letter_f() -> torch.Tensor:
        img = torch.zeros(3, 32, 32, dtype=torch.uint8)
        img[:, 6:26, 8:12] = 255  # the upright
        img[:, 6:10, 8:22] = 255  # the top bar
        img[:, 14:18, 8:18] = 255  # the middle bar
        return img

    letter_f = make_letter_f()

    def ink_centre(img):
        """Where is the bright ink, on average? A cheap fingerprint of the content."""
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
    mo.md(
        r"""
    ## [RandomHorizontalFlip](https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomHorizontalFlip.html)

    ```python
    v2.RandomHorizontalFlip(p=0.5)
    v2.RandomVerticalFlip(p=0.5)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `p` | `0.5` | probability of flipping; `p=1.0` always flips |

    5 calls across 2 demos. The `p` is a probability, not a strength — half the time it does nothing at all, which is the first thing to check when you cannot see any effect.
    """
    )
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
    print("note the row is unchanged and the column has mirrored about 16")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### The label preservation question

    A horizontal flip is the most common augmentation and the most commonly misapplied. Whether it is valid is a fact about your data, not about torchvision:

    | Data | Horizontal flip | Why |
    | --- | --- | --- |
    | photos of cats | fine | a mirrored cat is a cat |
    | handwritten letters | **no** | a flipped "b" is a "d" |
    | digits | **no** | and a vertical flip turns 6 into something like 9 |
    | ASL hand signs | fine | signing works with either dominant hand |
    | medical scans | usually **no** | organs are not symmetric |
    | road scenes for a UK model | debatable | it swaps which side of the road you drive on |

    `ASL/ASLPart3DataAugmentationMarimo.py:302` argues the ASL case explicitly rather than just applying the transform, and then at line 321 caps rotation at 20 degrees on the grounds that a sufficiently rotated sign stops meaning what it meant. That is the right shape of reasoning and worth pointing students at.

    The cell below shows what a bad flip does. The "F" and its mirror are different letters, and the model is being told they have the same label.
    """
    )
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
    mo.md(
        r"""
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

    Two defaults worth knowing. `expand=False` means the image stays the same size and the corners get cut off — usually what you want in a pipeline, since the shape must stay constant for batching. And `fill=0` means those corners become black, which on a dataset with white backgrounds inserts a black wedge that the model can learn from. On MNIST that is fine; on a document scan set `fill=255`.
    """
    )
    return


@app.cell
def _(letter_f, torch, v2):
    rotator = v2.RandomRotation(degrees=20)

    print("shape is preserved with expand=False:", tuple(rotator(letter_f).shape))
    print(
        "with expand=True it grows:           ",
        tuple(v2.RandomRotation(25, expand=True)(letter_f).shape),
    )
    print("  - which breaks batching, so it is rarely used in a Compose")
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
    print("so the answer depends entirely on your data. A centred subject with room")
    print("around it loses nothing; a full-frame one loses the corners every time.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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

    5 calls across 2 demos. All four default to `None`, meaning no change — so `ColorJitter()` with no arguments does precisely nothing, which is an easy thing to ship by accident.

    `hue` is the one to be careful with. Brightness and contrast changes are usually label-preserving; hue rotation is not, if colour is part of what distinguishes your classes. Shift the hue on a dataset of red and green peppers and you have relabelled them.
    """
    )
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
    print(
        "  if red versus green IS the label, that transform has just relabelled the data"
    )
    print(
        "  (white and grey are unaffected, which is why the test pattern above shows nothing)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## [RandomResizedCrop](https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomResizedCrop.html)

    ```python
    v2.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.333),
                         interpolation='bilinear', antialias=True)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `size` | required | output size, so the batch shape stays fixed |
    | `scale` | `(0.08, 1.0)` | fraction of the **original area** the crop covers |
    | `ratio` | `(0.75, 1.333)` | aspect ratio range of the crop before resizing |

    Crops a random region and resizes it to `size`. It is the standard ImageNet augmentation and it does the jobs of cropping, scaling and translating in one step.

    **Look at that `scale` default.** `0.08` means a crop can be 8% of the original area — under a third of the width and height. That default comes from ImageNet, where photographs are large and the subject usually fills a good part of the frame, so an aggressive crop still contains the subject. On a 28x28 MNIST digit or a tightly cropped ASL hand, an 8% crop is a few pixels of background with the label still attached, and you are training the model on noise.

    If you use this on anything other than ImageNet-like photographs, set `scale` yourself. Something like `(0.7, 1.0)` is a sane starting point.
    """
    )
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
        f"scale=(0.7, 1.0)   sensible    : mean {_gentle.mean():.1%}, range {_gentle.min():.1%} to {_gentle.max():.1%}"
    )
    print()
    _empty = (_default < 0.01).float().mean().item()
    print(f"crops that came back essentially blank with the default: {_empty:.1%}")
    print("every one of those is a training sample with a label and no content")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Applying augmentation sometimes: RandomApply and RandomChoice

    ```python
    v2.RandomApply([transform, ...], p=0.5)   # apply the whole list, or none of it
    v2.RandomChoice([transform, ...])         # pick exactly one
    v2.RandomOrder([transform, ...])          # all of them, shuffled
    ```

    Useful when an augmentation is strong enough that you only want it occasionally, or when two of them together would be too much. `RandomApply` with a low `p` is the usual way to include something aggressive without it dominating.
    """
    )
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
    mo.md(
        r"""
    ## The two pipelines

    This is the shape every image project takes. Both pipelines must end in the same conversion and normalisation, and only the training one is augmented.
    """
    )
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
    print("the last three are identical in both - that is not optional.")
    print("whatever the model was trained to expect, it must see at evaluation too.")
    return train_transform, valid_transform


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Why the validation set must not be augmented

    Two reasons, and the first is the one people miss. An augmented validation score is **not reproducible** — run it twice on identical data and get two different numbers, so you cannot tell whether your model improved or the dice rolled differently. The second is that it measures the wrong thing: you want performance on real images, not on randomly degraded ones.
    """
    )
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
    print(
        "if the validation pipeline were augmented, the second number would be non-zero"
    )
    print("and every evaluation would give a different accuracy on the same data")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Checking augmentation is worth it

    Augmentation is not free — it costs CPU time per sample, and too much of it slows convergence because the model is chasing a moving target. The honest way to decide is to train with and without and compare validation curves, but before that there are two cheap checks:

    1. **Look at the output.** `make_grid` from Part 1 over a batch of augmented copies of one image. If you cannot tell what the subject is, neither can the model.
    2. **Measure the spread**, as above. If the mean standard deviation per pixel is near zero your augmentation is doing nothing; if it is enormous you have probably destroyed the content.

    Below is the first check as numbers, since these notebooks have no plots.
    """
    )
    return


@app.cell
def _(letter_f, torch, v2):
    def ink_ratio(pipeline, source, trials=200):
        """Ink area after the transform, as a multiple of the original.

        Above 1 means the transform zoomed in; near 0 means the subject is gone.
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
    print("Read the two columns differently. A mean above 1 is the crop zooming in,")
    print("which is fine and is half the point of it. The column that matters is the")
    print("worst case: near zero means some training samples have no subject at all,")
    print("and they still carry a label.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Exercises

    1. Build a pipeline you believe is label-preserving for MNIST digits, and one that is not. Justify each choice in a sentence.
    2. Run `RandomResizedCrop(28)` with the default `scale` over a 28x28 digit 1000 times and count how many crops contain no ink at all. Would you have guessed that number?
    3. `ColorJitter(hue=0.5)` on a dataset where colour is the label — red versus green peppers, say. Construct the two-pixel example that proves it relabels them.
    4. Take the train pipeline above and move `Normalize` before the augmentation. Does it raise? If not, what is different about the result, and which is correct?
    5. Measure the per-sample cost of the train pipeline against the valid pipeline with `timeit`. At what batch size does augmentation become the bottleneck rather than the GPU?

    Part 4 is the last one: datasets, and the pre-trained models the transfer learning demos use.
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

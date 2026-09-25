#!/usr/bin/env uv run marimo edit

import marimo

__generated_with = "0.14.17"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # torchvision for Machine Learning, Part 2: transforms

    `torchvision.transforms` is 56 of the 63 torchvision calls in this repository, so this is the notebook that matters most.

    A transform is a callable that takes an image and returns a modified one. `Compose` chains them. That is the whole idea, and it would need very little explanation were it not for two things: there are two versions of the API in circulation, and the order you chain them in is not arbitrary.

    Both of those are visible in this repository right now — some demos import `torchvision.transforms`, others import `torchvision.transforms.v2`, and one still uses `ToTensor`. That is worth sorting out, and this notebook is partly me doing that.
    """
    )
    return


@app.cell
def _():
    import torch
    from torchvision.transforms import v2
    from torchvision.transforms.functional import to_pil_image

    torch.manual_seed(42)
    return to_pil_image, torch, v2


@app.cell
def _(torch):
    # the same test image as Part 1: red down, blue across, green square top left
    IMG_H, IMG_W = 64, 96
    _rows = torch.linspace(0, 255, IMG_H).unsqueeze(1).expand(IMG_H, IMG_W)
    _cols = torch.linspace(0, 255, IMG_W).unsqueeze(0).expand(IMG_H, IMG_W)

    image = torch.zeros(3, IMG_H, IMG_W, dtype=torch.uint8)
    image[0] = _rows.to(torch.uint8)
    image[2] = _cols.to(torch.uint8)
    image[1, 4:20, 4:20] = 255

    print("source image", tuple(image.shape), image.dtype)
    return IMG_H, IMG_W, image


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## v1 and v2

    ```python
    import torchvision.transforms as transforms        # v1, the original
    from torchvision.transforms import v2              # v2, since torchvision 0.15
    ```

    v2 is a rewrite with the same names and mostly the same behaviour. Three reasons it exists:

    - **it transforms more than images.** A v1 transform takes one image. A v2 transform takes an image *and* its bounding boxes, masks or keypoints, and applies the same geometric change to all of them consistently. If you flip the image you must flip the boxes, and v1 left you to do that yourself.
    - **it is faster**, particularly on batches and on tensors that are already on a GPU.
    - **it works on batches**, not just single images.

    For plain image classification — which is everything in this repository — the two behave identically, so the practical advice is simply to use v2 in new code because v1 is in maintenance. The [official guidance](https://pytorch.org/vision/stable/transforms.html) says the same.

    The one real difference you will hit is the conversion pair at the start of every pipeline.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The conversion pair: [ToImage](https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToImage.html) and [ToDtype](https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToDtype.html)

    ```python
    v2.ToImage()                              # to a tv_tensors.Image, no arguments
    v2.ToDtype(dtype, scale=False)            # cast, optionally rescaling the range
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `dtype` | required | usually `torch.float32` |
    | `scale` | `False` | map 0–255 onto 0–1 |

    Together these replace v1's `ToTensor()`, which did both jobs in one step and is now deprecated. `MNIST/PyTorchDataLoaders.ipynb:38` pairs them correctly.

    The split was not tidying up. `ToTensor` behaves **differently depending on what you hand it**, and that is the bug it was retired for:

    - given a PIL image or a NumPy array, it casts to `float32` and rescales to 0–1
    - given a tensor that is already `uint8`, it returns it unchanged — still `uint8`, still 0–255

    So the same line in your `Compose` either scales your data or does nothing at all, depending on how the image got loaded. Run the cell and watch it happen.
    """
    )
    return


@app.cell
def _(image, to_pil_image, torch, v2):
    import warnings

    as_pil = to_pil_image(image)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # it warns about deprecation, which we know
        from_pil = v2.ToTensor()(as_pil)
        from_tensor = v2.ToTensor()(image)

    print("ToTensor(), same picture, two source types:")
    print(
        f"  from a PIL image  -> {str(from_pil.dtype):14} {from_pil.min():.3f} to {from_pil.max():.3f}"
    )
    print(
        f"  from a uint8 tensor -> {str(from_tensor.dtype):12} {from_tensor.min():.3f} to {from_tensor.max():.3f}"
    )
    print()

    pair = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    print("ToImage + ToDtype(scale=True), the same two:")
    print(
        f"  from a PIL image  -> {str(pair(as_pil).dtype):14} {pair(as_pil).min():.3f} to {pair(as_pil).max():.3f}"
    )
    print(
        f"  from a uint8 tensor -> {str(pair(image).dtype):12} {pair(image).min():.3f} to {pair(image).max():.3f}"
    )
    print()
    print("the pair gives the same answer either way. ToTensor does not.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    That matters here specifically. This repository contains both `from torchvision.transforms import ToTensor` and `tv_io.read_image`, and `read_image` returns a `uint8` tensor. A dataset loaded through PIL and a dataset loaded through `read_image` will behave differently under the same `ToTensor()` line — one scaled to 0–1, the other left at 0–255 — with no warning and no exception, just a model that trains on one and not the other.

    Use `ToImage()` and `ToDtype(torch.float32, scale=True)`. The cost of the split is that `scale=True` is something you have to remember, and its default is `False`.
    """
    )
    return


@app.cell
def _(image, torch, v2):
    # and the one you must not forget
    unscaled = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)])(image)
    scaled = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(image)

    print("scale=False (the default):", f"{unscaled.min():.1f} to {unscaled.max():.1f}")
    print("scale=True               :", f"{scaled.min():.3f} to {scaled.max():.3f}")
    print()
    print("both are float32, both have the right shape, and nothing raises.")
    print("the first one just trains badly.")
    return (scaled,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## [Normalize](https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.Normalize.html)

    ```python
    v2.Normalize(mean, std, inplace=False)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `mean` | required | one value per channel |
    | `std` | required | one value per channel |

    It computes `(x - mean) / std` per channel. Nothing more.

    The point is to get every input feature onto a comparable scale centred near zero, which is where activation functions are most sensitive and where gradients behave. Feed a network values in 0–1 and it will train; centre them at zero and it usually trains faster.

    The numbers `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` appear everywhere. They are the per-channel statistics of the ImageNet training set. You use them when your model was pre-trained on ImageNet, because the model learned its weights on inputs distributed that way — and Part 4 shows that the weights themselves can hand you the right preset so you do not have to remember the numbers at all.

    Two rules. `Normalize` must come **after** the conversion to float — it will refuse on `uint8`. And it takes the data out of 0–1, so anything you do after it that assumes 0–1 will be wrong.
    """
    )
    return


@app.cell
def _(image, scaled, v2):
    imagenet_norm = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalised = imagenet_norm(scaled)

    print("before normalize:", f"{scaled.min():.3f} to {scaled.max():.3f}")
    print("after normalize :", f"{normalised.min():.3f} to {normalised.max():.3f}")
    print()
    print("per-channel means after normalising:")
    print("  ", normalised.mean(dim=(1, 2)).tolist())
    print("(not zero, because this test image is not an ImageNet photo -")
    print(" on real data these would sit near zero, which is the point)")

    print()
    try:
        imagenet_norm(image)  # uint8
    except (TypeError, RuntimeError) as e:
        print("on uint8 it refuses:", str(e).split("\n")[0][:90])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Geometry: [Resize](https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.Resize.html), CenterCrop and [Grayscale](https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.Grayscale.html)

    ```python
    v2.Resize(size, interpolation='bilinear', max_size=None, antialias=True)
    v2.CenterCrop(size)
    v2.Grayscale(num_output_channels=1)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `size` (Resize) | required | an int scales the **shorter** side, keeping aspect ratio; a tuple forces exact dimensions |
    | `interpolation` | `'bilinear'` | how pixels are sampled |
    | `antialias` | `True` | filter before downsampling |
    | `num_output_channels` | `1` | set to 3 to keep three identical channels |

    The `size` behaviour is the one to get right. `Resize(224)` on a 64x96 image gives you 224x336 — it scaled the shorter side and kept the proportions. `Resize((224, 224))` gives you exactly 224x224 and squashes the image. Both are used in practice; the standard ImageNet recipe is `Resize(256)` then `CenterCrop(224)`, which preserves the aspect ratio and then takes the middle.

    `Grayscale(num_output_channels=3)` looks odd but is common: you want greyscale content in a 3-channel tensor because the pre-trained model you are feeding expects 3 channels.
    """
    )
    return


@app.cell
def _(image, v2):
    print("source                       ", tuple(image.shape[1:]), "(h, w)")
    print(
        "Resize(224)  - shorter side  ",
        tuple(v2.Resize(224)(image).shape[1:]),
        "aspect kept",
    )
    print(
        "Resize((224, 224)) - exact   ",
        tuple(v2.Resize((224, 224))(image).shape[1:]),
        "squashed",
    )
    print()
    print("the ImageNet recipe:")
    _recipe = v2.Compose([v2.Resize(256), v2.CenterCrop(224)])
    print("  Resize(256) then CenterCrop(224) ->", tuple(_recipe(image).shape[1:]))
    print()
    print("Grayscale()                  ", tuple(v2.Grayscale()(image).shape))
    print(
        "Grayscale(3)                 ",
        tuple(v2.Grayscale(num_output_channels=3)(image).shape),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## [Compose](https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.Compose.html), and why the order is not arbitrary

    ```python
    v2.Compose([transform, transform, ...])
    ```

    12 calls across 5 demos. Chains transforms into one callable applied per sample, and it is nothing cleverer than a for loop over the list.

    The order that works, and the reason for each position:

    1. **geometry first** — `Resize`, `CenterCrop`, and any augmentation. Cheapest on `uint8`, and cropping before converting means converting fewer pixels.
    2. **then `ToImage` and `ToDtype(scale=True)`** — the conversion to float in 0–1.
    3. **then `Normalize`** — which needs float, and takes you off the 0–1 scale.

    Getting it wrong is usually an exception rather than a silent problem, which is a mercy. The cell below shows the two common mistakes.
    """
    )
    return


@app.cell
def _(image, torch, v2):
    good = v2.Compose(
        [
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    out_good = good(image)
    print(
        "correct order ->",
        tuple(out_good.shape),
        f"{out_good.min():.2f} to {out_good.max():.2f}",
    )

    print()
    normalize_too_early = v2.Compose(
        [
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # on uint8
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    try:
        normalize_too_early(image)
    except (TypeError, RuntimeError) as e:
        print("Normalize before the float conversion:")
        print("  ", str(e).split("\n")[0][:95])
    return (good,)


@app.cell
def _(good, image, torch, v2):
    # the subtler one: scaling after normalising, which does not raise
    scale_after = v2.Compose(
        [
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            v2.ToDtype(
                torch.float32, scale=True
            ),  # a no-op here, but people add it "to be safe"
        ]
    )
    both = scale_after(image)
    print("an extra ToDtype after Normalize:")
    print("  result identical?", torch.allclose(both, good(image)))
    print("  - harmless in this case, because scale only acts on integer input.")
    print("    But it reads as though it rescales, which is how the habit spreads.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Class transforms and functional transforms

    Every transform comes in two forms:

    ```python
    v2.RandomRotation(degrees=20)(img)        # the class: picks its own random angle
    v2.functional.rotate(img, angle=13.2)     # the function: you supply the angle
    ```

    The class form is what goes in a `Compose` — it holds the configuration and samples any randomness itself. The functional form does one thing with parameters you provide, and it is what you need when the same random operation has to be applied to more than one thing consistently.

    The classic case is an image and its segmentation mask. If you call the class transform twice you get two different random rotations and the mask no longer lines up. You sample the parameters once, then apply the functional form to both. (In v2 you can often pass both to the class transform together and it handles this — which is the main reason v2 exists — but the functional route is worth knowing.)

    `PreTrainedModelsPart1.ipynb:362` uses `functional.to_pil_image`, which is the other common use: a one-off conversion rather than part of a pipeline.
    """
    )
    return


@app.cell
def _(image, torch, v2):
    mask = torch.zeros(1, 64, 96, dtype=torch.uint8)
    mask[0, 4:20, 4:20] = 1  # marks the green square

    wobbly = v2.RandomRotation(degrees=25)
    img_a, mask_a = wobbly(image), wobbly(mask)  # two separate random angles

    angle = v2.RandomRotation(degrees=25).make_params([image])["angle"]
    img_b = v2.functional.rotate(image, angle)
    mask_b = v2.functional.rotate(mask, angle)  # same angle, so they still line up

    def square_centre(m):
        ys, xs = torch.nonzero(m[0], as_tuple=True)
        return (
            (ys.float().mean().item(), xs.float().mean().item())
            if len(ys)
            else (float("nan"),) * 2
        )

    print("the green square's centre in the image vs in the mask:")
    print(
        f"  class form twice : image {square_centre(img_a[1:2] > 200)}  mask {square_centre(mask_a)}"
    )
    print(f"  functional, one angle ({angle:.1f} deg):")
    print(
        f"                     image {square_centre(img_b[1:2] > 200)}  mask {square_centre(mask_b)}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## tv_tensors

    One last thing you will see in v2 code and in stack traces.

    `ToImage()` does not return a plain tensor — it returns a `tv_tensors.Image`, which is a subclass carrying a label saying "I am an image". That label is how a v2 transform knows to rotate the image and the mask together but to leave a plain tensor of class labels alone.

    It behaves as a tensor everywhere that matters, so you rarely think about it. Two places it shows up: `print(type(x))` in a debugging session, and `isinstance` checks that fail in surprising ways. `as_subclass(torch.Tensor)` gets you a plain one if something downstream is fussy.
    """
    )
    return


@app.cell
def _(image, torch, v2):
    tagged = v2.ToImage()(image)
    print("type after ToImage:", type(tagged).__name__)
    print("still a tensor?    ", isinstance(tagged, torch.Tensor))
    print("shape, dtype       ", tuple(tagged.shape), tagged.dtype)
    print("back to plain      ", type(tagged.as_subclass(torch.Tensor)).__name__)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The pipeline worth copying

    ```python
    from torchvision.transforms import v2

    train_transform = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        # augmentation goes here - Part 3
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    valid_transform = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ```

    Two pipelines, identical except that only the training one is augmented. That distinction is the subject of Part 3, and it is the one students most often get wrong.

    ## Exercises

    1. Take the correct pipeline and remove `scale=True`. Print the range going into `Normalize` and work out what the normalised values become. Would you notice this from the loss curve alone?
    2. `Resize(224)` on a 100x400 image — what comes out? Now `Resize((224, 224))`. Which would you use for a photograph, and which for a document scan?
    3. Compute the actual per-channel mean and std of a folder of your own images and normalise with those instead of the ImageNet numbers. When is that the better choice?
    4. Write a `Compose` that produces a 3-channel greyscale float image normalised with a single mean and std. How many channels does `Normalize` need values for?
    5. Rotate an image and a mask with the class transform inside one `v2.Compose([...])` call, passing both together. Does v2 keep them aligned? Compare with the two-separate-calls version above.

    Part 3 is augmentation.
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

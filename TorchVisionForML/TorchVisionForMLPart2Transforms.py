#!/usr/bin/env uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # torchvision for Machine Learning, Part 2: transforms

    In Part 1 we read images as tensors and checked their layout and range. We will now use transforms to prepare those images for a model.

    A transform takes an image and returns a modified version. `Compose` applies a list of transforms in order. I will use the v2 API here, with a few comparisons to the older API you may see in other examples.
    """)
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
    mo.md(r"""
    ## v1 and v2

    ```python
    import torchvision.transforms as transforms  # the original API
    from torchvision.transforms import v2
    ```

    The two APIs share many names. v2 also supports working with related inputs, such as an image and its bounding boxes or segmentation mask, in one transform call. This helps keep them aligned when we crop, rotate or flip them.

    We will use v2 for these examples. The [transforms guide](https://pytorch.org/vision/stable/transforms.html) covers the differences in more detail. First we need to separate converting an image to a tensor from scaling its values.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The conversion pair: [ToImage](https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToImage.html) and [ToDtype](https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToDtype.html)

    ```python
    v2.ToImage()                              # to a tv_tensors.Image, no arguments
    v2.ToDtype(dtype, scale=False)            # cast, optionally rescaling the range
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `dtype` | required | usually `torch.float32` |
    | `scale` | `False` | map 0–255 onto 0–1 |

    `ToImage` converts the input to a `tv_tensors.Image` without scaling the values. `ToDtype` changes the data type; for our `uint8` image, `scale=True` also converts 0–255 to 0–1.

    The next cell compares this pair with the deprecated [v2.ToTensor](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToTensor.html). That transform scales our RGB PIL image, but passes the existing tensor through unchanged. Check both the type and range in the output.

    This example uses `v2.ToTensor`. The older `torchvision.transforms.ToTensor` accepts PIL images and NumPy arrays, and rejects a tensor input.
    """)
    return


@app.cell
def _(image, to_pil_image, torch, v2):
    import warnings

    as_pil = to_pil_image(image)

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )  # the text above explains the deprecation warning
        from_pil = v2.ToTensor()(as_pil)
        from_tensor = v2.ToTensor()(image)

    print("v2.ToTensor(), same picture, two source types:")
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
    print("the conversion pair gives matching ranges for these two inputs")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The conversion pair gives us the same floating point image from either source. This is useful when one loader returns PIL images and another returns tensors.

    We still need `scale=True`. The next cell leaves it out, then compares the ranges. Both results have the same shape and type, so those checks alone would miss the difference.
    """)
    return


@app.cell
def _(image, torch, v2):
    # compare type conversion with and without scaling
    unscaled = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)])(image)
    scaled = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(image)

    print("scale=False (the default):", f"{unscaled.min():.1f} to {unscaled.max():.1f}")
    print("scale=True               :", f"{scaled.min():.3f} to {scaled.max():.3f}")
    print()
    print("both are float32, both have the right shape, and nothing raises.")
    print("check the value range as well as the type and shape")
    return (scaled,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [Normalize](https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.Normalize.html)

    ```python
    v2.Normalize(mean, std, inplace=False)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `mean` | required | one value per channel |
    | `std` | required | one value per channel |

    `Normalize` applies `(x - mean) / std` to each channel. It needs floating point input, so we convert and scale the image first.

    Here I have used the ImageNet normalisation values found in many pre-trained model examples. The preprocessing must match the weights we use; in Part 4 we will obtain it from the weights object.

    Normalised values can be negative or greater than one. Keep this in mind when displaying the result or applying a transform that expects values in 0–1.
    """)
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
    print("(these means describe one generated image;")
    print(" normalisation does not force each individual image to have zero mean)")

    print()
    try:
        imagenet_norm(image)  # uint8
    except (TypeError, RuntimeError) as e:
        print("on uint8 it refuses:", str(e).split("\n")[0][:90])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Resizing and greyscale conversion: [Resize](https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.Resize.html), CenterCrop and [Grayscale](https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.Grayscale.html)

    ```python
    v2.Resize(size, interpolation='bilinear', max_size=None, antialias=True)
    v2.CenterCrop(size)
    v2.Grayscale(num_output_channels=1)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `size` (Resize) | required | an int scales the shorter side, keeping aspect ratio; a tuple forces exact dimensions |
    | `interpolation` | `'bilinear'` | how pixels are sampled |
    | `antialias` | `True` | filter before downsampling |
    | `num_output_channels` | `1` | set to 3 to keep three identical channels |

    `Resize(224)` scales the shorter side to 224 and keeps the aspect ratio. Our 64 by 96 image becomes 224 by 336. `Resize((224, 224))` sets both dimensions, which changes the proportions of this image.

    We can resize whilst preserving the proportions, then use `CenterCrop` to get a fixed size. The example uses `Resize(256)` followed by `CenterCrop(224)`.

    `Grayscale(num_output_channels=3)` gives us three identical channels. This is useful when we have greyscale content but the model expects three input channels.
    """)
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
    print("a resize and centre crop:")
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
    mo.md(r"""
    ## [Compose](https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.Compose.html): applying transforms in order

    ```python
    v2.Compose([transform, transform, ...])
    ```

    `Compose` applies each transform to the output of the previous one. For the examples below we use this order:

    1. Resize and crop the image.
    2. Convert it with `ToImage` and `ToDtype(torch.float32, scale=True)`.
    3. Apply `Normalize`.

    This keeps the geometry operations on our original image and gives `Normalize` the floating point input it needs. Other pipelines can use a different order, but each step must accept what the previous one returns. The following cells show an ordering error and a redundant conversion.
    """)
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
    # repeating the same dtype conversion does not rescale the image
    scale_after = v2.Compose(
        [
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            v2.ToDtype(torch.float32, scale=True),  # the input is already float32
        ]
    )
    both = scale_after(image)
    print("an extra ToDtype after Normalize:")
    print("  result identical?", torch.allclose(both, good(image)))
    print("  the input is already float32, so this conversion leaves it unchanged.")
    print("  scale=True does not rescale a float32 tensor to the same dtype.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Class transforms and functional transforms

    ```python
    v2.RandomRotation(degrees=20)(img)
    v2.functional.rotate(img, angle=13.2)
    ```

    The class stores our settings and chooses a random angle each time we call it. With the functional version we supply the angle ourselves.

    If an image has a matching mask, separate random rotations can move them out of alignment. We can choose an angle once and use it for both, taking care to use suitable interpolation for the mask. v2 can also transform an image and a `tv_tensors.Mask` together in one call.
    """)
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
    mo.md(r"""
    ## tv_tensors

    `ToImage` returns a `tv_tensors.Image`, which is a subclass of `torch.Tensor`. The type tells v2 that this input represents an image. Other types identify masks and bounding boxes so a transform can handle each appropriately.

    We can still use tensor operations on an `Image`, and `isinstance(image, torch.Tensor)` is true. If an API needs a plain tensor, `as_subclass(torch.Tensor)` gives us one. The next cell prints the types before and after this conversion.
    """)
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
    mo.md(r"""
    ## A preprocessing pipeline

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

    These pipelines currently do the same thing. I have left a place for training augmentation, which we will add in Part 3. The validation pipeline will keep a fixed sequence of preprocessing steps.

    ## Exercises

    1. Remove `scale=True` and print the range before and after `Normalize`. Explain the difference using `(x - mean) / std`.
    2. Compare `Resize(224)` and `Resize((224, 224))` on a 100 by 400 image. Which preserves its proportions?
    3. Calculate the channel means and standard deviations for your own images. Discuss when to use these and when to keep a pre-trained model's preprocessing.
    4. Create a pipeline that produces three identical greyscale channels, then normalise it. Compare providing one mean and standard deviation with providing three identical values.
    5. Rotate an image and a `tv_tensors.Mask` together in one v2 call. Compare this with calling the random transform separately on each input.

    Part 3 covers data augmentation.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

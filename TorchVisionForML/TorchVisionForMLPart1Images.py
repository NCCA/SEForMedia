#!/usr/bin/env uv run marimo edit

import marimo

__generated_with = "0.14.17"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # torchvision for Machine Learning, Part 1: images as tensors

    Four notebooks on torchvision, following the NumPy and PyTorch sets in `NumPyForML/` and `PyTorchForML/`. Same method: I counted what the demos in this repository actually call. torchvision accounts for 63 call sites across 12 demos, and 56 of those are `transforms`, which tells you where the weight is.

    The four parts are:

    1. Images as tensors, and getting them on and off disk (this notebook)
    2. Transforms, and the v1 to v2 change
    3. Augmentation
    4. Datasets and pre-trained models

    This first one is the least glamorous and the one that causes the most wasted time. An image can be a `uint8` tensor, a `float32` tensor, a PIL image or a NumPy array, with the colour channel in one of two places and the values on one of two scales. Get any of that wrong and you either get an exception, or — much worse — a picture that looks like noise and a model that quietly learns nothing.

    Documentation is at [pytorch.org/vision](https://pytorch.org/vision/stable/index.html).
    """
    )
    return


@app.cell
def _():
    import torch
    import torchvision
    import torchvision.io as tv_io
    from torchvision.transforms.functional import to_pil_image
    from torchvision.utils import make_grid

    torch.manual_seed(42)
    print("torchvision", torchvision.__version__)
    return make_grid, to_pil_image, torch, tv_io


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The two conventions you have to hold in your head

    torchvision represents an image as a tensor of shape **(channels, height, width)** — `CHW`. Matplotlib, PIL, OpenCV and almost everything else outside deep learning use **(height, width, channels)** — `HWC`.

    Neither is wrong. `CHW` puts each colour channel in one contiguous block, which is what convolutions want; `HWC` puts each pixel's three values together, which is what a display wants. But you cross the boundary every time you plot something, and the crossing is a `permute`.

    | | Layout | Range | Type |
    | --- | --- | --- | --- |
    | `torchvision.io.read_image` | `CHW` | 0–255 | `uint8` tensor |
    | after `ToDtype(float32, scale=True)` | `CHW` | 0–1 | `float32` tensor |
    | what `plt.imshow` wants | `HWC` | 0–255 int or 0–1 float | array or tensor |
    | PIL image | — | 0–255 | `PIL.Image` |

    A batch adds a leading dimension, so a batch of images is `NCHW`. That is the shape every model in this repository expects.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Making a test image

    Nothing here downloads anything. I build a small image with an obvious structure — a red-to-blue gradient with a green square in one corner — so that if a layout goes wrong it is visible rather than subtle.
    """
    )
    return


@app.cell
def _(torch):
    import tempfile
    from pathlib import Path

    work_dir = Path(tempfile.mkdtemp())

    H, W = 64, 96
    _rows = torch.linspace(0, 255, H).unsqueeze(1).expand(H, W)
    _cols = torch.linspace(0, 255, W).unsqueeze(0).expand(H, W)

    test_image = torch.zeros(3, H, W, dtype=torch.uint8)
    test_image[0] = _rows.to(torch.uint8)  # red increases downwards
    test_image[2] = _cols.to(torch.uint8)  # blue increases rightwards
    test_image[1, 4:20, 4:20] = 255  # green square, top left

    print("shape", tuple(test_image.shape), "= (channels, height, width)")
    print(
        "dtype",
        test_image.dtype,
        "range",
        test_image.min().item(),
        "to",
        test_image.max().item(),
    )
    print()
    print("the green square is in the TOP LEFT - remember that, it is the layout check")
    return H, W, test_image, work_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## [torchvision.io](https://pytorch.org/vision/stable/io.html)

    ```python
    tv_io.read_image(path, mode=ImageReadMode.UNCHANGED, apply_exif_orientation=False)
    tv_io.decode_image(data, mode=...)      # from bytes already in memory
    tv_io.write_png(tensor, filename, compression_level=6)
    tv_io.write_jpeg(tensor, filename, quality=75)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `path` | required | the file to read |
    | `mode` | `UNCHANGED` | force a channel layout — see below |
    | `apply_exif_orientation` | `False` | honour the rotation flag phones write into JPEGs |

    `read_image` returns a `uint8` tensor in `CHW`. No PIL involved, no conversion, and it works inside a `DataLoader` worker without the overhead PIL brings. `PreTrainedModelsPart1.ipynb:323` uses it exactly this way.

    The `mode` parameter is the one to set deliberately. Leave it `UNCHANGED` and a folder of mixed images gives you 1-channel greyscale for some, 3-channel RGB for others, and 4-channel RGBA for any PNG with transparency — and then `torch.stack` in the DataLoader fails on the first batch that mixes them. Passing `ImageReadMode.RGB` makes every image 3 channels whatever it was.
    """
    )
    return


@app.cell
def _(H, W, test_image, to_pil_image, torch, tv_io, work_dir):
    from torchvision.io import ImageReadMode

    rgb_path = work_dir / "gradient.png"
    tv_io.write_png(test_image, str(rgb_path))

    # an RGBA image, as a PNG with transparency would be.
    # note write_png only handles 1 or 3 channels, so this one goes through PIL
    rgba_tensor = torch.cat([test_image, torch.full((1, H, W), 128, dtype=torch.uint8)])
    rgba_path = work_dir / "with_alpha.png"
    to_pil_image(rgba_tensor).save(rgba_path)

    print("read back UNCHANGED:")
    print("  gradient.png  ", tuple(tv_io.read_image(str(rgb_path)).shape))
    print(
        "  with_alpha.png",
        tuple(tv_io.read_image(str(rgba_path)).shape),
        "<- 4 channels",
    )
    print()
    print("read back with mode=ImageReadMode.RGB:")
    print(
        "  gradient.png  ",
        tuple(tv_io.read_image(str(rgb_path), mode=ImageReadMode.RGB).shape),
    )
    print(
        "  with_alpha.png",
        tuple(tv_io.read_image(str(rgba_path), mode=ImageReadMode.RGB).shape),
        "<- now consistent",
    )
    print()
    print(
        "  as greyscale  ",
        tuple(tv_io.read_image(str(rgb_path), mode=ImageReadMode.GRAY).shape),
    )
    return (rgb_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    That mixed-channel failure is worth seeing once, because the error appears in the DataLoader rather than anywhere near the file that caused it, and it only fires on the batch that happens to contain the odd image.
    """
    )
    return


@app.cell
def _(torch):
    grey = torch.randint(0, 255, (1, 64, 96), dtype=torch.uint8)
    colour = torch.randint(0, 255, (3, 64, 96), dtype=torch.uint8)

    try:
        torch.stack([colour, grey, colour])
    except RuntimeError as e:
        print("what the DataLoader raises when it hits a greyscale file:")
        print("  RuntimeError:", str(e).split("\n")[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Getting an image back out to look at it

    ```python
    to_pil_image(pic, mode=None)      # tensor -> PIL.Image
    pil_to_tensor(pic)                # PIL.Image -> uint8 CHW tensor
    tensor.permute(1, 2, 0)           # CHW -> HWC, for matplotlib
    ```

    Two routes. `to_pil_image` from `torchvision.transforms.functional` hands you a PIL image, which marimo and Jupyter will display directly — that is what `PreTrainedModelsPart1.ipynb:362` does. Or `permute` to `HWC` and give it to `plt.imshow`.

    A note on the import name. The repository writes `import torchvision.transforms.functional as F` in the pre-trained model demos, and `import torch.nn.functional as F` is the near-universal convention everywhere else. They are completely different modules and both are conventionally `F`. If you have both in one file, alias one of them to something else — I use `TF` — or you will eventually call the wrong one and get a confusing error.
    """
    )
    return


@app.cell
def _(rgb_path, to_pil_image, tv_io):
    loaded = tv_io.read_image(str(rgb_path))

    pil_version = to_pil_image(loaded)
    print(
        "as PIL:",
        pil_version.size,
        pil_version.mode,
        "- note PIL reports (width, height)",
    )

    for_matplotlib = loaded.permute(1, 2, 0)
    print("for imshow:", tuple(for_matplotlib.shape), "= (height, width, channels)")
    print()
    print("and the green square is still where it should be:")
    print("  top-left pixel  ", loaded[:, 10, 10].tolist(), "<- green channel high")
    print(
        "  bottom-right    ",
        loaded[:, 60, 90].tolist(),
        "<- red and blue high, green low",
    )
    return (loaded,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### What forgetting the permute looks like

    If you pass a `CHW` tensor straight to `imshow`, matplotlib reads the 3 as the height and the height as the width. With our 64x96 image it produces a 3-pixel-tall smear. On a square image it produces something that looks almost plausible, which is worse — you can stare at it for a while before realising.
    """
    )
    return


@app.cell
def _(loaded):
    print("what imshow would try to draw, given each:")
    print(
        "  correct (permuted):",
        tuple(loaded.permute(1, 2, 0).shape),
        "-> 64 x 96, 3 channels",
    )
    print("  forgotten permute :", tuple(loaded.shape), "-> 3 x 64, 96 'channels'")
    print()
    print("matplotlib either raises, or draws 3 rows of nonsense")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The other axis of confusion: dtype and range

    An image tensor is `uint8` in 0–255 or `float32` in 0–1. Both are normal; mixing them up is the problem.

    `plt.imshow` will take either, but it applies a different rule to each: integers are read as 0–255, floats as 0–1. So a `float32` tensor still holding 0–255 values gets clipped, and every pixel above 1.0 comes out pure white. If your image plots as a white rectangle, this is why — and it means the same values are going into your model unscaled, which is the loss-refuses-to-move bug from `PyTorchForML` Part 5.
    """
    )
    return


@app.cell
def _(loaded, torch):
    as_float_unscaled = loaded.to(torch.float32)  # 0-255 but float - wrong
    as_float_scaled = loaded.to(torch.float32) / 255.0  # 0-1 - right

    print(
        "uint8          ",
        loaded.dtype,
        f"{loaded.min().item()} to {loaded.max().item()}",
    )
    print(
        "float unscaled ",
        as_float_unscaled.dtype,
        f"{as_float_unscaled.min():.1f} to {as_float_unscaled.max():.1f}",
    )
    print(
        "float scaled   ",
        as_float_scaled.dtype,
        f"{as_float_scaled.min():.3f} to {as_float_scaled.max():.3f}",
    )
    print()
    print("fraction of pixels imshow would clip to white in the unscaled version:")
    print(f"  {(as_float_unscaled > 1.0).float().mean().item():.1%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## [make_grid](https://pytorch.org/vision/stable/generated/torchvision.utils.make_grid.html)

    ```python
    make_grid(tensor, nrow=8, padding=2, normalize=False, value_range=None,
              scale_each=False, pad_value=0.0)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `tensor` | required | a batch `NCHW`, or a list of images |
    | `nrow` | `8` | images **per row**, despite the name |
    | `padding` | `2` | pixels between images |
    | `normalize` | `False` | rescale to 0–1 first |

    Tiles a batch into one image so you can look at a whole batch at once. Not used in this repository, which is a shame — it is the fastest way to check that a data pipeline is producing what you think, and I would put it in every augmentation notebook.

    Watch `nrow`: it is the number of images in each row, not the number of rows. Everyone gets that backwards once.
    """
    )
    return


@app.cell
def _(loaded, make_grid, torch):
    batch = torch.stack([loaded] * 12)
    print("a batch of 12:", tuple(batch.shape))

    grid = make_grid(batch, nrow=4, padding=4, pad_value=255)
    print("as a grid:    ", tuple(grid.shape), "- one image, 3 rows of 4")
    print()
    print("still CHW, so it needs the same permute before plotting:")
    print("  ", tuple(grid.permute(1, 2, 0).shape))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## A helper worth keeping

    Every image notebook needs the same function, so here it is once. It takes anything — `uint8` or float, `CHW` or a batch — and returns something `imshow` will draw correctly.
    """
    )
    return


@app.cell
def _(H, W, loaded, make_grid, torch):
    def to_displayable(img: torch.Tensor) -> torch.Tensor:
        """CHW or NCHW tensor, any dtype, -> HWC float in 0-1 ready for imshow."""
        if img.ndim == 4:  # a batch, so tile it first
            img = make_grid(img, nrow=8)
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        else:
            img = img.float()
            if img.max() > 1.0:  # float but still on the 0-255 scale
                img = img / 255.0
        if img.shape[0] == 1:  # greyscale, drop the channel axis
            return img.squeeze(0).clamp(0, 1)
        return img.permute(1, 2, 0).clamp(0, 1)

    for label, candidate in [
        ("uint8 CHW      ", loaded),
        ("float 0-255    ", loaded.float()),
        ("float 0-1      ", loaded.float() / 255),
        ("greyscale      ", loaded[:1]),
        ("a batch of 5   ", torch.stack([loaded] * 5)),
    ]:
        out = to_displayable(candidate)
        print(f"{label} -> {str(tuple(out.shape)):18} {out.dtype}  max {out.max():.3f}")

    print()
    print(f"(the source image was {H} x {W})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Exercises

    1. Write the test image as a JPEG at quality 10 and read it back. Subtract it from the original and find the largest difference. Where in the image is it worst, and why?
    2. Read an image with `ImageReadMode.GRAY` and work out what weights torchvision used to combine the channels. (They are not equal thirds.)
    3. Build a batch from a folder of mixed greyscale and colour images without `mode=`, and produce the stacking error deliberately. Then fix it two different ways.
    4. Take `to_displayable` and break it: find an input for which it produces something wrong rather than raising. What would you add to catch that?
    5. `make_grid` with `normalize=True` and `scale_each=True` on a batch where one image is much brighter than the rest. What does `scale_each` change, and when would that mislead you?

    Part 2 covers the transforms themselves, and the v1 to v2 change you will see half-applied across the demos here.
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

#!/usr/bin/env uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # torchvision for Machine Learning, Part 1: images as tensors

    These four notebooks cover the torchvision code we use in the machine learning demos. They follow on from `NumPyForML/` and `PyTorchForML/`.

    1. Images as tensors, including reading and writing files
    2. Transforms and the v2 API
    3. Data augmentation
    4. Datasets and pre-trained models

    I will start with the image itself. We need to know its shape, data type and value range before passing it to a model or displaying it. The examples use a generated image so we can check the results without downloading a dataset.

    The [torchvision documentation](https://pytorch.org/vision/stable/index.html) has the full API reference.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import torch
    import torchvision
    import torchvision.io as tv_io
    from torchvision.transforms.functional import to_pil_image
    from torchvision.utils import make_grid

    torch.manual_seed(42)
    print("torchvision", torchvision.__version__)
    return make_grid, plt, to_pil_image, torch, tv_io


@app.cell
def _(plt, torch):
    def show_images(images: list[tuple[str, torch.Tensor]]):
        """Draw CHW tensors using their supplied values, without rescaling floats."""
        figure, axes = plt.subplots(
            1, len(images), figsize=(4 * len(images), 3.5), squeeze=False
        )
        for axis, (title, tensor) in zip(axes[0], images):
            data = tensor.detach().cpu()
            if data.shape[0] == 1:
                axis.imshow(
                    data[0],
                    cmap="gray",
                    vmin=0,
                    vmax=255 if data.dtype == torch.uint8 else 1,
                )
            else:
                axis.imshow(data.permute(1, 2, 0))
            axis.set_title(title)
            axis.set_axis_off()
        figure.tight_layout()
        plt.close(figure)
        return figure

    return (show_images,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Image layout

    For the colour images here, torchvision uses `(channels, height, width)`, usually written as `CHW`. To display one with Matplotlib we need `(height, width, channels)`, or `HWC`. We can reorder the axes using `permute(1, 2, 0)`.

    | Image | Layout | Range | Type |
    | --- | --- | --- | --- |
    | Our PNG loaded with `read_image` | `CHW` | 0–255 | `uint8` tensor |
    | After `ToDtype(float32, scale=True)` | `CHW` | 0–1 | `float32` tensor |
    | RGB input to `plt.imshow` | `HWC` | 0–255 integer or 0–1 float | array or tensor |
    | Our image converted to PIL | PIL image object | 0–255 | `PIL.Image` |

    A batch adds a dimension at the front, giving us `NCHW`. Keep an eye on this when we move from a single image to a `DataLoader`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Making a test image

    I have used a red gradient down the image, a blue gradient across it and a green square near the top left. This gives us something recognisable when checking the channel order. We generate it below using the tensor operations from the PyTorch notebooks.
    """)
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
    print("we will use the green square near the top left to check the layout")
    return H, W, test_image, work_dir


@app.cell
def _(show_images, test_image, torch):
    _panels = [("Generated RGB image", test_image)]
    for _index, _name in enumerate(
        ["Red: downwards", "Green: square", "Blue: rightwards"]
    ):
        _channel = torch.zeros_like(test_image)
        _channel[_index] = test_image[_index]
        _panels.append((_name, _channel))
    show_images(_panels)
    return


@app.cell
def _(mo):
    channel_order = mo.ui.dropdown(
        ["RGB", "BGR", "GRB"], value="RGB", label="Channel order"
    )
    mo.vstack(
        [
            mo.md(
                "Swap the channels and watch the colours change. The pixel positions stay the same."
            ),
            channel_order,
        ]
    )
    return (channel_order,)


@app.cell
def _(channel_order, show_images, test_image):
    _indices = ["RGB".index(_letter) for _letter in channel_order.value]
    show_images(
        [("Original RGB", test_image), (channel_order.value, test_image[_indices])]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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

    We will write our test image to a PNG and read it back as a tensor. `read_image` returns these 8-bit images in `CHW` layout.

    The `mode` argument lets us choose the colour channels. With `UNCHANGED`, a greyscale file can give us one channel, an RGB file three and an RGBA file four. These cannot be stacked into one batch without conversion. Here we use `ImageReadMode.RGB` to read both files with three channels.
    """)
    return


@app.cell
def _(H, W, test_image, to_pil_image, torch, tv_io, work_dir):
    from torchvision.io import ImageReadMode

    rgb_path = work_dir / "gradient.png"
    tv_io.write_png(test_image, str(rgb_path))

    # use PIL to save four channels; write_png accepts one or three
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
    return rgb_path, rgba_tensor


@app.cell
def _(rgb_path, rgba_tensor, show_images, test_image, torch, tv_io):
    _rgb = tv_io.read_image(str(rgb_path), mode=tv_io.ImageReadMode.RGB)
    _grey = tv_io.read_image(str(rgb_path), mode=tv_io.ImageReadMode.GRAY)
    _alpha = rgba_tensor[3:].float() / 255
    _foreground = rgba_tensor[:3].float() / 255
    _white = _foreground * _alpha + (1 - _alpha)
    _black = _foreground * _alpha
    assert torch.equal(test_image, _rgb)
    show_images(
        [
            ("PNG round trip: identical", _rgb),
            ("Read as greyscale", _grey),
            ("Alpha 128 over white", _white),
            ("Alpha 128 over black", _black),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can reproduce the channel mismatch without a dataset. The next cell tries to stack colour and greyscale tensors. This is also what can happen when a `DataLoader` collects images of different shapes into a batch.
    """)
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
    mo.md(r"""
    ## Displaying a tensor

    ```python
    to_pil_image(pic, mode=None)  # convert a tensor to a PIL image
    pil_to_tensor(pic)           # convert an 8-bit PIL image to a uint8 tensor
    tensor.permute(1, 2, 0)      # reorder CHW to HWC for Matplotlib
    ```

    We can convert the tensor to a PIL image with `to_pil_image`, or rearrange its axes for `plt.imshow`. The next cell checks both forms and prints a couple of pixels so we can check the green square is still in the same place.

    You will often see `F` used for both `torchvision.transforms.functional` and `torch.nn.functional`. I use `TF` for the torchvision module when both are needed in one file.
    """)
    return


@app.cell
def _(mo, plt, rgb_path, to_pil_image, tv_io):
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
    _figure, _axis = plt.subplots(figsize=(4, 3))
    _axis.imshow(for_matplotlib)
    _axis.set_title("Matplotlib: HWC")
    _axis.set_axis_off()
    plt.close(_figure)
    mo.hstack(
        [mo.vstack([mo.md("**PIL: RGB**"), mo.image(pil_version, width=300)]), _figure]
    )
    return (loaded,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Forgetting to reorder the axes

    Our tensor has shape `(3, 64, 96)`. Passing it directly to `imshow` would put 96 in the channel position, so Matplotlib rejects it. RGB and RGBA inputs need three or four channels in the last dimension, as described in the [imshow documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html).

    The next cell shows the correct layout alongside a tempting mistake: using `reshape` instead of `permute`. Reshaping changes which values belong to each pixel. It does not reorder the axes.
    """)
    return


@app.cell
def _(loaded, plt, show_images):
    print("what imshow would try to draw, given each:")
    print(
        "  correct (permuted):",
        tuple(loaded.permute(1, 2, 0).shape),
        "-> 64 x 96, 3 channels",
    )
    print("  forgotten permute :", tuple(loaded.shape), "-> 3 x 64, 96 'channels'")
    print()
    _figure, _axis = plt.subplots()
    try:
        _axis.imshow(loaded)
    except TypeError as _error:
        print("Actual imshow error:", _error)
    finally:
        plt.close(_figure)
    _reshaped = loaded.reshape(loaded.shape[1], loaded.shape[2], 3)
    show_images(
        [
            ("Correct: permute", loaded),
            ("Wrong: reshape to HWC", _reshaped.permute(2, 0, 1)),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data type and value range

    For these examples we use either `uint8` values from 0 to 255 or floating point values from 0 to 1. Converting the type with `.to(torch.float32)` leaves the values unchanged; we also need to divide by 255.

    For RGB data, `imshow` expects floating point components in 0–1 and clips values outside that range. An unscaled image can therefore look too bright or lose colour detail. Check the range going into the model as well as the range used for display.
    """)
    return


@app.cell
def _(loaded, show_images, torch):
    as_float_unscaled = loaded.to(
        torch.float32
    )  # converting the type leaves the range unchanged
    as_float_scaled = (
        loaded.to(torch.float32) / 255.0
    )  # scale for display as floating point RGB

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
    print("fraction of colour components above the floating point display range:")
    print(f"  {(as_float_unscaled > 1.0).float().mean().item():.1%}")
    show_images(
        [
            ("uint8: 0–255", loaded),
            ("Unscaled float: clipped to 0–1", as_float_unscaled.clamp(0, 1)),
            ("Float divided by 255", as_float_scaled),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [make_grid](https://pytorch.org/vision/stable/generated/torchvision.utils.make_grid.html)

    ```python
    make_grid(tensor, nrow=8, padding=2, normalize=False, value_range=None,
              scale_each=False, pad_value=0.0)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `tensor` | required | a batch `NCHW`, or a list of images |
    | `nrow` | `8` | images per row, despite the name |
    | `padding` | `2` | pixels between images |
    | `normalize` | `False` | rescale to 0–1 first |

    `make_grid` arranges a batch into a single image. I use this when checking several samples together, particularly after augmentation.

    `nrow` is the number of images in each row. Change the controls below to rearrange twelve images and adjust the gaps between them. The result is still `CHW`, so we need to reorder it before plotting.
    """)
    return


@app.cell
def _(mo):
    images_per_row = mo.ui.slider(
        1, 12, value=4, label="Images per row", show_value=True
    )
    grid_padding = mo.ui.slider(
        0, 12, value=4, label="Padding (pixels)", show_value=True
    )
    mo.hstack([images_per_row, grid_padding])
    return grid_padding, images_per_row


@app.cell
def _(grid_padding, images_per_row, loaded, make_grid, show_images, torch):
    batch = torch.stack([loaded] * 12)
    print("a batch of 12:", tuple(batch.shape))

    grid = make_grid(
        batch, nrow=images_per_row.value, padding=grid_padding.value, pad_value=255
    )
    print("as a grid:    ", tuple(grid.shape), "- one CHW image")
    print()
    print("still CHW, so it needs the same permute before plotting:")
    print("  ", tuple(grid.permute(1, 2, 0).shape))
    show_images([("Batch of twelve images", grid)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A display helper

    The following helper handles the image forms used in this notebook. It tiles batches, converts to floating point and rearranges the axes for display. For a single greyscale image it removes the channel dimension.

    I have kept the range check simple: a floating point maximum above 1 is treated as evidence of a 0–255 image. That is an assumption, and it will give the wrong result for some inputs (including normalised model inputs). We will come back to this in the exercises.
    """)
    return


@app.cell
def _(H, W, loaded, make_grid, plt, torch):
    def to_displayable(img: torch.Tensor) -> torch.Tensor:
        """
        Prepare the example images for display with imshow.

        Parameters
        ----------
        img : torch.Tensor
            CHW image or NCHW batch, with values in 0-1 or 0-255.

        Returns
        -------
        torch.Tensor
            Floating point values in 0-1, arranged as HWC or HW for a
            single greyscale image. Batches are tiled into a grid.

        Notes
        -----
        A floating point maximum above 1 is treated as a 0-255 input.
        This assumption does not handle normalised model inputs.
        """
        if img.ndim == 4:  # a batch, so tile it first
            img = make_grid(img, nrow=8)
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        else:
            img = img.float()
            if img.max() > 1.0:  # assume values above 1 indicate the 0-255 range
                img = img / 255.0
        if img.shape[0] == 1:  # greyscale, drop the channel axis
            return img.squeeze(0).clamp(0, 1)
        return img.permute(1, 2, 0).clamp(0, 1)

    _figure, _axes = plt.subplots(1, 5, figsize=(16, 3))
    for _axis, (label, candidate) in zip(
        _axes,
        [
            ("uint8 CHW      ", loaded),
            ("float 0-255    ", loaded.float()),
            ("float 0-1      ", loaded.float() / 255),
            ("greyscale      ", loaded[:1]),
            ("a batch of 5   ", torch.stack([loaded] * 5)),
        ],
    ):
        out = to_displayable(candidate)
        _axis.imshow(out, cmap="gray", vmin=0, vmax=1)
        _axis.set_title(label.strip())
        _axis.set_axis_off()
        print(f"{label} -> {tuple(out.shape)!s:18} {out.dtype}  max {out.max():.3f}")

    print()
    print(f"(the source image was {H} x {W})")
    _figure.tight_layout()
    plt.close(_figure)
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## JPEG compression

    PNG preserved every value. JPEG trades some detail for a smaller file. Move the quality slider and look at the green square and the gradients. We subtract floating point values so negative differences cannot wrap around as they would with `uint8`.

    The heatmap shows the largest absolute channel difference at each pixel, using a fixed 0–255 scale so we can compare settings. The mean and maximum below help us spot smaller changes.
    """)
    return


@app.cell
def _(mo):
    jpeg_quality = mo.ui.slider(1, 100, value=10, label="JPEG quality", show_value=True)
    jpeg_quality
    return (jpeg_quality,)


@app.cell
def _(jpeg_quality, loaded, mo, plt, tv_io):
    _encoded = tv_io.encode_jpeg(loaded, quality=jpeg_quality.value)
    _decoded = tv_io.decode_jpeg(_encoded)
    _difference = (loaded.float() - _decoded.float()).abs()
    _figure, _axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for _axis, _title, _image in zip(
        _axes[:2],
        ["Original", f"JPEG quality {jpeg_quality.value}"],
        [loaded, _decoded],
    ):
        _axis.imshow(_image.permute(1, 2, 0))
        _axis.set_title(_title)
        _axis.set_axis_off()
    _heatmap = _axes[2].imshow(_difference.amax(dim=0), cmap="magma", vmin=0, vmax=255)
    _axes[2].set_title("Largest channel difference")
    _axes[2].set_axis_off()
    _figure.colorbar(_heatmap, ax=_axes[2], label="Difference (0–255)")
    _figure.tight_layout()
    plt.close(_figure)
    mo.vstack(
        [
            _figure,
            mo.md(
                f"JPEG: **{_encoded.numel():,} bytes**; mean absolute channel difference: **{_difference.mean():.2f}**; maximum: **{_difference.max():.0f}**."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Write the test image as a JPEG at quality 10 and read it back. Convert both images to a signed or floating point type before subtracting them. Where is the largest difference?
    2. Read the image with `ImageReadMode.GRAY`. Find the channel weights used by torchvision and compare the result with an equal average of the channels.
    3. Try batching mixed greyscale and colour images. Reproduce the stacking error, then find two ways to make the channel counts agree.
    4. Give `to_displayable` an input for which its range assumption fails. How could we make the expected input range explicit?
    5. Use `make_grid` on a batch containing one much brighter image. Compare `normalize=True` with and without `scale_each=True`. Does the display still show the brightness difference?

    In Part 2 we will combine image conversions and transforms into a pipeline.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

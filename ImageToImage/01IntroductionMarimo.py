#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Image-to-image learning: introduction to U-Net

    We will build an encoder decoder that maps an image to another grid of values.
    The shared architecture gives us a starting point for three different problems:
    [denoising](DenoisingMarimo.py), [2× super resolution](SuperResolutionMarimo.py)
    and [pet segmentation](PetSegmentationMarimo.py).

    This notebook explains the common ideas such as feature maps, pooling, skip connections,
    upsampling, patches, validation and tiled inference. The other notebooks are stand alone examples which use their own loss functions and complete training and evaluation loops to demonstrate the principles described here.

    Start here, then open any of the three examples independently.

    This introduction uses small generated tensors on the CPU. It does not download
    a dataset or train a model.
    """)
    return


@app.cell
def _():
    import inspect
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    import torch.nn.functional as F

    lesson_directory = Path(mo.notebook_location())
    if str(lesson_directory) not in sys.path:
        sys.path.insert(0, str(lesson_directory))
    import image_models as core

    return F, core, inspect, mo, nn, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Keep the spatial grid

    An image classifier eventually reduces an image to class scores. Here we need an
    answer at each output location, so we keep spatial feature maps throughout the
    network. PyTorch stores each batch as **N × C × H × W**: batch, channels, height,
    width. RGB gives us three input channels; learned layers need not represent colours.

    A 3 × 3 convolution combines local information. Stacking convolutions lets a feature
    depend on more of the input. Pooling reduces the spatial resolution so deeper
    features can use a wider context at lower memory cost. It also loses fine detail.

    The encoder reduces resolution while increasing channels. The decoder increases
    resolution again. The skip connections carry features from the encoder to the
    matching decoder stage, which helps preserve the positions of edges and small
    structures. These features still have to be learned: an untrained U-Net does not
    produce a useful restored image or segmentation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(r"""
    flowchart TB
        input["RGB input<br/>N × 3 × 128 × 128"]
        e1["Encoder 1 · two 3 × 3 convolutions<br/>N × 32 × 128 × 128"]
        e2["Encoder 2 · two 3 × 3 convolutions<br/>N × 64 × 64 × 64"]
        e3["Encoder 3 · two 3 × 3 convolutions<br/>N × 128 × 32 × 32"]
        b["Bottleneck · two 3 × 3 convolutions<br/>N × 256 × 16 × 16"]
        d3["Decoder 3<br/>Concatenate: 128 + 128 = 256 channels<br/>Two 3 × 3 convolutions → N × 128 × 32 × 32"]
        d2["Decoder 2<br/>Concatenate: 64 + 64 = 128 channels<br/>Two 3 × 3 convolutions → N × 64 × 64 × 64"]
        d1["Decoder 1<br/>Concatenate: 32 + 32 = 64 channels<br/>Two 3 × 3 convolutions → N × 32 × 128 × 128"]
        output["1 × 1 convolution · output head<br/>N × C_out × 128 × 128<br/>An answer at every pixel"]

        input --> e1
        e1 -->|"2 × 2 max pool: halve H and W"| e2
        e2 -->|"2 × 2 max pool"| e3
        e3 -->|"2 × 2 max pool"| b
        b -->|"Upsample 2× · 256 → 128 channels"| d3
        d3 -->|"Upsample 2× · 128 → 64 channels"| d2
        d2 -->|"Upsample 2× · 64 → 32 channels"| d1
        d1 --> output
        e3 -. "Skip: same 32 × 32 grid" .-> d3
        e2 -. "Skip: same 64 × 64 grid" .-> d2
        e1 -. "Skip: same 128 × 128 grid" .-> d1

        classDef encoder fill:#dbeafe,stroke:#2563eb,color:#172554
        classDef decoder fill:#dcfce7,stroke:#15803d,color:#14532d
        classDef bridge fill:#fef3c7,stroke:#b45309,color:#78350f
        classDef endpoint fill:#f1f5f9,stroke:#475569,color:#0f172a
        class e1,e2,e3 encoder
        class d3,d2,d1 decoder
        class b bridge
        class input,output endpoint
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read each shape as **batch × channels × height × width**. The batch size `N`
    stays the same. Blue blocks reduce the spatial grid through pooling, while
    green blocks restore it through upsampling. Each convolution uses padding to
    keep its block's spatial size unchanged.

    The dotted arrows carry encoder features directly to a decoder at the same
    resolution. We concatenate along the channel axis, then use convolutions to
    combine the features. Channels such as 32 or 128 are learned features, not RGB
    colours. `C_out` is three for RGB/noise predictions or one for segmentation logits.

    This shows the core U-Net. For 2× super resolution, we first enlarge the
    64 × 64 input to 128 × 128 before entering this network. All of the convolution
    weights must be learned; the connections alone do not give useful predictions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Four resolutions and three skip connections

    Here, “four levels” means four spatial resolutions, including the bottleneck:

    | Stage | Channels | Size for a 128 × 128 patch |
    | --- | --- | --- |
    | Encoder 1 | 32 | 128 × 128 |
    | Encoder 2 | 64 | 64 × 64 |
    | Encoder 3 | 128 | 32 × 32 |
    | Bottleneck | 256 | 16 × 16 |
    | Decoder 3 | 128 | 32 × 32 |
    | Decoder 2 | 64 | 64 × 64 |
    | Decoder 1 | 32 | 128 × 128 |

    Pooling gives the deeper layers a larger view of the image, but loses fine spatial
    detail. We keep each encoder feature map and concatenate it with the decoder at
    the matching resolution. At the first decoder stage, 128 upsampled channels plus
    128 skip channels give 256 input channels to `double_conv`.

    The implementation below is the code we train. It lives in `image_models.py` so
    we can test it without downloading data or starting the notebook. There are no
    pre-trained layers. Padding to a multiple of eight lets us accept odd image sizes;
    we crop that padding off the final output.
    """)
    return


@app.cell
def _(core, inspect, mo):
    mo.md(
        "```python\n"
        + inspect.getsource(core.double_conv)
        + "\n"
        + inspect.getsource(core.UNet)
        + "\n```"
    )
    return


@app.cell
def _(core, mo, torch):
    with torch.random.fork_rng():
        torch.manual_seed(42)
        _rows = []
        for _mode in ("transpose", "resize"):
            _model = core.UNet(upsampling=_mode)
            with torch.no_grad():
                _output = _model(torch.zeros(1, 3, 128, 128))
            _rows.append(
                {
                    "Upsampling": _mode,
                    "Parameters": sum(p.numel() for p in _model.parameters()),
                    "Input": "1 × 3 × 128 × 128",
                    "Output": str(tuple(_output.shape)),
                }
            )
    mo.ui.table(_rows, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Concatenate along channels

    The batch, height and width must agree. Try concatenating an encoder feature map
    before upsampling the decoder: PyTorch catches the mismatched resolution. The
    second calculation shows the corrected shape.
    """)
    return


@app.cell
def _(F, torch):
    _encoder = torch.zeros(1, 128, 32, 32)
    _decoder = torch.zeros(1, 128, 16, 16)
    try:
        torch.cat((_encoder, _decoder), dim=1)
    except RuntimeError as error:
        print("Before upsampling:", error)
    _decoder = F.interpolate(_decoder, scale_factor=2, mode="nearest")
    print(
        "After upsampling and concatenation:",
        tuple(torch.cat((_encoder, _decoder), dim=1).shape),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why do checkerboards appear?

    A [transposed convolution](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)
    places a learned kernel around each input position. With a kernel of three and a
    stride of two, some output positions receive more contributions than others.
    [Odena et al.](https://distill.pub/2016/deconv-checkerboard/) explain how this uneven
    overlap produces a checkerboard pattern.

    We can see the overlap without training: use an input of ones and set every kernel
    weight to one. Compare it with nearest-neighbour `Upsample` followed by `Conv2d`.
    The colour limits are shared, and we crop the outer border to inspect the interior.
    Our U-Net uses kernel two, stride two, which avoids this particular uneven-overlap
    case. Even overlap does not guarantee that learned filters will avoid artefacts.
    """)
    return


@app.cell
def _(nn, plt, torch):
    _source = torch.ones(1, 1, 12, 12)
    _transpose = nn.ConvTranspose2d(
        1, 1, 3, stride=2, padding=1, output_padding=1, bias=False
    )
    _resize = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(1, 1, 3, padding=1, bias=False),
    )
    _even = nn.ConvTranspose2d(1, 1, 2, stride=2, bias=False)
    with torch.no_grad():
        _transpose.weight.fill_(1)
        _resize[1].weight.fill_(1)
        _even.weight.fill_(1)
        _panels = [
            _transpose(_source)[0, 0, 3:-3, 3:-3],
            _resize(_source)[0, 0, 3:-3, 3:-3],
            _even(_source)[0, 0, 3:-3, 3:-3],
        ]
    _figure, _axes = plt.subplots(1, 3, figsize=(11, 3), layout="constrained")
    for _axis, _pixels, _title in zip(
        _axes,
        _panels,
        (
            "Transpose: kernel 3, stride 2",
            "Nearest + 3 × 3 convolution",
            "Transpose: kernel 2, stride 2",
        ),
    ):
        _plot = _axis.imshow(
            _pixels.numpy(),
            vmin=0,
            vmax=9,
            cmap="viridis",
            interpolation="nearest",
        )
        _axis.set_title(_title, fontsize=10)
        _axis.set_xlabel("Output column")
        _axis.set_ylabel("Output row")
    _figure.colorbar(_plot, ax=_axes, label="Contributions to each output pixel")
    plt.close(_figure)
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adapting U-Net to Different Tasks

    The U-Net produces a grid of numbers. What those numbers represent depends on what we train it to predict.

    The decoder finishes with 32 features at each pixel. The final **1 × 1 convolution** combines those features into the required number of outputs without changing the image’s height or width:

    - Three outputs per pixel for RGB noise or RGB colour.
    - One output per pixel for pet segmentation.

    It performs a learned weighted combination of the features at each location. The training targets, loss and inference calculation establish what those outputs mean.

    ### Denoising: predict what to remove

    We add known noise to a clean image, then train the network to predict that noise:

    ```python
    noisy = clean + noise
    predicted_noise = model(noisy)
    loss = mse(predicted_noise, noise)
    restored = noisy - predicted_noise
    ```

    For example, if a noisy pixel is `0.7` and the predicted noise is `0.1`, the restored pixel is `0.6`. Noise can be negative, so we allow unrestricted output values.

    ### Super resolution: predict the larger image

    Here, the target is the clean, high-resolution image itself:

    ```python
    enlarged = interpolate(
        low_resolution,
        scale_factor=2,
        mode="bicubic",
        align_corners=False,
    )
    prediction = unet(enlarged)
    loss = mse(prediction, high_resolution)
    ```

    The wrapper enlarges the 64 × 64 input to 128 × 128 before the U-Net processes it. The U-Net preserves that enlarged size and learns to reconstruct the target’s RGB values.

    ESPCN takes a different route: it processes features at low resolution, then uses `PixelShuffle` to rearrange channels into a larger spatial grid.

    ### Segmentation: predict whether each pixel belongs to the pet

    We keep the input’s spatial size but output one number per pixel:

    ```python
    logits = model(image)
    loss = segmentation_loss(logits, mask, valid)
    probabilities = logits.sigmoid()
    prediction = probabilities >= 0.5
    ```

    A **logit** is an unrestricted score. Sigmoid converts it into a probability between zero and one. Thresholding produces the binary mask.

    `BCEWithLogitsLoss` takes the raw logits; the Dice part uses probabilities to measure foreground overlap. The validity mask excludes uncertain boundary pixels from both parts of the loss.

    ### Comparing the tasks

    | Example | Input patch | Network output | Training loss | Final result |
    | --- | --- | --- | --- | --- |
    | Denoising | RGB 128 × 128 | Three noise channels, 128 × 128 | Noise MSE | Input minus predicted noise |
    | 2× super resolution | RGB 64 × 64 | Three RGB channels, 128 × 128 | Image MSE | Predicted RGB image |
    | Pet segmentation | RGB 128 × 128 | One logit channel, 128 × 128 | BCE plus Dice | Sigmoid, then threshold |

    ### Skip connections and residual subtraction

    The encoder **skip connections** carry learned features into the decoder and concatenate them along the channel dimension. They operate inside the U-Net.

    The subtraction `noisy - predicted_noise` operates after the U-Net. It turns a noise estimate into a restored image.

    Keeping the skip connections does not, by itself, make a network predict noise. The training target establishes that meaning.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train on patches, split by photograph

    Activations often use more accelerator memory than the model weights. We train
    on random 128 × 128 target patches so a batch fits in memory and each photograph
    can supply different views. Super resolution derives a 64 × 64 input from its
    target crop; segmentation crops the image and mask at the same coordinates.

    Split the photographs into training, validation and test sets **before** taking
    patches. Two crops from the same photograph are related; putting one in training
    and the other in validation can give a misleading score.

    The examples split Oxford-IIIT Pet's official `trainval` set with seed 42 and keep
    its official `test` set for evaluation. Training samples fresh crops. Validation
    uses fixed crops (and fixed noise for denoising), so changes in its loss reflect
    the model rather than a different sample. `num_workers=0` keeps the random patch
    generator in one process. Small images are padded; padded segmentation labels
    are ignored.

    Below, the rectangle and extracted patch use the same coordinates. Moving the
    crop should change which features we see without rescaling the photograph.
    """)
    return


@app.cell
def _(plt, torch):
    from matplotlib.patches import Rectangle

    _y, _x = torch.meshgrid(
        torch.linspace(0, 1, 192), torch.linspace(0, 1, 256), indexing="ij"
    )
    _image = torch.stack((_x, _y, (_x > 0.5).float()))
    _generator = torch.Generator().manual_seed(42)
    _top = int(torch.randint(192 - 128 + 1, (), generator=_generator))
    _left = int(torch.randint(256 - 128 + 1, (), generator=_generator))
    _patch = _image[:, _top : _top + 128, _left : _left + 128]
    _figure, _axes = plt.subplots(1, 2, figsize=(9, 3), layout="constrained")
    _axes[0].imshow(_image.permute(1, 2, 0).numpy())
    _axes[0].add_patch(
        Rectangle(
            (_left - 0.5, _top - 0.5),
            128,
            128,
            fill=False,
            edgecolor="yellow",
            linewidth=2,
        )
    )
    _axes[0].set_title("Photograph before cropping")
    _axes[1].imshow(_patch.permute(1, 2, 0).numpy())
    _axes[1].set_title("128 × 128 training patch")
    for _axis in _axes:
        _axis.set_xlabel("Column")
        _axis.set_ylabel("Row")
    plt.close(_figure)
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The shared training workflow

    The example notebooks share the `data` directory and use
    [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/). Its photographs provide
    clean RGB references, and its trimaps provide the segmentation labels. The download
    is roughly 800 MB of archives and needs extra space for extraction. Point the data
    form at an existing copy if one is available.

    `Utils.get_device()` selects CUDA when available, otherwise MPS, otherwise CPU.
    Each example prints the detected device. Opening a notebook does not start a
    download or training: submit the dataset form, then the training form.

    Each run creates fresh weights and an Adam optimiser with seed 42. Within a batch
    we clear gradients, predict, calculate the task's loss, backpropagate and update
    the weights. After each epoch we evaluate fixed validation patches without
    recording gradients and retain the weights with the lowest validation loss.
    The test photographs do not take part in this choice.

    The short defaults are a workflow check. Increase the training budget before
    judging quality. Save the selected weights with the configuration, task, split
    seed and metric protocol so that we can interpret them later. The save control
    also writes the per-image scores; timestamped filenames keep runs separate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Full images: tiles with overlap

    A photograph can fit in CPU memory while its intermediate feature maps exhaust
    GPU memory. Keep the full image and output buffers on the CPU and send one tile
    at a time to the model. We include a final tile at each far edge, and pad images
    smaller than a tile, so no pixels are missed.

    Adjacent predictions overlap. A positive blending window gives less weight to a
    tile's edges; dividing the accumulated values by the accumulated weights restores
    the correct scale. The window never reaches zero, including at the frame boundary.

    Tile size and overlap are measured in **input** pixels. `scale=1` preserves spatial
    size; `scale=2` doubles it. Each task lesson explains what is blended: noise, RGB
    values or logits. The implementation is shared below.

    Tiling is an approximation for a convolutional model because its edge pixels see
    different context. Blending reduces visible seams but cannot recover context the
    network never received. CPU memory is still required for the complete frame and
    output buffers.
    """)
    return


@app.cell
def _(core, inspect, mo):
    mo.md("```python\n" + inspect.getsource(core.tiled_predict) + "\n```")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check the coverage before using a learned model

    An identity model returns its input unchanged. It has no context-dependent edge
    effects, so tiling and blending should reproduce every pixel, up to rounding.
    Using an odd image size checks that the final tiles cover both far edges. This
    checks stitching; it does not imply that a U-Net gives identical tiled and
    full-frame predictions.
    """)
    return


@app.cell
def _(core, nn, torch):
    _image = torch.rand(3, 137, 181, generator=torch.Generator().manual_seed(42))
    _reconstructed = core.tiled_predict(nn.Identity(), _image, tile=64, overlap=16)
    intro_tile_error = (_reconstructed - _image).abs().max().item()
    print("Input shape:", tuple(_image.shape))
    print("Stitched shape:", tuple(_reconstructed.shape))
    print("Largest stitching difference:", intro_tile_error)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluate the final result

    Compare methods on the same held-out photographs and the same degraded inputs.
    Choose settings on validation data first. Calculate each photograph's score, then
    report the mean and sample standard deviation across photographs. A few images
    are useful for checking the pipeline, not for establishing generalisation.

    The restoration lessons follow the existing [evaluation notebook](../EvaluationOfResults/BaselinesEvaluationMarimo.py):
    clip RGB predictions to [0, 1], remove two pixels from every border, and report MSE
    and PSNR. PSNR is a logarithmic transformation of MSE, so they are not independent
    votes for quality. Segmentation uses Dice and IoU with uncertain labels excluded.
    The example notebooks explain the relevant baselines and scoring details.

    Inspect the images as well as the numbers. Also consider parameter count and time.
    For timing, the notebooks warm each neural model once and synchronise the device.
    Their measurements include tile transfers and CPU stitching but exclude loading
    photographs from disk. These are end-to-end inference measurements, not isolated
    kernel benchmarks.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Trace the channel counts at the first decoder stage. If we add the skip features instead of concatenating them, which convolution must change?
    2. Explain why the kernel-three, stride-two transposed convolution has uneven overlap. Why does kernel two remove this particular pattern but not guarantee artefact-free learned outputs?
    3. Inspect a trained U-Net with `use_skips=False`, then train a model without skips. Why are these different experiments?
    4. Why must we split photographs before sampling patches? What would happen if two nearby crops reached different splits?
    5. Change the image size, tile size and overlap in the identity check. Predict the output shape. Then compare tiled and full-frame inference for a trained U-Net on a small image.

    Continue with [denoising](DenoisingMarimo.py), [2× super resolution](SuperResolutionMarimo.py)
    or [pet segmentation](PetSegmentationMarimo.py). Each example runs independently.
    """)
    return


if __name__ == "__main__":
    app.run()

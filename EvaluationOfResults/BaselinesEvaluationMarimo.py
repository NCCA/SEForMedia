#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Evaluaiton of image results

    In this demo I use 2× image upscaling because we can see the mistakes as well as measure them.

    We will compare four ways of enlarging an image: nearest-neighbour, bilinear, bicubic, and bicubic followed by sharpening with an unsharp mask. None requires training or a nural network but can be used to discuss results as if they were from a network.

    We evaluate each method on all 24 test images, then inspect the eight images with the largest reconstruction errors for the selected method. This lets us compare overall performance and see where the methods struggle (which is the "honest baseline evaluation" mentioned in the assignment ideas.)

    The question I want to answer is: does the extra processing actually improve the
    result enough to justify using it? A baseline gives that question a concrete reference.
    If a complicated method cannot beat a cheap resize, that is useful information. It can
    save us spending weeks improving a system whose main advantage is its complexity.

    There are three separate judgements here: how closely we reproduce the reference pixels,
    whether we prefer the appearance, and whether the difference matters for the intended use.
    We should expect those judgements to disagree sometimes. The purpose of this notebook is
    to make that disagreement visible and explain it, rather than find one impressive number.

    The test set is the 24-image [Kodak suite](https://r0k.us/graphics/kodak/).
    The first run downloads about 16 MB into `data/kodak` in this notebook folder.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    from urllib.request import urlopen
    import hashlib
    import json
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image
    import torch
    import torch.nn.functional as F

    return F, Image, Path, hashlib, json, mo, np, pd, plt, torch, urlopen


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment outline

    Before running the comparison, we choose the test images, method settings and scoring rules. Every method receives the same low-resolution input and is compared with the same reference image.

    We keep these choices consistent so that differences in the results come from the methods being tested. If we use the results to tune a method, we should evaluate that revised method on a separate test set.

    We use all 24 images at their original resolution. Inputs are RGB floats in [0, 1].
    We synthesise the low-resolution input using antialiased bicubic downsampling by 2,
    with `align_corners=False`, then clip to [0, 1]. Every method receives that same input. Note that real low-resolution images may also contain camera blur, noise or JPEG compression artefacts, which this experiment does not simulate.


    Halving width and height leaves a quarter of the original pixel locations. Antialiasing smooths away fine detail before we sample the smaller image, reducing misleading patterns caused by undersampling. Upscaling restores the dimensions; it does not undo that loss of information. Several different high-resolution images could produce a similar small
    image, so there is no unique answer waiting to be recovered by interpolation.

    We start from a known large image so that we have a reference to compare against. This makes the experiment repeatable, but also defines its limits: we are testing how well
    methods reverse *our* reduction process. A real small photograph may have unknown blur, noise, sharpening and compression. A result on this notebook is evidence for this task, not an example of what super resolution may actually produce, it does, however, give you an idea of what to look for.

    Nearest, bilinear and bicubic use PyTorch's
    [`interpolate`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html).
    The fourth method adds 0.5 times the difference between bicubic and a 3×3 box blur,
    using replicated padding. We use a sharpening strength of 0.5 for every image. If we adjust this setting after inspecting the Kodak results, we should test our final choice on a separate set of images.

    We also limit the output pixel values to [0, 1], the valid intensity range used here. Any value below 0 is set to 0, and any value above 1 is set to 1

    | Method | What it does | What I would look for |
    | --- | --- | --- |
    | Nearest | Copies the nearest input sample into each output location. | Blocks and stepped diagonals; potentially desirable for pixel art. |
    | Bilinear | Blends the four surrounding samples using distance weights. | Smoother edges, but soft texture and thin lines. |
    | Bicubic | Uses a wider neighbourhood with cubic weights. | Better edge definition, with possible ringing near strong transitions. |
    | Bicubic + unsharp | Boosts the difference between the resized image and a blurred copy. | More apparent sharpness, but also halos and emphasis of unwanted texture. |

    The unsharp mask is deliberately simple. The blurred copy contains slowly changing
    structure, subtracting it leaves local variation, which we add back with a fixed weight.
    It can make an edge look crisper without recovering a single missing piece of texture.
    I would not describe that as recovered detail unless the reference supports the claim.

    In the code, an image changes from `(height, width, channels)` to
    `(batch, channels, height, width)` because that is the layout `interpolate` expects.
    The batch has one image. `inference_mode()` avoids recording gradients; using PyTorch
    for these array operations does not make this a trained model. We convert the output
    back to the image layout for plotting and scoring.

    We remove two high-resolution pixels at each border for every metric. MSE averages
    squared error over the remaining RGB pixels (lower is better). PSNR is
    $10\log_{10}(1/\mathrm{MSE})$ in dB (higher is better). These are gamma-encoded RGB
    scores, not luminance-only benchmark scores. PSNR and MSE measure the same errors in
    different ways; neither tells us whether an image looks convincing.

    Squaring the error gives large pixel differences more influence than small ones.
    A slight displacement of a sharp edge can therefore receive a substantial penalty,
    even when a viewer still recognises a clean edge. Conversely, a smooth result can avoid
    some large errors whilst looking disappointingly soft. MSE has no concept of a face,
    readable lettering or the particular detail that attracted our attention.

    For one image, a 3.01 dB PSNR gain corresponds to roughly halving MSE. It does **not**
    mean the image looks twice as good. Also, MSE and PSNR are not two independent votes:
    PSNR is a transformation of MSE. Their dataset averages can order methods differently
    because we apply that transformation before averaging.

    The common border crop reduces the influence of boundary handling. It is a stated
    convention, not permission to hide an inconvenient region. Clipping matters too:
    bicubic interpolation and sharpening can produce values outside the display range.
    We score the clipped images that we show, so the numbers and pictures refer to the
    same result. Changing the crop, colour space or clipping rule changes the experiment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MSE: work through four pixels

    MSE stands for *Mean Squared Error*. We compare corresponding values in the reference
    and reconstructed image, square their differences, then take the mean. Lower is better.
    For a simple grey scale image we would get the following where, 0 is black and 1 is white.

    ```text
    Reference               Reconstructed image
    ┌──────┬──────┐          ┌──────┬──────┐
    │ 0.20 │ 0.40 │          │ 0.30 │ 0.40 │
    ├──────┼──────┤          ├──────┼──────┤
    │ 0.60 │ 0.80 │          │ 0.50 │ 1.00 │
    └──────┴──────┘          └──────┴──────┘

    Result − reference      Squared differences
    ┌──────┬──────┐          ┌──────┬──────┐
    │ +0.10│  0.00│          │ 0.01 │ 0.00 │
    ├──────┼──────┤    →     ├──────┼──────┤
    │ −0.10│ +0.20│          │ 0.01 │ 0.04 │
    └──────┴──────┘          └──────┴──────┘
    ```

    $$\mathrm{MSE}=\frac{0.01+0+0.01+0.04}{4}=0.015$$

    More generally,

    $$\mathrm{MSE}=\frac{1}{N}\sum_{i=1}^{N}(I_i-\hat I_i)^2$$

    $I_i$ is a reference value, $\hat I_i$ is its reconstructed value, and $N$ counts
    the values compared. For an RGB image we include all three channels, so after our
    border crop $N=(H-4)(W-4)\times3$. The four-pixel example has no border crop.

    Squaring stops positive and negative differences cancelling. It also gives large errors
    more influence: a difference of 0.2 contributes four times as much as a difference of
    0.1. Identical images have MSE zero. The square root of MSE, called RMSE, would express
    the error on the original intensity scale; MSE itself is in squared intensity units.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PSNR: express the same error on a logarithmic scale

    PSNR stands for *Peak Signal-to-Noise Ratio*. Here, noise means reconstruction error:
    blur, blocks and displaced edges count as well as grain. Higher PSNR is better.

    $$\mathrm{PSNR}=10\log_{10}\left(\frac{L^2}{\mathrm{MSE}}\right)$$

    $L$ is the maximum permitted pixel value, not the brightest pixel we happen to find
    in a photograph. Use 1 for our normalised images and 255 for 8-bit values in [0, 255].
    For the four-pixel example:

    $$\mathrm{PSNR}=10\log_{10}(1/0.015)\approx18.24\ \mathrm{dB}$$

    | MSE for values in [0, 1] | PSNR |
    | ---: | ---: |
    | 0.1 | 10 dB |
    | 0.01 | 20 dB |
    | 0.001 | 30 dB |
    | 0.0001 | 40 dB |
    | 0 | $+\infty$ |

    Dividing MSE by ten adds 10 dB. Halving it adds about 3.01 dB. Neither change means
    the image looks a corresponding number of times better: decibels describe an error
    ratio, not a scale of visual satisfaction. There is no universal PSNR threshold for
    an acceptable image; the task and image content matter.

    If we multiply both images by 255, MSE grows by $255^2$. Using $L=255$ cancels that
    factor, so PSNR stays the same. Mixing normalised MSE with $L=255$ gives an incorrect
    score. Pixel alignment, colour space and value range must agree before comparing images.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid("""
    flowchart TD
        A[Reference and reconstructed image] --> B[Subtract corresponding values]
        B --> C[Square each difference]
        C --> D[Average: MSE — lower is better]
        D --> E[Divide peak value squared by MSE]
        E --> F[Take log base 10 and multiply by 10]
        F --> G[PSNR in dB : higher is better]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A sharper edge can get a worse score

    Consider six samples across a black-to-white edge. One reconstruction keeps the edge
    sharp but shifts it one pixel. Another keeps its position but blurs the transition.
    These are schematic signals, not outputs from our four upscaling methods.

    ```text
    Reference:       0   0   0   1   1   1
    Shifted edge:    0   0   0   0   1   1
    Blurred edge:    0   0  0.3 0.7  1   1
    ```

    The shifted edge gets one sample completely wrong:

    $$\mathrm{MSE}_{\mathrm{shift}}=1/6\approx0.167,
    \qquad\mathrm{PSNR}_{\mathrm{shift}}\approx7.78\ \mathrm{dB}$$

    The blurred edge gets two samples slightly wrong:

    $$\mathrm{MSE}_{\mathrm{blur}}=(0.3^2+0.3^2)/6=0.03,
    \qquad\mathrm{PSNR}_{\mathrm{blur}}\approx15.23\ \mathrm{dB}$$

    The score strongly prefers the blurred edge. I might prefer the crisp appearance in
    some contexts, but I would still need to acknowledge that its position is wrong. If
    the task requires accurate geometry, that displacement may be the more serious defect.
    The metric is doing exactly what we asked: measuring agreement at matching locations.
    Our visual judgement may be asking a different question.

    In a photograph, MSE has no concept of a face, readable lettering or the particular
    detail that catches our attention. It measures pixel fidelity. I use the gallery to
    understand the character of the errors, then decide whether they matter for the task.
    A subjective preference is useful evidence when labelled as such; it is not a substitute
    for reporting the measured error.

    As you read the results below, keep three questions separate: how much error is there,
    how stable is the mean difference across resampled images, and what do the failures
    look like? The metric table, paired bootstrap and failure gallery answer those questions
    respectively. We need all three to make a useful judgement.
    """)
    return


@app.cell
def _(F, np, torch):
    def upscale(low: torch.Tensor, method: str) -> np.ndarray:
        mode = "bicubic" if method == "bicubic + unsharp" else method
        options = {} if mode == "nearest" else {"align_corners": False}
        result = F.interpolate(low, scale_factor=2, mode=mode, **options)
        if method == "bicubic + unsharp":
            blurred = F.avg_pool2d(
                F.pad(result, (1, 1, 1, 1), mode="replicate"), 3, stride=1
            )
            result = result + 0.5 * (result - blurred)
        return result.clamp(0, 1)[0].permute(1, 2, 0).numpy()

    def metrics(reference: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
        difference = reference[2:-2, 2:-2].astype(np.float64) - prediction[2:-2, 2:-2]
        mse = float(np.mean(difference**2))
        psnr = float("inf") if mse == 0 else float(10 * np.log10(1 / mse))
        return mse, psnr

    methods = ("nearest", "bilinear", "bicubic", "bicubic + unsharp")
    return methods, metrics, upscale


@app.cell
def _(Image, Path, hashlib, mo, np, urlopen):
    data_dir = Path(mo.notebook_location()) / "data" / "kodak"
    data_dir.mkdir(parents=True, exist_ok=True)
    references = {}
    manifest = []
    for _index in range(1, 25):
        _name = f"kodim{_index:02d}.png"
        _path = data_dir / _name
        _url = f"https://r0k.us/graphics/kodak/kodak/{_name}"
        if not _path.exists():
            with urlopen(_url, timeout=60) as _response:
                _payload = _response.read()
            _temporary = _path.with_suffix(".part")
            _temporary.write_bytes(_payload)
            with Image.open(_temporary) as _check:
                _check.verify()
            _temporary.replace(_path)
        with Image.open(_path) as _image:
            if _image.size not in ((768, 512), (512, 768)):
                raise ValueError(f"Unexpected Kodak dimensions: {_name}: {_image.size}")
            references[_name] = (
                np.asarray(_image.convert("RGB"), dtype=np.float32) / 255
            )
        manifest.append(
            {
                "image": _name,
                "url": _url,
                "sha256": hashlib.sha256(_path.read_bytes()).hexdigest(),
            }
        )
    mo.md(f"Loaded **{len(references)} / 24** images. No images excluded.")
    return manifest, references


@app.cell
def _(F, methods, metrics, pd, references, torch, upscale):
    predictions = {}
    _rows = []
    with torch.inference_mode():
        for _name, _reference in references.items():
            _tensor = torch.from_numpy(_reference).permute(2, 0, 1).unsqueeze(0)
            _low = F.interpolate(
                _tensor,
                scale_factor=0.5,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).clamp(0, 1)
            for _method in methods:
                _prediction = upscale(_low, _method)
                predictions[(_name, _method)] = _prediction
                _mse, _psnr = metrics(_reference, _prediction)
                _rows.append(
                    {
                        "image": _name,
                        "method": _method,
                        "MSE": _mse,
                        "PSNR": _psnr,
                    }
                )
    scores = pd.DataFrame(_rows)
    summary = scores.groupby("method", sort=False)[["MSE", "PSNR"]].agg(["mean", "std"])
    summary
    return predictions, scores, summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Look at individual images as well as the average

    We calculate MSE and PSNR separately for each image, then summarise the 24 results for each method. The table shows the mean and sample standard deviation:

    - The **mean** is the average score across the images. Each image contributes equally.
    - The **standard deviation** describes how much the scores vary between images. A larger value means the scores are more spread out.

    For example, “30 ± 3 dB” means a mean PSNR of 30 dB and a standard deviation of 3 dB. It does not mean every image scores between 27 and 33 dB, and it is not a confidence interval for the mean. The `ddof=1` setting calculates the sample standard deviation by dividing the sum of squared deviations by \(n-1\).

    We average the individual PSNR scores directly. Because PSNR uses a logarithm, this gives a different result from averaging the MSE values first and converting that average to PSNR.

    ### Reading the scatter plots

    Each point is one image evaluated by one method. At a particular image number, compare the methods: lower MSE or higher PSNR means closer agreement with the reference. Then follow one method across the images to see which photographs it handles well and which it struggles with.

    An average can hide a serious failure. A method might improve most images slightly but make one much worse. I would want to inspect that image before deciding whether the method is useful.

    Also look for images that are difficult for every method. Fine textures or thin lines may have been lost during downsampling, leaving all four methods with the same underlying problem.

    ### Compare methods on the same images

    Suppose A scores 25 dB on one image and 35 dB on another, whilst B scores 24 dB and 34 dB. Both methods have a wide spread of scores, but A is consistently 1 dB ahead.

    This is why we compare A and B on each matching image. The next section uses these paired differences to investigate how consistent the average improvement is.
    """)
    return


@app.cell
def _(methods, plt, scores):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for _method in methods:
        _part = scores[scores.method == _method]
        for _axis, _metric in zip(_axes, ("MSE", "PSNR")):
            _axis.scatter(range(1, 25), _part[_metric], label=_method, s=22)
            _axis.set(xlabel="Kodak image number", ylabel=_metric)
    _axes[0].legend(fontsize=8)
    _fig
    return


@app.cell
def _(methods, mo):
    method_a = mo.ui.dropdown(methods, value="bicubic + unsharp", label="Method A")
    method_b = mo.ui.dropdown(methods, value="bicubic", label="Method B / baseline")
    mo.hstack([method_a, method_b])
    return method_a, method_b


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Is A's average PSNR better than B's?

    First, we compare the two methods on each image by subtracting B’s PSNR from A’s:

    | Image | A PSNR | B PSNR | Difference: A − B |
    | --- | ---: | ---: | ---: |
    | 1 | 30 dB | 29 dB | +1 dB |
    | 2 | 27 dB | 28 dB | −1 dB |
    | 3 | 35 dB | 33 dB | +2 dB |

    Positive differences favour A; negative differences favour B. In this small example, A’s mean advantage is approximately 0.67 dB, but it does not win on every image.

    Our notebook makes the same calculation for all 24 Kodak images. We then ask how much that average changes when we change the mix of images.

    ### Resampling the images

    We use a technique called a *paired bootstrap*:

    1. Randomly select 24 image indices **with replacement**. An image can be selected more than once, whilst another may not be selected at all.
    2. Take the A − B differences for those selected images.
    3. Calculate their mean.
    4. Repeat this process 10,000 times.

    For the three-image example, one resample might select images `[1, 1, 3]`. Its differences are `[+1, +1, +2]`, giving a mean of 1.33 dB. Another might select `[2, 2, 1]`, giving differences `[−1, −1, +1]` and a mean of −0.33 dB.

    These different averages show how the result depends on the images selected. We use the same NumPy random generator and array indexing introduced in NumPyForML.

    The comparison stays *paired*: whenever we select an image, we use both methods’ scores for that image. We do not compare A on one photograph with B on a different photograph. We also sample whole images rather than individual pixels, because nearby pixels are related and should not be treated as independent test examples.

    ### Reading the confidence interval

    We sort the 10,000 bootstrap means and take the 2.5th and 97.5th percentiles. These mark the central 95% of the resampled means and give an approximate **95% bootstrap confidence interval for the mean PSNR difference**.

    For example:

    | Example interval | Interpretation |
    | --- | --- |
    | +0.2 to +0.8 dB | Supports a positive mean advantage for A. |
    | −0.3 to +0.5 dB | The interval includes zero, so the direction of the mean advantage is uncertain. |
    | −0.8 to −0.2 dB | Supports a positive mean advantage for B. |

    An interval containing zero does not prove the methods are equal. An interval above zero does not mean A wins on every image, or that it has a 95% chance of winning on the next photograph.

    ### Reading the plots

    In the left plot, each point compares the two methods on the same image. The diagonal represents equal PSNR. Points above it favour A; points below it favour B.

    The right plot shows the 10,000 bootstrap means. A narrow distribution means the average difference changes little when we resample these images. A wider distribution means it is more sensitive to the mix of images.

    ### Does the improvement matter?

    I would still look at the images before choosing a method. A small, consistent PSNR improvement might be difficult to see at the intended display size. It might also come with sharpening artefacts that I find distracting. The confidence interval does not measure visual preference or processing cost.

    The bootstrap reuses the photographs we already have; it does not create new evidence from new photographs. Kodak is a small, curated collection, so this interval cannot establish performance on every kind of image. Increasing the number of resamples makes the calculation more stable, but does not make the dataset more representative. The fixed seed lets us repeat the calculation.

    If we use these results to choose settings or select a favourable comparison, we should check that choice on a fresh test set.
    """)
    return


@app.cell
def _(np):
    def paired_bootstrap(a: np.ndarray, b: np.ndarray, seed: int = 42) -> tuple:
        difference = np.asarray(a) - np.asarray(b)
        if (
            difference.ndim != 1
            or difference.size < 2
            or not np.isfinite(difference).all()
        ):
            raise ValueError("Use at least two finite, paired image scores")
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, difference.size, size=(10_000, difference.size))
        means = difference[indices].mean(axis=1)
        low, high = np.quantile(means, [0.025, 0.975])
        return float(difference.mean()), float(low), float(high), means

    return (paired_bootstrap,)


@app.cell
def _(method_a, method_b, mo, paired_bootstrap, plt, scores):
    paired = scores.pivot(index="image", columns="method", values="PSNR")
    gain, ci_low, ci_high, bootstrap_means = paired_bootstrap(
        paired[method_a.value].to_numpy(), paired[method_b.value].to_numpy()
    )
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    _axes[0].scatter(paired[method_b.value], paired[method_a.value])
    _limits = (paired.min().min() - 1, paired.max().max() + 1)
    _axes[0].plot(_limits, _limits, "k--")
    _axes[0].set(
        xlabel=f"{method_b.value} PSNR (dB)",
        ylabel=f"{method_a.value} PSNR (dB)",
    )
    _axes[1].hist(bootstrap_means, bins=40)
    _axes[1].axvline(0, color="black", linestyle="--")
    _axes[1].axvspan(ci_low, ci_high, alpha=0.2, color="orange")
    _axes[1].set(xlabel="Mean paired PSNR gain A − B (dB)", ylabel="Bootstrap samples")
    mo.vstack(
        [
            mo.md(
                f"Mean gain: **{gain:.3f} dB**, 95% percentile interval **[{ci_low:.3f}, {ci_high:.3f}] dB**."
            ),
            _fig,
        ]
    )
    return ci_high, ci_low, gain


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Failure gallery: inspect the eight largest errors

    We sort the 24 images by method A’s MSE, from highest to lowest, and display the first eight. These are the images where A differs most from the reference according to this metric.

    The ranking uses all pixels remaining after removing the two-pixel border. It does not use only the small crop shown in the gallery.

    ### Reading each row

    Each row contains five views:

    | Column | What it shows |
    | --- | --- |
    | Full reference | The original photograph, to give context. |
    | Reference crop | A central 128×128 region of the original. |
    | Method B crop | The same region reconstructed by the baseline. |
    | Method A crop | The same region reconstructed by the method being evaluated. |
    | Error map | The size of A’s pixel differences from the reference in that region. |

    The crops show exactly the same location, making direct comparison easier. We always use the centre rather than choosing a region that makes one method look good. However, this means the crop may miss a serious error elsewhere in the photograph.

    ### What should we look for?

    Look for specific changes rather than just deciding that an image looks “bad”:

    - **Lost texture:** leaves, brickwork or fabric become smooth or smeared.
    - **Stepped edges:** diagonals become visibly jagged.
    - **Blur:** boundaries soften and thin features merge.
    - **Ringing or halos:** light or dark bands appear beside strong edges.

    I would first judge an image at its intended viewing size, then use enlarged crops to investigate anything distracting. A defect visible only under extreme magnification may matter less than one visible during normal use. Keep the displayed sizes consistent when comparing methods; the browser can also resize these figures.

    With the default comparison, inspect the foliage in `kodim13.png` and the brickwork in `kodim01.png`. I find the sharpened result a little crisper, but the foliage still looks smeared and the fine brick texture has not returned. I would describe this as stronger edge contrast, rather than recovery of the missing texture. That is a visual judgement, separate from the PSNR result.

    ### Reading the error map

    At each pixel, we calculate the absolute difference from the reference in red, green and blue, then average those three differences. Brighter areas indicate larger errors.

    Every row uses the same colour scale, from 0 to 0.2, so colours can be compared across images. Errors above 0.2 all use the brightest colour.

    The map shows **where pixels disagree**, not how distracting the disagreement is. A bright error around a small feature may matter more to a viewer than a larger error spread across an uninteresting background.

    Also note that the map uses absolute differences, whilst MSE uses squared differences. The map helps locate errors; the MSE score determines the image ranking.

    ### What does “worst” mean here?

    These are A’s largest reconstruction errors. They are not necessarily the images where A performs worse than B. Both methods might struggle on the same photograph, and A could still be the better of the two.

    To find where A loses most to B, we would instead rank images by the difference between their scores.

    The gallery deliberately shows difficult cases, so it does not represent typical performance. Read it alongside the results for all 24 images.
    """)
    return


@app.cell
def _(method_a, method_b, np, plt, predictions, references, scores):
    worst = (
        scores[scores.method == method_a.value]
        .sort_values(["MSE", "image"], ascending=[False, True])
        .head(8)
    )
    gallery, _axes = plt.subplots(8, 5, figsize=(14, 20), layout="constrained")
    for _row, _record in enumerate(worst.itertuples()):
        _ref = references[_record.image]
        _y, _x = (_ref.shape[0] - 128) // 2, (_ref.shape[1] - 128) // 2
        _crop = (
            _slice_y := slice(_y, _y + 128),
            _slice_x := slice(_x, _x + 128),
        )
        _a = predictions[(_record.image, method_a.value)]
        _b = predictions[(_record.image, method_b.value)]
        _images = [
            _ref,
            _ref[_crop],
            _b[_crop],
            _a[_crop],
            np.abs(_ref[_crop] - _a[_crop]).mean(axis=2),
        ]
        _titles = [
            f"{_record.image} / MSE {_record.MSE:.5f}",
            "Reference crop",
            method_b.value,
            method_a.value,
            "A absolute error (0–0.2)",
        ]
        for _column, (_image, _title) in enumerate(zip(_images, _titles)):
            _axis = _axes[_row, _column]
            _axis.imshow(
                _image,
                **({"cmap": "magma", "vmin": 0, "vmax": 0.2} if _column == 4 else {}),
            )
            _axis.set_title(_title, fontsize=8)
            _axis.axis("off")
    gallery
    return (worst,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Report template

    Use the results table and failure gallery to write a short conclusion. Include:

    - **The comparison:** which methods you compared, using which images and settings.
    - **The numerical result:** the mean scores and the paired bootstrap confidence interval.
    - **A visual example:** name an image and describe a specific improvement or failure.
    - **Your judgement:** which method you would choose for the intended task, and why.
    - **The limits:** what this experiment does not tell us.

    Keep the evaluation settings and software versions with the report so someone else can repeat the comparison.

    ### Describe what you can see

    Separate an observation from an explanation:

    | Type of statement | Example |
    | --- | --- |
    | Observation | “The mortar lines look clearer in the sharpened crop.” |
    | Possible explanation | “The unsharp mask increases local contrast around those edges.” |
    | Claim requiring more evidence | “The method recovers the original brick texture.” |

    Clearer edges do not necessarily mean missing detail has been recovered. Compare with the reference before making that claim.

    ### Explain your choice

    The more complicated method does not have to win. If sharpening adds little visible benefit or produces distracting halos, keeping bicubic is a reasonable conclusion.

    I would explain the decision in terms of the intended use. For displaying a photograph, I care about its appearance at the viewing size. For faithful reproduction, I also care whether the apparent detail agrees with the reference. A higher PSNR alone does not settle either decision.

    ### State what remains untested

    We tested images created by shrinking the Kodak originals with bicubic downsampling. The ranking may change for inputs containing camera blur, noise or JPEG compression.

    If we adjust the sharpening strength after inspecting these results, Kodak has helped us choose the settings. We should then evaluate the final choice on a separate set of images that we have not used for tuning.

    You can generate reports and tables in code demonstrated below. Note the elements to be filled in. You can then generate downloadable reports in various formats using the marimo download buttons.
    """)
    return


@app.cell
def _(ci_high, ci_low, gain, method_a, method_b, mo, summary, worst):
    _lines = [
        "| Method | MSE ↓ (mean ± SD) | PSNR dB ↑ (mean ± SD) |",
        "| --- | ---: | ---: |",
    ]
    for _method, _row in summary.iterrows():
        _lines.append(
            f"| {_method} | {_row[('MSE', 'mean')]:.6f} ± {_row[('MSE', 'std')]:.6f} | {_row[('PSNR', 'mean')]:.3f} ± {_row[('PSNR', 'std')]:.3f} |"
        )
    report = (
        "\n".join(_lines)
        + f"""

    We evaluated 2× upscaling on all 24 Kodak images using antialiased bicubic degradation,
    clipped RGB floats, and a two-pixel scoring border. {method_a.value} changed mean PSNR
    by {gain:.3f} dB relative to {method_b.value} (paired percentile bootstrap 95% CI
    [{ci_low:.3f}, {ci_high:.3f}], 10,000 resamples, seed 42). The worst eight for A were
    {", ".join(worst.image)}; see the failure gallery. On [image], I can see [specific failure],
    which may be caused by [reason supported by the crop]. This matters because [effect on
    intended use]. These results apply to this dataset and degradation; they do not establish
    perceptual quality or performance on real low-resolution photographs.
    """
    )
    mo.md(report)
    return (report,)


@app.cell
def _(json, manifest, mo, np, pd, report, scores, torch):
    provenance = {
        "dataset": manifest,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "pandas": pd.__version__,
        "device": "CPU",
        "scale": 2,
        "border": 2,
        "unsharp_amount": 0.5,
        "bootstrap_seed": 42,
        "bootstrap_samples": 10000,
    }
    mo.hstack(
        [
            mo.download(
                scores.to_csv(index=False).encode(),
                filename="per-image-scores.csv",
                label="Download all scores",
            ),
            mo.download(
                report.encode(),
                filename="report-template.md",
                label="Download report template",
            ),
            mo.download(
                json.dumps(provenance, indent=2).encode(),
                filename="protocol.json",
                label="Download image hashes and settings",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()

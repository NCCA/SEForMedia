#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Image-to-image learning, Part 2: 2× super resolution

    We will train a U-Net and a small ESPCN to reconstruct a 128 × 128 RGB patch from a 64 × 64 input, then compare their quality and processing cost.

    [Introduction](IntroductionMarimo.py) · [Denoising](DenoisingMarimo.py) · [Super resolution](SuperResolutionMarimo.py) · [Pet segmentation](PetSegmentationMarimo.py)

    ## What changes for this task?

    Unlike denoising, the target is the clean image itself, and it has twice the width and height of the input. Unlike segmentation, the output still has three RGB channels. U-Net first enlarges the input; ESPCN learns features at low resolution and rearranges its final channels with PixelShuffle.

    | Item | This notebook |
    | --- | --- |
    | Input patch (C × H × W) | Low-resolution RGB, 3 × 64 × 64 |
    | Training target | Clean RGB, 3 × 128 × 128 |
    | Loss | MSE between predicted and clean RGB pixels |
    | Inference | Use the predicted RGB image at twice the input size |

    Read the [introduction](IntroductionMarimo.py) first for the U-Net architecture,
    skip connections, checkerboard comparison and shared patch/tiling workflow.
    This notebook runs independently and uses the same `data` directory.
    """)
    return


@app.cell
def _():
    import inspect
    import sys
    from pathlib import Path
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torchvision import datasets
    from torchvision.transforms import functional as TF

    lesson_directory = Path(mo.notebook_location())
    if str(lesson_directory) not in sys.path:
        sys.path.insert(0, str(lesson_directory))
    if str(lesson_directory.parent) not in sys.path:
        sys.path.append(str(lesson_directory.parent))
    import image_models as core
    from Utils import get_device

    return (
        F,
        TF,
        core,
        datasets,
        get_device,
        inspect,
        lesson_directory,
        mo,
        nn,
        pd,
        plt,
        time,
        torch,
    )


@app.cell
def _(get_device):
    device = get_device()
    print(f"Training device: {device}")
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From low-resolution features to RGB pixels

    We crop a 128 × 128 target first, then use area averaging to make its 64 × 64
    input. The U-Net baseline enlarges the input with bicubic interpolation before
    processing it. ESPCN does its convolutions at the smaller resolution and moves
    channels into spatial positions with
    [`PixelShuffle(2)`](https://docs.pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html).
    For RGB output it needs $3\times2^2=12$ channels before the shuffle.

    This is an RGB teaching variant of [ESPCN](https://arxiv.org/abs/1609.05158).
    It has far fewer parameters than our U-Net. Both receive exactly the same low
    resolution patches and minimise MSE against the same targets. We train both networks in this notebook. The small example makes the channel-to-pixel movement visible.
    """)
    return


@app.cell
def _(nn, torch):
    _channels = torch.arange(4.0).reshape(1, 4, 1, 1)
    print("Input shape:", tuple(_channels.shape))
    print("Channel values:", _channels.flatten())
    print("PixelShuffle(2):\n", nn.PixelShuffle(2)(_channels)[0, 0])
    return


@app.cell
def _(core, inspect, mo):
    mo.md(
        "```python\n"
        + inspect.getsource(core.SuperResolutionUNet)
        + "\n"
        + inspect.getsource(core.ESPCN)
        + "\n```"
    )
    return


@app.cell
def _(core):
    _small = core.ESPCN()
    print("ESPCN parameters:", f"{sum(p.numel() for p in _small.parameters()):,}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prepare the training pair

    We crop the clean 128 × 128 target first, then area-downsample it to 64 × 64.
    This keeps the input and target aligned. The trimap is not a training target.
    Both networks receive the same low-resolution tensors and target pixels.

    We use the photograph-level split and fixed validation protocol from the
    [introduction](IntroductionMarimo.py). The following generated example shows
    exactly what this task receives as input and target before we load photographs.
    """)
    return


@app.cell
def _(F, core, plt, torch):
    _y, _x = torch.meshgrid(
        torch.linspace(0, 1, 160), torch.linspace(0, 1, 192), indexing="ij"
    )
    _synthetic = torch.stack((_x, _y, (_x > 0.5).float()))
    _trimap = torch.where((_x - 0.5).square() + (_y - 0.5).square() < 0.12, 1, 2)[
        None
    ].byte()
    _inputs, _target, _valid = core.make_patch(
        _synthetic, _trimap, "sr", torch.Generator().manual_seed(42)
    )
    _panels = [
        ("Low-resolution input: 64 × 64", _inputs),
        (
            "Bicubic baseline: 128 × 128",
            F.interpolate(
                _inputs[None], scale_factor=2, mode="bicubic", align_corners=False
            )[0].clamp(0, 1),
        ),
        ("Clean target: 128 × 128", _target),
    ]

    print("Input shape:", tuple(_inputs.shape), "Target shape:", tuple(_target.shape))
    _figure, _axes = plt.subplots(1, 3, figsize=(11, 3), layout="constrained")
    for _axis, (_title, _pixels) in zip(_axes, _panels):
        _axis.imshow(
            (_pixels[0] if _pixels.shape[0] == 1 else _pixels.permute(1, 2, 0)).numpy(),
            vmin=0,
            vmax=1,
            cmap="gray",
        )
        _axis.set_title(_title, fontsize=10)
        _axis.axis("off")
    plt.close(_figure)
    _figure
    return


@app.cell
def _(lesson_directory, mo):
    data_form = mo.ui.dictionary(
        {
            "root": mo.ui.text(
                value=str(lesson_directory / "data"), label="Dataset directory"
            ),
            "download": mo.ui.checkbox(
                value=False, label="Download Oxford-IIIT Pet if missing"
            ),
        }
    ).form(submit_button_label="Load dataset")
    data_form
    return (data_form,)


@app.cell
def _(data_form, datasets, mo, torch):
    mo.stop(
        data_form.value is None,
        mo.md("Choose a directory and load the dataset when you are ready."),
    )
    _settings = data_form.value
    try:
        pets_trainval = datasets.OxfordIIITPet(
            _settings["root"],
            split="trainval",
            target_types="segmentation",
            download=_settings["download"],
        )
        pets_test = datasets.OxfordIIITPet(
            _settings["root"],
            split="test",
            target_types="segmentation",
            download=_settings["download"],
        )
    except RuntimeError as error:
        mo.stop(
            True,
            mo.md(
                f"Dataset could not be loaded: {error}. Check the directory or enable the download."
            ),
        )
    _order = torch.randperm(
        len(pets_trainval), generator=torch.Generator().manual_seed(42)
    ).tolist()
    _n_validation = max(1, len(_order) // 5)
    validation_indices, train_indices = (
        _order[:_n_validation],
        _order[_n_validation:],
    )
    print(
        f"Photographs: {len(train_indices)} training, {len(validation_indices)} validation, {len(pets_test)} test"
    )
    return pets_test, pets_trainval, train_indices, validation_indices


@app.cell
def _(F, TF, core, pets_trainval, plt, torch, train_indices):
    _image, _trimap = pets_trainval[train_indices[0]]
    _inputs, _target, _valid = core.make_patch(
        TF.to_tensor(_image.convert("RGB")),
        TF.pil_to_tensor(_trimap),
        "sr",
        torch.Generator().manual_seed(42),
    )
    _panels = [
        ("Low-resolution input: 64 × 64", _inputs),
        (
            "Bicubic baseline: 128 × 128",
            F.interpolate(
                _inputs[None], scale_factor=2, mode="bicubic", align_corners=False
            )[0].clamp(0, 1),
        ),
        ("Clean target: 128 × 128", _target),
    ]

    print("Input shape:", tuple(_inputs.shape), "Target shape:", tuple(_target.shape))
    _figure, _axes = plt.subplots(1, 3, figsize=(11, 3), layout="constrained")
    for _axis, (_title, _pixels) in zip(_axes, _panels):
        _axis.imshow(
            (_pixels[0] if _pixels.shape[0] == 1 else _pixels.permute(1, 2, 0)).numpy(),
            vmin=0,
            vmax=1,
            cmap="gray",
        )
        _axis.set_title(_title, fontsize=10)
        _axis.axis("off")
    plt.close(_figure)
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train a fresh model

    U-Net and ESPCN train on the same sequence of patches with the same number of optimiser steps. Their processing time differs.

    Submit the settings to train. The loop below uses the device detected above and
    retains the best validation weights. See the [introduction](IntroductionMarimo.py)
    for the shared training and validation protocol.
    """)
    return


@app.cell
def _(mo):
    training_form = mo.ui.dictionary(
        {
            "upsampling": mo.ui.dropdown(
                ["transpose", "resize"], value="transpose", label="U-Net decoder"
            ),
            "epochs": mo.ui.number(1, 200, value=2, label="Epochs"),
            "samples": mo.ui.number(
                8, 8192, step=8, value=64, label="Patches per epoch"
            ),
            "batch": mo.ui.dropdown([1, 2, 4, 8], value=4, label="Batch size"),
            "lr": mo.ui.number(
                1e-05, 0.01, step=1e-05, value=0.001, label="Learning rate"
            ),
        }
    ).form(submit_button_label="Train")
    training_form
    return (training_form,)


@app.cell
def _(
    core,
    device,
    mo,
    nn,
    pets_trainval,
    time,
    torch,
    train_indices,
    training_form,
    validation_indices,
):
    mo.stop(training_form.value is None, mo.md("Submit the training form to begin."))
    config = dict(training_form.value)
    config["task"] = "sr"
    config["device"] = str(device)
    trained_models, history, training_seconds = ({}, [], {})
    _names = ["U-Net", "ESPCN"]
    for _name in _names:
        torch.manual_seed(42)
        if _name == "ESPCN":
            _model = core.ESPCN()
        else:
            _model = core.SuperResolutionUNet(config["upsampling"])
        _model = _model.to(device)
        _optimiser = torch.optim.Adam(_model.parameters(), lr=config["lr"])
        _train_data = core.PetPatches(
            pets_trainval, train_indices, "sr", config["samples"], seed=42
        )
        _validation_data = core.PetPatches(
            pets_trainval, validation_indices, "sr", samples=32, seed=10000, fixed=True
        )
        _loader = torch.utils.data.DataLoader(
            _train_data, batch_size=config["batch"], num_workers=0
        )
        _validation_loader = torch.utils.data.DataLoader(
            _validation_data, batch_size=config["batch"], num_workers=0
        )
        _best_loss = float("inf")
        _best_weights = None
        _start = time.perf_counter()
        for _epoch in mo.status.progress_bar(
            range(config["epochs"]), title=f"Training {_name}"
        ):
            _model.train()
            _train_total = 0.0
            for _inputs, _targets, _valid in _loader:
                _inputs, _targets, _valid = (
                    _tensor.to(device) for _tensor in (_inputs, _targets, _valid)
                )
                _optimiser.zero_grad()
                _predictions = _model(_inputs)
                _loss = nn.functional.mse_loss(_predictions, _targets)
                _loss.backward()
                _optimiser.step()
                _train_total += _loss.item() * len(_inputs)
            _model.eval()
            _validation_total = 0.0
            with torch.inference_mode():
                for _inputs, _targets, _valid in _validation_loader:
                    _inputs, _targets, _valid = (
                        _tensor.to(device) for _tensor in (_inputs, _targets, _valid)
                    )
                    _predictions = _model(_inputs)
                    _loss = nn.functional.mse_loss(_predictions, _targets)
                    _validation_total += _loss.item() * len(_inputs)
            _validation_loss = _validation_total / len(_validation_data)
            history.append(
                {
                    "Model": _name,
                    "Epoch": _epoch + 1,
                    "Training": _train_total / len(_train_data),
                    "Validation": _validation_loss,
                }
            )
            if _validation_loss < _best_loss:
                _best_loss = _validation_loss
                _best_weights = {
                    key: value.detach().cpu().clone()
                    for key, value in _model.state_dict().items()
                }
        if _best_weights is None:
            raise RuntimeError(
                "No finite validation loss; reduce the learning rate and inspect the inputs"
            )
        _model.load_state_dict(_best_weights)
        trained_models[_name] = _model.cpu().eval()
        training_seconds[_name] = time.perf_counter() - _start
    print("Best validation weights are ready. Training seconds:", training_seconds)
    return (config, history, trained_models, training_seconds)


@app.cell
def _(history, plt):
    _figure, _axis = plt.subplots(figsize=(8, 3), layout="constrained")
    for _name in dict.fromkeys(row["Model"] for row in history):
        _rows = [row for row in history if row["Model"] == _name]
        _axis.plot(
            [row["Epoch"] for row in _rows],
            [row["Training"] for row in _rows],
            label=f"{_name} training",
        )
        _axis.plot(
            [row["Epoch"] for row in _rows],
            [row["Validation"] for row in _rows],
            "--",
            label=f"{_name} validation",
        )
    _axis.set(xlabel="Epoch", ylabel="Loss")
    _axis.legend()
    plt.close(_figure)
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Apply the model to full images

    A 128-pixel low-resolution input tile produces a 256-pixel RGB tile (`scale=2`). We blend RGB predictions directly. Both tile size and overlap are measured in low-resolution input pixels.

    The [introduction](IntroductionMarimo.py) explains `core.tiled_predict`, blending
    and the context lost at tile edges. Here we use it for this task's output.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare with baselines

    We compare U-Net and ESPCN with nearest, bilinear, bicubic and bicubic plus unsharp interpolation. These are the resizing baselines from the evaluation lesson. For odd-sized photographs we remove the last row or column, then area-downsample the reference. Every method receives that same low-resolution image.

    We report RGB MSE and PSNR using the two-pixel scoring border described in the introduction.

    The [introduction](IntroductionMarimo.py) explains per-image aggregation, timing
    and how to interpret the comparison. The gallery below shows the first test image.
    """)
    return


@app.cell
def _(mo):
    evaluation_form = mo.ui.dictionary(
        {
            "count": mo.ui.number(
                2,
                3669,
                value=4,
                label="Test photographs (first N in official order)",
            ),
            "tile": mo.ui.dropdown([64, 128, 256], value=128, label="Input tile size"),
            "overlap": mo.ui.dropdown([0, 16, 32, 48], value=32, label="Overlap"),
        }
    ).form(submit_button_label="Evaluate full images")
    evaluation_form
    return (evaluation_form,)


@app.cell
def _(
    F,
    TF,
    config,
    core,
    device,
    evaluation_form,
    mo,
    pd,
    pets_test,
    time,
    torch,
    trained_models,
):
    mo.stop(
        evaluation_form.value is None,
        mo.md("Train a model, then request a full-image comparison."),
    )
    _eval = evaluation_form.value
    score_rows, preview_panels = ([], [])
    for _name, _model in trained_models.items():
        _model.to(device)
        with torch.inference_mode():
            _model(torch.zeros(1, 3, _eval["tile"], _eval["tile"], device=device))

    def synchronise() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    for _index in mo.status.progress_bar(
        range(min(_eval["count"], len(pets_test))), title="Full-image evaluation"
    ):
        _pil, _trimap = pets_test[_index]
        _reference = TF.to_tensor(_pil.convert("RGB"))
        _height, _width = _reference.shape[-2:]
        _reference = _reference[:, : _height - _height % 2, : _width - _width % 2]
        _input = F.interpolate(_reference[None], scale_factor=0.5, mode="area")[0]
        _baselines = ("nearest", "bilinear", "bicubic", "bicubic + unsharp")
        if _index == 0:
            preview_panels.append(("Reference", _reference))
            preview_panels.append(("Input", _input.clamp(0, 1)))
        for _name in (*_baselines, *trained_models):
            synchronise()
            _start = time.perf_counter()
            if _name in trained_models:
                _prediction = core.tiled_predict(
                    trained_models[_name],
                    _input,
                    tile=_eval["tile"],
                    overlap=_eval["overlap"],
                    scale=2,
                    device=device,
                )
            else:
                _mode = "bicubic" if _name == "bicubic + unsharp" else _name
                _options = {} if _mode == "nearest" else {"align_corners": False}
                _prediction = F.interpolate(
                    _input[None], scale_factor=2, mode=_mode, **_options
                )[0]
                if _name == "bicubic + unsharp":
                    _blur = F.avg_pool2d(
                        F.pad(_prediction[None], (1, 1, 1, 1), mode="replicate"),
                        3,
                        stride=1,
                    )[0]
                    _prediction = _prediction + 0.5 * (_prediction - _blur)
            _prediction = _prediction.clamp(0, 1)
            synchronise()
            _seconds = time.perf_counter() - _start
            _row = {
                "Image": _index,
                "Method": _name,
                "Seconds": _seconds,
                "Parameters": sum(
                    (p.numel() for p in trained_models[_name].parameters())
                )
                if _name in trained_models
                else 0,
            }
            _row["MSE"], _row["PSNR"] = core.image_scores(_reference, _prediction)
            score_rows.append(_row)
            if _index == 0:
                preview_panels.append((_name, _prediction))
    for _model in trained_models.values():
        _model.cpu()
    scores = pd.DataFrame(score_rows)
    _metric_names = ["MSE", "PSNR"]
    summary = scores.groupby("Method", sort=False)[[*_metric_names, "Seconds"]].agg(
        ["mean", "std"]
    )
    mo.vstack(
        [
            mo.md("Per-image scores (lower MSE and higher PSNR are better):"),
            mo.ui.table(scores, selection=None),
            mo.ui.table(summary.reset_index(), selection=None),
        ]
    )
    return (preview_panels, scores)


@app.cell
def _(plt, preview_panels):
    _columns = 3
    _rows = (len(preview_panels) + _columns - 1) // _columns
    _figure, _axes = plt.subplots(
        _rows,
        _columns,
        figsize=(12, 3.5 * _rows),
        squeeze=False,
        layout="constrained",
    )
    for _axis in _axes.flat:
        _axis.axis("off")
    for _axis, (_title, _image) in zip(_axes.flat, preview_panels):
        _axis.imshow(
            (_image[0] if _image.shape[0] == 1 else _image.permute(1, 2, 0)).numpy(),
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        _axis.set_title(_title)
    plt.close(_figure)
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Keep this run

    Save the selected weights, configuration and per-image scores to `checkpoints`.
    The filename includes the task and a timestamp. See the [introduction](IntroductionMarimo.py)
    for the saved metadata and why we keep it.
    """)
    return


@app.cell
def _(mo):
    save_button = mo.ui.run_button(label="Save checkpoint and scores")
    save_button
    return (save_button,)


@app.cell
def _(
    config,
    lesson_directory,
    mo,
    save_button,
    scores,
    torch,
    trained_models,
    training_seconds,
):
    mo.stop(not save_button.value)
    from datetime import datetime

    _output_directory = lesson_directory / "checkpoints"
    _output_directory.mkdir(exist_ok=True)
    _stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    _checkpoint = _output_directory / f"sr-{_stamp}.pth"
    torch.save(
        {
            "config": config,
            "split_seed": 42,
            "metric_protocol": "RGB [0,1], two-pixel crop, per-image MSE/PSNR",
            "torch_version": str(torch.__version__),
            "training_seconds": training_seconds,
            "weights": {
                name: {key: value.cpu() for key, value in model.state_dict().items()}
                for name, model in trained_models.items()
            },
        },
        _checkpoint,
    )
    scores.to_csv(_checkpoint.with_suffix(".csv"), index=False)
    print("Saved:", _checkpoint)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Trace a 64 × 64 input through each network. At which line does the spatial resolution double?
    2. Change PixelShuffle to 3× enlargement. How many output channels are needed, and what else must change in the data and tiling code?
    3. Compare U-Net and ESPCN on PSNR, parameter count and time. Which would you choose for a live preview?
    4. Train with area downsampling, then test with a different degradation. Why might the scores change even at the same scale factor?
    5. Compare tile overlaps 0, 16 and 32. Remember that overlap is measured in low-resolution pixels.

    Next, [pet segmentation](PetSegmentationMarimo.py) replaces the RGB output with one logit per pixel.
    """)
    return


if __name__ == "__main__":
    app.run()

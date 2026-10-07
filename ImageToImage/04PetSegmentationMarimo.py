#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Image-to-image learning, Part 3: binary pet segmentation

    We will train a U-Net to label each pixel as pet or background. The output is a mask, so matching the colours of the photograph is no longer our objective.

    [Introduction](IntroductionMarimo.py) · [Denoising](DenoisingMarimo.py) · [Super resolution](SuperResolutionMarimo.py) · [Pet segmentation](PetSegmentationMarimo.py)

    ## What changes for this task?

    Unlike denoising and super resolution, the target is an annotation rather than a photograph. We change the U-Net head from three RGB channels to one logit channel. Image MSE and PSNR are replaced by mask overlap scores, and uncertain boundary pixels must be ignored in both the loss and evaluation.

    | Item | This notebook |
    | --- | --- |
    | Input patch (C × H × W) | Clean RGB, 3 × 128 × 128 |
    | Training target | Binary mask, 1 × 128 × 128 |
    | Loss | BCEWithLogitsLoss plus soft Dice, ignoring uncertain pixels |
    | Inference | Apply sigmoid to logits, then threshold at 0.5 |

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
    ## One logit per pixel

    Oxford's trimap has label 1 for the pet, 2 for the background and 3 for the uncertain
    boundary. Our target is `(trimap == 1).float()`. We ignore label 3, including in the
    scores; counting it as background would change the task.

    The U-Net output has one channel with no sigmoid layer. `BCEWithLogitsLoss` includes
    the sigmoid calculation in a numerically stable form. We apply sigmoid separately
    for soft Dice and for display:

    $$L=L_{\mathrm{BCE}}+1-\frac{2\sum_i p_i t_i+\epsilon}{\sum_i p_i+\sum_i t_i+\epsilon}.$$

    The sums contain only valid pixels. Dice is computed per patch, then averaged over
    the batch. An all-ignored patch contributes zero loss. For scoring, we threshold
    at 0.5 and report hard Dice and intersection over union (IoU). We define an empty
    prediction and empty target as a score of one.
    """)
    return


@app.cell
def _(core, inspect, mo):
    mo.md("```python\n" + inspect.getsource(core.segmentation_loss) + "\n```")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prepare the training pair

    We crop the RGB photograph and trimap at the same coordinates. Label 1 becomes
    foreground, label 2 becomes background, and label 3 is excluded with a validity
    mask. Small photographs are padded and padded mask pixels are ignored.

    We use the photograph-level split and fixed validation protocol from the
    [introduction](IntroductionMarimo.py). The following generated example shows
    exactly what this task receives as input and target before we load photographs.
    """)
    return


@app.cell
def _(core, plt, torch):
    _y, _x = torch.meshgrid(
        torch.linspace(0, 1, 160), torch.linspace(0, 1, 192), indexing="ij"
    )
    _synthetic = torch.stack((_x, _y, (_x > 0.5).float()))
    _trimap = torch.where((_x - 0.5).square() + (_y - 0.5).square() < 0.12, 1, 2)[
        None
    ].byte()
    _distance = (_x - 0.5).square() + (_y - 0.5).square()
    _trimap[:, (_distance >= 0.11) & (_distance <= 0.13)] = 3
    _inputs, _target, _valid = core.make_patch(
        _synthetic, _trimap, "segment", torch.Generator().manual_seed(42)
    )
    _panels = [
        ("RGB input: 128 × 128", _inputs),
        ("Binary target: 1 channel", _target),
        ("Valid pixels (white)", _valid),
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
def _(TF, pets_trainval, plt, train_indices):
    _image, _mask = pets_trainval[train_indices[0]]
    _mask = TF.pil_to_tensor(_mask)[0]
    _figure, _axes = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
    _axes[0].imshow(_image)
    _axes[1].imshow(
        _mask.numpy(), vmin=1, vmax=3, cmap="viridis", interpolation="nearest"
    )
    _axes[2].imshow((_mask == 1).float().numpy(), vmin=0, vmax=1, cmap="gray")
    for _axis, _title in zip(
        _axes,
        (
            "Training photograph",
            "Trimap: 1 pet, 2 background, 3 boundary",
            "Binary target (boundary ignored)",
        ),
    ):
        _axis.set_title(_title, fontsize=10)
        _axis.axis("off")
    plt.close(_figure)
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train a fresh model

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
    config["task"] = "segment"
    config["device"] = str(device)
    trained_models, history, training_seconds = ({}, [], {})
    _name = "U-Net"
    torch.manual_seed(42)
    _model = core.UNet(1, config["upsampling"])
    _model = _model.to(device)
    _optimiser = torch.optim.Adam(_model.parameters(), lr=config["lr"])
    _train_data = core.PetPatches(
        pets_trainval, train_indices, "segment", config["samples"], seed=42
    )
    _validation_data = core.PetPatches(
        pets_trainval, validation_indices, "segment", samples=32, seed=10000, fixed=True
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
            _loss = core.segmentation_loss(_predictions, _targets, _valid)
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
                _loss = core.segmentation_loss(_predictions, _targets, _valid)
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

    The model predicts one logit per input pixel (`scale=1`). We blend logits, apply sigmoid to the assembled output, then threshold at 0.5. We do not blend already-thresholded binary masks.

    The [introduction](IntroductionMarimo.py) explains `core.tiled_predict`, blending
    and the context lost at tile edges. Here we use it for this task's output.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare with baselines

    We compare the U-Net mask with all-background and all-pet predictions. These expose class imbalance, but are not strong segmentation methods. Hard Dice and IoU are computed at threshold 0.5 with trimap label 3 excluded. We score masks rather than RGB colours; a high pixel accuracy can hide a model that misses the pet.

    Higher Dice and IoU indicate greater overlap. We ignore trimap label 3 in both masks and define two empty masks as a score of one.

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
        _trimap = TF.pil_to_tensor(_trimap)
        _valid = (_trimap != 3).float()
        _mask = (_trimap == 1).float()
        _input = _reference
        _baselines = ("All background", "All pet")
        if _index == 0:
            preview_panels.append(("Photograph", _reference))
            preview_panels.append(
                ("Target (grey = ignored)", _mask * _valid + 0.5 * (1 - _valid))
            )
        for _name in (*_baselines, *trained_models):
            synchronise()
            _start = time.perf_counter()
            if _name in trained_models:
                _prediction = core.tiled_predict(
                    trained_models[_name],
                    _input,
                    tile=_eval["tile"],
                    overlap=_eval["overlap"],
                    scale=1,
                    device=device,
                )
                _prediction = _prediction.sigmoid()
            else:
                _prediction = (
                    torch.zeros_like(_mask)
                    if _name == "All background"
                    else torch.ones_like(_mask)
                )
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
            _row["Dice"], _row["IoU"] = core.mask_scores(_prediction, _mask, _valid)
            score_rows.append(_row)
            if _index == 0:
                preview_panels.append(
                    (_name, (_prediction >= 0.5).float() * _valid + 0.5 * (1 - _valid))
                )
    for _model in trained_models.values():
        _model.cpu()
    scores = pd.DataFrame(score_rows)
    _metric_names = ["Dice", "IoU"]
    summary = scores.groupby("Method", sort=False)[[*_metric_names, "Seconds"]].agg(
        ["mean", "std"]
    )
    mo.vstack(
        [
            mo.md("Per-image scores (higher Dice and IoU are better):"),
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
    _checkpoint = _output_directory / f"segment-{_stamp}.pth"
    torch.save(
        {
            "config": config,
            "split_seed": 42,
            "metric_protocol": "Trimap 3 ignored, per-image Dice/IoU, threshold 0.5",
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

    1. Explain why we pass logits to BCEWithLogitsLoss but probabilities to Dice. What happens if we apply sigmoid twice?
    2. Compare pixel accuracy with Dice and IoU for an all-background prediction. Which score reveals the missing pet?
    3. Treat the uncertain boundary as background, then compare with ignoring it. Explain how this changes both the target and evaluation.
    4. Compare a model trained with BCE alone with BCE plus Dice using the same split and seed.
    5. Vary the prediction threshold using validation photographs. Keep the selected threshold fixed for the final test comparison.

    Return to [denoising](DenoisingMarimo.py) or [super resolution](SuperResolutionMarimo.py) and explain why their three-channel MSE objective would not describe this task.
    """)
    return


if __name__ == "__main__":
    app.run()

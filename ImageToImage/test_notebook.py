import importlib
from types import SimpleNamespace

import marimo
import numpy as np
from PIL import Image
import pytest
import torch

from ImageToImage.IntroductionMarimo import app as introduction

NOTEBOOKS = {
    "denoise": "DenoisingMarimo",
    "sr": "SuperResolutionMarimo",
    "segment": "PetSegmentationMarimo",
}


@pytest.fixture(params=NOTEBOOKS)
def task(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def notebook(task: str) -> marimo.App:
    return importlib.import_module(f"ImageToImage.{NOTEBOOKS[task]}").app


@pytest.fixture
def synthetic_pets() -> list[tuple[Image.Image, Image.Image]]:
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(4):
        pixels = rng.integers(0, 256, (137, 149, 3), dtype=np.uint8)
        mask = np.full((137, 149), 2, dtype=np.uint8)
        mask[30:110, 35:115] = 1
        mask[29:31, 35:115] = 3
        samples.append((Image.fromarray(pixels), Image.fromarray(mask)))
    return samples


def test_introduction_runs_without_data_or_training() -> None:
    _, definitions = introduction.run()

    assert "training_form" not in definitions
    assert "pets_trainval" not in definitions
    assert definitions["intro_tile_error"] < 1e-6


def test_default_run_waits_for_user(notebook: marimo.App) -> None:
    _, definitions = notebook.run()

    assert definitions["device"] == definitions["get_device"]()
    assert "Device" not in definitions["training_form"].text
    assert "A — denoising" not in definitions["training_form"].text
    assert definitions["training_form"].value is None
    assert "trained_models" not in definitions


def test_task_trains_and_evaluates_without_downloads(
    notebook: marimo.App,
    task: str,
    synthetic_pets: list[tuple[Image.Image, Image.Image]],
) -> None:
    _, definitions = notebook.run(
        defs={
            "device": torch.device("cpu"),
            "pets_trainval": synthetic_pets,
            "pets_test": synthetic_pets[:2],
            "train_indices": [0, 1],
            "validation_indices": [2, 3],
            "training_form": SimpleNamespace(
                value={
                    "upsampling": "transpose",
                    "epochs": 1,
                    "samples": 8,
                    "batch": 4,
                    "lr": 0.001,
                }
            ),
            "evaluation_form": SimpleNamespace(
                value={"count": 2, "tile": 128, "overlap": 32}
            ),
        }
    )

    assert definitions["config"]["task"] == task
    assert definitions["history"]
    scores = definitions["scores"]
    metrics = ["Dice", "IoU"] if task == "segment" else ["MSE", "PSNR"]
    assert np.isfinite(scores[metrics].to_numpy()).all()
    assert set(scores["Image"]) == {0, 1}
    assert len(definitions["trained_models"]) == (2 if task == "sr" else 1)
    assert not definitions["save_button"].value
    with torch.inference_mode():
        for model in definitions["trained_models"].values():
            prediction = model(torch.zeros(1, 3, 32, 40))
            expected = (
                (1, 3, 64, 80)
                if task == "sr"
                else (1, 1 if task == "segment" else 3, 32, 40)
            )
            assert tuple(prediction.shape) == expected

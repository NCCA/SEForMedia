import importlib.util
from pathlib import Path

import pytest


def run_notebook(suffix: str) -> dict:
    path = Path(__file__).resolve().parents[1] / f"TorchAudioForML{suffix}.py"
    assert path.exists(), f"Missing notebook: {path.name}"
    spec = importlib.util.spec_from_file_location(suffix, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, definitions = module.app.run()
    return definitions


def test_waveforms():
    d = run_notebook("Part1Waveforms")
    assert d["roundtrip_error"] < 1 / 32768
    assert d["resampled"].shape[-1] == 8000
    assert d["waveform"].shape == (2, 16000)


def test_features():
    d = run_notebook("Part2Features")
    assert d["mel_power"].shape[-2] == 64
    assert d["mfcc"].shape[-2] == 13
    assert d["reconstruction_error"] < 1e-5
    assert d["torch"].isfinite(d["mel_db"]).all()


def test_augmentation():
    d = run_notebook("Part3Augmentation")
    assert d["measured_snr"] == pytest.approx(15.0, abs=0.0005)
    assert d["torch"].equal(d["clean_power"], d["original_power"])
    assert (d["masked_power"] == 0).any()
    assert d["noisy"].shape == d["clean"].shape


def test_dataset_and_classifier():
    d = run_notebook("Part4Datasets")
    assert d["test_accuracy"] >= 0.9
    assert d["loss_history"][-1] < d["loss_history"][0]
    assert d["torch"].isfinite(d["batch_features"]).all()
    for row, length in zip(d["padded"], d["lengths"]):
        assert (row[int(length) :] == 0).all()
    assert d["restored_predictions"].tolist() == d["predictions"].tolist()

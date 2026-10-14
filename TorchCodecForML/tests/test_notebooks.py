import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def run_notebook():
    """Execute a notebook, including its generated media and plots."""
    media_directories = []

    def run(suffix: str) -> dict:
        path = Path(__file__).resolve().parents[1] / f"TorchCodecForML{suffix}.py"
        assert path.exists(), f"Missing notebook: {path.name}"
        spec = importlib.util.spec_from_file_location(suffix, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _, definitions = module.app.run()
        media_directories.append(definitions["media_directory"])
        return definitions

    yield run
    for directory in media_directories:
        directory.cleanup()


def test_video_pixels_and_presentation_times(run_notebook) -> None:
    d = run_notebook("Part1Video")
    assert len(d["decoder"]) == 24
    assert tuple(d["frames"].data.shape) == (4, 3, 64, 96)
    assert d["torch"].equal(d["decoded"], d["source_frames"])
    assert d["torch"].equal(d["from_bytes"], d["decoded"][0])
    for requested, pts, duration in zip(
        d["requested_times"],
        d["timed_frames"].pts_seconds,
        d["timed_frames"].duration_seconds,
    ):
        assert float(pts) <= requested
        assert requested < float(pts + duration) + 1e-6


def test_sampling_and_end_policies(run_notebook) -> None:
    d = run_notebook("Part2Sampling")
    assert tuple(d["regular"].data.shape) == (3, 4, 3, 64, 96)
    assert d["torch"].equal(d["random_a"].data, d["random_b"].data)
    assert d["torch"].equal(d["repeat"].data[0, -1], d["repeat"].data[0, -2])
    assert d["torch"].equal(d["wrapped"].data[0, 0], d["wrapped"].data[0, 2])
    assert d["policy_error"]
    assert tuple(d["model_input"].shape) == (3, 3, 4, 32, 48)
    assert float(d["model_input"].min()) >= 0
    assert float(d["model_input"].max()) <= 1


def test_audio_roundtrip_resampling_and_ranges(run_notebook) -> None:
    d = run_notebook("Part3Audio")
    assert tuple(d["samples"].data.shape) == (2, 32000)
    assert d["roundtrip_error"] < 1e-4
    assert d["resampled"].sample_rate == 8000
    assert tuple(d["resampled"].data.shape) == (1, 16000)
    assert tuple(d["segment"].data.shape) == (2, 8000)
    assert d["segment"].pts_seconds == pytest.approx(0.5, abs=0.00005)
    assert d["torch"].allclose(d["segment"].data, d["samples"].data[:, 8000:16000])
    assert d["flac_error"] < 1e-4


def test_dataset_batches_and_model(run_notebook) -> None:
    d = run_notebook("Part4Datasets")
    assert tuple(d["batch_clips"].shape) == (4, 3, 4, 32, 48)
    assert tuple(d["logits"].shape) == (4, 2)
    assert d["torch"].isfinite(d["logits"]).all()
    assert set(d["train_paths"]).isdisjoint(d["test_paths"])
    assert d["batch_labels"].tolist() == [0, 0, 1, 1]
    assert d["torch"].equal(d["dataset"][0][0], d["dataset"][0][0])
    assert d["gradient_norm"] > 0
    assert d["final_loss"] < d["initial_loss"]

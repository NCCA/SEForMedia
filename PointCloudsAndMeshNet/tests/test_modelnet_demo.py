"""Checks ModelNet10 preparation, the cache clean-up and the Part 7 lesson."""

import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import modelnet_data
import pytest
from scripts import clean_data

CATEGORIES = (
    "bathtub",
    "bed",
    "chair",
    "desk",
    "dresser",
    "monitor",
    "night_stand",
    "sofa",
    "table",
    "toilet",
)
# a tetrahedron, the smallest closed mesh trimesh will sample
OFF = "OFF\n4 4 0\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n3 0 1 2\n3 0 1 3\n3 0 2 3\n3 1 2 3\n"


@pytest.fixture
def modelnet_archive(tmp_path: Path) -> Path:
    """A tiny stand-in for the ModelNet10 zip: 4 train and 2 test meshes per class."""
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for category in CATEGORIES:
            for split, count in [("train", 4), ("test", 2)]:
                for index in range(count):
                    archive.writestr(
                        f"ModelNet10/{category}/{split}/{category}_{index:04}.off", OFF
                    )
    return path


def test_preparation_extracts_the_archive_into_the_cache(tmp_path, modelnet_archive):
    result = modelnet_data.prepare_modelnet10(
        tmp_path / "data", url=modelnet_archive.as_uri()
    )
    assert result.name == "ModelNet10"
    assert (result / "chair" / "test" / "chair_0000.off").is_file()
    assert modelnet_data.modelnet10_ready(tmp_path / "data")


def test_preparation_reuses_a_complete_cache_without_downloading(
    tmp_path, modelnet_archive
):
    first = modelnet_data.prepare_modelnet10(
        tmp_path / "data", url=modelnet_archive.as_uri()
    )
    again = modelnet_data.prepare_modelnet10(
        tmp_path / "data", url="https://invalid.example/no-network-needed"
    )
    assert again == first


def test_preparation_rejects_archive_paths_outside_the_cache(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("../outside.txt", "wrong location")
    with pytest.raises(ValueError):
        modelnet_data.prepare_modelnet10(tmp_path / "data", url=bad.as_uri())
    assert not (tmp_path / "outside.txt").exists()
    assert not modelnet_data.modelnet10_ready(tmp_path / "data")


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway git repo holding a downloaded cache and some trained weights."""
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    lesson = tmp_path / "PointCloudsAndMeshNet"
    (lesson / "data").mkdir(parents=True)
    (lesson / "data" / "ModelNet10.zip").write_bytes(b"archive")
    (lesson / "outputs").mkdir()
    (lesson / "outputs" / "modelnet10_classifier.pt").write_bytes(b"weights")
    monkeypatch.setattr(clean_data, "REPO_ROOT", tmp_path)
    return tmp_path


def test_cleanup_targets_the_download_cache_but_not_trained_weights(fake_repo):
    cache = fake_repo / "PointCloudsAndMeshNet" / "data"
    assert clean_data.find_targets() == [(cache, len(b"archive"))]


def test_cleanup_skips_the_cache_once_git_tracks_it(fake_repo, capsys):
    subprocess.run(
        ["git", "add", "PointCloudsAndMeshNet/data/ModelNet10.zip"],
        cwd=fake_repo,
        check=True,
    )
    assert clean_data.find_targets() == []
    assert "tracked by git" in capsys.readouterr().out


@pytest.mark.slow
def test_identification_lesson_runs_on_an_off_mesh_fixture(
    tmp_path, modelnet_archive, run_lesson
):
    modelnet_data.prepare_modelnet10(tmp_path / "data", url=modelnet_archive.as_uri())
    definitions = run_lesson(
        7,
        {
            "dataset_cache": tmp_path / "data",
            "training_form": SimpleNamespace(
                value=dict(
                    points=32, limit=4, test_limit=2, epochs=1, lr=0.003, seed=42
                )
            ),
        },
    )
    assert definitions["class_names"] == list(CATEGORIES)
    assert definitions["test_logits"].shape == (20, 10)
    assert [trace.type for trace in definitions["identification_view"].data] == [
        "mesh3d",
        "scatter3d",
    ]
    assert len(definitions["test_sources"]) == 20
    assert definitions["test_sources"][0]["path"].parent.name == "test"
    assert float(definitions["identification_probabilities"].sum()) == pytest.approx(
        1.0, abs=1e-5
    )

"""Checks dataset preparation and the object-identification lesson."""

import contextlib
import importlib.util
import io
import subprocess
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from scripts import clean_data

try:
    from PointCloudForML import modelnet_data
except ImportError:
    modelnet_data = None

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
OFF = "OFF\n4 4 0\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n3 0 1 2\n3 0 1 3\n3 0 2 3\n3 1 2 3\n"


def fixture_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for category in CATEGORIES:
            for split, count in [("train", 4), ("test", 2)]:
                for index in range(count):
                    archive.writestr(
                        f"ModelNet10/{category}/{split}/{category}_{index:04}.off", OFF
                    )


class ModelNetDemoTests(unittest.TestCase):
    def test_preparation_downloads_extracts_and_reuses_a_complete_cache(self):
        self.assertIsNotNone(modelnet_data, "Dataset preparation is not implemented")
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            fixture_archive(root / "source.zip")
            result = modelnet_data.prepare_modelnet10(
                root / "data", url=(root / "source.zip").as_uri()
            )
            self.assertEqual(result.name, "ModelNet10")
            self.assertTrue((result / "chair" / "test" / "chair_0000.off").is_file())
            self.assertTrue(modelnet_data.modelnet10_ready(root / "data"))
            self.assertEqual(
                modelnet_data.prepare_modelnet10(
                    root / "data", url="https://invalid.example/no-network-needed"
                ),
                result,
            )

    def test_preparation_rejects_archive_paths_outside_the_cache(self):
        self.assertIsNotNone(modelnet_data, "Dataset preparation is not implemented")
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with zipfile.ZipFile(root / "bad.zip", "w") as archive:
                archive.writestr("../outside.txt", "wrong location")
            with self.assertRaises(ValueError):
                modelnet_data.prepare_modelnet10(
                    root / "data", url=(root / "bad.zip").as_uri()
                )
            self.assertFalse((root / "outside.txt").exists())
            self.assertFalse(modelnet_data.modelnet10_ready(root / "data"))

    def test_cleanup_registry_covers_the_local_download_cache(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            subprocess.run(["git", "init", "--quiet", str(root)], check=True)
            cache = root / "PointCloudForML" / "data"
            cache.mkdir(parents=True)
            (cache / "ModelNet10.zip").write_bytes(b"archive")
            outputs = root / "PointCloudForML" / "outputs"
            outputs.mkdir()
            (outputs / "modelnet10_classifier.pt").write_bytes(b"weights")
            previous_root = clean_data.REPO_ROOT
            try:
                clean_data.REPO_ROOT = root
                targets = clean_data.find_targets()
                self.assertEqual([path for path, _ in targets], [cache])
                self.assertEqual(targets[0][1], len(b"archive"))
                subprocess.run(
                    ["git", "add", "PointCloudForML/data/ModelNet10.zip"],
                    cwd=root,
                    check=True,
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(clean_data.find_targets(), [])
                self.assertTrue((outputs / "modelnet10_classifier.pt").exists())
            finally:
                clean_data.REPO_ROOT = previous_root

    def test_identification_notebook_runs_on_an_off_mesh_fixture(self):
        path = Path(__file__).parent / "PointCloudForMLPart7ModelNet.py"
        self.assertTrue(path.exists(), "The object-identification notebook is missing")
        self.assertIsNotNone(modelnet_data)
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                fixture_archive(root / "source.zip")
                modelnet_data.prepare_modelnet10(
                    root / "data", url=(root / "source.zip").as_uri()
                )
                spec = importlib.util.spec_from_file_location(path.stem, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                with contextlib.redirect_stdout(io.StringIO()):
                    _, definitions = module.app.run(
                        defs={
                            "dataset_cache": root / "data",
                            "training_form": SimpleNamespace(
                                value=dict(
                                    points=32,
                                    limit=4,
                                    test_limit=2,
                                    epochs=1,
                                    lr=0.003,
                                    seed=42,
                                )
                            ),
                        }
                    )
                self.assertEqual(definitions["class_names"], list(CATEGORIES))
                self.assertEqual(definitions["test_logits"].shape, (20, 10))
                self.assertEqual(
                    [trace.type for trace in definitions["identification_view"].data],
                    ["mesh3d", "scatter3d"],
                )
                self.assertEqual(len(definitions["test_sources"]), 20)
                self.assertEqual(
                    definitions["test_sources"][0]["path"].parent.name, "test"
                )
                self.assertAlmostEqual(
                    float(definitions["identification_probabilities"].sum()),
                    1.0,
                    places=5,
                )
        finally:
            torch.set_num_threads(previous)


if __name__ == "__main__":
    unittest.main()

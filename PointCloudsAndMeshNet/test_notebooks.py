"""Runs the lesson cells, including the submitted generated-data forms."""

import contextlib
import importlib.util
import io
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def run_lesson(self, part: int, overrides: dict | None = None) -> dict:
        path = next(Path(__file__).parent.glob(f"PointCloudForMLPart{part}*.py"))
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with contextlib.redirect_stdout(io.StringIO()):
            _, definitions = module.app.run(defs=overrides)
        return definitions

    def test_all_six_default_lessons_run_without_training_or_optional_data(self):
        for part in range(1, 7):
            with self.subTest(part=part):
                definitions = self.run_lesson(part)
                self.assertNotIn("trained_model", definitions)
                self.assertNotIn("segmentation_model", definitions)
                self.assertNotIn("comparison_models", definitions)

    def test_classifier_learns_and_keeps_held_out_output_dimensions(self):
        settings = dict(root="", points=128, limit=24, epochs=12, lr=0.003, seed=42)
        definitions = self.run_lesson(
            4, {"training_form": SimpleNamespace(value=settings)}
        )
        history = definitions["history"]
        self.assertLess(history[-1]["train_loss"], history[0]["train_loss"])
        self.assertEqual(definitions["test_logits"].shape, (36, 3))
        self.assertTrue(torch.isfinite(definitions["test_logits"]).all())
        expected = torch.bincount(
            definitions["test_y"] * 3 + definitions["predictions"], minlength=9
        ).reshape(3, 3)
        np.testing.assert_array_equal(
            definitions["confusion_plot"].data[0].z, expected.numpy()
        )
        self.assertEqual(
            definitions["confusion_plot"].layout.yaxis.autorange, "reversed"
        )

    def test_robustness_comparison_preserves_shuffle_accuracy(self):
        settings = dict(root="", points=128, limit=24, epochs=30, lr=0.003, seed=42)
        definitions = self.run_lesson(
            5, {"training_form": SimpleNamespace(value=settings)}
        )
        rows = definitions["robustness_rows"]
        for model in definitions["comparison_models"]:
            clean = next(
                row
                for row in rows
                if row["model"] == model and row["condition"] == "Clean"
            )
            shuffled = next(
                row
                for row in rows
                if row["model"] == model and row["condition"] == "Shuffle"
            )
            self.assertEqual(clean["accuracy"], shuffled["accuracy"])
        self.assertTrue(all(0 <= row["accuracy"] <= 1 for row in rows))
        for trace in definitions["robustness_plot"].data:
            condition = "Noise" if trace.xaxis == "x" else "Points"
            expected = [
                row
                for row in rows
                if row["model"] == trace.name and row["condition"] == condition
            ]
            np.testing.assert_allclose(
                trace.error_y.array, [row["std"] for row in expected]
            )
            np.testing.assert_allclose(trace.y, [row["accuracy"] for row in expected])

    def test_sampling_view_uses_matching_mesh_faces_and_cloud_coordinates(self):
        try:
            import plotly  # noqa: F401
            import trimesh  # noqa: F401
        except ImportError:
            self.skipTest(
                "Run with --with plotly --with trimesh to check the mesh views"
            )
        definitions = self.run_lesson(
            1, {"mesh_form": SimpleNamespace(value={"path": "", "count": 256})}
        )
        self.assertIn("mesh_comparison", definitions)
        self.assertIn("mesh_overlay", definitions)
        import numpy as np

        mesh = definitions["mesh"]
        centre = definitions["mesh_points"].mean(axis=0)
        radius = np.linalg.norm(definitions["mesh_points"] - centre, axis=1).max()
        np.testing.assert_allclose(
            definitions["display_vertices"], (mesh.vertices - centre) / radius
        )
        np.testing.assert_allclose(
            definitions["display_points"],
            (definitions["mesh_points"] - centre) / radius,
        )
        for figure in (definitions["mesh_comparison"], definitions["mesh_overlay"]):
            self.assertEqual(
                [trace.type for trace in figure.data], ["mesh3d", "scatter3d"]
            )
            np.testing.assert_array_equal(figure.data[0].i, mesh.faces[:, 0])
            np.testing.assert_allclose(
                figure.data[1].x, definitions["display_points"][:, 0]
            )
            self.assertEqual(figure.layout.scene.dragmode, "orbit")

    def test_feature_channels_and_contributors_are_plotly_views(self):
        definitions = self.run_lesson(3)
        self.assertIn("feature_plot", definitions)
        self.assertEqual(
            [trace.type for trace in definitions["feature_plot"].data],
            ["scatter3d"] * 3,
        )
        self.assertTrue(definitions["feature_plot"].data[2].marker.showscale)

    def test_segmentation_loss_demo_runs_before_training(self):
        definitions = self.run_lesson(6)
        self.assertIn("point_losses", definitions)
        torch.testing.assert_close(
            definitions["point_losses"], definitions["manual_point_losses"]
        )
        torch.testing.assert_close(
            definitions["point_losses"].mean(), definitions["flattened_point_loss"]
        )

    def test_segmentation_checkpoint_reloads_the_same_predictions(self):
        settings = dict(root="", points=192, limit=24, epochs=2, lr=0.003, seed=42)
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "segmentation.pt"
            definitions = self.run_lesson(
                6,
                {
                    "segmentation_form": SimpleNamespace(value=settings),
                    "seg_checkpoint_path": SimpleNamespace(value=str(destination)),
                    "seg_save_button": SimpleNamespace(value=True),
                },
            )
            self.assertTrue(destination.is_file())
            checkpoint = torch.load(destination, weights_only=True)
            self.assertEqual(checkpoint["parts"], definitions["seg_names"])
            with torch.no_grad():
                torch.testing.assert_close(
                    definitions["reloaded_segmentation_model"](
                        definitions["seg_test_x"]
                    ),
                    definitions["segmentation_model"](definitions["seg_test_x"]),
                )

    def test_segmentation_learns_aligned_point_labels(self):
        settings = dict(root="", points=192, limit=24, epochs=12, lr=0.003, seed=42)
        definitions = self.run_lesson(
            6, {"segmentation_form": SimpleNamespace(value=settings)}
        )
        self.assertEqual(
            definitions["seg_predictions"].shape, definitions["seg_test_y"].shape
        )
        self.assertGreater(definitions["seg_ious"].mean().item(), 0.7)
        view = definitions["segmentation_plot"]
        for scene in (view.layout.scene, view.layout.scene2, view.layout.scene3):
            self.assertEqual(scene.dragmode, "orbit")
            self.assertEqual(scene.xaxis.range, (-1.1, 1.1))
        self.assertEqual(
            [trace.name for trace in view.data[:3]], definitions["seg_names"]
        )
        self.assertEqual(
            [trace.name for trace in view.data[-2:]], ["Correct", "Incorrect"]
        )
        self.assertTrue(all(trace.type == "scatter3d" for trace in view.data))


if __name__ == "__main__":
    unittest.main()

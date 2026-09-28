import importlib.util
import unittest
from pathlib import Path


class NotebookTests(unittest.TestCase):
    def run_notebook(self, suffix: str) -> dict:
        """Execute the notebook, including its generated media and plots."""
        path = Path(__file__).resolve().parents[1] / f"TorchCodecForML{suffix}.py"
        self.assertTrue(path.exists(), f"Missing notebook: {path.name}")
        spec = importlib.util.spec_from_file_location(suffix, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _, definitions = module.app.run()
        self.addCleanup(definitions["media_directory"].cleanup)
        return definitions

    def test_video_pixels_and_presentation_times(self) -> None:
        d = self.run_notebook("Part1Video")
        self.assertEqual(len(d["decoder"]), 24)
        self.assertEqual(tuple(d["frames"].data.shape), (4, 3, 64, 96))
        self.assertTrue(d["torch"].equal(d["decoded"], d["source_frames"]))
        self.assertTrue(d["torch"].equal(d["from_bytes"], d["decoded"][0]))
        for requested, pts, duration in zip(
            d["requested_times"],
            d["timed_frames"].pts_seconds,
            d["timed_frames"].duration_seconds,
        ):
            self.assertLessEqual(float(pts), requested)
            self.assertLess(requested, float(pts + duration) + 1e-6)

    def test_sampling_and_end_policies(self) -> None:
        d = self.run_notebook("Part2Sampling")
        self.assertEqual(tuple(d["regular"].data.shape), (3, 4, 3, 64, 96))
        self.assertTrue(d["torch"].equal(d["random_a"].data, d["random_b"].data))
        self.assertTrue(
            d["torch"].equal(d["repeat"].data[0, -1], d["repeat"].data[0, -2])
        )
        self.assertTrue(
            d["torch"].equal(d["wrapped"].data[0, 0], d["wrapped"].data[0, 2])
        )
        self.assertTrue(d["policy_error"])
        self.assertEqual(tuple(d["model_input"].shape), (3, 3, 4, 32, 48))
        self.assertGreaterEqual(float(d["model_input"].min()), 0)
        self.assertLessEqual(float(d["model_input"].max()), 1)

    def test_audio_roundtrip_resampling_and_ranges(self) -> None:
        d = self.run_notebook("Part3Audio")
        self.assertEqual(tuple(d["samples"].data.shape), (2, 32000))
        self.assertLess(d["roundtrip_error"], 1e-4)
        self.assertEqual(d["resampled"].sample_rate, 8000)
        self.assertEqual(tuple(d["resampled"].data.shape), (1, 16000))
        self.assertEqual(tuple(d["segment"].data.shape), (2, 8000))
        self.assertAlmostEqual(d["segment"].pts_seconds, 0.5, places=4)
        self.assertTrue(
            d["torch"].allclose(d["segment"].data, d["samples"].data[:, 8000:16000])
        )
        self.assertLess(d["flac_error"], 1e-4)

    def test_dataset_batches_and_model(self) -> None:
        d = self.run_notebook("Part4Datasets")
        self.assertEqual(tuple(d["batch_clips"].shape), (4, 3, 4, 32, 48))
        self.assertEqual(tuple(d["logits"].shape), (4, 2))
        self.assertTrue(d["torch"].isfinite(d["logits"]).all())
        self.assertTrue(set(d["train_paths"]).isdisjoint(d["test_paths"]))
        self.assertEqual(d["batch_labels"].tolist(), [0, 0, 1, 1])
        self.assertTrue(d["torch"].equal(d["dataset"][0][0], d["dataset"][0][0]))
        self.assertGreater(d["gradient_norm"], 0)
        self.assertLess(d["final_loss"], d["initial_loss"])


if __name__ == "__main__":
    unittest.main()

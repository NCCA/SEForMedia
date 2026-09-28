import importlib.util
from pathlib import Path
import unittest


class NotebookTests(unittest.TestCase):
    def run_notebook(self, suffix: str):
        path = Path(__file__).resolve().parents[1] / f"TorchAudioForML{suffix}.py"
        self.assertTrue(path.exists(), f"Missing notebook: {path.name}")
        spec = importlib.util.spec_from_file_location(suffix, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _, definitions = module.app.run()
        return definitions

    def test_waveforms(self):
        d = self.run_notebook("Part1Waveforms")
        self.assertLess(d["roundtrip_error"], 1 / 32768)
        self.assertEqual(d["resampled"].shape[-1], 8000)
        self.assertEqual(d["waveform"].shape, (2, 16000))

    def test_features(self):
        d = self.run_notebook("Part2Features")
        self.assertEqual(d["mel_power"].shape[-2], 64)
        self.assertEqual(d["mfcc"].shape[-2], 13)
        self.assertLess(d["reconstruction_error"], 1e-5)
        self.assertTrue(d["torch"].isfinite(d["mel_db"]).all())

    def test_augmentation(self):
        d = self.run_notebook("Part3Augmentation")
        self.assertAlmostEqual(d["measured_snr"], 15.0, places=3)
        self.assertTrue(d["torch"].equal(d["clean_power"], d["original_power"]))
        self.assertTrue((d["masked_power"] == 0).any())
        self.assertEqual(d["noisy"].shape, d["clean"].shape)

    def test_dataset_and_classifier(self):
        d = self.run_notebook("Part4Datasets")
        self.assertGreaterEqual(d["test_accuracy"], 0.9)
        self.assertLess(d["loss_history"][-1], d["loss_history"][0])
        self.assertTrue(d["torch"].isfinite(d["batch_features"]).all())
        for row, length in zip(d["padded"], d["lengths"]):
            self.assertTrue((row[int(length) :] == 0).all())
        self.assertEqual(d["restored_predictions"].tolist(), d["predictions"].tolist())


if __name__ == "__main__":
    unittest.main()

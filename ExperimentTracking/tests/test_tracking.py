import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

from mlflow.tracking import MlflowClient
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    assert (ROOT / f"{name}.py").exists(), f"Missing demo: {name}"
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrackingTests(unittest.TestCase):
    def test_notebooks_run_from_a_fresh_directory(self):
        for name in ("TensorBoardMarimo", "MLflowMarimo"):
            with (
                self.subTest(notebook=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                module = load(name)
                previous = Path.cwd()
                try:
                    os.chdir(directory)
                    _, definitions = module.app.run()
                finally:
                    os.chdir(previous)
                self.assertLess(definitions["result"]["loss"], 0.00001)
                self.assertAlmostEqual(definitions["result"]["weight"], 2.0, places=2)

    def test_tensorboard_records_loss_and_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            result = load("tensorboard_demo").run_experiment(Path(directory), 0.1)
            events = EventAccumulator(str(result["path"])).Reload()
            losses = events.Scalars("loss/train")
            self.assertEqual(len(losses), 101)
            self.assertLess(losses[-1].value, losses[0].value / 1000)
            self.assertAlmostEqual(result["weight"], 2.0, places=2)
            self.assertIn("model/weight", events.Tags()["scalars"])

    def test_mlflow_records_parameters_history_and_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            module = load("mlflow_demo")
            result = module.run_experiment(Path(directory), 0.1)
            client = MlflowClient(tracking_uri=result["tracking_uri"])
            run = client.get_run(result["run_id"])
            history = client.get_metric_history(result["run_id"], "loss")
            self.assertEqual(run.info.status, "FINISHED")
            self.assertEqual(float(run.data.params["learning_rate"]), 0.1)
            self.assertEqual(len(history), 101)
            self.assertLess(history[-1].value, history[0].value / 1000)
            self.assertEqual(
                client.list_artifacts(result["run_id"])[0].path, "line.json"
            )


if __name__ == "__main__":
    unittest.main()

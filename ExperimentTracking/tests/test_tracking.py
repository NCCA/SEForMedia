import importlib.util
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    assert (ROOT / f"{name}.py").exists(), f"Missing demo: {name}"
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["TensorBoardMarimo", "MLflowMarimo"])
def test_notebooks_run_from_a_fresh_directory(name, tmp_path, monkeypatch):
    module = load(name)
    monkeypatch.chdir(tmp_path)
    _, definitions = module.app.run()
    assert definitions["result"]["loss"] < 0.00001
    assert definitions["result"]["weight"] == pytest.approx(2.0, abs=0.005)


def test_tensorboard_records_loss_and_parameters(tmp_path):
    result = load("tensorboard_demo").run_experiment(tmp_path, 0.1)
    events = EventAccumulator(str(result["path"])).Reload()
    losses = events.Scalars("loss/train")
    assert len(losses) == 101
    assert losses[-1].value < losses[0].value / 1000
    assert result["weight"] == pytest.approx(2.0, abs=0.005)
    assert "model/weight" in events.Tags()["scalars"]


def test_mlflow_records_parameters_history_and_artifact(tmp_path):
    result = load("mlflow_demo").run_experiment(tmp_path, 0.1)
    client = MlflowClient(tracking_uri=result["tracking_uri"])
    run = client.get_run(result["run_id"])
    history = client.get_metric_history(result["run_id"], "loss")
    assert run.info.status == "FINISHED"
    assert float(run.data.params["learning_rate"]) == 0.1
    assert len(history) == 101
    assert history[-1].value < history[0].value / 1000
    assert client.list_artifacts(result["run_id"])[0].path == "line.json"

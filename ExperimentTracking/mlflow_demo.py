import argparse
from pathlib import Path

import mlflow
import numpy as np


def run_experiment(root: Path, learning_rate: float = 0.1) -> dict:
    """
    Fit a line and record its parameters, loss history and fitted values.

    Parameters
    ----------
    root : Path
        Directory for our SQLite database and artifacts.
    learning_rate : float
        Size of each gradient descent step.
    """
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{root / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name("fit-a-line")
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else mlflow.create_experiment(
            "fit-a-line", artifact_location=(root / "artifacts").as_uri()
        )
    )
    x = np.linspace(-1.0, 1.0, 21)
    y = 2.0 * x + 1.0
    weight, bias = 0.0, 0.0
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=f"lr-{learning_rate}"
    ) as run:
        mlflow.log_params(
            {"learning_rate": learning_rate, "steps": 100, "samples": len(x)}
        )
        for step in range(101):
            error = weight * x + bias - y
            loss = float(np.mean(error**2))
            mlflow.log_metric("loss", loss, step=step)
            if step < 100:
                weight -= learning_rate * float(2.0 * np.mean(error * x))
                bias -= learning_rate * float(2.0 * np.mean(error))
        mlflow.log_dict({"weight": weight, "bias": bias}, "line.json")
        run_id = run.info.run_id
    return {
        "tracking_uri": tracking_uri,
        "run_id": run_id,
        "loss": loss,
        "weight": weight,
        "bias": bias,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit y = 2x + 1 with MLflow logging")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument(
        "--output", type=Path, default=Path("ExperimentTracking/output/mlflow")
    )
    args = parser.parse_args()
    print(run_experiment(args.output, args.learning_rate))

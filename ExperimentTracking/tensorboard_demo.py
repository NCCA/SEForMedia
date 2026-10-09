import argparse
from pathlib import Path
from uuid import uuid4

import numpy as np
from torch.utils.tensorboard import SummaryWriter


def run_experiment(root: Path, learning_rate: float = 0.1) -> dict:
    """
    Fit a line and record the loss, weight and bias in TensorBoard.

    Parameters
    ----------
    root : Path
        Directory for our event files.
    learning_rate : float
        Size of each gradient descent step.
    """
    run_path = root.resolve() / f"lr-{learning_rate}-{uuid4().hex[:8]}"
    x = np.linspace(-1.0, 1.0, 21)
    y = 2.0 * x + 1.0
    weight, bias = 0.0, 0.0
    with SummaryWriter(log_dir=str(run_path)) as writer:
        for step in range(101):
            error = weight * x + bias - y
            loss = float(np.mean(error**2))
            writer.add_scalar("loss/train", loss, global_step=step)
            writer.add_scalar("model/weight", weight, global_step=step)
            writer.add_scalar("model/bias", bias, global_step=step)
            if step < 100:
                weight -= learning_rate * float(2.0 * np.mean(error * x))
                bias -= learning_rate * float(2.0 * np.mean(error))
    return {"path": run_path, "loss": loss, "weight": weight, "bias": bias}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit y = 2x + 1 with TensorBoard logging"
    )
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument(
        "--output", type=Path, default=Path("ExperimentTracking/output/tensorboard")
    )
    args = parser.parse_args()
    print(run_experiment(args.output, args.learning_rate))

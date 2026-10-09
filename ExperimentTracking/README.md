# Experiment tracking

I use these examples to demonstrate TensorBoard and MLflow, both as standalone tools and within marimo. We fit `y = 2x + 1` with NumPy so we can check the result without downloading a dataset.

Run from the repository root, or the root of this worktree. It is expected you will use `uv` and the flag `--with` which supplies the tracking package without adding it to the project dependencies (see [uv docs](https://docs.astral.sh/uv/guides/dependency-management/#supplying-dependencies))

| Tool                                                            | Notebook                                     | Standalone script                          |
| --------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------ |
| [TensorBoard](https://docs.pytorch.org/docs/stable/tensorboard) | [TensorBoardMarimo.py](TensorBoardMarimo.py) | [tensorboard_demo.py](tensorboard_demo.py) |
| [MLflow](https://mlflow.org/docs/latest/ml/tracking/)           | [MLflowMarimo.py](MLflowMarimo.py)           | [mlflow_demo.py](mlflow_demo.py)           |

```bash
uv run --with tensorboard marimo edit ExperimentTracking/TensorBoardMarimo.py
uv run --with mlflow marimo edit ExperimentTracking/MLflowMarimo.py
```

Each notebook explains the logging calls, includes a learning-rate control, and prints a viewer command using the same output location. Start the viewer in another terminal, then open its link or show the dashboard inside the notebook. These localhost examples expect the notebook, viewer and browser to run on the same computer.

## Standalone examples

```bash
uv run --with tensorboard python ExperimentTracking/tensorboard_demo.py
uv run --with tensorboard tensorboard --logdir ExperimentTracking/output/tensorboard --host 127.0.0.1 --port 6006
```

Open [TensorBoard](http://localhost:6006). Each script call writes a new event directory.

```bash
uv run --with mlflow python ExperimentTracking/mlflow_demo.py
uv run --with mlflow mlflow server --backend-store-uri sqlite:///ExperimentTracking/output/mlflow/mlflow.db --host 127.0.0.1 --port 5000
```

Open [MLflow](http://localhost:5000). Each script call creates a run in `fit-a-line`. Both scripts accept `--learning-rate` and `--output`. Press Ctrl+C in the viewer terminal when finished.

Generated data stays in `ExperimentTracking/output/` and is ignored by Git. The notebooks run once on opening; further experiments use the form's submit button. MLflow records the learning rate as a parameter, loss as a metric, and the fitted line as a JSON artifact. The notebook's server command adds `--x-frame-options NONE` to allow the local iframe; the standalone command keeps the default framing restriction.

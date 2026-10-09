#!/usr/bin/env -S uv run marimo edit

# ruff: noqa: B018, PLR1711

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MLflow: recording a training experiment

    I will use a small line-fitting example to show how we record an experiment with
    [MLflow](https://mlflow.org/docs/latest/ml/tracking/).
    We will fit `y = 2x + 1`, so we already know the weight and bias we want.
    We will run it as a Python script first, then run the same calculation here in marimo.

    MLflow groups parameters, metrics and artifacts into runs. We will keep the learning rate, loss history and a JSON file containing our fitted line.
    No dataset download or GPU is needed. The gradient descent calculation uses NumPy;
    MLflow uses a local SQLite database.
    The gradient calculation follows the [autograd lesson](https://github.com/NCCA/SEForMedia/blob/main/PyTorchForML/PyTorchForMLPart2Autograd.py),
    though here we calculate the two gradients by hand.

    ## Run it on its own

    Run these commands from the repository root (or this worktree root).
    `uv --with` supplies the extra package for the command.

    ```bash
    uv run --with mlflow python ExperimentTracking/mlflow_demo.py --learning-rate 0.1
    uv run --with mlflow python ExperimentTracking/mlflow_demo.py --learning-rate 0.02
    ```

    Each call creates a new run in the `fit-a-line` experiment under `ExperimentTracking/output/mlflow`.
    The standalone script accepts `--output` if we want to store results elsewhere.
    Keep the viewer pointed at that same location.

    We can open this notebook with:

    ```bash
    uv run --with mlflow marimo edit ExperimentTracking/MLflowMarimo.py
    ```
    """)
    return


@app.cell
def _():
    import shlex
    from pathlib import Path

    import matplotlib.pyplot as plt
    import mlflow
    import numpy as np

    return Path, mlflow, np, plt, shlex


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What are we recording?

    We start with weight and bias both zero, using 21 evenly spaced samples between
    -1 and 1. Our loss is the mean squared error, `mean((weight * x + bias - y)**2)`.
    For this data the starting loss is about 2.467. It should fall whilst the weight
    approaches 2 and the bias approaches 1.

    We calculate both gradients from the same prediction error before updating the
    parameters. Step 0 is before training; step 100 is after 100 updates.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### [Parameters, metrics and artifacts](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html)

    | Call | What we record | Why we use it |
    | --- | --- | --- |
    | `mlflow.log_params(...)` | Learning rate, update count, sample count | Settings fixed for one run. |
    | `mlflow.log_metric("loss", loss, step=step)` | Loss at each step | A value that changes during training. |
    | `mlflow.log_dict(..., "line.json")` | Fitted weight and bias | A file we can inspect or download. |

    The JSON file describes our line; it is not a registered MLflow model.
    The `start_run` context manager finishes the run when training ends.
    """)
    return


@app.cell
def _(Path, mlflow, np):
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

    return (run_experiment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run it within marimo

    Change the learning rate and press **Run experiment**. Each submission creates a
    new run, including when we submit the same learning rate again. The default
    settings run once when the notebook opens. The calculation is small, but batching
    the control means dragging the slider does not create lots of unwanted runs.
    """)
    return


@app.cell
def _(mo):
    settings = mo.ui.batch(
        mo.md("Learning rate: {learning_rate}"),
        {
            "learning_rate": mo.ui.slider(
                0.01, 0.3, step=0.01, value=0.1, show_value=True
            )
        },
    ).form(submit_button_label="Run experiment")
    settings
    return (settings,)


@app.cell
def _(Path, run_experiment, settings):
    learning_rate = (settings.value or {"learning_rate": 0.1})["learning_rate"]
    output_root = Path("ExperimentTracking/output/mlflow").resolve()
    result = run_experiment(output_root, learning_rate)
    print(f"Final loss: {result['loss']:.8f}")
    print(f"Weight: {result['weight']:.4f} (wanted 2.0)")
    print(f"Bias: {result['bias']:.4f} (wanted 1.0)")
    return output_root, result


@app.cell
def _(mlflow, mo, result):
    print("Tracking URI:", result["tracking_uri"])
    print("Run ID:", result["run_id"])
    client = mlflow.tracking.MlflowClient(tracking_uri=result["tracking_uri"])
    loss_history = [
        (metric.step, metric.value)
        for metric in client.get_metric_history(result["run_id"], "loss")
    ]
    runs = client.search_runs([client.get_run(result["run_id"]).info.experiment_id])
    mo.ui.table(
        [
            {
                "run_id": run.info.run_id,
                "learning_rate": run.data.params["learning_rate"],
                "final_loss": run.data.metrics["loss"],
            }
            for run in runs
        ],
        selection=None,
    )
    return (loss_history,)


@app.cell
def _(loss_history, plt):
    figure, axes = plt.subplots()
    axes.plot(
        [step for step, loss in loss_history],
        [loss for step, loss in loss_history],
        label="Training loss",
    )
    axes.set(
        xlabel="Update step", ylabel="Mean squared error", title="Fitting y = 2x + 1"
    )
    axes.legend()
    plt.close(figure)
    figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Open the dashboard

    Run the command below in a second terminal. It uses the absolute path selected
    above, so we can launch the viewer from another directory. Leave it running whilst
    we inspect the results; press Ctrl+C in that terminal when we finish.

    Open [http://localhost:5000](http://localhost:5000) to use the dashboard on its
    own, or tick the box below to show it here. The notebook and viewer must run on
    the same computer as our browser for these localhost links to work.

    In MLflow, open `fit-a-line`, select two runs and compare their parameters and loss. Open a run to inspect `line.json` under Artifacts.

    MLflow normally restricts iframe embedding. The command uses `--x-frame-options NONE` for this local teaching example, with the server bound to loopback. Keep the default setting for a shared server.
    """)
    return


@app.cell(hide_code=True)
def _(mo, output_root, result, shlex):
    server_command = shlex.join(
        [
            "uv",
            "run",
            "--with",
            "mlflow",
            "mlflow",
            "server",
            "--backend-store-uri",
            result["tracking_uri"],
            "--host",
            "127.0.0.1",
            "--port",
            "5000",
            "--x-frame-options",
            "NONE",
        ]
    )
    mo.md(f"```bash\n{server_command}\n```")
    return


@app.cell
def _(mo):
    show_dashboard = mo.ui.checkbox(
        label="Show the running MLflow dashboard", value=False
    )
    show_dashboard
    return (show_dashboard,)


@app.cell(hide_code=True)
def _(mo, show_dashboard):
    mo.Html(
        '<iframe src="http://localhost:5000" title="MLflow dashboard" width="100%" height="650" style="border: 1px solid #aaa;"></iframe>'
    ) if show_dashboard.value else mo.md(
        "Start the viewer in a terminal, then tick the box above."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Things to check

    If a run is missing, compare the printed database URI with the viewer command. Two relative SQLite paths launched from different working directories can refer to different databases. We use an absolute path to avoid this. A parameter is fixed within a run: if we want another learning rate, we start another run.

    A blank embedded view usually means the viewer is not running yet, its port is in
    use, or the browser has refused the iframe. The Codex in-app browser may block
    local embedded dashboards; use the notebook in a normal browser in that case. Try the standalone link first. If we
    change the port in the terminal, change the link and iframe source as well.

    ## Exercises

    1. Compare learning rates 0.02 and 0.2. Which reaches a small loss in fewer updates?
    2. Change the target to `y = 3x - 2`. Predict the final weight and bias before running it.
    3. Log the final weight and bias as metrics as well as an artifact. Find them in the run comparison.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

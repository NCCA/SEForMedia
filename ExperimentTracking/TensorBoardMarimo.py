#!/usr/bin/env -S uv run marimo edit

# ruff: noqa: B018, PLR1711

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TensorBoard: recording a training experiment

    I will use a small line-fitting example to show how we record an experiment with
    [TensorBoard](https://docs.pytorch.org/docs/stable/tensorboard).
    We will fit `y = 2x + 1`, so we already know the weight and bias we want.
    We will run it as a Python script first, then run the same calculation here in marimo.

    TensorBoard plots values from event files. We will record loss, weight and bias at each step.
    No dataset download or GPU is needed. The gradient descent calculation uses NumPy;
    PyTorch supplies the TensorBoard writer.
    The gradient calculation follows the [autograd lesson](https://github.com/NCCA/SEForMedia/blob/main/PyTorchForML/PyTorchForMLPart2Autograd.py),
    though here we calculate the two gradients by hand.

    ## Run it on its own

    Run these commands from the repository root (or this worktree root).
    `uv --with` supplies the extra package for the command.

    ```bash
    uv run --with tensorboard python ExperimentTracking/tensorboard_demo.py --learning-rate 0.1
    uv run --with tensorboard python ExperimentTracking/tensorboard_demo.py --learning-rate 0.02
    ```

    Each call writes a separate event directory under `ExperimentTracking/output/tensorboard`.
    The standalone script accepts `--output` if we want to store results elsewhere.
    Keep the viewer pointed at that same location.

    We can open this notebook with:

    ```bash
    uv run --with tensorboard marimo edit ExperimentTracking/TensorBoardMarimo.py
    ```
    """)
    return


@app.cell
def _():
    import shlex
    from pathlib import Path
    from uuid import uuid4

    import matplotlib.pyplot as plt
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter

    return Path, SummaryWriter, np, plt, uuid4, shlex


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
    ### [SummaryWriter.add_scalar](https://docs.pytorch.org/docs/stable/tensorboard)

    ```python
    writer.add_scalar(tag, scalar_value, global_step)
    ```

    | Parameter | What we pass | What it does |
    | --- | --- | --- |
    | `tag` | `"loss/train"` | Names the plot; slashes group related plots. |
    | `scalar_value` | `loss` | Records a number. For a PyTorch loss use `loss.item()`. |
    | `global_step` | `step` | Places the value on the horizontal axis. |

    The writer buffers data. Its context manager closes it and flushes the remaining
    events, so the last values are available when we open the viewer.
    """)
    return


@app.cell
def _(Path, SummaryWriter, np, uuid4):
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
    output_root = Path("ExperimentTracking/output/tensorboard").resolve()
    result = run_experiment(output_root, learning_rate)
    print(f"Final loss: {result['loss']:.8f}")
    print(f"Weight: {result['weight']:.4f} (wanted 2.0)")
    print(f"Bias: {result['bias']:.4f} (wanted 1.0)")
    return output_root, result


@app.cell
def _(result):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(result["path"])).Reload()
    loss_history = [(event.step, event.value) for event in events.Scalars("loss/train")]
    print("Event directory:", result["path"])
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

    Open [http://localhost:6006](http://localhost:6006) to use the dashboard on its
    own, or tick the box below to show it here. The notebook and viewer must run on
    the same computer as our browser for these localhost links to work.

    In TensorBoard, open Scalars and select both learning-rate runs. Turn smoothing to zero to see the values we actually recorded.

    The dashboard reads the same event files as the plot above.
    """)
    return


@app.cell(hide_code=True)
def _(mo, output_root, result, shlex):
    server_command = shlex.join(
        [
            "uv",
            "run",
            "--with",
            "tensorboard",
            "tensorboard",
            "--logdir",
            str(output_root),
            "--host",
            "127.0.0.1",
            "--port",
            "6006",
        ]
    )
    mo.md(f"```bash\n{server_command}\n```")
    return


@app.cell
def _(mo):
    show_dashboard = mo.ui.checkbox(
        label="Show the running TensorBoard dashboard", value=False
    )
    show_dashboard
    return (show_dashboard,)


@app.cell(hide_code=True)
def _(mo, show_dashboard):
    mo.Html(
        '<iframe src="http://localhost:6006" title="TensorBoard dashboard" width="100%" height="650" style="border: 1px solid #aaa;"></iframe>'
    ) if show_dashboard.value else mo.md(
        "Start the viewer in a terminal, then tick the box above."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Things to check

    If curves appear to restart or overlap, check that separate experiments have separate log directories. Reusing the same directory and restarting the step counter mixes events. Our UUID suffix gives every call its own directory. If a new run is missing, check the viewer log directory and reload its data.

    A blank embedded view usually means the viewer is not running yet, its port is in
    use, or the browser has refused the iframe. The Codex in-app browser may block
    local embedded dashboards; use the notebook in a normal browser in that case. Try the standalone link first. If we
    change the port in the terminal, change the link and iframe source as well.

    ## Exercises

    1. Compare learning rates 0.02 and 0.2. Which reaches a small loss in fewer updates?
    2. Change the target to `y = 3x - 2`. Predict the final weight and bias before running it.
    3. Add a scalar for the learning rate, then find it in the dashboard.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

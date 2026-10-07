#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Offloading Training to the Cloud (with Podman)

    So far everything we have trained has run inside the notebook on the machine in front of us. That is fine for prototyping, but for real models it doesn't scale: training can take hours on a GPU we don't own, the notebook (and our laptop) is tied up for the whole time, and it is very hard to say exactly which code and settings produced a given set of weights.

    The usual answer is to *offload* training. Services such as [Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html), [Google Vertex AI](https://cloud.google.com/vertex-ai/docs/training/overview) and [Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/) all work in roughly the same way. We package our training code in a [container image](https://docs.podman.io/en/latest/Introduction.html), point it at some data in cloud storage, and they run it on a machine of our choosing, write the results back to storage and shut the machine down. Inference works the same way: another container serving the trained model over HTTP.

    We don't have any cloud credit, but that turns out not to matter very much. The cloud is (mostly) just running containers, so we can run exactly the same workflow on our own machine with [podman](https://podman.io/). This is also what you should do before spending real money, as a job that crashes after an hour on a GPU instance still costs you the hour!

    In this notebook we will

    1. Prototype a small classifier in the notebook (the bit we already know how to do)
    2. Turn the training into a standalone script, `train.py`, and smoke test it locally
    3. Package that script as a training image and run it as a "cloud" job
    4. Collect and check the results
    5. Build a separate, much smaller inference image and deploy it as an endpoint
    6. Call the endpoint like any other client would, then tear it down

    This is the whole ML DevOps (or MLOps) cycle in miniature. The model is deliberately tiny so the focus stays on the process rather than the network.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("## The cycle"),
            mo.mermaid(
                """
    flowchart LR
        A[Prototype in notebook] --> B[Standalone train.py]
        B --> C[Smoke test locally]
        C --> D[Build training image]
        D --> E[Run training job]
        E --> F[(Bucket: model and metrics)]
        F --> G[Check and choose a run]
        G --> H[Build serving image]
        H --> I[Deploy endpoint]
        I --> J[Clients call /predict]
        J -. new data and ideas .-> A
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulating the cloud

    Each part of a cloud ML platform has a simple local stand in. The commands change when you move to a real provider, the shape of the workflow does not.

    | In the cloud | Typical service | Here |
    | --- | --- | --- |
    | Object storage | S3, Google Cloud Storage | a `bucket/` folder next to this notebook, mounted into containers with `-v` |
    | Container registry | ECR, Artifact Registry, ghcr.io | podman's local image store (images tagged `localhost/...`) |
    | Training job | SageMaker training job, Vertex AI custom job | `podman run --rm` with the bucket mounted |
    | Instance type | CPU / GPU machine sizes | `--cpus` and `--memory` limits |
    | Job logs | CloudWatch, Cloud Logging | the container's stdout, `podman logs` |
    | Model registry | SageMaker / Vertex AI model registry | `bucket/runs/<run_id>/` with a `metrics.json` per run |
    | Inference endpoint | SageMaker endpoint, Vertex AI endpoint, Cloud Run | `podman run -d -p 8080:8080` |

    The files for this demo live alongside the notebook

    | File | What it does |
    | --- | --- |
    | `spiralnet/` | the data, model and training loop, shared by the notebook **and** the job |
    | `train.py` | the standalone training job |
    | `Containerfile.train` | builds the training image (PyTorch on CPU) |
    | `serve.py` | the FastAPI inference service |
    | `Containerfile.serve` | builds the inference image (onnxruntime, no PyTorch) |
    | `requirements-*.txt` | pinned dependencies for each image |

    We start with the imports. Note `spiralnet` is imported just like any other package, it is our own code sitting in the folder next to the notebook.
    """)
    return


@app.cell
def _():
    import json
    import os
    import shlex
    import shutil
    import subprocess
    import sys
    import time
    from datetime import datetime
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import onnxruntime as ort
    import requests
    import torch
    from spiralnet import (
        build_model,
        fit,
        make_spiral,
        predict_classes,
        save_dataset,
        train_val_split,
    )

    print(f"torch {torch.__version__}, onnxruntime {ort.__version__}")
    return (
        Path,
        build_model,
        datetime,
        fit,
        json,
        make_spiral,
        np,
        ort,
        os,
        plt,
        predict_classes,
        requests,
        save_dataset,
        shlex,
        shutil,
        subprocess,
        sys,
        time,
        torch,
        train_val_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The container engine

    Everything below drives [podman](https://podman.io/docs/installation) from Python with [`subprocess`](https://docs.python.org/3/library/subprocess.html). On macOS and Windows podman runs containers inside a small Linux virtual machine, so you need to run `podman machine init` and `podman machine start` once before any of this will work. On Linux it runs natively and *rootless*, meaning the containers run as you rather than as root.

    The commands are almost identical in [Docker](https://docs.docker.com/), so if you only have Docker you can select it below. The one difference we have to handle is file ownership, which I explain when we get to it.
    """)
    return


@app.cell
def _(mo, shutil):
    _available = [name for name in ("podman", "docker") if shutil.which(name)]
    engine = mo.ui.dropdown(
        options=_available or ["podman"],
        value=(_available or ["podman"])[0],
        label="Container engine",
    )
    engine
    return (engine,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We are going to run a lot of commands, so a small helper prints each command (so you can copy it into a terminal and try it yourself), streams its output into the notebook as it runs and returns the exit status. Exit statuses matter: `0` means success and anything else is a failure, which is how every cloud service and CI system decides whether your job worked.
    """)
    return


@app.cell
def _(Path, mo, shlex, subprocess, sys):
    HERE = mo.notebook_dir()

    # On macOS podman runs containers in a Linux VM which can only see the folders
    # shared into it, by default /Users, /private and /var/folders. If the notebook
    # lives anywhere else (an external drive under /Volumes for example) a -v mount
    # fails with "statfs ... no such file or directory", so in that case we put the
    # bucket in the home folder instead. On Linux there is no VM, so anywhere works.
    MAC_VM_SHARED = ("/Users/", "/private/", "/var/folders/")
    if sys.platform == "darwin" and not str(HERE.resolve()).startswith(MAC_VM_SHARED):
        BUCKET = Path.home() / "spiral-bucket"
    else:
        BUCKET = HERE / "bucket"
    BUCKET.mkdir(parents=True, exist_ok=True)
    print(f"bucket is {BUCKET}")

    def run(cmd: list[str]) -> int:
        """
        Run a command in the notebook folder, echoing it and streaming its output.

        Parameters
        ----------
            cmd : list[str]
                the command and its arguments

        Returns
        -------
            int
                the exit status of the command
        """
        print("$", shlex.join(cmd), flush=True)
        with subprocess.Popen(
            cmd,
            cwd=HERE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
        print(f"[exit status {process.returncode}]")
        return process.returncode

    return BUCKET, HERE, run


@app.cell
def _(engine, run):
    _status = run([engine.value, "--version"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If the engine isn't installed (or isn't on your `PATH`) `subprocess` doesn't return an exit status at all, it raises a `FileNotFoundError` because there is no program to run. It is worth recognising this one as it looks quite different from a command that ran and failed.
    """)
    return


@app.cell
def _(run):
    try:
        run(["podman", "--version"])  # deliberate typo
    except FileNotFoundError as error:
        print(f"FileNotFoundError: {error}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally some names we use throughout. The `localhost/` prefix on the image names says these images live in our local store rather than a registry such as Docker Hub. The `:0.1` tag is a version, in a real project this is often the git commit hash so every image can be traced back to the exact code that built it.

    There are two platform details hidden in here.

    * On Linux, systems with [SELinux](https://docs.podman.io/en/latest/markdown/podman-run.1.html#volume-v-source-volume-host-dir-container-dir-options) (such as the lab machines) block containers from reading your files unless the volume is relabelled with `:Z`. On macOS the files are shared into the podman virtual machine differently and `:Z` isn't needed.
    * Rootless podman maps root in the container to *you* on the host, so files the training job writes are owned by you. Docker really does run as root, so without `--user` the results would be owned by root and you couldn't delete them!
    """)
    return


@app.cell
def _(engine, os, sys):
    TRAIN_IMAGE = "localhost/spiral-train:0.1"
    SERVE_IMAGE = "localhost/spiral-serve:0.1"
    ENDPOINT_NAME = "spiral-endpoint"
    PORT = 8080

    selinux_label = ",Z" if sys.platform.startswith("linux") else ""
    if engine.value == "docker" and hasattr(os, "getuid"):
        user_flags = ["--user", f"{os.getuid()}:{os.getgid()}"]
    else:
        user_flags = []
    print(f"volume options: rw{selinux_label}, extra user flags: {user_flags}")
    return (
        ENDPOINT_NAME,
        PORT,
        SERVE_IMAGE,
        TRAIN_IMAGE,
        selinux_label,
        user_flags,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part 1: Prototype in the notebook

    ## The data

    We use the classic spiral problem from the [CS231n notes](https://cs231n.github.io/neural-networks-case-study/): three interleaved spiral arms, one per class. No straight line can separate them so it needs a (small) neural network, but it trains in seconds on a CPU.

    The data is generated rather than downloaded, and then saved into `bucket/data/`. This is our equivalent of uploading the dataset to cloud storage, which in real life would be something like `aws s3 cp spiral.npz s3://my-bucket/data/` or `gcloud storage cp`. The training job never sees the notebook's variables, only what is in the bucket.
    """)
    return


@app.cell
def _(BUCKET, make_spiral, np, save_dataset):
    points, labels = make_spiral(points_per_class=300, classes=3, noise=0.2, seed=1234)
    DATA_FILE = save_dataset(BUCKET / "data" / "spiral.npz", points, labels)

    print(f"points {points.shape} {points.dtype}, labels {labels.shape} {labels.dtype}")
    print(f"points per class {np.bincount(labels)}")
    print(f"saved to {DATA_FILE} ({DATA_FILE.stat().st_size / 1024:.1f} KiB)")
    return DATA_FILE, labels, points


@app.cell
def _(labels, plt, points):
    _fig, _ax = plt.subplots(figsize=(4, 4), layout="constrained")
    for _c in range(labels.max() + 1):
        _ax.scatter(*points[labels == _c].T, s=8, label=f"class {_c}")
    _ax.set(title="Spiral dataset", xlabel="x", ylabel="y", aspect="equal")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The model lives in a package, not the notebook

    The single most useful habit for moving work off the notebook is to keep the model and training loop in ordinary `.py` files that the notebook imports. If the model is defined in a notebook cell and *copied* into `train.py` the two will drift apart, and you will eventually spend an afternoon trying to work out why the cloud model doesn't match the one you tested.

    Here `spiralnet/model.py` holds a two hidden layer MLP and `spiralnet/training.py` holds a `fit` function. The training loop itself is the same one we have written before (see the [Linear Model](../LinearModel/LinearModelMarimo.py) notebook), so I won't go through it again. The point is that `fit` doesn't know or care whether it is called from a notebook or a container, it just reports each epoch through a callback.
    """)
    return


@app.cell
def _(build_model, labels, points, torch):
    _model = build_model(hidden=64, classes=3)
    print(_model)
    print(f"parameters: {sum(p.numel() for p in _model.parameters())}")
    with torch.no_grad():
        _logits = _model(torch.from_numpy(points[:4]))
    print(
        f"input {tuple(points[:4].shape)} -> logits {tuple(_logits.shape)}, labels {labels[:4]}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A quick training run

    Before shipping anything we check the idea works at all. This trains on the CPU (for a model this small a GPU would be slower, as copying the data to it costs more than the maths) using the same `train_val_split` and seed the job will use.
    """)
    return


@app.cell
def _(mo):
    notebook_settings = mo.ui.dictionary(
        {
            "epochs": mo.ui.number(start=1, stop=500, value=100, label="Epochs"),
            "learning_rate": mo.ui.number(
                start=0.0001, stop=0.1, step=0.0001, value=0.01, label="Learning rate"
            ),
        }
    ).form(submit_button_label="Train")
    notebook_settings
    return (notebook_settings,)


@app.cell
def _(
    build_model,
    fit,
    labels,
    mo,
    notebook_settings,
    points,
    torch,
    train_val_split,
):
    mo.stop(notebook_settings.value is None, mo.md("Press **Train** above to begin."))

    torch.manual_seed(1234)
    train_data, val_data = train_val_split(points, labels, seed=1234)
    notebook_model = build_model(hidden=64, classes=3)
    _epochs = notebook_settings.value["epochs"]
    with mo.status.progress_bar(total=_epochs, title="Training") as _bar:
        notebook_history = fit(
            notebook_model,
            train_data,
            val_data,
            epochs=_epochs,
            lr=notebook_settings.value["learning_rate"],
            on_epoch=lambda _metrics: _bar.update(),
        )
    print(f"final validation accuracy {notebook_history[-1]['val_accuracy']:.1%}")
    return notebook_history, notebook_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two plotting helpers we will reuse to compare the notebook model, the model from the training job and the deployed endpoint. `plot_decision_boundary` takes a `predict` function rather than a model, so it works the same whether the prediction comes from PyTorch, onnxruntime or an HTTP request.
    """)
    return


@app.cell
def _(np):
    def plot_history(ax, history: list[dict], title: str) -> None:
        """
        Plot training and validation loss from a fit history.

        Parameters
        ----------
            ax : matplotlib.axes.Axes
                axes to draw on
            history : list[dict]
                history returned by fit (or read from metrics.json)
            title : str
                axes title
        """
        epochs = [row["epoch"] for row in history]
        ax.plot(epochs, [row["train_loss"] for row in history], label="train loss")
        ax.plot(epochs, [row["val_loss"] for row in history], label="validation loss")
        ax.set(title=title, xlabel="epoch", ylabel="loss", yscale="log")
        ax.legend()

    def plot_decision_boundary(
        ax, predict, points: np.ndarray, labels: np.ndarray, title: str
    ) -> None:
        """
        Shade the class predicted at every point of a 100 x 100 grid and overlay the data.

        Parameters
        ----------
            ax : matplotlib.axes.Axes
                axes to draw on
            predict : Callable[[np.ndarray], np.ndarray]
                maps an (N, 2) float32 array to N class indices
            points : np.ndarray
                data points to overlay
            labels : np.ndarray
                their labels
            title : str
                axes title
        """
        axis = np.linspace(-1.2, 1.2, 100)
        grid_x, grid_y = np.meshgrid(axis, axis)
        grid = np.c_[grid_x.ravel(), grid_y.ravel()].astype(np.float32)
        classes = np.asarray(predict(grid)).reshape(grid_x.shape)
        levels = np.arange(labels.max() + 2) - 0.5
        ax.contourf(grid_x, grid_y, classes, levels=levels, alpha=0.3, cmap="viridis")
        ax.scatter(
            *points.T, c=labels, s=6, cmap="viridis", edgecolors="k", linewidths=0.2
        )
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")

    return plot_decision_boundary, plot_history


@app.cell
def _(
    labels,
    notebook_history,
    notebook_model,
    plot_decision_boundary,
    plot_history,
    plt,
    points,
    predict_classes,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    plot_history(_axes[0], notebook_history, "Notebook training")
    plot_decision_boundary(
        _axes[1],
        lambda _grid: predict_classes(notebook_model, _grid),
        points,
        labels,
        "Notebook model",
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part 2: From notebook to job

    The idea works, so now we write it as something that can run without us. A training job has a very different set of constraints from a notebook

    * **No hidden state.** It starts from nothing, every input comes from a file path, an argument or an environment variable.
    * **No interaction.** No widgets, no plots on screen, no `input()`. Nobody is watching.
    * **Configuration from outside.** Hyper-parameters are arguments (we use [`argparse`](https://docs.python.org/3/library/argparse.html)) and each one can also be set from an environment variable, as this is how most cloud services pass configuration. SageMaker for example sets `SM_MODEL_DIR` and Vertex AI sets `AIP_MODEL_DIR` to tell your code where to save the model.
    * **Logs to stdout.** The cloud collects whatever the job prints. We print one JSON object per line so people can read it and log tools can search it, and we `flush` so lines appear as they happen.
    * **Everything it makes goes in one folder.** The weights, an exported model and a `metrics.json` recording the settings, versions and results, so we can always say what produced a given model.
    * **Fail loudly.** If something is wrong exit with a non zero status and a clear message, rather than carrying on and saving rubbish.

    Have a look through `train.py` below with that list in mind. It also exports the model to [ONNX](https://onnx.ai/) at the end, which we will need for deployment.
    """)
    return


@app.cell
def _(HERE, mo):
    def show_file(name: str, language: str = "python"):
        return mo.md(f"```{language}\n{(HERE / name).read_text()}\n```")

    mo.accordion({"train.py": show_file("train.py")})
    return (show_file,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Smoke testing

    Before building any images, we run the script locally using the notebook’s Python environment for a handful of training epochs. This is a *smoke test*: a short run to check that the main parts of the program work together. We want to see that the script starts, finds its data, runs the training loop and writes its outputs.

    Passing this test doesn’t tell us whether the model is any good, or whether a full training run will succeed. It catches basic problems such as missing dependencies, incorrect file paths and errors when saving results. These are cheaper to find on our own machine than after building an image or starting a paid GPU session.
    """)
    return


@app.cell
def _(mo):
    smoke_button = mo.ui.run_button(label="Run smoke test")
    smoke_button
    return (smoke_button,)


@app.cell
def _(BUCKET, DATA_FILE, mo, run, set_runs_changed, smoke_button, sys):
    mo.stop(
        not smoke_button.value,
        mo.md("Press **Run smoke test** to run `train.py` for 5 epochs."),
    )
    _status = run(
        [
            sys.executable,
            "train.py",
            "--data",
            str(DATA_FILE),
            "--out",
            str(BUCKET / "runs" / "smoke"),
            "--run-id",
            "smoke",
            "--epochs",
            "5",
        ]
    )
    set_runs_changed(lambda _count: _count + 1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It is also worth checking that it fails properly. Here we point it at a dataset that doesn't exist. Look at the exit status: `1`, not `0`, so anything running this job (a cloud service, a CI pipeline, a shell script) knows it failed.
    """)
    return


@app.cell
def _(mo, run, smoke_button, sys):
    mo.stop(not smoke_button.value)
    _status = run([sys.executable, "train.py", "--data", "bucket/data/missing.npz"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part 3: Package the job as an image

    A container image bundles our code with the exact Python, PyTorch and libraries it needs, so it runs the same on our laptop, a lab machine or a cloud GPU. The recipe is a `Containerfile` (podman's name for a [Dockerfile](https://docs.docker.com/reference/dockerfile/), the format is identical).

    Things to notice in `Containerfile.train`

    * **uv, not pip.** We use [uv](https://docs.astral.sh/uv/) in the image just as we do everywhere else in the unit. Rather than installing it, the `uv` binary is copied straight out of Astral's [published uv image](https://docs.astral.sh/uv/guides/integration/docker/), pinned to a version like any other dependency. Inside a container there is only one Python and nothing else to keep separate from it, so `UV_SYSTEM_PYTHON=1` tells `uv pip install` to install into it directly rather than looking for a virtual environment.
    * **Layer order.** Each instruction makes a cached layer. The requirements are copied and installed *before* our code, so editing `train.py` only rebuilds the last cheap layers rather than downloading PyTorch again.
    * **A cache mount.** `RUN --mount=type=cache` gives uv a download cache that lives on the build machine, not in the image. When the requirements *do* change, uv reuses the wheels it already has instead of downloading torch again, and the image doesn't grow by the size of the cache.
    * **CPU PyTorch.** Torch is installed from the CPU wheel index which keeps the image to around a gigabyte rather than the five or six that the CUDA libraries add. The base image and wheel index are build arguments so the same file can build a GPU image for the real cloud.
    * **Pinned versions.** `requirements-train.txt` pins exact versions so the image built next month is the same as the one built today.
    * **No data or code baked in that changes per run.** The dataset and results go through the bucket, mounted at `/bucket` when the job runs.

    `.containerignore` keeps the build context small, it stops the bucket (which could be gigabytes of data) and this notebook being copied into the build.
    """)
    return


@app.cell
def _(mo, show_file):
    mo.accordion(
        {
            "Containerfile.train": show_file("Containerfile.train", "dockerfile"),
            "requirements-train.txt": show_file("requirements-train.txt", "text"),
            ".containerignore": show_file(".containerignore", "text"),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first build downloads the base image and PyTorch so it takes a few minutes, after that it is quick unless the requirements change. Try editing a comment in `train.py` and building again, you will see the dependency layers reported as cached.
    """)
    return


@app.cell
def _(mo):
    build_train_button = mo.ui.run_button(label="Build training image")
    build_train_button
    return (build_train_button,)


@app.cell
def _(TRAIN_IMAGE, build_train_button, engine, mo, run):
    mo.stop(
        not build_train_button.value,
        mo.md("Press **Build training image** to build it."),
    )
    _status = run(
        [engine.value, "build", "-t", TRAIN_IMAGE, "-f", "Containerfile.train", "."]
    )
    if _status == 0:
        run([engine.value, "image", "ls", TRAIN_IMAGE])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part 4: Run the training job

    Submitting a job to a cloud service is essentially "run this image, on this size of machine, with this storage attached and these settings". With podman that is one `podman run`

    | Option | What it does | Cloud equivalent |
    | --- | --- | --- |
    | `--rm` | delete the container when the job finishes | the machine is released after the job |
    | `--name` | a name to find it by in `podman ps` and `podman logs` | job name |
    | `--cpus`, `--memory` | limit the resources the job can use | choosing an instance type |
    | `-v bucket:/bucket` | mount our bucket folder inside the container | attaching cloud storage |
    | `-e NAME=value` | set environment variables read by `train.py` | job hyper-parameters |

    Every run gets its own `run_id` (here just the date and time) and writes to its own folder, so runs never overwrite each other. Try a few runs with different settings.

    If you get an error mentioning *cgroup* or *controller* your system doesn't allow rootless containers to limit CPU or memory. This is common on Linux without systemd [cgroup delegation](https://github.com/containers/podman/blob/main/troubleshooting.md), untick the limits and the job will run without them.
    """)
    return


@app.cell
def _(mo):
    job_settings = mo.ui.dictionary(
        {
            "epochs": mo.ui.number(start=1, stop=2000, value=200, label="Epochs"),
            "learning_rate": mo.ui.number(
                start=0.0001, stop=0.1, step=0.0001, value=0.01, label="Learning rate"
            ),
            "limits": mo.ui.checkbox(value=True, label="Apply resource limits"),
            "cpus": mo.ui.number(start=1, stop=16, value=2, label="CPUs"),
            "memory": mo.ui.dropdown(
                options=["1g", "2g", "4g"], value="2g", label="Memory"
            ),
        }
    ).form(submit_button_label="Submit job")
    job_settings
    return (job_settings,)


@app.cell
def _(
    BUCKET,
    TRAIN_IMAGE,
    datetime,
    engine,
    job_settings,
    mo,
    run,
    selinux_label,
    set_runs_changed,
    user_flags,
):
    mo.stop(
        job_settings.value is None,
        mo.md("Choose the job settings and press **Submit job**."),
    )

    run_id = datetime.now().strftime("run-%Y%m%d-%H%M%S")
    _settings = job_settings.value
    _limits = (
        ["--cpus", str(_settings["cpus"]), "--memory", _settings["memory"]]
        if _settings["limits"]
        else []
    )
    _cmd = [
        engine.value,
        "run",
        "--rm",
        "--name",
        f"spiral-train-{run_id}",
        *_limits,
        *user_flags,
        "-v",
        f"{BUCKET}:/bucket:rw{selinux_label}",
        "-e",
        f"RUN_ID={run_id}",
        "-e",
        f"EPOCHS={_settings['epochs']}",
        "-e",
        f"LEARNING_RATE={_settings['learning_rate']}",
        "-e",
        f"OUTPUT_DIR=/bucket/runs/{run_id}",
        TRAIN_IMAGE,
    ]
    job_status = run(_cmd)
    set_runs_changed(lambda _count: _count + 1)
    mo.stop(
        job_status != 0,
        mo.callout(
            mo.md(f"Job **{run_id}** failed, see the log above."),
            kind="danger",
        ),
    )
    mo.callout(
        mo.md(f"Job **{run_id}** finished, results in `{BUCKET / 'runs' / run_id}`"),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Forgetting the storage

    The most common first mistake is forgetting to attach the storage. The image runs perfectly happily, but `/bucket` inside the container is just an empty folder, so `train.py` can't find the data. Because we wrote the script to fail loudly the job stops immediately with a clear message and exit status `1`, instead of a confusing traceback (or worse, training on nothing).
    """)
    return


@app.cell
def _(mo):
    no_volume_button = mo.ui.run_button(label="Run without the bucket")
    no_volume_button
    return (no_volume_button,)


@app.cell
def _(TRAIN_IMAGE, engine, mo, no_volume_button, run):
    mo.stop(not no_volume_button.value)
    _status = run([engine.value, "run", "--rm", TRAIN_IMAGE, "--epochs", "5"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that we passed `--epochs 5` after the image name. Anything after the image is handed to the `ENTRYPOINT`, so it becomes `python train.py --epochs 5`, which is a handy way to override settings for a one off test.

    # Part 5: Collect and check the results

    Getting the results back is just reading the bucket (in the cloud `aws s3 cp` or `gcloud storage cp` the other way). Each run folder holds the model, the ONNX export and `metrics.json`. Together these act as a very simple *model registry*: we can compare runs and choose which one to deploy.
    """)
    return


@app.cell
def _(mo):
    get_runs_changed, set_runs_changed = mo.state(0)
    return get_runs_changed, set_runs_changed


@app.cell
def _(BUCKET, get_runs_changed, json, mo):
    get_runs_changed()  # re-run this cell whenever a job or smoke test finishes
    runs = {}
    # oldest first, so the newest run is last and becomes the default choice
    for _metrics_file in sorted(
        (BUCKET / "runs").glob("*/metrics.json"), key=lambda _f: _f.stat().st_mtime
    ):
        runs[_metrics_file.parent.name] = json.loads(_metrics_file.read_text())
    mo.stop(not runs, mo.md("No runs yet, run the smoke test or submit a job above."))

    run_choice = mo.ui.dropdown(
        options=list(runs), value=list(runs)[-1], label="Run to inspect and deploy"
    )
    mo.vstack(
        [
            mo.ui.table(
                [
                    {
                        "run": _id,
                        "epochs": _run["args"]["epochs"],
                        "learning rate": _run["args"]["lr"],
                        "val accuracy": f"{_run['final']['val_accuracy']:.1%}",
                        "seconds": _run["duration_seconds"],
                        "host": _run["host"],
                        "torch": _run["torch_version"],
                    }
                    for _id, _run in runs.items()
                ],
                selection=None,
            ),
            run_choice,
        ]
    )
    return run_choice, runs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `host` column is a nice check that the job really ran somewhere else: the smoke test reports your machine's name, the container runs report the container's ID. The torch version can differ too, the image has its own copy of PyTorch which need not match the notebook's.
    """)
    return


@app.cell
def _(
    BUCKET,
    build_model,
    labels,
    plot_decision_boundary,
    plot_history,
    plt,
    points,
    predict_classes,
    run_choice,
    runs,
    torch,
):
    run_dir = BUCKET / "runs" / run_choice.value
    _checkpoint = torch.load(run_dir / "model.pt", weights_only=True)
    job_model = build_model(**_checkpoint["config"])
    job_model.load_state_dict(_checkpoint["model_state"])

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    plot_history(_axes[0], runs[run_choice.value]["history"], f"Job {run_choice.value}")
    plot_decision_boundary(
        _axes[1],
        lambda _grid: predict_classes(job_model, _grid),
        points,
        labels,
        "Model from the job",
    )
    _fig
    return job_model, run_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check the export before deploying

    The inference service will run `model.onnx` with [onnxruntime](https://onnxruntime.ai/docs/get-started/with-python.html), not the PyTorch weights. Exporting can go wrong in subtle ways (an unsupported operation, a fixed batch size), so before deploying we run both on the same points and compare. The logits may not be bit for bit identical, as the two libraries can do the floating point maths in a slightly different order, but any difference should be tiny (around `1e-6` or less) and they should give the same class every time.
    """)
    return


@app.cell
def _(job_model, np, ort, points, run_dir, torch):
    _session = ort.InferenceSession(
        str(run_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    print("onnx inputs ", [(_i.name, _i.shape) for _i in _session.get_inputs()])
    print("onnx outputs", [(_o.name, _o.shape) for _o in _session.get_outputs()])

    (_onnx_logits,) = _session.run(["logits"], {"points": points})
    with torch.no_grad():
        _torch_logits = job_model(torch.from_numpy(points)).numpy()
    print(f"max difference in logits {np.abs(_onnx_logits - _torch_logits).max():.2e}")
    print(
        f"same class for {np.mean(_onnx_logits.argmax(1) == _torch_logits.argmax(1)):.1%} of {len(points)} points"
    )
    print(
        f"model.pt {(run_dir / 'model.pt').stat().st_size / 1024:.1f} KiB, model.onnx {(run_dir / 'model.onnx').stat().st_size / 1024:.1f} KiB"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note the batch dimension shows as `batch` rather than `1`. That comes from the `dynamic_shapes` argument in `export_onnx`, without it the exported model would only ever accept one point at a time. Try removing it from `train.py`, rebuilding and re-running a job to see what breaks.

    # Part 6: Deploy for inference

    Serving has very different needs from training. It runs for days rather than hours, has to start quickly, should be small and secure, and needs a stable API that other people's code can call. So it gets its own image rather than reusing the training one.

    * `serve.py` is a small [FastAPI](https://fastapi.tiangolo.com/) app with a `/health` route (used by the platform to check the service is alive) and a `/predict` route.
    * The request is described with [pydantic](https://docs.pydantic.dev/) types, so a malformed request is rejected with a clear error before it gets anywhere near the model.
    * The model is loaded once at start up, not per request.
    * It only needs `onnxruntime` and `numpy`, there is no PyTorch in this image at all, and it runs as a normal user rather than root.
    """)
    return


@app.cell
def _(mo, show_file):
    mo.accordion(
        {
            "serve.py": show_file("serve.py"),
            "Containerfile.serve": show_file("Containerfile.serve", "dockerfile"),
            "requirements-serve.txt": show_file("requirements-serve.txt", "text"),
        }
    )
    return


@app.cell
def _(mo):
    build_serve_button = mo.ui.run_button(label="Build inference image")
    build_serve_button
    return (build_serve_button,)


@app.cell
def _(SERVE_IMAGE, TRAIN_IMAGE, build_serve_button, engine, mo, run):
    mo.stop(
        not build_serve_button.value,
        mo.md("Press **Build inference image** to build it."),
    )
    _status = run(
        [engine.value, "build", "-t", SERVE_IMAGE, "-f", "Containerfile.serve", "."]
    )
    if _status == 0:
        # compare the sizes of the two images
        run([engine.value, "image", "ls", "--filter", "reference=localhost/spiral-*"])
    print(f"(built {SERVE_IMAGE}, training image is {TRAIN_IMAGE})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Start the endpoint

    This time the container is long running, so the options are a little different

    | Option | What it does |
    | --- | --- |
    | `-d` | detach, run in the background and return straight away |
    | `-p 8080:8080` | publish the container's port 8080 on our machine's port 8080 |
    | `-v run_dir:/models:ro` | mount only the chosen run, **read only** |

    Deploying a model is then just choosing which run folder to mount. Pick a different run in the table above and redeploy to roll forwards or back. The cell waits until `/health` answers before carrying on, just as a cloud load balancer won't send traffic to a new container until its health check passes.

    If port 8080 is already in use on your machine change `PORT` in the settings cell near the top.
    """)
    return


@app.cell
def _(mo):
    deploy_button = mo.ui.run_button(label="Deploy endpoint")
    stop_button = mo.ui.run_button(label="Stop endpoint", kind="danger")
    mo.hstack([deploy_button, stop_button], justify="start")
    return deploy_button, stop_button


@app.cell
def _(
    ENDPOINT_NAME,
    PORT,
    SERVE_IMAGE,
    deploy_button,
    engine,
    mo,
    requests,
    run,
    run_dir,
    selinux_label,
    subprocess,
    time,
):
    mo.stop(
        not deploy_button.value,
        mo.md("Press **Deploy endpoint** to serve the selected run."),
    )

    # remove any previous endpoint, ignoring the error if there isn't one
    subprocess.run([engine.value, "rm", "--force", ENDPOINT_NAME], capture_output=True)
    _status = run(
        [
            engine.value,
            "run",
            "-d",
            "--name",
            ENDPOINT_NAME,
            "-p",
            f"{PORT}:8080",
            "-v",
            f"{run_dir}:/models:ro{selinux_label}",
            SERVE_IMAGE,
        ]
    )
    mo.stop(
        _status != 0,
        mo.callout("The container didn't start, see above.", kind="danger"),
    )

    endpoint_url = f"http://localhost:{PORT}"
    for _attempt in range(30):
        try:
            _health = requests.get(f"{endpoint_url}/health", timeout=1)
            _health.raise_for_status()
            print(f"healthy after {_attempt + 1} attempt(s): {_health.json()}")
            break
        except requests.RequestException:
            time.sleep(1)
    else:
        run([engine.value, "logs", ENDPOINT_NAME])
        mo.stop(
            True,
            mo.callout(
                "The endpoint never became healthy, its logs are above.", kind="danger"
            ),
        )
    return (endpoint_url,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Call it like a client

    From here on the notebook is just another client. It sends JSON over HTTP with [requests](https://requests.readthedocs.io/) and has no idea what is behind the endpoint; the same calls would work from a web page, a game engine or a phone app. We send the validation points and check the accuracy matches what the job reported.
    """)
    return


@app.cell
def _(endpoint_url, labels, np, points, requests, train_val_split):
    _, (_val_points, _val_labels) = train_val_split(points, labels, seed=1234)
    _response = requests.post(
        f"{endpoint_url}/predict", json={"points": _val_points.tolist()}, timeout=10
    )
    _response.raise_for_status()
    _result = _response.json()
    print(f"model version {_result['model_version']}")
    print(
        f"first point {_val_points[0]} -> class {_result['classes'][0]}, probabilities {_result['probabilities'][0]}"
    )
    print(
        f"endpoint accuracy on {len(_val_points)} validation points: {np.mean(np.array(_result['classes']) == _val_labels):.1%}"
    )
    return


@app.cell
def _(endpoint_url, labels, np, plot_decision_boundary, plt, points, requests):
    def endpoint_predict(grid: np.ndarray) -> np.ndarray:
        response = requests.post(
            f"{endpoint_url}/predict", json={"points": grid.tolist()}, timeout=30
        )
        response.raise_for_status()
        return np.array(response.json()["classes"])

    _fig, _ax = plt.subplots(figsize=(4, 4), layout="constrained")
    plot_decision_boundary(
        _ax, endpoint_predict, points, labels, "Predicted by the endpoint"
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That plot needed 10,000 predictions, which went in a single request. The decision boundary should look identical to the one from the job's PyTorch model above.

    ### Bad requests

    Someone will always send you something unexpected. Here a point has three coordinates. The endpoint replies with status [422](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/422) and a message saying exactly what is wrong, and the model never runs. `raise_for_status` turns that into an exception on the client side, which we catch to look at the details.
    """)
    return


@app.cell
def _(endpoint_url, requests):
    _response = requests.post(
        f"{endpoint_url}/predict", json={"points": [[0.1, 0.2, 0.3]]}, timeout=5
    )
    try:
        _response.raise_for_status()
    except requests.HTTPError as error:
        print(f"HTTPError: {error}")
        for _problem in _response.json()["detail"]:
            print(f"  {_problem['loc']}: {_problem['msg']}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Batching

    Every request has a fixed cost (opening the connection, parsing JSON, a trip through the web framework) on top of the model itself. For a model this small that overhead *is* the cost, which is easy to see by sending the same 50 points one at a time and then as one batch. On a real cloud endpoint, with network latency in between, the gap is even bigger.
    """)
    return


@app.cell
def _(endpoint_url, points, requests, time):
    _sample = points[:50].tolist()
    _start = time.perf_counter()
    for _point in _sample:
        requests.post(
            f"{endpoint_url}/predict", json={"points": [_point]}, timeout=5
        ).raise_for_status()
    _one_at_a_time = time.perf_counter() - _start

    _start = time.perf_counter()
    requests.post(
        f"{endpoint_url}/predict", json={"points": _sample}, timeout=5
    ).raise_for_status()
    _batched = time.perf_counter() - _start
    print(
        f"50 single requests {_one_at_a_time * 1000:.1f} ms, one batch of 50 {_batched * 1000:.1f} ms"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Logs and tear down

    The service logs every request to stdout, which is where a cloud platform would collect it. `podman logs` shows the same thing locally.

    Finally, stop the endpoint. This matters much more in the cloud than here: an endpoint costs money for every hour it is running, whether anyone calls it or not, and a forgotten GPU endpoint left running over the summer is a very expensive mistake.
    """)
    return


@app.cell
def _(ENDPOINT_NAME, endpoint_url, engine, run):
    print(f"logs for the endpoint at {endpoint_url}")
    _status = run([engine.value, "logs", "--tail", "5", ENDPOINT_NAME])
    return


@app.cell
def _(ENDPOINT_NAME, engine, mo, run, stop_button):
    mo.stop(not stop_button.value)
    run([engine.value, "stop", ENDPOINT_NAME])
    run([engine.value, "rm", "--force", ENDPOINT_NAME])
    _status = run([engine.value, "ps", "--all", "--filter", "name=spiral"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Moving to a real cloud

    Nothing above is specific to podman, so moving to a real provider is mostly a matter of swapping the local stand ins for the real thing

    1. **Push the images to a registry** the cloud can reach, for example `podman tag localhost/spiral-train:0.1 ghcr.io/<user>/spiral-train:0.1` then `podman push ghcr.io/<user>/spiral-train:0.1`.
    2. **Upload the data** to object storage and give the job that location instead of our `bucket/` folder.
    3. **Build a GPU training image** by changing the `BASE_IMAGE` and `TORCH_INDEX` build arguments. Our CPU image would run on a GPU machine but never use the GPU.
    4. **Submit the job** with the provider's SDK or command line tool, passing the same environment variables.
    5. **Deploy** the serving image to a managed endpoint or a container service such as Cloud Run, pointing it at the chosen model in storage.

    It is worth being honest about what podman *doesn't* simulate: credentials and permissions (IAM), the time and cost of moving large datasets, jobs being interrupted part way through on cheaper *spot* machines, and endpoints scaling up and down with demand. These are where most of the real world pain is, but they all sit on top of the workflow we have just built.

    # Exercises

    1. Add a `--hidden` setting to the job form and submit runs with 8, 64 and 256 hidden units. Compare them in the runs table, then deploy the smallest model that still gets above 95%.
    2. Change a comment in `train.py` and rebuild the training image. Which steps are cached? Now add a package to `requirements-train.txt` and rebuild. Which steps run again this time, and what does uv print about downloading torch? (This is the cache mount at work.)
    3. Predict what happens if you remove `dynamic_shapes` from `export_onnx`, then try it. Which of our checks catches the problem first?
    4. Cloud *spot* machines can be stopped at any time. Make `train.py` save a checkpoint every few epochs and resume from it if one exists (see the [Checkpoints](../Checkpoints/CheckPointsMarimo.py) notebook). Test it by running `podman stop` on a job part way through, then submitting it again with the same `RUN_ID`.
    5. Add a `/model` route to `serve.py` that returns the run's `metrics.json` settings, so a client can find out exactly which model it is talking to.
    6. Write a short shell script that does the whole cycle (data, build, train, deploy, one request, stop) without the notebook. This is the first step towards a CI pipeline.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

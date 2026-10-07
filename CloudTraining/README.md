# Cloud Training with Podman

This demo walks through the whole ML DevOps cycle (prototype, standalone training job, container image, training run, model check, inference endpoint) without using any cloud credit. The "cloud" is simulated with [podman](https://podman.io/) on your own machine, which is also how you should test a job before paying to run it for real.

The model is a small MLP classifying a generated spiral dataset, so it trains in seconds on a CPU and the focus stays on the process.

## Running it

You need podman installed (or Docker, the notebook lets you choose). On macOS and Windows start the podman virtual machine first

```bash
podman machine init
podman machine start
```

Then from the root of the repo

```bash
uv run marimo edit CloudTraining/CloudTrainingMarimo.py
```

The first image build downloads the Python base image and PyTorch so takes a few minutes, after that rebuilds are quick. Both images install their dependencies with [uv](https://docs.astral.sh/uv/guides/integration/docker/), copied in from Astral's uv image, so there is no pip anywhere.

## What is here

| File | Description |
| :--- | :--- |
| [CloudTrainingMarimo.py](CloudTrainingMarimo.py) | The notebook, drives everything else |
| [spiralnet/](spiralnet/) | Data, model and training loop shared by the notebook and the job |
| [train.py](train.py) | Standalone training job, configured by arguments or environment variables |
| [Containerfile.train](Containerfile.train) | Training image, CPU PyTorch |
| [serve.py](serve.py) | FastAPI inference service running the exported ONNX model |
| [Containerfile.serve](Containerfile.serve) | Inference image, onnxruntime only |
| `requirements-*.txt` | Pinned dependencies for each image |

The notebook writes the dataset and every run to `bucket/` (our pretend cloud storage), which is gitignored.

## Doing it by hand

Everything the notebook runs is printed so you can copy it into a terminal. The short version, from this folder, is

```bash
uv run python -c "from spiralnet import *; save_dataset('bucket/data/spiral.npz', *make_spiral())"
uv run python train.py --data bucket/data/spiral.npz --out bucket/runs/smoke --epochs 5

podman build -t localhost/spiral-train:0.1 -f Containerfile.train .
podman run --rm -v $PWD/bucket:/bucket:Z -e RUN_ID=run1 -e OUTPUT_DIR=/bucket/runs/run1 localhost/spiral-train:0.1

podman build -t localhost/spiral-serve:0.1 -f Containerfile.serve .
podman run -d --name spiral-endpoint -p 8080:8080 -v $PWD/bucket/runs/run1:/models:ro,Z localhost/spiral-serve:0.1
curl -X POST localhost:8080/predict -H "Content-Type: application/json" -d '{"points": [[0.1, 0.2]]}'
podman rm --force spiral-endpoint
```

Leave off the `:Z` (and the `,Z`) on macOS.

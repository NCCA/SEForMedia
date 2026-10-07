#!/usr/bin/env -S uv run
"""
Standalone training job for the spiral classifier.

This is the script we hand to "the cloud". It has no notebook state, no plots and
no prompts: it reads its inputs from paths and arguments, logs to stdout, writes
everything it produces to one output folder and exits with a non zero status if
anything goes wrong. That is all a cloud training service (or podman) needs.

Every argument can also be set with an environment variable, as most cloud
services pass configuration that way, for example

    python train.py --data bucket/data/spiral.npz --out bucket/runs/test --epochs 5
    EPOCHS=5 python train.py
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch
from spiralnet import build_model, fit, load_dataset, train_val_split


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    env = os.environ.get
    parser = argparse.ArgumentParser(description="Train the spiral classifier")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(env("DATA_PATH", "/bucket/data/spiral.npz")),
        help="dataset .npz file (env DATA_PATH)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(env("OUTPUT_DIR", "/bucket/runs/latest")),
        help="folder for the model and metrics (env OUTPUT_DIR)",
    )
    parser.add_argument(
        "--run-id",
        default=env("RUN_ID", "local"),
        help="name for this run (env RUN_ID)",
    )
    parser.add_argument("--epochs", type=int, default=int(env("EPOCHS", 200)))
    parser.add_argument("--lr", type=float, default=float(env("LEARNING_RATE", 0.01)))
    parser.add_argument("--batch-size", type=int, default=int(env("BATCH_SIZE", 64)))
    parser.add_argument("--hidden", type=int, default=int(env("HIDDEN", 64)))
    parser.add_argument("--seed", type=int, default=int(env("SEED", 1234)))
    parser.add_argument(
        "--device",
        default=env("DEVICE", "auto"),
        choices=["auto", "cpu", "cuda"],
        help="auto uses CUDA when the machine has it",
    )
    return parser.parse_args(argv)


def export_onnx(model: torch.nn.Module, path: Path) -> None:
    """
    Export the model to ONNX with a dynamic batch size.

    The inference container only needs onnxruntime to run this file, not the
    whole of PyTorch, which keeps the serving image small.

    Parameters
    ----------
        model : torch.nn.Module
            trained model
        path : Path
            .onnx file to write
    """
    model = model.cpu().eval()
    example = torch.zeros(1, 2)
    torch.onnx.export(
        model,
        (example,),
        path,
        input_names=["points"],
        output_names=["logits"],
        dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
        external_data=False,
        verbose=False,
    )


def log(**fields) -> None:
    # one JSON object per line, readable by people and by cloud log tools.
    # flush so the lines appear in `podman logs` as they happen, not at exit
    print(json.dumps(fields), flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.data.exists():
        print(
            f"error: dataset {args.data} not found (is the bucket mounted?)",
            file=sys.stderr,
        )
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(args.seed)

    points, labels = load_dataset(args.data)
    train, val = train_val_split(points, labels, seed=args.seed)
    config = {"hidden": args.hidden, "classes": int(labels.max()) + 1}
    model = build_model(**config)
    log(
        event="start",
        run_id=args.run_id,
        host=platform.node(),
        device=str(device),
        torch=torch.__version__,
        train_size=len(train[0]),
        val_size=len(val[0]),
    )

    start = time.perf_counter()
    report_every = max(1, args.epochs // 10)

    def on_epoch(metrics: dict) -> None:
        if metrics["epoch"] % report_every == 0 or metrics["epoch"] == args.epochs:
            log(event="epoch", **metrics)

    history = fit(
        model,
        train,
        val,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        on_epoch=on_epoch,
    )
    duration = time.perf_counter() - start

    torch.save(
        {
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "config": config,
            # str() because __version__ is a TorchVersion, which weights_only loading rejects
            "torch_version": str(torch.__version__),
        },
        args.out / "model.pt",
    )
    export_onnx(model, args.out / "model.onnx")
    summary = {
        "run_id": args.run_id,
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "config": config,
        "device": str(device),
        "host": platform.node(),
        "torch_version": torch.__version__,
        "duration_seconds": round(duration, 2),
        "final": history[-1],
        "history": history,
    }
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2))
    log(
        event="done",
        seconds=round(duration, 2),
        val_accuracy=history[-1]["val_accuracy"],
        out=str(args.out),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

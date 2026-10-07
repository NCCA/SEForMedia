#!/usr/bin/env -S uv run --script
"""
Remove the datasets the notebooks download, to reclaim disk space.

Most of the training notebooks download their data into a folder beside the
notebook when they aren't running in the lab (in the lab it goes in
/transfer, which this script doesn't touch). Over a few weeks that adds up to
a couple of gigabytes, so this deletes them all again. Every notebook will
download its data again the next time it is run.

The folders to remove are listed in DATA_PATHS, relative to the repo root.
When you add a demo that downloads something, add its folder there.

By default I list what will be removed and how big it is, then ask before
deleting anything. Use ``--dry-run`` to only list, or ``--yes`` to skip the
question. Anything git tracks is skipped, so a committed file can't be
deleted by a typo in the list.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# folders (or files) the notebooks download into, relative to the repo root
DATA_PATHS = [
    "ASL/mnist_asl",  # ASL/ASLPart1Marimo.py
    "DataLeakage/ESC50",  # DataLeakage/DataLeakageMarimo.py
    "EvaluationOfResults/data",  # EvaluationOfResults/BaselinesEvaluationMarimo.py (kodak images)
    "FreeSpokenDigits/data",  # FreeSpokenDigits/FSDDMarimoPt1.py (kagglehub)
    "ImageToImage/data",  # ImageToImage/0*Marimo.py (Oxford-IIIT Pet)
    "Lecture6/data",  # Lecture6/IntroductionToPandasMarimo.py and IntroductionToPolarsMarimo.py
    "MNIST/MNIST",  # MNIST/TheMNISTDataSetMarimo.py
    "MNIST/data",  # MNIST/TheMNISTDataSetMarimo.py (torchvision datasets.MNIST)
    "MNIST/mnist_data",  # MNIST/PyTorchDataLoadersMarimo.py
    "PreTrainedModels/dog_door",  # PreTrainedModels/TransferLearningMarimo.py
    "TextProcessingRNN/data",  # TextProcessingRNN/RNNPart1.py (Hugging Face cache)
]


def size_of(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def human_size(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def is_tracked(path: Path) -> bool:
    """
    Check whether git tracks the path, or anything inside it.

    Parameters
    ----------
        path : Path
            the file or folder to check

    Returns
    -------
        bool
            True if ``git ls-files`` lists anything under the path
    """
    result = subprocess.run(
        ["git", "ls-files", "--", str(path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip() != ""


def find_targets() -> list[tuple[Path, int]]:
    """
    Work out which of the DATA_PATHS exist and are safe to delete.

    Returns
    -------
        list[tuple[Path, int]]
            each path that will be removed along with its size in bytes
    """
    targets = []
    for entry in DATA_PATHS:
        path = (REPO_ROOT / entry).resolve()
        # guard against a ../ in the list pointing outside the repo
        if not path.is_relative_to(REPO_ROOT):
            print(f"skipping {entry}: outside the repo")
            continue
        if not path.exists():
            continue
        if is_tracked(path):
            print(f"skipping {entry}: contains files tracked by git")
            continue
        targets.append((path, size_of(path)))
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--dry-run", action="store_true", help="list what would be removed and stop"
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", help="delete without asking first"
    )
    args = parser.parse_args()

    targets = find_targets()
    if not targets:
        print("Nothing to remove.")
        return

    total = 0
    for path, size in targets:
        print(f"{human_size(size):>10}  {path.relative_to(REPO_ROOT)}")
        total += size
    print(f"{human_size(total):>10}  total")

    if args.dry_run:
        return
    if not args.yes and input("Delete these? [y/N] ").strip().lower() != "y":
        print("Nothing removed.")
        sys.exit(1)

    for path, _ in targets:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        print(f"removed {path.relative_to(REPO_ROOT)}")
    print(f"Reclaimed {human_size(total)}.")


if __name__ == "__main__":
    main()

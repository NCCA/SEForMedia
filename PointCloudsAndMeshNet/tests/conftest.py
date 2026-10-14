"""
Shared fixtures for the point-cloud lesson tests.

The notebooks import ``pointcloud`` and ``modelnet_data`` from their own
folder, so I put that folder on ``sys.path`` here in the same way. The repo
root goes on too so the tests can reach ``scripts/clean_data.py``.
"""

import contextlib
import importlib.util
import io
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import torch

LESSON_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = LESSON_DIR.parent
sys.path.insert(0, str(LESSON_DIR))
sys.path.insert(0, str(REPO_ROOT))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: trains a model in a notebook (deselect with -m 'not slow')"
    )


@pytest.fixture
def single_thread() -> Iterator[None]:
    """Run torch on one thread so the seeded training runs repeat exactly."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def run_lesson(single_thread: None) -> Callable[..., dict]:
    """
    Return a function which runs one of the marimo lessons headless.

    The returned function takes the lesson number and an optional dict of
    cell definitions to override (for example a submitted form) and returns
    every name the notebook defined.
    """

    def run(part: int, overrides: dict | None = None) -> dict:
        path = next(LESSON_DIR.glob(f"PointCloudForMLPart{part}*.py"))
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with contextlib.redirect_stdout(io.StringIO()):
            _, definitions = module.app.run(defs=overrides)
        return definitions

    return run

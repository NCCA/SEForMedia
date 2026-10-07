from collections.abc import Iterator

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def torch_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def torch_seed() -> Iterator[None]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        yield

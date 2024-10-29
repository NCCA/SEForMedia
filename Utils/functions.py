#!/usr/bin/env python

"""
Module to provide utility functions for lab examples
"""

import re
import platform
import requests
from tqdm import tqdm
from pathlib import Path
import IPython


def in_lab() -> bool:
    """
    Returns True if the code is running in one of the NCCA labs
    Basically we are checking if the hostname is one of the lab machines which begin with either
    pg or w followed by a number. We use a regular expression to do this
    """
    hostname = platform.node()
    # Regular expression pattern
    return re.search(r"(pg|w)\d+", hostname) is not None


def download(url: str, fname: str):
    """
    Downloads a file from the specified url and saves it to the specified file name
        param url: the url of the file to download
        param fname: the file name to save the file as

    """

    resp = requests.get(url, stream=True, verify=False)
    total = int(resp.headers.get("content-length", 0))
    # Can also replace 'file' with a io.BytesIO object
    with open(fname, "wb") as file, tqdm(
        desc=fname, total=total, unit="iB", unit_scale=True, unit_divisor=1024
    ) as progress_bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)


def shutdown_kernel():
    """
    Shutdown the current IPython kernel
    """
    app = IPython.Application.instance()
    app.kernel.do_shutdown(True)


def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N


if __name__ == "__main__":
    print(f"running in lab {in_lab()}")

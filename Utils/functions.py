#!/usr/bin/env python

"""
Module to provide utility functions for lab examples
"""

import platform
import re
import zipfile
from pathlib import Path

import IPython
import requests
from tqdm import tqdm
import torch


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


def unzip_file(zip_file: Path, dest_dir: Path):
    """
    Unzips a file to the specified directory
        param zip_file: the zip file to unzip
        param dest_dir: the directory to unzip the file to
    """

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(dest_dir)


def shutdown_kernel():
    """
    Shutdown the current IPython kernel
    """
    app = IPython.Application.instance()
    app.kernel.do_shutdown(True)


def get_batch_accuracy(output: torch.Tensor, y: torch.Tensor, N: int) -> float:
    """
    Calculate the accuracy of a batch of predictions.

    Args:
        output (torch.Tensor): The output predictions from the model.
        y (torch.Tensor): The true labels.
        N (int): The total number of samples in the batch.

    Returns:
        float: The accuracy of the batch.
    """
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate the accuracy of predictions.

    Args:
        y_true (torch.Tensor): The true labels.
        y_pred (torch.Tensor): The predicted labels.

    Returns:
        float: The accuracy as a percentage.
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


if __name__ == "__main__":
    print(f"running in lab {in_lab()}")

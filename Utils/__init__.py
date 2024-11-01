# import functions into the package
from .functions import download, get_batch_accuracy, in_lab, shutdown_kernel, unzip_file
from .TorchUtils import get_device

__all__ = [
    "get_device",
    "in_lab",
    "download",
    "shutdown_kernel",
    "get_batch_accuracy",
    "unzip_file",
]

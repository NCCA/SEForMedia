# import functions into the package
from .TorchUtils import get_device
from .functions import in_lab, download, shutdown_kernel

__all__ = ["get_device", "in_lab", "download", "shutdown_kernel"]

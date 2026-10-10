"""Downloads and prepares the mesh archive used in the ModelNet10 lesson."""

import shutil
import urllib.request
import zipfile
from pathlib import Path

MODELNET10_URL = (
    "https://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
)
MODELNET10_CATEGORIES = (
    "bathtub",
    "bed",
    "chair",
    "desk",
    "dresser",
    "monitor",
    "night_stand",
    "sofa",
    "table",
    "toilet",
)


def modelnet10_ready(cache: Path) -> bool:
    root = cache / "ModelNet10"
    return (cache / ".modelnet10-complete").is_file() and all(
        any((root / category / split).glob("*.off"))
        for category in MODELNET10_CATEGORIES
        for split in ("train", "test")
    )


def prepare_modelnet10(cache: Path, url: str = MODELNET10_URL) -> Path:
    """
    Downloads the archive once and extracts its training and test meshes.

    Parameters
    ----------
        cache : Path
            directory for the archive, extracted meshes and completion marker
        url : str
            source archive URL; defaults to the official HTTPS download
    """
    cache = Path(cache)
    if modelnet10_ready(cache):
        return cache / "ModelNet10"
    cache.mkdir(parents=True, exist_ok=True)
    archive_path = cache / "ModelNet10.zip"
    if not archive_path.exists():
        partial = cache / "ModelNet10.zip.part"
        print(f"Downloading ModelNet10 into {cache} (about 451 MiB)", flush=True)
        with (
            urllib.request.urlopen(url, timeout=60) as response,
            partial.open("wb") as output,
        ):
            shutil.copyfileobj(response, output, length=1024 * 1024)
        partial.replace(archive_path)
    print("Extracting ModelNet10 meshes...", flush=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (cache / member.filename).resolve()
            if not destination.is_relative_to(cache.resolve()):
                raise ValueError(f"Archive entry outside the cache: {member.filename}")
        archive.extractall(cache)
    root = cache / "ModelNet10"
    if not all(
        any((root / category / split).glob("*.off"))
        for category in MODELNET10_CATEGORIES
        for split in ("train", "test")
    ):
        raise ValueError(
            "The archive does not contain all ten ModelNet10 train/test categories"
        )
    (cache / ".modelnet10-complete").write_text(url + "\n", encoding="utf-8")
    return root

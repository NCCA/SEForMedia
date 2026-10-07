"""
Generate, save and load the spiral dataset.

The data is generated rather than downloaded so the demo has no external
dependencies, and it is saved to a single .npz file so we have a real file to
"upload" to our pretend cloud bucket.
"""

from pathlib import Path

import numpy as np


def make_spiral(
    points_per_class: int = 300,
    classes: int = 3,
    noise: float = 0.2,
    seed: int = 1234,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate interleaved spiral arms, one arm per class.

    This is the classic toy problem from the Stanford CS231n notes, a straight
    line can't separate the classes so a linear model will fail but a small MLP
    will not.

    Parameters
    ----------
        points_per_class : int
            number of points in each spiral arm
        classes : int
            number of arms (and therefore classes)
        noise : float
            standard deviation of the noise added to the angle
        seed : int
            seed for the random generator so the data is reproducible

    Returns
    -------
        tuple[np.ndarray, np.ndarray]
            points as float32 with shape (N, 2) and labels as int64 with shape (N,)
    """
    rng = np.random.default_rng(seed)
    points = np.zeros((points_per_class * classes, 2), dtype=np.float32)
    labels = np.zeros(points_per_class * classes, dtype=np.int64)
    for c in range(classes):
        index = slice(points_per_class * c, points_per_class * (c + 1))
        radius = np.linspace(0.0, 1.0, points_per_class)
        theta = np.linspace(c * 4.0, (c + 1) * 4.0, points_per_class)
        theta += rng.normal(0.0, noise, points_per_class)
        points[index] = np.c_[radius * np.sin(theta), radius * np.cos(theta)]
        labels[index] = c
    return points, labels


def save_dataset(path: Path, points: np.ndarray, labels: np.ndarray) -> Path:
    """
    Save the dataset as a compressed .npz file, creating the folder if needed.

    Parameters
    ----------
        path : Path
            file to write
        points : np.ndarray
            (N, 2) array of points
        labels : np.ndarray
            (N,) array of class labels
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points=points, labels=labels)
    return path


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a dataset written by save_dataset.

    Parameters
    ----------
        path : Path
            .npz file to read

    Returns
    -------
        tuple[np.ndarray, np.ndarray]
            points and labels
    """
    with np.load(path) as data:
        return data["points"], data["labels"]


def train_val_split(
    points: np.ndarray, labels: np.ndarray, val_fraction: float = 0.2, seed: int = 1234
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Shuffle and split the data into training and validation sets.

    Parameters
    ----------
        points : np.ndarray
            (N, 2) array of points
        labels : np.ndarray
            (N,) array of labels
        val_fraction : float
            fraction of the data held back for validation
        seed : int
            seed for the shuffle so the split is the same everywhere

    Returns
    -------
        tuple
            ((train_points, train_labels), (val_points, val_labels))
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(points))
    split = int(len(points) * (1.0 - val_fraction))
    train, val = order[:split], order[split:]
    return (points[train], labels[train]), (points[val], labels[val])

"""
Small geometry and model helpers shared by the point-cloud lessons.

This holds the two PointNet style models (classification and part segmentation),
the synthetic shape and chair generators, loaders for ModelNet and ShapeNet Part,
a per-part IoU metric and a plotly helper for viewing clouds.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    import plotly.graph_objects as go


class PointNet(nn.Module):
    """
    A cut-down PointNet classifier.

    Each point goes through the same small MLP (written as 1x1 Conv1d layers), then a
    max over all the points gives a single global feature that doesn't depend on the
    order of the points. A linear head turns that into class scores.

    Attributes
    ----------
        local : nn.Sequential
            per-point layers lifting xyz to 64 features
        features : nn.Sequential
            per-point layers lifting 64 to 128 features before the max-pool
        head : nn.Sequential
            fully connected layers mapping the global feature to class scores
    """

    def __init__(self, classes: int) -> None:
        """
        Parameters
        ----------
            classes : int
                number of output classes
        """
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv1d(3, 32, 1), nn.ReLU(), nn.Conv1d(32, 64, 1), nn.ReLU()
        )
        self.features = nn.Sequential(nn.Conv1d(64, 128, 1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, classes))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Classifies a batch of clouds.

        Parameters
        ----------
            points : torch.Tensor
                clouds with shape (batch, points, 3)

        Returns
        -------
            torch.Tensor
                unnormalised class scores with shape (batch, classes)
        """
        # Conv1d wants channels first, so (batch, 3, points)
        local = self.local(points.transpose(1, 2))
        global_features = self.features(local).amax(dim=2)
        return self.head(global_features)


class SegmentNet(nn.Module):
    """
    A cut-down PointNet part segmentation network.

    This works like PointNet but rather than classifying the whole cloud, the global
    feature is copied back to every point and joined to that point's local feature,
    so each point is labelled knowing both where it is and what shape it belongs to.

    Attributes
    ----------
        local : nn.Sequential
            per-point layers lifting xyz to 64 features
        features : nn.Sequential
            per-point layers lifting 64 to 128 features before the max-pool
        head : nn.Sequential
            per-point layers mapping the 192 joined features to part scores
    """

    def __init__(self, parts: int) -> None:
        """
        Parameters
        ----------
            parts : int
                number of part labels
        """
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv1d(3, 32, 1), nn.ReLU(), nn.Conv1d(32, 64, 1), nn.ReLU()
        )
        self.features = nn.Sequential(nn.Conv1d(64, 128, 1), nn.ReLU())
        self.head = nn.Sequential(
            nn.Conv1d(192, 64, 1), nn.ReLU(), nn.Conv1d(64, parts, 1)
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Labels every point in a batch of clouds.

        Parameters
        ----------
            points : torch.Tensor
                clouds with shape (batch, points, 3)

        Returns
        -------
            torch.Tensor
                unnormalised part scores with shape (batch, parts, points), which is
                the layout nn.CrossEntropyLoss expects
        """
        local = self.local(points.transpose(1, 2))
        global_features = self.features(local).amax(dim=2, keepdim=True)
        context = global_features.expand(-1, -1, points.shape[1])
        return self.head(torch.cat([local, context], dim=1))


def normalise(points: np.ndarray) -> np.ndarray:
    """
    Centres a cloud and scales its furthest point to radius one.

    Parameters
    ----------
        points : np.ndarray
            finite coordinates with shape (points, 3)

    Returns
    -------
        np.ndarray
            float32 cloud with the same shape, centred on the origin

    Raises
    ------
        ValueError
            if the cloud is empty, not (N, 3) or contains NaN or inf
    """
    points = np.asarray(points, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not len(points)
        or not np.isfinite(points).all()
    ):
        raise ValueError("Expected a non-empty, finite (N, 3) cloud")
    centred = points - points.mean(axis=0)
    radius = np.linalg.norm(centred, axis=1).max()
    return (centred / radius if radius > 0 else centred).astype(np.float32)


def sample_triangles(
    triangles: np.ndarray, count: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Samples triangles in proportion to their surface area.

    Parameters
    ----------
        triangles : np.ndarray
            vertices with shape (faces, 3, 3)
        count : int
            number of points to sample
        seed : int
            seed for a local random generator

    Returns
    -------
        tuple[np.ndarray, np.ndarray]
            the sampled points with shape (count, 3) and the index of the face each
            one came from

    Raises
    ------
        ValueError
            if count is less than one or the mesh has no surface area
    """
    rng = np.random.default_rng(seed)
    areas = (
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        )
        / 2
    )
    if count < 1 or areas.sum() <= 0:
        raise ValueError("Need a positive point count and non-zero surface area")
    faces = rng.choice(len(triangles), count, p=areas / areas.sum())
    uv = rng.random((count, 2))
    # fold points from the far half of the parallelogram back into the triangle
    uv[uv.sum(axis=1) > 1] = 1 - uv[uv.sum(axis=1) > 1]
    selected = triangles[faces]
    points = (
        selected[:, 0]
        + uv[:, :1] * (selected[:, 1] - selected[:, 0])
        + uv[:, 1:] * (selected[:, 2] - selected[:, 0])
    )
    return points, faces


def box_surface(
    count: int, extents: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    Samples points on the surface of a box centred on the origin.

    Each point picks a face pair in proportion to its area, so a long thin box
    doesn't end up with crowded ends.

    Parameters
    ----------
        count : int
            number of points to sample
        extents : np.ndarray
            box size along x, y and z
        rng : np.random.Generator
            random generator to draw from

    Returns
    -------
        np.ndarray
            points with shape (count, 3)
    """
    points = rng.uniform(-0.5, 0.5, (count, 3))
    areas = np.array(
        [extents[1] * extents[2], extents[0] * extents[2], extents[0] * extents[1]]
    )
    axes = rng.choice(3, count, p=areas / areas.sum())
    # push one coordinate of each point onto a face of the unit cube
    points[np.arange(count), axes] = rng.choice([-0.5, 0.5], count)
    return points * extents


def make_shapes(
    per_class: int, count: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Generates independent sphere, cube and upright rod examples.

    Parameters
    ----------
        per_class : int
            number of shapes in each class
        count : int
            points per shape
        seed : int
            seed for geometry and sampling

    Returns
    -------
        tuple[torch.Tensor, torch.Tensor, list[str]]
            normalised clouds with shape (3 * per_class, count, 3), integer labels
            and the class names
    """
    rng = np.random.default_rng(seed)
    clouds: list[np.ndarray] = []
    labels: list[int] = []
    for label in range(3):
        for _ in range(per_class):
            if label == 0:
                points = rng.normal(size=(count, 3))
                points /= np.linalg.norm(points, axis=1, keepdims=True)
            else:
                extents = np.ones(3) if label == 1 else np.array([0.35, 2.0, 0.35])
                extents *= rng.uniform(0.9, 1.1, 3)
                points = box_surface(count, extents, rng)
            clouds.append(normalise(points))
            labels.append(label)
    return (
        torch.tensor(np.stack(clouds)),
        torch.tensor(labels),
        ["sphere", "cube", "rod"],
    )


def make_parts(
    shapes: int, count: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Generates chairs with seat, back and leg labels.

    Parameters
    ----------
        shapes : int
            number of independent chairs
        count : int
            labelled points per chair
        seed : int
            seed for proportions and sampling

    Returns
    -------
        tuple[torch.Tensor, torch.Tensor, list[str]]
            normalised clouds with shape (shapes, count, 3), per-point part labels
            with shape (shapes, count) and the part names
    """
    rng = np.random.default_rng(seed)
    clouds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for _ in range(shapes):
        width = rng.uniform(0.8, 1.2)
        seat = box_surface(count // 3, np.array([width, 0.18, 1.0]), rng)
        back = box_surface(count // 3, np.array([width, 1.0, 0.15]), rng) + [
            0,
            0.6,
            0.45,
        ]
        leg_count = count - len(seat) - len(back)
        legs = box_surface(leg_count, np.array([0.12, 0.8, 0.12]), rng)
        corners = np.array(
            [
                [-width / 2, -0.5, -0.4],
                [width / 2, -0.5, -0.4],
                [-width / 2, -0.5, 0.4],
                [width / 2, -0.5, 0.4],
            ]
        )
        legs += corners[rng.integers(0, 4, leg_count)]
        points = np.concatenate([seat, back, legs])
        parts = np.repeat([0, 1, 2], [len(seat), len(back), leg_count])
        # shuffle so the model can't learn the part from the point index
        order = rng.permutation(count)
        clouds.append(normalise(points)[order])
        labels.append(parts[order])
    return (
        torch.tensor(np.stack(clouds)),
        torch.tensor(np.stack(labels)),
        ["seat", "back", "legs"],
    )


def rotate(
    points: torch.Tensor, generator: torch.Generator, full: bool = False
) -> torch.Tensor:
    """
    Applies an independent rotation to each cloud, preserving its scale.

    Parameters
    ----------
        points : torch.Tensor
            CPU clouds with shape (batch, points, 3)
        generator : torch.Generator
            local random generator
        full : bool
            use uniform SO(3) rotations instead of rotations about y

    Returns
    -------
        torch.Tensor
            rotated clouds with the same shape as points
    """
    batch = len(points)
    if full:
        # a normalised Gaussian quaternion is a uniform random rotation
        q = torch.randn(batch, 4, generator=generator)
        q /= q.norm(dim=1, keepdim=True)
        w, x, y, z = q.unbind(1)
        matrix = torch.stack(
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
            dim=1,
        ).reshape(batch, 3, 3)
    else:
        angle = torch.rand(batch, generator=generator) * (2 * torch.pi)
        c, s = angle.cos(), angle.sin()
        zero, one = torch.zeros_like(c), torch.ones_like(c)
        matrix = torch.stack([c, zero, s, zero, one, zero, -s, zero, c], dim=1).reshape(
            batch, 3, 3
        )
    return points @ matrix.transpose(1, 2)


def split_indices(labels: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Splits shapes into training and validation sets, class by class.

    Roughly 20% of each class goes to validation (at least one shape), so every class
    shows up in both sets.

    Parameters
    ----------
        labels : torch.Tensor
            class label for each shape
        seed : int
            seed for the shuffle

    Returns
    -------
        tuple[torch.Tensor, torch.Tensor]
            indices of the training shapes and of the validation shapes

    Raises
    ------
        ValueError
            if any class has fewer than two shapes
    """
    generator = torch.Generator().manual_seed(seed)
    train: list[torch.Tensor] = []
    val: list[torch.Tensor] = []
    for label in labels.unique():
        indices = torch.where(labels == label)[0]
        if len(indices) < 2:
            raise ValueError("Need at least two training shapes per class")
        indices = indices[torch.randperm(len(indices), generator=generator)]
        size = max(1, int(len(indices) * 0.2))
        val.append(indices[:size])
        train.append(indices[size:])
    return torch.cat(train), torch.cat(val)


def load_modelnet(
    root: str,
    split: str,
    count: int,
    per_class: int,
    seed: int,
    return_sources: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, list[str]]
    | tuple[torch.Tensor, torch.Tensor, list[str], list[dict[str, Any]]]
):
    """
    Reads an extracted ModelNet directory without downloading data.

    Parameters
    ----------
        root : str
            directory containing category/train and category/test folders
        split : str
            train or test
        count : int
            sampled points per mesh
        per_class : int
            maximum meshes per category; zero means all
        seed : int
            seed for file selection and surface sampling
        return_sources : bool
            also return mesh paths and the sampled centroid and radius for display

    Returns
    -------
        tuple
            normalised clouds with shape (meshes, count, 3), integer labels and the
            category names. When return_sources is set a fourth item is added, a list
            of dicts with path, centre and radius keys so the original mesh can be
            drawn in the same space as its cloud

    Raises
    ------
        ValueError
            if split is not train or test, no categories are found or a category
            has no OFF files for the split
    """
    import trimesh

    if split not in {"train", "test"}:
        raise ValueError("ModelNet split must be train or test")
    root_path = Path(root).expanduser()
    categories = sorted(
        p
        for p in root_path.iterdir()
        if (p / "train").is_dir() and (p / "test").is_dir()
    )
    if not categories:
        raise ValueError(
            "Select the extracted ModelNet root containing category/train and category/test"
        )
    rng = np.random.default_rng(seed)
    clouds: list[np.ndarray] = []
    labels: list[int] = []
    sources: list[dict[str, Any]] = []
    for label, category in enumerate(categories):
        files = sorted((category / split).glob("*.off"))
        if not files:
            raise ValueError(f"No OFF meshes in {category / split}")
        files = [files[i] for i in rng.permutation(len(files))]
        if per_class:
            files = files[:per_class]
        for path in files:
            mesh = trimesh.load_mesh(path, process=False)
            points, _ = trimesh.sample.sample_surface(
                mesh, count, seed=int(rng.integers(2**31))
            )
            clouds.append(normalise(points))
            labels.append(label)
            if return_sources:
                centre = points.mean(axis=0)
                sources.append(
                    dict(
                        path=path,
                        centre=centre,
                        radius=float(np.linalg.norm(points - centre, axis=1).max()),
                    )
                )
    result = (
        torch.tensor(np.stack(clouds)),
        torch.tensor(labels),
        [p.name for p in categories],
    )
    return (*result, sources) if return_sources else result


def load_shapenet(
    root: str, split: str, count: int, limit: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Loads labelled chairs from a prepared PyTorch Geometric cache.

    Parameters
    ----------
        root : str
            existing ShapeNet cache; raw data must already be present
        split : str
            train, val or test
        count : int
            sampled points per shape
        limit : int
            maximum shapes; zero means all
        seed : int
            seed for file selection and matched point/label sampling

    Returns
    -------
        tuple[torch.Tensor, torch.Tensor, list[str]]
            normalised clouds with shape (shapes, count, 3), per-point part labels
            remapped to 0 to 3 and the part names

    Raises
    ------
        ValueError
            if the cache is missing or incomplete, or the chair labels are not 12
            through 15
    """
    cache = Path(root).expanduser()
    processed = [
        cache / "processed" / f"cha_{name}.pt"
        for name in ("train", "val", "test", "trainval")
    ]
    if (
        not all(path.is_file() for path in processed)
        or not (cache / "raw" / "train_test_split").is_dir()
    ):
        raise ValueError(
            "Prepare the ShapeNet cache first; this lesson does not download it automatically"
        )
    from torch_geometric.datasets import ShapeNet

    if not all(
        (cache / "raw" / name).exists() for name in ShapeNet.category_ids.values()
    ):
        raise ValueError(
            "The raw ShapeNet cache is incomplete; prepare it explicitly before this lesson"
        )
    dataset = ShapeNet(
        str(cache), categories=["Chair"], split=split, include_normals=False
    )
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))
    if limit:
        indices = indices[:limit]
    clouds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for index in indices:
        item = dataset[int(index)]
        selected = rng.choice(len(item.pos), count, replace=len(item.pos) < count)
        clouds.append(normalise(item.pos.numpy()[selected]))
        targets.append(item.y.numpy()[selected])
    # Chair labels in ShapeNet Part are 12, 13, 14 and 15.
    stacked = np.stack(targets) - 12
    if stacked.min() < 0 or stacked.max() > 3:
        raise ValueError("Expected ShapeNet Chair labels 12 through 15")
    return (
        torch.tensor(np.stack(clouds)),
        torch.tensor(stacked),
        ["part 12", "part 13", "part 14", "part 15"],
    )


def part_iou(
    prediction: torch.Tensor, target: torch.Tensor, parts: int
) -> torch.Tensor:
    """
    Works out the intersection over union of each part for every shape.

    If a part is missing from both the prediction and the target for a shape, the
    score is 1, because the model correctly predicted that the part isn't there.

    Parameters
    ----------
        prediction : torch.Tensor
            predicted part labels with shape (batch, points)
        target : torch.Tensor
            true part labels with shape (batch, points)
        parts : int
            number of part labels

    Returns
    -------
        torch.Tensor
            IoU scores with shape (batch, parts)
    """
    scores: list[torch.Tensor] = []
    for part in range(parts):
        predicted, actual = prediction == part, target == part
        intersection = (predicted & actual).sum(dim=1)
        union = (predicted | actual).sum(dim=1)
        scores.append(torch.where(union > 0, intersection / union.clamp_min(1), 1.0))
    return torch.stack(scores, dim=1)


def cloud_figure(
    points: np.ndarray,
    title: str,
    colours: np.ndarray | None = None,
    labels: list[str] | None = None,
) -> "go.Figure":
    """
    Creates a rotatable point-cloud view with y as the upright axis.

    Parameters
    ----------
        points : np.ndarray
            coordinates with shape (points, 3)
        title : str
            title displayed above the view
        colours : np.ndarray, optional
            scalar activations, or integer part IDs when labels are supplied
        labels : list[str], optional
            names for categorical part IDs, starting at zero

    Returns
    -------
        go.Figure
            plotly figure with one trace per part, or a single trace when no labels
            are given
    """
    import plotly.graph_objects as go

    figure = go.Figure()
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    if labels is not None:
        for index, name in enumerate(labels):
            selected = np.asarray(colours) == index
            figure.add_trace(
                go.Scatter3d(
                    x=points[selected, 0],
                    y=points[selected, 1],
                    z=points[selected, 2],
                    mode="markers",
                    name=name,
                    legendgroup=name,
                    marker=dict(size=3, color=palette[index % len(palette)]),
                    customdata=np.flatnonzero(selected),
                    hovertemplate="Point %{customdata}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>%{fullData.name}</extra>",
                )
            )
    else:
        marker = (
            dict(size=3, color=palette[0])
            if colours is None
            else dict(
                size=3,
                color=colours,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Activation"),
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                name="Points",
                marker=marker,
            )
        )
    figure.update_layout(
        title=title,
        height=550,
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(
            aspectmode="cube",
            dragmode="orbit",
            xaxis=dict(title="x", range=[-1.1, 1.1]),
            yaxis=dict(title="y", range=[-1.1, 1.1]),
            zaxis=dict(title="z", range=[-1.1, 1.1]),
            camera=dict(up=dict(x=0, y=1, z=0), eye=dict(x=1.5, y=1.2, z=1.5)),
        ),
    )
    return figure

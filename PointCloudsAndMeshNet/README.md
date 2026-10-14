# Point Clouds for Machine Learning

Seven marimo lessons on learning from unordered 3D points. We start with surface sampling and small arrays, build a simplified PointNet, then investigate classification, rotation augmentation and part segmentation.

| Notebook | Covers |
| --- | --- |
| [Part 1: sampling](PointCloudForMLPart1Sampling.py) | Triangle area, normalisation, interactive Plotly mesh/cloud comparison and overlay |
| [Part 2: sets](PointCloudForMLPart2Sets.py) | Flattened MLP baseline, shared mappings, invariance and equivariance |
| [Part 3: PointNet](PointCloudForMLPart3PointNet.py) | Kernel-size-one convolutions, tensor layouts, pooling contributors |
| [Part 4: classification](PointCloudForMLPart4Classification.py) | Generated shapes or ModelNet, validation selection, confusion matrix, checkpoint |
| [Part 5: robustness](PointCloudForMLPart5Robustness.py) | Paired augmentation experiment, rotations, noise, point removal and spatial cropping |
| [Part 6: segmentation](PointCloudForMLPart6Segmentation.py) | Local/global features, generated chairs or ShapeNet Part, label alignment, worked loss, part IoU, checkpoint save/reload |
| [Part 7: ModelNet10 identification](PointCloudForMLPart7ModelNet.py) | Download and cache the dataset, train a classifier, inspect held-out object labels and class scores |

It is expected you will use uv to run the notebooks from the repository root:

```bash
uv run --with plotly --with 'trimesh>=4.6' marimo edit PointCloudForML/PointCloudForMLPart1Sampling.py
```

Part 1 uses Plotly for mouse rotation, zoom and point inspection, with trimesh for the mesh-loading form. Its mesh and cloud share one centring/scale transform; the side-by-side panels rotate independently and the overlay rotates both representations together. Leave the mesh path blank and press Sample mesh for a generated box. The remaining figures also use Plotly, including feature activations, classification results, robustness error bars and labelled segmentation. Add `--with plotly` when launching Parts 3–7. Part 2 contains numerical examples without plots. Multiple 3D panels start with matching cameras and rotate independently; part colours stay consistent between ground truth and predictions. Training waits for a form submission; opening a notebook does not start training. The generated shapes exercise the pipeline and are not substitutes for benchmark results. CPU is the default. Geometry uses y as the upright axis. Training forms use a short budget which students can increase.

For trimesh examples and ModelNet loading:

```bash
uv run --with plotly --with 'trimesh>=4.6' marimo edit PointCloudForML/PointCloudForMLPart4Classification.py
```

Part 7 downloads and extracts [ModelNet10](https://modelnet.cs.princeton.edu/download.html) from the official HTTPS archive when its download button is pressed. It caches the archive and meshes under `PointCloudForML/data` (or `/transfer/modelnet10` in the labs), then reuses them on later runs. Start it with:

```bash
uv run --with plotly --with 'trimesh>=4.6' marimo edit PointCloudForML/PointCloudForMLPart7ModelNet.py
```

Press Download and prepare ModelNet10, then submit Sample ModelNet10 and train. The default limits the training subset to 40 meshes per category and uses the full test split. Its identification viewer shows the actual category, predicted category and softmax scores. The archive is about 451 MiB; extraction needs additional disk space.

For Parts 4 and 5, download and extract ModelNet10 separately or reuse Part 7's prepared directory. Enter the directory containing category folders such as `chair/train` and `chair/test`. The loader selects meshes before sampling and keeps training, validation and official test objects separate. The training limit is per category; zero uses all available training meshes. ModelNet40 uses the same directory layout, with a larger training cost. [PyTorch Geometric also provides ModelNet](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.ModelNet.html), but we use trimesh here to keep the sampling operation visible.

For ShapeNet Part, Part 6 explains how to prepare a [PyTorch Geometric ShapeNet cache](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.ShapeNet.html) explicitly and run with `--with plotly --with torch-geometric`. We begin with Chair only. Sampling uses the same indices for points and labels; normals are excluded. The notebook uses all validation and test chairs, whilst the training limit is adjustable. Preparing data can require a download and local processing; check the dataset access conditions.

Checkpoints are saved only after pressing their Save button. Classification uses `PointCloudForML/outputs/classifier.pt`; Part 6 defaults to `PointCloudForML/outputs/segmentation.pt` and verifies a reload against the model’s predictions. Part 7 saves its checkpoint as `PointCloudForML/outputs/modelnet10_classifier.pt`. Use `uv run scripts/clean_data.py --dry-run` to list downloaded caches, then run without `--dry-run` to remove them after confirmation. The cleanup script includes `PointCloudForML/data`, preserves saved models, and does not touch `/transfer`. The shared model and data helpers are in `pointcloud.py`; the lessons show their source where the calculation matters. They omit the original PointNet transformation networks, batch normalisation and dropout.

By the end students should be able to prepare geometry, explain set-based architectures, train with separate validation and test data, design a controlled robustness experiment, and interpret numerical results alongside 3D figures. Each lesson ends with exercises which change or diagnose the examples.

## Developer notes

The tests are in `tests/` and use [pytest](https://docs.pytest.org/). The ones which train a model in a notebook are marked `slow`, so I skip them when I just want a quick check.

```bash
uv run pytest PointCloudsAndMeshNet/tests
uv run pytest PointCloudsAndMeshNet/tests -m "not slow"
ruff check PointCloudsAndMeshNet
ruff format --check PointCloudsAndMeshNet
uv run marimo check --strict PointCloudsAndMeshNet/PointCloudForMLPart*.py
```

`marimo export html` runs the default examples and produces a standalone preview. Forms remain unsubmitted in those exports, so training and optional datasets need a separate interactive or programmatic check.

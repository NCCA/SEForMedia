# Point Clouds for Machine Learning

Seven marimo lessons on learning from unordered 3D points. We start with surface sampling and small arrays, build a simplified PointNet, then investigate classification, rotation augmentation and part segmentation.

| Notebook                                                             | Covers                                                                                                                   |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| [Part 1: sampling](PointCloudForMLPart1Sampling.py)                  | Triangle area, normalisation, interactive Plotly mesh/cloud comparison and overlay                                       |
| [Part 2: sets](PointCloudForMLPart2Sets.py)                          | Flattened MLP baseline, shared mappings, invariance and equivariance                                                     |
| [Part 3: PointNet](PointCloudForMLPart3PointNet.py)                  | Kernel-size-one convolutions, tensor layouts, pooling contributors                                                       |
| [Part 4: classification](PointCloudForMLPart4Classification.py)      | Generated shapes or ModelNet, validation selection, confusion matrix, checkpoint                                         |
| [Part 5: robustness](PointCloudForMLPart5Robustness.py)              | Paired augmentation experiment, rotations, noise, point removal and spatial cropping                                     |
| [Part 6: segmentation](PointCloudForMLPart6Segmentation.py)          | Local/global features, generated chairs or ShapeNet Part, label alignment, worked loss, part IoU, checkpoint save/reload |
| [Part 7: ModelNet10 identification](PointCloudForMLPart7ModelNet.py) | Download and cache the dataset, train a classifier, inspect held-out object labels and class scores                      |

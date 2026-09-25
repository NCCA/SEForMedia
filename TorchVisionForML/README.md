# torchvision for Machine Learning

Four marimo notebooks covering the torchvision functions used in the machine learning demos in this repository, following on from `NumPyForML/` and `PyTorchForML/`.

torchvision accounts for 63 call sites across 12 demos here, and 56 of those are `transforms`, so that is where the weight of these notebooks sits. Each function gets its signature, its parameters with defaults, a link to the pytorch.org page, a runnable example, and a note on what goes wrong when it is used carelessly.

| Notebook | Covers |
| --- | --- |
| `TorchVisionForMLPart1Images.py` | `read_image`, `ImageReadMode`, CHW vs HWC, uint8 vs float, `to_pil_image`, `make_grid` |
| `TorchVisionForMLPart2Transforms.py` | v1 vs v2, `Compose`, `ToImage`/`ToDtype`, `Normalize`, `Resize`, `Grayscale`, ordering, `tv_tensors` |
| `TorchVisionForMLPart3Augmentation.py` | `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`, `RandomResizedCrop`, label preservation, train vs validation pipelines |
| `TorchVisionForMLPart4Datasets.py` | `ImageFolder`, the weights enums, `weights.transforms()`, freezing, transfer learning |

Run them with uv:

```
uv run marimo edit TorchVisionForML/TorchVisionForMLPart2Transforms.py
```

Nothing downloads — the images are generated, and Part 4 builds the model architectures with `weights=None` so it works offline in the labs. The weights metadata, including the preprocessing preset and the 1000 ImageNet category names, is available locally without fetching anything, so the interesting half of the pre-trained model story runs regardless.

Two things in here are worth reading even if you know torchvision. Part 2 demonstrates that `ToTensor()` rescales a PIL image to 0–1 but leaves an existing `uint8` tensor at 0–255 — the same line in your `Compose` doing different things depending on how the image was loaded, silently. Part 3 measures what `RandomResizedCrop`'s default `scale=(0.08, 1.0)` actually does to a small subject, which is not what most people assume they are asking for.

# torchvision for Machine Learning

These four marimo notebooks cover the torchvision code used in the machine learning demos. I use generated images so we can work through the examples without downloading a dataset. They follow on from `NumPyForML/` and `PyTorchForML/`.

| Notebook | Covers |
| --- | --- |
| [Part 1: images](TorchVisionForMLPart1Images.py) | Reading images, tensor layout, value ranges and display |
| [Part 2: transforms](TorchVisionForMLPart2Transforms.py) | Conversion, resizing, normalisation and composing transforms |
| [Part 3: augmentation](TorchVisionForMLPart3Augmentation.py) | Random transforms, checking labels and validation preprocessing |
| [Part 4: datasets](TorchVisionForMLPart4Datasets.py) | Class folders, model weights and transfer learning |

Run a notebook from the repository root using uv:

```sh
uv run marimo edit TorchVisionForML/TorchVisionForMLPart1Images.py
```

Part 4 uses randomly initialised models to inspect their structure without downloading weights. Its code snippets show how to load trained weights when preparing a training run.

The [torchvision documentation](https://pytorch.org/vision/stable/index.html) has the full API reference.

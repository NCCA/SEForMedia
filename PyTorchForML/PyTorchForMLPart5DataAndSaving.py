#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PyTorch for Machine Learning, Part 5: data pipelines and saving models

    In this final notebook, we look at loading and preparing data for a model, then saving the trained model for later use.

    Our smaller demos keep the entire dataset in memory as a single tensor. Larger datasets may need to be loaded in batches, whilst images often need resizing, conversion to tensors and augmentation during training. We will use `torchvision.transforms` as part of this pipeline.

    We will also look at saving model parameters and loading them again for inference or further training.

    The examples use synthetic images, so no dataset downloads are needed.
    """)
    return


@app.cell
def _():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import v2

    torch.manual_seed(42)
    return DataLoader, Dataset, nn, torch, v2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)

    An abstract class asking you for two methods:

    ```python
    class MyDataset(Dataset):
        def __len__(self):            # how many samples
            ...
        def __getitem__(self, idx):   # return sample idx, usually (data, label)
            ...
    ```

    Implementing that pair means the dataset can be indexed lazily, one sample at a time. The file is read, decoded and transformed only when that index is actually asked for, so a dataset of 100,000 images costs you a list of filenames in memory rather than 100,000 images.

    `PreTrainedModels/TransferLearning.py:51` defines one over a directory of images. The one below fakes the file reading with a print so you can see when the work happens.
    """)
    return


@app.cell
def _(Dataset, torch):
    class FakeImageDataset(Dataset):
        """Pretends to load images from disk, so we can see when it does the work."""

        def __init__(self, count: int, transform=None, noisy: bool = False):
            self.filenames = [f"image_{i:03d}.png" for i in range(count)]
            self.labels = torch.randint(0, 3, (count,))
            self.transform = transform
            self.noisy = noisy

        def __len__(self) -> int:
            return len(self.filenames)

        def __getitem__(self, idx: int):
            if self.noisy:
                print(f"    reading {self.filenames[idx]}")
            # stand-in for reading and decoding a real file
            image = torch.randint(0, 256, (3, 16, 16), dtype=torch.uint8)
            if self.transform is not None:
                image = self.transform(image)
            return image, self.labels[idx]

    chatty = FakeImageDataset(100, noisy=True)
    print("the dataset holds", len(chatty), "samples but has read nothing yet")
    print()
    print("asking for two of them:")
    _img, _label = chatty[0]
    _img2, _label2 = chatty[7]
    print()
    print(
        "sample 0 shape",
        tuple(_img.shape),
        "dtype",
        _img.dtype,
        "label",
        _label.item(),
    )
    return (FakeImageDataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### [TensorDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset) and [random_split](https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split)

    When our inputs and targets are already tensors in memory, `TensorDataset` lets us access them together without writing a custom `Dataset` class. Each index returns the corresponding entry from every tensor.

    ```python
    TensorDataset(*tensors)
    random_split(dataset, lengths, generator=None)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `*tensors` | required | Tensors to index together. Their first dimensions must have the same size. |
    | `lengths` | required | Split sizes as counts that total `len(dataset)`, or fractions that sum to 1. |
    | `generator` | `None` | Controls the random split. Use a seeded generator for repeatable results. |

    We can create a dataset and divide it into training and validation subsets:

    ```python
    dataset = TensorDataset(X, y)
    generator = torch.Generator().manual_seed(42)

    train_data, validation_data = random_split(
        dataset, [0.8, 0.2], generator=generator
    )
    ```

    The returned `Subset` objects reference the original dataset and store the selected indices. They do not copy the underlying samples.

    Keeping the split fixed makes validation scores easier to compare between runs. Recreating the seeded generator reproduces the split, provided the dataset size and ordering stay the same.

    Changing the split does not automatically cause data leakage when we train a  a fresh model. However, if we continue training an existing model and move previously seen samples into validation, that validation set is no longer independent of training.
    """)
    return


@app.cell
def _(torch):
    from torch.utils.data import TensorDataset, random_split

    features = torch.randn(100, 4)
    targets = torch.randint(0, 3, (100,))
    in_memory = TensorDataset(features, targets)

    sample_x, sample_y = in_memory[0]
    print("samples:", len(in_memory))
    print("one sample:", tuple(sample_x.shape), "label", sample_y.item())

    split_generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(in_memory, [0.8, 0.2], generator=split_generator)
    print()
    print(
        "split into",
        len(train_set),
        "training and",
        len(val_set),
        "validation",
    )

    # same seed, same split - which is the whole reason to pass a generator
    again = random_split(
        in_memory, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
    )
    print(
        "indices repeat with the same seed:",
        again[0].indices == train_set.indices,
    )
    print(
        "and the two subsets do not overlap:",
        not (set(train_set.indices) & set(val_set.indices)),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

    `DataLoader` takes a dataset and provides batches that we can iterate over during training or evaluation.

    ```python
    DataLoader(dataset, batch_size=1, shuffle=None, num_workers=0,
               pin_memory=False, drop_last=False)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `batch_size` | `1` | Number of samples per batch. |
    | `shuffle` | `None` | Set to `True` to shuffle the sample order on each pass through the loader. |
    | `num_workers` | `0` | Number of worker processes used to load data. With `0`, loading happens in the main process. |
    | `pin_memory` | `False` | Places tensors in page-locked host memory, which can speed up transfers to a CUDA GPU. |
    | `drop_last` | `False` | Discards the final batch if it contains fewer than `batch_size` samples. |

    The loader handles three useful jobs:

    - **Batching:** by default, it stacks matching sample tensors along a new batch dimension. Processing several samples together helps us use the hardware efficiently.
    - **Shuffling:** for training, changing the order each epoch varies which samples appear together and reduces the effects of ordering in the dataset. We normally leave validation data unshuffled, as in `TransferLearning.ipynb`.
    - **Parallel loading:** with worker processes enabled, data loading and preparation can overlap with model computation, reducing the time spent waiting for the next batch.

    The best batch size and worker count depend on the dataset and available hardware. More workers can help when loading is slow, but their overhead may outweigh the benefit for a small dataset already held in memory.
    """)
    return


@app.cell
def _(DataLoader, FakeImageDataset):
    quiet = FakeImageDataset(100)
    loader = DataLoader(quiet, batch_size=16, shuffle=True)

    print("100 samples at batch_size=16 gives", len(loader), "batches")

    for batch_no, (images, labels) in enumerate(loader):
        print(
            f"  batch {batch_no}: images {tuple(images.shape)}  labels {tuple(labels.shape)}"
        )
        if batch_no >= 1:
            print("  ...")
            break

    last = list(loader)[-1]
    print()
    print(
        "the final batch is short:",
        tuple(last[0].shape),
        "- 100 is not a multiple of 16",
    )
    print("drop_last=True would discard it")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That short final batch is worth a thought. It is harmless for most training, but if your model uses `BatchNorm` and the last batch happens to contain one sample, you get the error from Part 4. `drop_last=True` is the usual fix.

    On `num_workers`: more is not always better, and on Windows and in notebooks it can misbehave because of how subprocesses start. Start at 0, and only raise it if you can show the GPU is waiting for data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [transforms.v2](https://pytorch.org/vision/stable/transforms.html)

    `torchvision.transforms.v2` provides operations for preparing images. We can combine them into a single callable using `Compose`.

    ```python
    v2.Compose([...])             # apply transforms in sequence
    v2.ToImage()                  # convert to a tv_tensors.Image
    v2.ToDtype(dtype, scale=False) # change dtype, optionally scaling values
    v2.Resize(size, antialias=True)
    v2.Grayscale(num_output_channels=1)
    ```

    The usual v2 replacement for `ToTensor()` combines image conversion with conversion to floating point:

    ```python
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    ```

    `MNIST/PyTorchDataLoaders.py` uses this pattern.

    Pay attention to `scale=True`. For a `uint8` image, it converts pixel values from the range 0–255 to floating-point values in the range 0–1. The default, `scale=False`, changes the data type but leaves the values unchanged.

    The input range should match what the model expects. Accidentally supplying values up to 255 instead of 1 changes the scale of its inputs and can make training harder. `ToImage()` alone does not rescale the pixels.
    """)
    return


@app.cell
def _(torch, v2):
    raw = torch.randint(0, 256, (3, 16, 16), dtype=torch.uint8)

    without_scale = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)])
    with_scale = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    print("raw uint8        min", raw.min().item(), "max", raw.max().item())
    print(
        "scale=False      min",
        without_scale(raw).min().item(),
        "max",
        without_scale(raw).max().item(),
    )
    print(
        "scale=True       min",
        round(with_scale(raw).min().item(), 4),
        "max",
        round(with_scale(raw).max().item(), 4),
    )
    print()
    print("only the second one is what a network expects")
    return (raw,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Augmentation

    Data augmentation introduces random variations when we load training images. These can include flips, rotations, colour changes and crops:

    ```python
    v2.RandomHorizontalFlip(p=0.5)
    v2.RandomRotation(degrees=20)
    v2.ColorJitter(brightness=0.2, contrast=0.2)
    v2.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.333))
    ```

    The ASL and transfer-learning demos use these transforms to vary the images presented during training. This can help the model generalise to changes in lighting, orientation and framing.

    We need to consider two things when choosing augmentations:

    1. **Keep routine validation preprocessing deterministic.** Random augmentation makes scores vary with the sampled transformations, which makes comparisons harder. Deliberately testing robustness to transformations is a separate evaluation.
    2. **Preserve the target label.** A transformed image must still represent the class we assign to it.

    The ASL notebook discusses this second point using handedness. It treats horizontal flipping as a way to represent signing with the other hand, whilst limiting rotation to 20 degrees to avoid changing the meaning of the gesture.

    That choice depends on the task. Mirroring a printed “b” can make it resemble a “d”, so the same transform would be unsuitable if we kept the original label. We need to inspect the transformed samples and use our knowledge of the subject to decide which variations are valid.
    """)
    return


@app.cell
def _(raw, torch, v2):
    augment = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=20),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    print("the same source image through the same pipeline four times:")
    for run in range(4):
        out = augment(raw)
        print(f"  run {run}: shape {tuple(out.shape)}  mean {out.mean().item():.4f}")
    print()
    print("different every time, which is the point - and the shape never changes,")
    print("which is what lets them be batched")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [state_dict](https://pytorch.org/docs/stable/generated/torch.nn.Module.state_dict.html) and saving


    ```python
    model.state_dict()                      # OrderedDict of parameter name -> tensor
    torch.save(obj, f)
    torch.load(f, map_location=None, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    ```

    A `state_dict` is **not the model**. It carries the learned numbers and nothing about the architecture, which is exactly why it is the recommended thing to save, the code that defines the layers stays in version control where it belongs, and the file holds only what the code cannot reproduce.

    `LinearModel.py:120` saves the recommended way:

    ```python
    torch.save(obj=model.state_dict(), f="LRModel.pth")
    ```

    Restoring is two steps and the order matters: build the architecture first, then assings the numbers.
    """)
    return


@app.cell
def _(nn, torch):
    import tempfile
    from pathlib import Path

    tmp_dir = Path(tempfile.mkdtemp())

    original = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    print("what a state_dict actually contains:")
    for key, value in original.state_dict().items():
        print(f"  {key:10} {tuple(value.shape)}")

    save_path = tmp_dir / "model.pth"
    torch.save(original.state_dict(), save_path)
    print()
    print("saved", save_path.stat().st_size, "bytes")

    # architecture first, then the numbers
    restored = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    restored.load_state_dict(torch.load(save_path))

    probe = torch.randn(3, 4)
    print(
        "restored model matches the original:",
        torch.allclose(original(probe), restored(probe)),
    )
    return save_path, tmp_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `strict`, and loading into a different architecture

    ```python
    model.load_state_dict(state_dict, strict=True)
    ```

    | Parameter | Default | What it does |
    | --- | --- | --- |
    | `strict` | `True` | Requires the saved keys to match the keys expected by the model, with none missing or unexpected. |

    `CheckPointsMarimo.py` uses `strict=True` to restore a matching model, then `strict=False` to load compatible entries into a different architecture. Loading matches parameters and buffers by name; it does not work out which layers serve similar purposes.

    With `strict=False`, missing and unexpected keys are allowed and returned for us to inspect:

    ```python
    result = model.load_state_dict(state_dict, strict=False)
    print(result.missing_keys)
    print(result.unexpected_keys)
    ```

    Missing entries retain their current values in the model, whilst unexpected entries from the saved dictionary are ignored.

    Shape mismatches still raise an error. If the same key refers to a tensor of shape `(2, 8)` in the saved dictionary and `(5, 8)` in the model, `strict=False` cannot load it. We must remove that entry before loading or change the model to match.

    The examples below show both cases, starting with one that `strict=False` permits: loading a saved dictionary that has no entries for the new model’s additional layers.
    """)
    return


@app.cell
def _(nn, save_path, torch):
    # same first two layers, plus an extra one the saved file knows nothing about
    deeper = nn.Sequential(
        nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2), nn.ReLU(), nn.Linear(2, 2)
    )

    try:
        deeper.load_state_dict(torch.load(save_path), strict=True)
    except RuntimeError as e:
        print("strict=True raises:", str(e).split("\n")[0])

    report = deeper.load_state_dict(torch.load(save_path), strict=False)
    print()
    print("strict=False reports instead of raising:")
    print("  missing keys   :", report.missing_keys)
    print("  unexpected keys:", report.unexpected_keys)
    print()
    print("layers 0 and 2 were loaded; layer 4 is still randomly initialised.")
    print("Always read this return value - loading nothing at all looks identical.")
    return


@app.cell
def _(nn, save_path, torch):
    # and the case strict=False does NOT rescue: a key that matches by name but not by shape
    wrong_width = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 5))

    try:
        wrong_width.load_state_dict(torch.load(save_path), strict=False)
    except RuntimeError as e:
        print("even with strict=False:")
        for line in str(e).split("\n")[:3]:
            print("  ", line)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can use `strict=False` when reusing part of a model, provided the entries with matching names also have compatible shapes. It allows missing and unexpected keys, but does not automatically skip tensors with incompatible shapes.

    If we replace a classification head but keep its parameter names, a shape mismatch will still raise an error. We can omit those entries to keep the new head’s initial values, or adapt the saved tensors where that makes sense. `CheckPointsMarimo.py` demonstrates the latter by slicing tensors in its loop over `loaded_state_dict.items()`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `weights_only`, and why the default changed

    Since PyTorch 2.6, `torch.load()` defaults to `weights_only=True` when no custom `pickle_module` is supplied. The Checkpoints demo explicitly uses `weights_only=False` when loading a complete model object. See the [serialization notes](https://docs.pytorch.org/docs/stable/notes/serialization.html#torch-load-with-weights-only-true).

    This matters because PyTorch uses Python’s pickle machinery for serialization. Unrestricted unpickling can execute arbitrary code from a malicious file, so loading a checkpoint can have consequences beyond reading its tensors.

    `weights_only=True` restricts loading to tensors, basic types, dictionaries and explicitly allowlisted types. This reduces the risk of code execution, but does not make an untrusted checkpoint completely safe.

    For our models, we normally save the `state_dict()` and load it with the restriction enabled:

    ```python
    torch.save(model.state_dict(), "weights.pth")
    state_dict = torch.load("weights.pth", weights_only=True)
    model.load_state_dict(state_dict)
    ```

    This also keeps the model definition separate from its saved parameters. Loading a complete pickled model usually requires `weights_only=False`; we should only do that when we trust the checkpoint’s source.
    """)
    return


@app.cell
def _(nn, tmp_dir, torch):
    whole_model_path = tmp_dir / "whole.pth"
    torch.save(nn.Sequential(nn.Linear(2, 2)), whole_model_path)  # the whole object

    try:
        torch.load(whole_model_path)  # weights_only=True by default
    except Exception as e:
        print(type(e).__name__, ":", str(e).split("\n")[0][:110])

    loaded_whole = torch.load(whole_model_path, weights_only=False)
    print()
    print("with weights_only=False it loads:", loaded_whole)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Saving compiled models

    As we saw in Part 3, `torch.compile()` wraps the original model. The wrapper’s `state_dict()` can contain keys prefixed with `_orig_mod.`, which will not match the names expected by an uncompiled model.

    We can avoid this by keeping a reference to the original model and saving its state:

    ```python
    model = MyModel()
    compiled_model = torch.compile(model)

    # train using compiled_model, then save the shared parameters
    torch.save(model.state_dict(), "weights.pth")
    ```

    Training the compiled model updates the original model’s parameters too. Saving through `model` keeps the usual parameter names, so we can load the checkpoint into an ordinary `MyModel` instance.
    """)
    return


@app.cell
def _(nn, torch):
    base = nn.Linear(3, 2)
    wrapped = torch.compile(base)

    print("wrapped keys:", list(wrapped.state_dict().keys()))
    print("the fix     :", list(wrapped._orig_mod.state_dict().keys()))
    print()
    fresh = nn.Linear(3, 2)
    fresh.load_state_dict(wrapped._orig_mod.state_dict())
    print(
        "loads cleanly into an uncompiled model:",
        torch.allclose(base.weight, fresh.weight),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try it

    The short final batch is the thing students trip over, and it is much easier to see than to describe. 100 samples divided by a batch size that does not go into 100 leaves a remainder, and that remainder becomes one last undersized batch.

    Drag the batch size and watch the last number in the list. Then turn on `drop_last` and watch it disappear — along with those samples, which is the cost. At `batch_size=64` dropping the tail throws away 36 of your 100 samples for that epoch.

    Two sizes are worth stopping at. **64** gives you batches of 64 and 36, the most lopsided pair available here. **1** gives a hundred batches of one, which is the setting that makes `BatchNorm` raise the error from Part 4 — it has no batch to compute statistics over.
    """)
    return


@app.cell
def _(mo):
    size_slider = mo.ui.slider(1, 64, value=16, label="Batch size", show_value=True)
    drop_last_switch = mo.ui.switch(value=False, label="drop_last")
    mo.vstack([size_slider, drop_last_switch])
    return drop_last_switch, size_slider


@app.cell
def _(DataLoader, FakeImageDataset, drop_last_switch, size_slider):
    tail_loader = DataLoader(
        FakeImageDataset(100),
        batch_size=size_slider.value,
        shuffle=False,
        drop_last=drop_last_switch.value,
    )

    tail_sizes = [len(batch_labels) for _, batch_labels in tail_loader]

    print(
        f"100 samples, batch_size={size_slider.value}, drop_last={drop_last_switch.value}"
    )
    print("  batches   ", len(tail_sizes))
    print("  sizes     ", tail_sizes)
    print("  samples used", sum(tail_sizes), "of 100")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Add a `transform` to `FakeImageDataset` and confirm from the printed reads that it runs per sample rather than up front.
    2. Build a loader with `batch_size=32` over 100 samples. How many batches, and what shape is the last one? Now set `drop_last=True`.
    3. Train anything for one epoch, save the `state_dict`, build a fresh model, and check the outputs differ. Load the weights and check they now match.
    4. Save a model, load it with `strict=False` into an architecture with an extra layer, and print what comes back. Which layer is still random? Now change a shared layer's width instead and explain why `strict=False` no longer helps.
    5. Take the augmentation pipeline and apply it to the same image 100 times, collecting the means. How much variation are you actually introducing? Is it enough to be worth the extra epochs?

    That is the set. Between `NumPyForML/` and these five you have every NumPy and PyTorch function used in the machine learning demos in this repository, with the exception of the pre-trained model loading in `PreTrainedModels/`, which is worth reading next.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

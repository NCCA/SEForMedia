#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # torchaudio for Machine Learning, Part 4: datasets and a classifier

    We can now turn waveforms into model inputs. I will generate three classes of tones, use a `Dataset` and `DataLoader`, and train a small classifier on log mel features. This is a deliberately easy task for checking the pipeline, not evidence that the model can recognise speech or music.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torchaudio
    from torchaudio import transforms as T

    print("torch", torch.__version__, "torchaudio", torchaudio.__version__)
    return T, mo, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The pieces of our learning task

    A **classifier** predicts which category, or **class**, an input belongs to. Our classes are low, middle and high tones. Each clip has a **label**, stored as an integer: 0, 1 or 2. A **feature** is a measurement supplied to the model; here we measure energy in groups of frequencies called mel bands, take logarithms and average over time. [Part 2](TorchAudioForMLPart2Features.py) explains these steps.

    | Term | What it does here |
    | --- | --- |
    | Dataset | Holds examples and their labels, and lets us request one by index. |
    | DataLoader | Fetches examples from a dataset and groups them into batches. |
    | Batch | A small group of examples processed together, here 15 training clips. |
    | Preprocessing | Operations that prepare raw samples for the model, such as cropping and feature extraction. |
    | Training | Adjusting the model's numerical weights using labelled examples. |
    | Inference | Using the learned weights to make predictions without updating them. |

    The [PyTorch data guide](https://docs.pytorch.org/docs/stable/data.html) explains `Dataset` and `DataLoader`. A **tensor** stores numbers along named-by-position axes; its **shape** gives the size of each axis. A batch of 15 clips with 32 features has shape `(15, 32)`.

    We train on one set and evaluate on **held-out** examples that were not used to adjust the weights. A **validation set** would be used to choose settings such as model size. The **test set** is reserved for final evaluation; repeatedly tuning against it would make it part of model development.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Different lengths in a batch

    Real recordings do not all have the same duration. **Collation** is the step that groups individual examples into a batch. By default PyTorch stacks their tensors, which requires equal shapes. **Padding** extends shorter sequences, here with zeros. **Cropping** keeps only a selected section. We can pad a batch and retain its lengths, or crop/pad every recording to a fixed duration.

    The example below shows padding separately. Lengths tell a model processing a sequence which samples are real. **Pooling** combines values, for example by taking an average. A **loss** is the numerical error used during training. Neither operation automatically ignores padding; the length information or a **mask** (true/false flags for valid positions) must be used by the calculation.
    """)
    return


@app.cell
def _(torch):
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader, Dataset
    from torch import nn

    _clips = [torch.ones(3200), torch.ones(5600), torch.ones(4000)]
    lengths = torch.tensor([clip.numel() for clip in _clips])
    padded = pad_sequence(_clips, batch_first=True)
    valid_samples = torch.arange(padded.shape[1])[None, :] < lengths[:, None]
    print("padded", tuple(padded.shape), "lengths", lengths.tolist())
    print("valid samples per row", valid_samples.sum(dim=1).tolist())
    return DataLoader, Dataset, nn


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A generated audio dataset

    Each class has a different centre frequency. We vary duration, phase, amplitude and frequency slightly. Separate local random generators give us repeatable training and test recordings without reusing the same examples.

    For this fixed-duration classifier, we retain the first half-second of every clip. All generated recordings are at least that long. Cropping might remove the event we need in a real task, so this choice needs checking when replacing the data.

    The feature transform belongs to the dataset here and runs on the **CPU**, the computer's general-purpose processor. In a larger system we could batch it on the **GPU**, a graphics processor that can carry out many tensor operations in parallel. The sample rate is samples per second, the FFT analyses frequencies within a frame, and the hop is the step between frames. The `sample_rate`, FFT size, hop and mel band count must stay consistent between training and inference.
    """)
    return


@app.cell
def _(Dataset, T, torch):
    sample_rate = 8000
    class_names = ["Low tone", "Middle tone", "High tone"]
    feature_settings = {
        "sample_rate": sample_rate,
        "n_fft": 256,
        "hop_length": 128,
        "n_mels": 32,
    }

    class ToneDataset(Dataset):
        def __init__(self, count: int, seed: int) -> None:
            generator = torch.Generator().manual_seed(seed)
            self.examples = []
            self.transform = T.MelSpectrogram(**feature_settings)
            for index in range(count):
                label = index % 3
                frequency = [300, 900, 1800][label] + 40 * (
                    torch.rand((), generator=generator).item() - 0.5
                )
                length = int(torch.randint(4000, 6001, (), generator=generator))
                time = torch.arange(length) / sample_rate
                phase = 2 * torch.pi * torch.rand((), generator=generator).item()
                amplitude = 0.2 + 0.2 * torch.rand((), generator=generator).item()
                waveform = amplitude * torch.sin(
                    2 * torch.pi * frequency * time + phase
                )
                waveform += 0.01 * torch.randn(length, generator=generator)
                self.examples.append((waveform, label))

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            waveform, label = self.examples[index]
            mel = self.transform(waveform[:4000])
            features = torch.log(mel.clamp_min(1e-6)).mean(dim=-1)
            return features, label

    train_data = ToneDataset(90, seed=42)
    test_data = ToneDataset(30, seed=123)
    print("train", len(train_data), "test", len(test_data))
    print("feature shape", tuple(train_data[0][0].shape))
    return class_names, feature_settings, test_data, train_data


@app.cell
def _(DataLoader, test_data, torch, train_data):
    train_loader = DataLoader(
        train_data,
        batch_size=15,
        shuffle=True,
        generator=torch.Generator().manual_seed(7),
    )
    test_loader = DataLoader(test_data, batch_size=30, shuffle=False)
    batch_features, batch_labels = next(iter(train_loader))
    print("features", tuple(batch_features.shape), "labels", tuple(batch_labels.shape))
    _all_train = torch.stack([train_data[index][0] for index in range(len(train_data))])
    feature_mean = _all_train.mean(dim=0)
    feature_std = _all_train.std(dim=0).clamp_min(1e-6)
    return feature_mean, feature_std, test_loader, train_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training

    We average log mel features over time to get 32 values per clip. This loses the ordering of events, which is acceptable for our steady tones but would be a poor choice for recognising a spoken sentence.

    **Normalisation** here subtracts each feature's training-set mean and divides by its standard deviation, a measure of how spread out its values are. This puts features on comparable scales. We keep a small minimum divisor to avoid dividing by zero. These statistics come from the training set only; computing them from test clips would let test information influence preprocessing.

    Our [linear layer](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) computes three scores by multiplying the 32 features by learned weights, adding them, and adding a bias (an adjustable offset). These scores are called **logits**. The largest score selects the predicted class.

    A **loss function** measures disagreement with the correct labels. [Cross-entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) penalises assigning low probability to the correct class. **Softmax** converts logits into positive values that sum to one; PyTorch's loss handles the relevant calculation internally. `CrossEntropyLoss` takes raw class scores and integer targets; we do not add softmax before the loss. The training loop follows [PyTorch Part 2](../PyTorchForML/PyTorchForMLPart2Autograd.py). A **gradient** tells us how the loss changes when a weight changes; `backward()` calculates these gradients. `zero_grad()` clears the previous batch's gradients. An **optimiser**, here [Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html), uses them to update the weights. The **learning rate** controls the update scale. An **epoch** is one pass through the training set; we make 12 passes.
    """)
    return


@app.cell
def _(feature_mean, feature_std, nn, torch, train_loader):
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = nn.Linear(32, 3)
    _optimiser = torch.optim.Adam(model.parameters(), lr=0.03)
    _loss_fn = nn.CrossEntropyLoss()
    loss_history = []
    model.train()
    for _epoch in range(12):
        _total = 0.0
        for _features, _labels in train_loader:
            _scores = model((_features - feature_mean) / feature_std)
            _loss = _loss_fn(_scores, _labels)
            _optimiser.zero_grad()
            _loss.backward()
            _optimiser.step()
            _total += _loss.item() * len(_labels)
        loss_history.append(_total / len(train_loader.dataset))
    print("first loss", loss_history[0], "last loss", loss_history[-1])
    return loss_history, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading the evaluation

    `model.eval()` selects evaluation behaviour for layers which distinguish training from evaluation. Our linear layer behaves the same in either mode, but the call is useful when extending the model. `torch.inference_mode()` turns off gradient bookkeeping because we are only making predictions.

    `argmax` returns the position of the largest score, which is our predicted label. **Accuracy** is the fraction of clips classified correctly. A **confusion matrix** counts the actual and predicted classes together: each row is the true class, each column is the prediction. The diagonal contains correct predictions; off-diagonal entries show which classes were confused. Here the matrix contains counts, not percentages.

    The loss plot shows training progress, whilst the confusion matrix uses separate test clips. A low training loss alone does not demonstrate success on new examples: a model can **overfit**, learning details specific to the training recordings.
    """)
    return


@app.cell
def _(feature_mean, feature_std, model, test_loader, torch):
    model.eval()
    with torch.inference_mode():
        test_features, test_labels = next(iter(test_loader))
        predictions = model((test_features - feature_mean) / feature_std).argmax(dim=-1)
        test_accuracy = (predictions == test_labels).float().mean().item()
    confusion = torch.zeros(3, 3, dtype=torch.int64)
    for _actual, _predicted in zip(test_labels, predictions):
        confusion[_actual, _predicted] += 1
    print(f"held-out accuracy: {test_accuracy:.1%}")
    return confusion, test_features


@app.cell
def _(class_names, confusion, loss_history, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    _axes[0].plot(range(1, len(loss_history) + 1), loss_history, marker="o")
    _axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Training loss")
    _axes[1].imshow(confusion.numpy(), cmap="Blues", vmin=0)
    _axes[1].set(
        xticks=range(3),
        yticks=range(3),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="Actual",
        title="Held-out recordings",
    )
    for _row in range(3):
        for _col in range(3):
            _axes[1].text(
                _col,
                _row,
                str(int(confusion[_row, _col])),
                ha="center",
                va="center",
                color="white"
                if confusion[_row, _col] > confusion.max() / 2
                else "black",
            )
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saving the preprocessing with the weights

    A **checkpoint** is a saved snapshot needed to reuse a model. PyTorch's [`state_dict`](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) maps parameter names to their tensors of learned weights and biases. Saving a `state_dict` alone does not tell us how the model expects audio to be prepared. I keep the feature settings, class names, duration and normalisation statistics in the checkpoint. The feature reduction below is `log(clamp(mel, 1e-6)).mean(time)`; it must also remain consistent when loading new audio.
    """)
    return


@app.cell
def _(
    class_names,
    feature_mean,
    feature_settings,
    feature_std,
    model,
    nn,
    test_features,
    torch,
):
    import io

    _buffer = io.BytesIO()
    torch.save(
        {
            "model": model.state_dict(),
            "class_names": class_names,
            "feature_settings": feature_settings,
            "clip_samples": 4000,
            "feature_reduction": "log-clamp-1e-6-mean-time",
            "mean": feature_mean,
            "std": feature_std,
        },
        _buffer,
    )
    _buffer.seek(0)
    checkpoint = torch.load(_buffer, weights_only=True)
    _restored = nn.Linear(
        checkpoint["feature_settings"]["n_mels"], len(checkpoint["class_names"])
    )
    _restored.load_state_dict(checkpoint["model"])
    _restored.eval()
    with torch.inference_mode():
        restored_predictions = _restored(
            (test_features - checkpoint["mean"]) / checkpoint["std"]
        ).argmax(dim=-1)
    print("checkpoint bytes", _buffer.getbuffer().nbytes)
    return (restored_predictions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Moving to recorded audio

    [torchaudio.datasets](https://docs.pytorch.org/audio/stable/datasets.html) includes datasets such as Speech Commands. This is a reference snippet, not a cell that downloads a dataset:

    ```python
    from torchaudio.datasets import SPEECHCOMMANDS

    data = SPEECHCOMMANDS("data", subset="training", download=True)
    waveform, sample_rate, label, speaker_id, utterance_number = data[0]
    ```

    Use the provided training, validation and test subsets. When organising your own speech recordings, split by speaker as well as recording so the model does not get an easy route through recognising the speaker. A **label mapping** associates class names with integer IDs, such as “Low tone” with 0. Build the label mapping from training labels and keep it with the checkpoint. Check sample rates and channels before stacking clips, then resample explicitly if needed.

    **Pre-trained** models already have weights learned from another dataset. Speech models also have specific sample rates and input layouts. Read the model's preprocessing requirements before replacing our linear classifier. A **transcript** is written text representing spoken words. A transcript model's output is not a clip classification label.

    ## Exercises

    1. Reduce the separation between the tone classes and inspect the confusion matrix.
    2. Add noise to training recordings only. Evaluate on a separate set with stronger noise.
    3. Replace the mean over time with a small [convolutional model](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) on log mel spectrograms. Its learned filters look for local patterns in neighbouring time-frequency positions. What shape does it expect?
    4. Reload the checkpoint and prepare a newly generated waveform using its saved settings.
    5. Plan a speech command experiment with separate speakers and an untouched test set. Which augmentations from Part 3 preserve the labels?
    """)
    return


if __name__ == "__main__":
    app.run()

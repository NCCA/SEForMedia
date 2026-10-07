#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full", app_title="Data Leakage and Group Splits")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Leakage and Group Splits

    On the [project ideas page](https://nccastaff.bournemouth.ac.uk/jmacey/SEForMedia/Assignment/ideas/) I ask you to keep related samples together when splitting your data. This includes frames from a video, clips from a recording and takes from a performer. In this notebook we will look at why this matters.

    We will train the same model twice using different splits of the same dataset:

    1. A random split using `torch.utils.data.random_split`, which assigns individual clips to training, validation and test sets.
    2. A group split using [`GroupShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html), which keeps clips from each source recording together.

    We will then compare some simpler models using [`GroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) and listen to examples from the data.

    I am using [ESC-50](https://github.com/karolpiczak/ESC-50) (Piczak 2015), which contains 2000 clips of environmental sounds in 50 classes. Each clip is five seconds long and the download is about 600 MB. The metadata includes the source recording for each clip, so we already have a group identifier to work with. See the dataset's [licence](https://github.com/karolpiczak/ESC-50#license) for the terms of use.

    ## Setup

    As in the other notebooks we import our `Utils` library and check whether we are in the lab. In the lab the data goes in `/transfer`, otherwise we use the current folder.
    """)
    return


@app.cell
def _():
    import pathlib
    import sys

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import soundfile as sf
    import torch
    import torch.nn as nn
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import (
        GroupKFold,
        GroupShuffleSplit,
        KFold,
        PredefinedSplit,
        cross_val_score,
    )
    from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, Dataset, random_split

    sys.path.append("../")
    import Utils

    print(f"{Utils.in_lab()=}")
    return (
        DataLoader,
        Dataset,
        GroupKFold,
        GroupShuffleSplit,
        KFold,
        KNeighborsClassifier,
        LogisticRegression,
        NearestNeighbors,
        PredefinedSplit,
        StandardScaler,
        Utils,
        cross_val_score,
        make_pipeline,
        nn,
        np,
        pathlib,
        pd,
        plt,
        random_split,
        sf,
        torch,
    )


@app.cell
def _(Utils):
    device = Utils.get_device()
    print(f"using {device}")
    return (device,)


@app.cell
def _(Utils, pathlib):
    DATASET_LOCATION = ""
    if Utils.in_lab():
        DATASET_LOCATION = "/transfer/ESC50/"
    else:
        DATASET_LOCATION = "./ESC50/"
    # now we will create the folder if it does not exist
    pathlib.Path(DATASET_LOCATION).mkdir(parents=True, exist_ok=True)
    return (DATASET_LOCATION,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We download the dataset as a zip of the GitHub repository and extract it if the metadata is not already present. The audio files are in `ESC-50-master/audio` and `ESC-50-master/meta/esc50.csv` describes each clip. We keep the download locally so we do not need to fetch it again.
    """)
    return


@app.cell
def _(DATASET_LOCATION, Utils, pathlib):
    URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    ESC50_ROOT = pathlib.Path(DATASET_LOCATION) / "ESC-50-master"
    _zip = pathlib.Path(DATASET_LOCATION) / "ESC-50-master.zip"

    if not (ESC50_ROOT / "meta" / "esc50.csv").exists():
        if not _zip.exists():
            print("Downloading ESC-50")
            Utils.download(URL, str(_zip))
        print("Unzipping")
        Utils.unzip_file(_zip, pathlib.Path(DATASET_LOCATION))
    print(f"data in {ESC50_ROOT}")
    return (ESC50_ROOT,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Samples come in groups

    The filenames use the format `{fold}-{src_file}-{take}-{target}.wav`. For example, `1-100038-A-14.wav` is take A from source recording 100038, with class 14 (chirping birds).

    Several clips can come from one longer recording. Whilst they are separate files, they can share the same background sounds and recording conditions. We will use `src_file` to keep track of this.

    Let's load the metadata and see how many recordings have more than one clip.
    """)
    return


@app.cell
def _(ESC50_ROOT, pd):
    meta = pd.read_csv(ESC50_ROOT / "meta" / "esc50.csv")
    meta.head(10)
    return (meta,)


@app.cell
def _(meta):
    takes_per_recording = meta.groupby("src_file").size()
    clips_with_siblings = (meta["src_file"].map(takes_per_recording) > 1).mean()
    print(f"{len(meta)} clips cut from {meta['src_file'].nunique()} source recordings")
    print(
        f"{clips_with_siblings:.0%} of clips share their recording with at least one other clip"
    )
    print(f"largest group is {takes_per_recording.max()} clips from one recording")
    return (takes_per_recording,)


@app.cell
def _(plt, takes_per_recording):
    _counts = takes_per_recording.value_counts().sort_index()
    _fig, _ax = plt.subplots(figsize=(8, 3))
    _ax.bar(_counts.index.astype(str), _counts.values, color="#4C72B0")
    _ax.set_xlabel("clips cut from the same recording")
    _ax.set_ylabel("number of recordings")
    _ax.set_title("How big are the groups?")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Have a listen to one of the larger groups. Our `Dataset` treats these as separate samples, but they come from the same recording. Listen for shared background sounds such as traffic, hiss or room tone as well as the sound being labelled.
    """)
    return


@app.cell
def _(ESC50_ROOT, meta, mo, sf, takes_per_recording):
    _src = takes_per_recording[takes_per_recording >= 3].index[0]
    _rows = meta[meta["src_file"] == _src].head(4)
    _players = []
    for _name, _cat in zip(_rows["filename"], _rows["category"]):
        _audio, _rate = sf.read(ESC50_ROOT / "audio" / _name)
        _players.append(
            mo.vstack([mo.md(f"`{_name}` ({_cat})"), mo.audio(_audio, rate=_rate)])
        )
    mo.hstack(_players, justify="start", gap=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Turning audio into images

    We will convert each clip into a log-mel spectrogram. With the settings below, five seconds of audio gives us a 64 × 216 image: 64 mel bands and 216 time steps. We can then use a small CNN to classify it.

    I build the mel filter bank here so we can see how it works. The FreeSpokenDigits and TorchAudioForML demos show the same approach using torchaudio transforms.

    The spectrograms take some time to calculate, so we cache them next to the audio files.
    """)
    return


@app.cell
def _(np, torch):
    SAMPLE_RATE = 44100
    N_FFT = 2048
    HOP = 1024
    N_MELS = 64

    def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> torch.Tensor:
        """
        Triangular filters evenly spaced on the mel scale (HTK formula).

        Parameters
        ----------
            sample_rate : int
                sample rate of the audio
            n_fft : int
                size of the FFT, gives n_fft // 2 + 1 frequency bins
            n_mels : int
                number of mel bands to produce
        """

        def hz_to_mel(f):
            return 2595.0 * np.log10(1.0 + f / 700.0)

        def mel_to_hz(m):
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        fft_freqs = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)
        mel_points = np.linspace(hz_to_mel(0), hz_to_mel(sample_rate / 2), n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bank = np.zeros((n_mels, len(fft_freqs)), dtype=np.float32)
        for m in range(n_mels):
            left, centre, right = hz_points[m : m + 3]
            rising = (fft_freqs - left) / (centre - left)
            falling = (right - fft_freqs) / (right - centre)
            bank[m] = np.maximum(0, np.minimum(rising, falling))
        return torch.from_numpy(bank)

    MEL_BANK = mel_filterbank(SAMPLE_RATE, N_FFT, N_MELS)

    def log_mel(audio: np.ndarray) -> np.ndarray:
        """
        Convert audio to a log-mel spectrogram.

        Parameters
        ----------
            audio : np.ndarray
                mono audio sampled at SAMPLE_RATE

        Returns
        -------
            np.ndarray
                mel-band power in decibels, shape (n_mels, frames)
        """
        x = torch.from_numpy(audio.astype(np.float32))
        spec = torch.stft(
            x, N_FFT, HOP, window=torch.hann_window(N_FFT), return_complex=True
        )
        mel = MEL_BANK @ spec.abs() ** 2
        return (10.0 * torch.log10(mel.clamp(min=1e-10))).numpy()

    return (log_mel,)


@app.cell
def _(DATASET_LOCATION, ESC50_ROOT, log_mel, meta, mo, np, pathlib, sf):
    _cache = pathlib.Path(DATASET_LOCATION) / "logmel64.npz"
    if _cache.exists() and list(np.load(_cache)["names"]) == list(meta["filename"]):
        features = np.load(_cache)["features"]
    else:
        _specs = []
        for _name in mo.status.progress_bar(
            meta["filename"], title="log-mel spectrograms"
        ):
            _audio, _ = sf.read(ESC50_ROOT / "audio" / _name)
            _specs.append(log_mel(_audio))
        features = np.stack(_specs).astype(np.float32)
        # dtype=str stores the names as plain unicode, otherwise they are pickled objects
        np.savez(
            _cache,
            features=features,
            names=meta["filename"].to_numpy(dtype=str),
        )
    labels = meta["target"].to_numpy()
    groups = meta["src_file"].to_numpy()
    print(f"{features.shape=} {labels.shape=}")
    return features, groups, labels


@app.cell
def _(features, meta, plt):
    _fig, _axes = plt.subplots(1, 4, figsize=(16, 3))
    for _ax, _i in zip(_axes, [0, 1, 40, 80]):
        _ax.imshow(features[_i], origin="lower", aspect="auto", cmap="magma")
        _ax.set_title(f"{meta['category'][_i]}\n{meta['filename'][_i]}", fontsize=9)
        _ax.set_xlabel("time step")
    _axes[0].set_ylabel("mel band")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The dataset and the model

    The `Dataset` returns a spectrogram and its label. I have left normalisation out of this class as we need to calculate the mean and standard deviation from the training set for each split. Using all the data would let the validation and test sets influence preprocessing.

    I store these values as buffers in the model so they are saved with its weights. The model itself uses four convolution blocks followed by global average pooling and a linear classifier. This is enough for us to compare the two ways of splitting the data.
    """)
    return


@app.cell
def _(Dataset, features, labels, nn, torch):
    class SpectrogramDataset(Dataset):
        """
        Wraps the cached log-mel features, returns (1, mels, frames) tensors.

        Attributes
        ----------
            features : torch.Tensor
                every spectrogram, shape (clips, mels, frames)
            labels : torch.Tensor
                class index for every clip
        """

        def __init__(self, features, labels) -> None:
            self.features = torch.from_numpy(features)
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index):
            return self.features[index].unsqueeze(0), self.labels[index]

    class SmallCNN(nn.Module):
        """
        Four conv blocks and a linear classifier.

        Attributes
        ----------
            mean : torch.Tensor
                training set mean, a buffer so it is saved with the weights
            std : torch.Tensor
                training set standard deviation
        """

        def __init__(self, n_classes: int, mean: float, std: float) -> None:
            super().__init__()
            self.register_buffer("mean", torch.tensor(mean))
            self.register_buffer("std", torch.tensor(std))

            def block(c_in: int, c_out: int) -> nn.Sequential:
                return nn.Sequential(
                    nn.Conv2d(c_in, c_out, 3, padding=1),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )

            self.features = nn.Sequential(
                block(1, 16), block(16, 32), block(32, 64), block(64, 128)
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(128, n_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = (x - self.mean) / self.std
            return self.classifier(self.features(x))

    def random_time_shift(x: torch.Tensor) -> torch.Tensor:
        """Shift every spectrogram in the batch by the same random time offset."""
        return torch.roll(x, shifts=int(torch.randint(0, x.shape[-1], (1,))), dims=-1)

    esc_dataset = SpectrogramDataset(features, labels)
    return SmallCNN, esc_dataset, random_time_shift


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We use the `train_epoch` and `evaluate` functions from `Utils`, as in the earlier notebooks. We keep the weights from the epoch with the best validation accuracy and evaluate those weights once on the test set.

    We also split the validation set by the same method as the test set. If related recordings cross into validation, model selection can favour a model that recognises those recordings. Keeping the test set separate is only part of the job.
    """)
    return


@app.cell
def _(
    DataLoader,
    SmallCNN,
    Utils,
    device,
    esc_dataset,
    features,
    mo,
    nn,
    np,
    random_time_shift,
    torch,
):
    EPOCHS = 30
    BATCH_SIZE = 32

    def run_experiment(name: str, train_idx, val_idx, test_idx) -> dict:
        """
        Train a fresh SmallCNN on train_idx, pick the best epoch on val_idx, test on test_idx.

        Parameters
        ----------
            name : str
                shown on the progress bar
            train_idx, val_idx, test_idx : array like
                row indices into esc_dataset
        """
        torch.manual_seed(42)
        train_idx, val_idx, test_idx = (
            np.asarray(i) for i in (train_idx, val_idx, test_idx)
        )

        def _subset(idx):
            return torch.utils.data.Subset(esc_dataset, idx)

        train_loader = DataLoader(
            _subset(train_idx), batch_size=BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(_subset(val_idx), batch_size=128)
        test_loader = DataLoader(_subset(test_idx), batch_size=128)

        train_feats = features[train_idx]
        model = SmallCNN(50, float(train_feats.mean()), float(train_feats.std())).to(
            device
        )
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        history = {"train": [], "val": []}
        best_acc, best_weights = -1.0, None
        for _epoch in mo.status.progress_bar(range(EPOCHS), title=name):
            _, train_acc = Utils.train_epoch(
                model,
                train_loader,
                loss_fn,
                optimiser,
                device,
                transform=random_time_shift,
            )
            _, val_acc = Utils.evaluate(model, val_loader, loss_fn, device)
            history["train"].append(train_acc)
            history["val"].append(val_acc)
            if val_acc > best_acc:
                best_acc, best_weights = val_acc, Utils.copy_weights(model)

        model.load_state_dict(best_weights)
        model.eval()
        with torch.inference_mode():
            preds = torch.cat(
                [model(x.to(device)).argmax(1).cpu() for x, _ in test_loader]
            )
        test_labels = esc_dataset.labels[test_idx]
        return {
            "name": name,
            "model": model,
            "history": history,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "preds": preds.numpy(),
            "test_acc": (preds == test_labels).float().mean().item(),
        }

    return (run_experiment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The random split

    First we assign 60% of the clips to training, 20% to validation and 20% to testing using `random_split`. This treats each clip as an independent sample and does not use the source recording information.
    """)
    return


@app.cell
def _(esc_dataset, random_split, run_experiment, torch):
    _train, _val, _test = random_split(
        esc_dataset,
        [0.6, 0.2, 0.2],
        generator=torch.Generator().manual_seed(42),
    )
    random_result = run_experiment(
        "random split", _train.indices, _val.indices, _test.indices
    )
    print(f"random split test accuracy {random_result['test_acc']:.1%}")
    return (random_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before interpreting that accuracy, let's check how many test clips have another take from the same recording in the training set. I will call these related takes siblings in the plots below.
    """)
    return


@app.cell
def _(groups, np, random_result):
    _train_groups = set(groups[random_result["train_idx"]])
    test_has_sibling = np.array(
        [g in _train_groups for g in groups[random_result["test_idx"]]]
    )
    print(
        f"{test_has_sibling.mean():.0%} of random split test clips have a sibling in train"
    )
    return (test_has_sibling,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The group split

    Now we keep every clip from a source recording in the same set. `GroupShuffleSplit` takes a `groups` array and keeps each group on one side of the split. We use it once to set aside the test groups, then again to split the remaining groups into training and validation.

    The first split sets aside 20% of the groups. The second uses 25% of the remaining groups for validation, giving roughly 60/20/20 overall. These are proportions of groups, so the clip counts can differ. This splitter does not balance the classes either.

    The assertions check that no source recording appears in more than one set. I would include this check in the project's tests so that later changes to the data loading do not introduce an overlap.
    """)
    return


@app.cell
def _(GroupShuffleSplit, groups, labels, np):
    def check_no_group_overlap(groups, *index_sets) -> None:
        """Fail loudly if any group appears in more than one of the index sets."""
        seen: set = set()
        for idx in index_sets:
            these = set(groups[idx])
            assert seen.isdisjoint(these), (
                f"{len(seen & these)} groups leak across the split"
            )
            seen |= these

    _all = np.arange(len(labels))
    _outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _rest, group_test_idx = next(_outer.split(_all, labels, groups))
    _inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    _tr, _va = next(_inner.split(_rest, labels[_rest], groups[_rest]))
    group_train_idx, group_val_idx = _rest[_tr], _rest[_va]

    check_no_group_overlap(groups, group_train_idx, group_val_idx, group_test_idx)
    print(
        f"train {len(group_train_idx)}, val {len(group_val_idx)}, test {len(group_test_idx)}"
    )
    return (
        check_no_group_overlap,
        group_test_idx,
        group_train_idx,
        group_val_idx,
    )


@app.cell
def _(group_test_idx, group_train_idx, group_val_idx, run_experiment):
    group_result = run_experiment(
        "group split", group_train_idx, group_val_idx, group_test_idx
    )
    print(f"group split test accuracy {group_result['test_acc']:.1%}")
    return (group_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparing the results

    We use the same model architecture, dataset, epoch limit and random seed for both runs. The clips in each set differ, and the group split can also change the set sizes and class balance.

    On the left we compare the test accuracies, including the random split's clips with and without a sibling in training. On the right we plot training and validation accuracy for both runs.
    """)
    return


@app.cell
def _(esc_dataset, group_result, plt, random_result, test_has_sibling):
    _correct = (
        random_result["preds"] == esc_dataset.labels[random_result["test_idx"]].numpy()
    )
    _bars = {
        "random split\n(all test clips)": random_result["test_acc"],
        "random split\nsibling in train": _correct[test_has_sibling].mean(),
        "random split\nno sibling": _correct[~test_has_sibling].mean(),
        "group split\n(all test clips)": group_result["test_acc"],
    }
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 4))
    _b = _ax1.bar(
        _bars.keys(),
        _bars.values(),
        color=["#C44E52", "#C44E52", "#8C8C8C", "#4C72B0"],
    )
    _ax1.bar_label(_b, labels=[f"{v:.0%}" for v in _bars.values()])
    _ax1.set_ylim(0, 1.05)
    _ax1.set_ylabel("test accuracy")
    _ax1.set_title("Where does the random split's score come from?")
    for _r, _c in ((random_result, "#C44E52"), (group_result, "#4C72B0")):
        _ax2.plot(
            _r["history"]["train"],
            color=_c,
            linestyle="--",
            label=f"{_r['name']} train",
        )
        _ax2.plot(_r["history"]["val"], color=_c, label=f"{_r['name']} validation")
    _ax2.set_xlabel("epoch")
    _ax2.set_ylabel("accuracy")
    _ax2.set_ylim(0, 1.05)
    _ax2.legend()
    _ax2.set_title("Training and validation accuracy")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(group_result, mo, random_result):
    mo.md(f"""
    The random split gives {random_result["test_acc"]:.0%} test accuracy and the group split gives {group_result["test_acc"]:.0%}. The group split estimates performance on source recordings held out from training. This is the relevant check if we want to classify clips from new recordings.

    Look at the curves on the right as well. Training and validation curves alone cannot tell us whether recordings overlap between sets. We need to check the metadata and understand how the clips were collected.

    These results come from one split of each kind, so we should not attribute every difference to leakage. Next we will use cross-validation with smaller models and compare the mean accuracy and variation across folds.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Looking for recording-specific clues

    A 1-nearest-neighbour classifier assigns each test clip the label of its closest training clip. This gives us a way to inspect which clips look similar using our chosen features. We summarise each spectrogram with the mean and standard deviation of each mel band, giving 128 values per clip.

    We also try logistic regression using the quietest quarter of the frames, ranked by their mean log-mel value. I use this as a rough way to look for background information. These frames can still contain the labelled sound, so this does not isolate the background.

    We compare five-fold `KFold` with shuffled clips, `GroupKFold` grouped by recording, and the official ESC-50 folds. The [official folds](https://github.com/karolpiczak/ESC-50#esc-50-dataset-for-environmental-sound-classification) keep clips from the same source together and should be used when comparing against published ESC-50 results. The scaler is fitted within each training fold using a pipeline.
    """)
    return


@app.cell
def _(features, np):
    summary_features = np.concatenate(
        [features.mean(axis=2), features.std(axis=2)], axis=1
    )

    # use quiet frames as a rough proxy for background; they can still contain the labelled sound
    _frame_energy = features.mean(axis=1)
    _quiet = np.argsort(_frame_energy, axis=1)[:, : features.shape[2] // 4]
    background_features = np.stack(
        [f[:, q].mean(axis=1) for f, q in zip(features, _quiet)]
    )
    print(f"{summary_features.shape=} {background_features.shape=}")
    return background_features, summary_features


@app.cell
def _(
    GroupKFold,
    KFold,
    KNeighborsClassifier,
    LogisticRegression,
    PredefinedSplit,
    StandardScaler,
    background_features,
    cross_val_score,
    groups,
    labels,
    make_pipeline,
    meta,
    np,
    pd,
    summary_features,
):
    _splitters = {
        "KFold (random)": (KFold(5, shuffle=True, random_state=42), None),
        "GroupKFold (by recording)": (GroupKFold(5), groups),
        "official ESC-50 folds": (
            PredefinedSplit(meta["fold"].to_numpy() - 1),
            None,
        ),
    }
    _models = {
        "1-NN, whole clip": (
            make_pipeline(StandardScaler(), KNeighborsClassifier(1)),
            summary_features,
        ),
        "logistic regression, quiet frames": (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
            background_features,
        ),
    }
    _rows = []
    for _model_name, (_model, _X) in _models.items():
        for _split_name, (_cv, _g) in _splitters.items():
            _scores = cross_val_score(_model, _X, labels, groups=_g, cv=_cv)
            _rows.append(
                {
                    "model": _model_name,
                    "split": _split_name,
                    "mean accuracy": np.round(_scores.mean(), 3),
                    "std": np.round(_scores.std(), 3),
                }
            )
    cv_results = pd.DataFrame(_rows)
    cv_results
    return (cv_results,)


@app.cell(hide_code=True)
def _(cv_results, mo):
    def _get(m, s):
        return cv_results.query("model == @m and split == @s")["mean accuracy"].item()

    _bg_random = _get("logistic regression, quiet frames", "KFold (random)")
    _bg_group = _get("logistic regression, quiet frames", "GroupKFold (by recording)")
    mo.md(f"""
    The quiet-frame model gives {_bg_random:.0%} mean accuracy with a random split and {_bg_group:.0%} with a group split. Uniform random guessing across 50 classes has an expected accuracy of 2%.

    If the random split scores higher, shared recording conditions are one possible explanation. However, the quiet frames can also contain useful class information, such as a continuous rain sound. This experiment does not prove which features the CNN uses.

    Compare the official folds with `GroupKFold` as well. Both keep source recordings together, but their scores need not match as they use different assignments of recordings to folds.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Listening to similar clips

    For each test clip in the random split we find the closest training clip using the standardised summary features. Let's check how often this is another take from the same recording, then listen to a few pairs.

    This comparison uses the summary features from above, rather than features learnt by the CNN.
    """)
    return


@app.cell
def _(
    NearestNeighbors,
    StandardScaler,
    groups,
    random_result,
    summary_features,
    test_has_sibling,
):
    _tr, _te = random_result["train_idx"], random_result["test_idx"]
    _scaler = StandardScaler().fit(summary_features[_tr])
    _finder = NearestNeighbors(n_neighbors=1).fit(
        _scaler.transform(summary_features[_tr])
    )
    _, _nearest = _finder.kneighbors(_scaler.transform(summary_features[_te]))
    nearest_train = _tr[_nearest[:, 0]]
    same_recording = groups[nearest_train] == groups[_te]
    print(
        f"for test clips with a sibling in train, the nearest training clip is "
        f"from the same recording {same_recording[test_has_sibling].mean():.0%} of the time"
    )
    return nearest_train, same_recording


@app.cell
def _(
    ESC50_ROOT,
    features,
    meta,
    mo,
    nearest_train,
    plt,
    random_result,
    same_recording,
    sf,
):
    _te = random_result["test_idx"]
    _picks = [i for i in range(len(_te)) if same_recording[i]][:3]
    _panels = []
    for _i in _picks:
        _a, _b = _te[_i], nearest_train[_i]
        _columns = []
        for _idx, _role in zip((_a, _b), ("test clip", "nearest training clip")):
            _fig, _ax = plt.subplots(figsize=(4.5, 2.4))
            _ax.imshow(features[_idx], origin="lower", aspect="auto", cmap="magma")
            _fig.subplots_adjust(left=0.1, right=0.98, bottom=0.15, top=0.95)
            _audio, _rate = sf.read(ESC50_ROOT / "audio" / meta["filename"][_idx])
            _columns.append(
                mo.vstack(
                    [
                        mo.md(f"**{_role}**"),
                        mo.md(f"`{meta['filename'][_idx]}`"),
                        mo.as_html(_fig),
                        mo.audio(_audio, rate=_rate).style({"width": "100%"}),
                    ],
                    align="stretch",
                )
            )
            plt.close(_fig)
        _panels.append(
            mo.vstack(
                [
                    mo.md(f"**{meta['category'][_a]}**"),
                    mo.hstack(_columns, widths="equal", align="start", gap=1),
                ]
            )
        )
    mo.vstack(_panels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Where the group split model fails

    Finally we list the most common mistakes from the CNN trained with the group split. For your writeup, look at which classes are confused and listen to some examples. Can you hear why they might be difficult to distinguish? This is the sort of analysis I am asking for on the project ideas page.
    """)
    return


@app.cell
def _(esc_dataset, group_result, meta, pd):
    _names = meta.drop_duplicates("target").set_index("target")["category"]
    _true = esc_dataset.labels[group_result["test_idx"]].numpy()
    _pred = group_result["preds"]
    _wrong = pd.DataFrame(
        {
            "true": _names[_true].to_numpy(),
            "predicted": _names[_pred].to_numpy(),
        }
    )[_true != _pred]
    confusions = _wrong.value_counts().rename("count").reset_index().head(15)
    confusions
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Applying this to your project

    Before splitting the data, decide what you want the model to handle when it is used. New recordings, new performers and new scenes each suggest a different grouping. We need to keep related samples together at the level we want to evaluate.

    These are some starting points:

    | Project type | Possible group |
    |---|---|
    | Video | the source video or shot, keeping related frames together |
    | Audio cut into clips | the source recording |
    | Speech, dialogue cleanup, diarisation | the speaker or recording session, depending on the task |
    | Mocap, gesture, facial animation | the performer or capture session |
    | Music tagging | the artist or album |
    | Synthetic renders | the scene, considering shared assets such as HDRIs and textures |
    | Patches cut from images or textures | the source image |
    | Augmented data | the original sample, splitting before augmentation |

    For your project:

    1. Store the group identifier in the metadata, as ESC-50 does with `src_file`.
    2. Use `GroupShuffleSplit` or `GroupKFold`, or an official split that matches the evaluation you need.
    3. Keep groups separate in validation as well as testing.
    4. Fit normalisation statistics, vocabularies and PCA on the training set only.
    5. Add a test for group overlap, like the check below.
    6. Explain the split and the choice of groups in your write-up.

    If the dataset has no group information, say so and explain what you have been able to check. Look for duplicate or near-duplicate samples as well.
    """)
    return


@app.cell
def _(
    check_no_group_overlap,
    group_test_idx,
    group_train_idx,
    groups,
    random_result,
):
    # The sort of test that belongs in tests/test_data.py. The group split passes...
    check_no_group_overlap(groups, group_train_idx, group_test_idx)

    # ...and the random split fails
    try:
        check_no_group_overlap(
            groups, random_result["train_idx"], random_result["test_idx"]
        )
    except AssertionError as error:
        print(f"random split rejected: {error}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Things to try

    - Change `EPOCHS` or the model and compare the results. A different model does not remove overlap in the data, even if the accuracy gap changes.
    - Use the official folds, holding out each fold in turn for testing. Choose validation data from the remaining folds and report the mean test accuracy across all five runs.
    - Try a saliency map ([Grad-CAM](https://arxiv.org/abs/1610.02391) or occlusion) to investigate which parts of the spectrogram affect the CNN's predictions.
    - Remove the quietest frames before training and compare the two splits again. Bear in mind that this may remove some of the labelled sound as well.

    ## Reference

    Piczak, Karol J. "ESC: Dataset for Environmental Sound Classification." *Proceedings of the 23rd ACM International Conference on Multimedia*, 2015. <https://doi.org/10.1145/2733373.2806390>
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()

#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full", app_title="Data Leakage and Group Splits")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Leakage and Group Splits

    The [project ideas page](https://nccastaff.bournemouth.ac.uk/jmacey/SEForMedia/Assignment/ideas/) keeps telling you to keep related samples on one side of the split. Frames from the same video, clips from the same recording, takes from the same performer. In this notebook we are going to see *why*.

    We will train exactly the same model on exactly the same data twice. The only thing that changes is how we split it into train, validation and test sets.

    1. A random split using `torch.utils.data.random_split`, which is what most tutorials do.
    2. A group split using [`GroupShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html) and [`GroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html), where every clip cut from the same source recording stays together.

    Then we will examine what the model actually learnt, and finish with a checklist you should apply to your own project.

    The dataset is [ESC-50](https://github.com/karolpiczak/ESC-50) (Piczak 2015), 2000 five second environmental sound clips in 50 classes (dog, rain, door knock, chainsaw...). It is about 600MB, licensed CC BY-NC, and crucially it tells us which source recording every clip was cut from, so the group information is built in.

    ## Setup

    As in the other notebooks we import our Utils library and check if we are in the lab, if so the data goes on /transfer, otherwise into the current folder.
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
    The whole dataset is a single zip of the GitHub repository. We only download and unzip it if it is not already there, as it is 600MB. Once unzipped we have `ESC-50-master/audio` containing the wav files and `ESC-50-master/meta/esc50.csv` describing them.
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

    Each file name tells us where the clip came from, `{fold}-{src_file}-{take}-{target}.wav`. For example `1-100038-A-14.wav` is take A, cut from Freesound recording 100038, of class 14 (chirping birds). When the authors built the dataset they often cut several clips out of one longer recording, so take A, B and C are the same birds, in the same garden, recorded on the same microphone.

    Let's load the metadata and see how common that is.
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
    Have a listen to one of the bigger groups. These are separate samples as far as a `Dataset` is concerned, but they are really the same event. Listen to the background (traffic, hiss, room tone) rather than the thing being labelled. That is what will give the game away later.
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

    We will use a log-mel spectrogram for each clip, which turns 5 seconds of audio into a 64 x 216 image (64 frequency bands by 216 time steps) that a small CNN can classify. I build the mel filter bank by hand here so there is nothing hidden, torchaudio has a `MelSpectrogram` transform that does the same thing (see the FreeSpokenDigits and TorchAudioForML demos for examples)

    Computing these takes a minute or so, so the result is cached next to the data.
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
        """Power spectrogram -> mel bands -> decibels, returns (n_mels, frames)."""
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
        np.savez(_cache, features=features, names=meta["filename"].to_numpy(dtype=str))
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

    The `Dataset` just indexes into the feature array. Notice there is no normalisation in here. The mean and standard deviation have to come from the **training set only** (working them out over everything is a smaller leak of its own), so I store them inside the model as buffers. That way they get saved with the weights and can never drift out of step with them.

    The model is a deliberately ordinary small CNN, four conv blocks then global average pooling. Nothing in this notebook depends on it being clever.
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
        """Augmentation: roll each batch along time by a random amount."""
        return torch.roll(x, shifts=int(torch.randint(0, x.shape[-1], (1,))), dims=-1)

    esc_dataset = SpectrogramDataset(features, labels)
    return SmallCNN, esc_dataset, random_time_shift


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The training loop is the same `train_epoch` / `evaluate` pair from the Utils library that we have used before. We keep the weights from the epoch with the best **validation** accuracy, then report accuracy once on the test set.

    Note that the validation set is split the same way as the test set in each experiment. If the validation set leaked we would be choosing the epoch that memorised best, which is the same mistake one step removed.
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

    This is the split you will see in most tutorials, 60% train, 20% validation, 20% test, chosen at random. Each clip is treated as if it were independent of every other clip.
    """)
    return


@app.cell
def _(esc_dataset, random_split, run_experiment, torch):
    _train, _val, _test = random_split(
        esc_dataset, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(42)
    )
    random_result = run_experiment(
        "random split", _train.indices, _val.indices, _test.indices
    )
    print(f"random split test accuracy {random_result['test_acc']:.1%}")
    return (random_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That looks like a good result. Before we celebrate, let's check how many of the test clips have a sibling (another take from the same recording) sitting in the training set.
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

    Now the same thing, but every recording goes wholly into train, validation or test. `GroupShuffleSplit` works like `train_test_split` but takes a `groups` array and never puts one group on both sides. We use it twice, once to carve off the test set and again to split what is left into train and validation.

    Because groups vary in size the proportions will not be exactly 60/20/20, which is fine.

    The `assert` lines are the important bit. This is the kind of thing that should be in your project's tests, it costs nothing and it catches the mistake for good.
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
    ### Side by side

    Same model, same data, same number of epochs, same seed. On the left we break the random split's test accuracy down into clips that had a sibling in training and clips that did not. On the right are the validation curves.
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
        _bars.keys(), _bars.values(), color=["#C44E52", "#C44E52", "#8C8C8C", "#4C72B0"]
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
    _ax2.set_title("The random split's validation curve gives no warning")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(group_result, mo, random_result):
    mo.md(f"""
    The random split reports **{random_result["test_acc"]:.0%}**, the group split **{group_result["test_acc"]:.0%}**. Only the second number tells you how the model will do on a recording it has never heard, which is the only situation anyone will ever use it in.

    Look at the right hand plot as well. Nothing in the random split's curves looks wrong, the validation set leaks in exactly the same way as the test set so it agrees with it. **You cannot spot this from the training curves.** You have to know how your data was made.

    One split is a single sample though. The next section uses cross validation to get a mean and spread, using models fast enough to run five times.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. What did the model actually learn?

    The easiest way to see memorisation is with a model that does nothing *but* memorise. A 1-nearest-neighbour classifier labels each test clip with the label of the single most similar training clip. For features we squash each spectrogram down to the mean and standard deviation of every mel band (128 numbers per clip).

    We run it with 5 fold `KFold` (random) and `GroupKFold` (by recording), and also with the folds ESC-50 ships with. The official folds were built so that each recording only appears in one fold, which is exactly why you should use them when reporting results on this dataset.
    """)
    return


@app.cell
def _(features, np):
    summary_features = np.concatenate(
        [features.mean(axis=2), features.std(axis=2)], axis=1
    )

    # The quietest quarter of each clip is mostly background, not the sound being labelled
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
        "official ESC-50 folds": (PredefinedSplit(meta["fold"].to_numpy() - 1), None),
    }
    _models = {
        "1-NN, whole clip": (
            make_pipeline(StandardScaler(), KNeighborsClassifier(1)),
            summary_features,
        ),
        "logistic regression, background only": (
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

    _bg_random = _get("logistic regression, background only", "KFold (random)")
    _bg_group = _get(
        "logistic regression, background only", "GroupKFold (by recording)"
    )
    mo.md(f"""
    Look at the background only row. That model never sees the loud part of the clip, the bit with the dog in it. It is given the noise floor of the quietest quarter of the clip and nothing else. Chance on 50 classes is 2%.

    With a random split it scores **{_bg_random:.0%}**. With a group split it drops to **{_bg_group:.0%}**.

    The background is a fingerprint for the recording, and because the recording's other takes are in the training set, recognising the fingerprint is enough to get the label right. A CNN will happily use the same shortcut, it has no way of knowing which part of the spectrogram you care about. Also note the official folds and `GroupKFold` agree with each other, and both disagree with `KFold`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Listening to the shortcut

    For each test clip in the random split we can ask which training clip it is closest to. If the model were learning "what a dog sounds like" the nearest clip would just be another dog. Let's see how often it is literally another take from the same recording.
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
        _fig, _axes = plt.subplots(1, 2, figsize=(9, 2.4), sharey=True)
        for _ax, _idx, _role in zip(
            _axes, (_a, _b), ("test clip", "nearest training clip")
        ):
            _ax.imshow(features[_idx], origin="lower", aspect="auto", cmap="magma")
            _ax.set_title(f"{_role}: {meta['filename'][_idx]}", fontsize=9)
        _fig.tight_layout()
        _players = []
        for _idx in (_a, _b):
            _audio, _rate = sf.read(ESC50_ROOT / "audio" / meta["filename"][_idx])
            _players.append(mo.audio(_audio, rate=_rate))
        _panels.append(
            mo.vstack([mo.md(f"**{meta['category'][_a]}**"), _fig, mo.hstack(_players)])
        )
        plt.close(_fig)
    mo.vstack(_panels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Where the honest model fails

    Finally the group split CNN's mistakes. These are the interesting results for a write up: which classes get confused, and does it make sense when you listen? This is what "show where it fails" on the ideas page means.
    """)
    return


@app.cell
def _(esc_dataset, group_result, meta, pd):
    _names = meta.drop_duplicates("target").set_index("target")["category"]
    _true = esc_dataset.labels[group_result["test_idx"]].numpy()
    _pred = group_result["preds"]
    _wrong = pd.DataFrame(
        {"true": _names[_true].to_numpy(), "predicted": _names[_pred].to_numpy()}
    )[_true != _pred]
    confusions = _wrong.value_counts().rename("count").reset_index().head(15)
    confusions
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Checklist for your project

    Before you split anything, write down the answer to one question: **what is a group in my data?** A group is anything that makes two samples more alike than two random samples from the same class would be.

    | Project type | Likely group |
    |---|---|
    | Anything built from video | the video, or at least the shot. Never split individual frames |
    | Audio cut into clips | the source recording, and often the speaker or performer |
    | Speech, dialogue cleanup, diarisation | the speaker |
    | Mocap, gesture, facial animation | the performer (and the capture session) |
    | Music tagging | the artist, sometimes the album |
    | Synthetic renders (denoising, depth, materials) | the scene, plus any shared assets such as HDRIs and textures |
    | Patches cut from images or textures | the source image |
    | Augmented data | the original sample. Augment *after* splitting, never before |

    Then:

    1. Put the group in your metadata as a column, the same way ESC-50 has `src_file`.
    2. Split with `GroupShuffleSplit` / `GroupKFold` (or the dataset's official split if it has one).
    3. Split the validation set the same way as the test set.
    4. Compute normalisation statistics, vocabularies, PCA etc. on the training set only.
    5. Add a test like the one below to your test suite so it stays fixed.
    6. Report how you split in your write up, and why.

    If your dataset has no group information at all, say so, and think about what the hidden groups might be. Near duplicates are common in scraped datasets.
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

    - Change `EPOCHS` or the model. The gap between the two splits does not go away with a better model, a bigger model usually memorises *better*.
    - Train on the official folds 1-4 and test on fold 5, then do all five and report the mean.
    - Add a saliency map ([Grad-CAM](https://arxiv.org/abs/1610.02391) or occlusion) to see which parts of the spectrogram the random split model relies on.
    - Remove the quietest frames before training and see if the random split gap shrinks.

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

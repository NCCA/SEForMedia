import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Free Spoken Digits Dataset Part 2 DataLoaders

    In the previous notebook we downloaded the data and explored its structure. In this one we will use pytorch and Dataloaders to generate a dataset for our training.

    We left the last notebook looking at two appraches to how we can process the data,

    - 4 people for train with all the numbers 2 people for validation with all the numbers
    - Split randomly on number rather than speaker

    It will be good to do both but I will leave this as an exercise for later, for now we will use the 2nd option and generate our 80 / 20 split based on sampling each speaker and their 10 digits.

    First let us get the data path. If you recall I named a cell in the last notebook, we can now re-use this code in the current notebook by importing it and running the cell. The data can then be read from the definitions dictionary. More details can be found [here](https://docs.marimo.io/api/cell/)
    """)
    return


@app.cell
def _():
    from FSDDMarimoPt1 import download_digits

    output, definitions = download_digits.run()
    dataset_path = definitions["dataset_path"]
    print(dataset_path)
    return (dataset_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Torch Codec

    We will use torch codec to load in the audio data, we could use torchaudio to do the same thing as well.
    """)
    return


@app.cell
def _():
    import torch
    import torchcodec
    from torchcodec.decoders import AudioDecoder
    from pathlib import Path
    import matplotlib.pyplot as plt

    print("torch", torch.__version__, "torchcodec", torchcodec.__version__)
    return AudioDecoder, Path, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lets see if we can load one of the files and visualise it.

    First let's generate a slider to make it more interactive, this will select and index for the file so we can see what the different data files look like.
    """)
    return


@app.cell
def _(mo):
    file_index = mo.ui.slider(start=0, stop=3000 - 1, value=0, label="File Index")
    file_index
    return (file_index,)


@app.cell
def _(AudioDecoder, Path, dataset_path, file_index, plt, torch):
    # grab the first file in the folder
    root = Path(f"{dataset_path}/recordings")
    loaded_file = Path(sorted(root.glob("*.wav"))[file_index.value])
    label = int(loaded_file.stem.split("_", 1)[0])
    print(f"Audio number is {label}")
    audio_decoder = AudioDecoder(loaded_file)
    samples = audio_decoder.get_all_samples()

    _start = audio_decoder.metadata.begin_stream_seconds
    _duration = audio_decoder.metadata.duration_seconds

    if _start is not None and _duration is not None:
        _end = _start + _duration
        print(f"Start: {_start:.3f}s, end: {_end:.3f}s")
        print(f"Duration: {_duration:.3f}s")
    else:
        print("Timing information is missing from the metadata.")

    segment = audio_decoder.get_samples_played_in_range(
        start_seconds=_start,
        stop_seconds=min(_start + 1.0, _end),
    )

    _segment_time = (
        segment.pts_seconds + torch.arange(segment.data.shape[-1]) / segment.sample_rate
    )
    _fig, _axes = plt.subplots(2, 1, figsize=(10, 4), layout="constrained")
    _axes[0].plot(
        (torch.arange(samples.data.shape[-1]) / samples.sample_rate).numpy(),
        samples.data[0].numpy(),
        linewidth=0.4,
    )

    _axes[0].set(
        xlabel="Source time (s)",
        ylabel="Amplitude",
        title=f"Left channel digit {label}",
    )
    _axes[1].plot(_segment_time[:160].numpy(), segment.data[0, :160].numpy(), ".-")
    _axes[1].set(
        xlabel="Source time (s)",
        ylabel="Amplitude",
        title=f"First 10 ms of the excerpt digit {label}",
    )
    plt.close(_fig)
    _fig
    return label, loaded_file, root


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##From audio to features
    AudioDecoder returns a tensor with shape [channels, samples]. We request mono audio at 8 kHz, then pad short clips with zeros or trim long clips to one second.  We can then process this to give us a mel spectrogram and plot this as a feature. There is a good article about why this is a good idea [here](https://towardsdatascience.com/audio-deep-learning-made-simple-part-2-why-mel-spectrograms-perform-better-aad889a93505/) I also have a full work book in the TorchAudioForML section of the repo [GitHub](https://github.com/NCCA/SEForMedia/blob/main/TorchAudioForML/TorchAudioForMLPart2Features.py)
    """)
    return


@app.cell
def _(AudioDecoder, loaded_file, torch):
    from torchaudio.transforms import MelSpectrogram

    mel = MelSpectrogram(
        sample_rate=8000,
        n_fft=256,
        hop_length=80,
        n_mels=64,
    )
    _samples = AudioDecoder(
        loaded_file,
        sample_rate=8000,
        num_channels=1,
    ).get_all_samples()
    audio = _samples.data[:, :8000]
    # padd the audio to a set size of 8000
    audio = torch.nn.functional.pad(audio, (0, 8000 - audio.shape[-1]))
    feature = mel(audio).clamp_min(1e-10).log()
    feature = (feature - feature.mean()) / feature.std().clamp_min(1e-6)
    return (feature,)


@app.cell
def _(feature, label, loaded_file, mo, plt):
    _fig, _ax = plt.subplots(figsize=(9, 3), layout="constrained")
    _image = _ax.imshow(
        feature[0].numpy(), origin="lower", aspect="auto", extent=[0, 1, 0, 64]
    )
    _ax.set(
        xlabel="Time (s)",
        ylabel="Mel band",
        title=f"Normalised log-mel features: digit {label} ",
    )
    _fig.colorbar(_image, ax=_ax)
    plt.close(_fig)
    mo.vstack([mo.audio(loaded_file.read_bytes()), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This image is basically what we are going to train our network on. Scrubbing through the files you will see that nearly all of them start at 0 but the purple elements at the end are different. This is basically the silence we added when padding the data.  The reason we do this is so we can stack our training data into equal size tensors when training.

    It is possible to optimize this more but for now it will be fine.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A Dataloader

    We are going to develop a dataloader for this data, the filenames contain the digit, speaker and take number. We will use takes 0–4
    for testing, as specified by the [FSDD project](https://github.com/Jakobovski/free-spoken-digit-dataset#usage). From the remaining recordings we reserve takes 5–9 for validation and train on takes 10 onwards.

    As we are going to re-use this code in the next notebook I will develop it in it's own module called FSDDataLoader.py in the same folder as this notebook.

    The contents of the file follow (or you can open it in the zed editor)
    """)
    return


@app.cell
def _(Path, mo):
    _source = Path("FSDDataLoader.py").read_text(encoding="utf-8")
    mo.md(f"```python\n{_source}\n```")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The split_recording function just splits the data in the folder as mentioned above and the SpokenDigits class is a typical Dataset class used by the data loaders. We can now generate a data set.
    """)
    return


@app.cell
def _(root):
    from FSDDataLoader import split_recordings, SpokenDigits

    recordings = sorted(root.glob("*.wav"))
    train_paths, validation_paths, test_paths = split_recordings(recordings)
    train_data = SpokenDigits(train_paths)
    return (train_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can now plot the data from the data set, lets add a slider then do the plots.
    """)
    return


@app.cell
def _(mo, train_data):
    dataloader_index = mo.ui.slider(
        start=0, stop=len(train_data) - 1, value=0, label="Dataloader Index"
    )
    dataloader_index
    return (dataloader_index,)


@app.cell
def _(dataloader_index, mo, plt, train_data):
    _feature, _label = train_data[dataloader_index.value]
    _fig, _ax = plt.subplots(figsize=(9, 3), layout="constrained")
    _image = _ax.imshow(
        _feature[0].numpy(),
        origin="lower",
        aspect="auto",
        extent=[0, 1, 0, 64],
    )
    _ax.set(
        xlabel="Time (s)",
        ylabel="Mel band",
        title=f"Normalised log-mel features: digit {_label}",
    )
    _fig.colorbar(_image, ax=_ax)
    plt.close(_fig)
    mo.vstack([mo.audio(train_data.paths[dataloader_index.value].read_bytes()), _fig])
    return


if __name__ == "__main__":
    app.run()

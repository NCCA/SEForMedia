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
    """)
    return


@app.cell
def _(AudioDecoder, Path, dataset_path, plt, torch):
    # grab the first file in the folder
    root = Path(f"{dataset_path}/recordings")
    _file0 = Path(sorted(root.glob("*.wav"))[0])

    audio_decoder = AudioDecoder(_file0)
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

    _axes[0].set(xlabel="Source time (s)", ylabel="Amplitude", title="Left channel")
    _axes[1].plot(_segment_time[:160].numpy(), segment.data[0, :160].numpy(), ".-")
    _axes[1].set(
        xlabel="Source time (s)",
        ylabel="Amplitude",
        title="First 10 ms of the excerpt",
    )
    plt.close(_fig)
    _fig
    return


if __name__ == "__main__":
    app.run()

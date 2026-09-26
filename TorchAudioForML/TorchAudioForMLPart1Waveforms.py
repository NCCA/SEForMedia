#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # torchaudio for Machine Learning, Part 1: waveforms

    These four notebooks follow on from [PyTorchForML](../PyTorchForML/) and [TorchVisionForML](../TorchVisionForML/). We will work through waveforms, frequency features, augmentation and a small classification task. Each notebook runs on its own and generates its own audio.

    I will start with the tensor itself. A sample rate tells us how to interpret the samples in time; it is not stored in a plain tensor. Keep it alongside the waveform.

    The [torchaudio documentation](https://docs.pytorch.org/audio/stable/index.html) describes the library. Start playback at a comfortable volume using the audio controls below.
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
    ## What are we storing when we record a sound?

    Sound is a changing pressure in the air. A microphone converts that motion into an electrical signal, and a digital recording stores measurements of that signal taken at regular intervals. Each measurement is a **sample**. A **waveform** is the sequence of those samples, or a graph showing their values over time.

    | Term | Meaning here |
    | --- | --- |
    | Amplitude | The signed height of the waveform. Larger excursions generally mean a stronger signal, but amplitude alone does not determine perceived loudness. |
    | Frequency | How many times a repeating wave completes a cycle each second. **Hertz (Hz)** means cycles per second; 1 kHz is 1000 Hz. |
    | Pitch | How high or low a sound seems to us. A higher-frequency pure tone usually sounds higher in pitch. |
    | Sine wave | A smooth repeating wave with one frequency, useful as a simple test tone. |
    | Sample rate | How many measurements we store each second. This is different from the frequency of the sound. |
    | Channel | One stream of samples. **Mono** has one channel; **stereo** normally has left and right channels. |

    A 440 Hz tone recorded at 16,000 samples per second has about 36 samples per cycle. Playing a tone and choosing how often to measure it are separate decisions.

    A **tensor** is PyTorch's container for numbers arranged along one or more axes, rather like a table that can have more than two dimensions. Its **shape** lists the length of each axis. Here `(2, 16000)` means two channels with 16,000 samples each. `float32` stores each value as a 32-bit floating point number, which can represent fractions. A **batch** is a group of clips processed together. The [PyTorch tensor introduction](../PyTorchForML/PyTorchForMLPart1Tensors.py) revisits indexing, shapes and data types.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Samples, channels and time

    | Value | Meaning in this notebook |
    | --- | --- |
    | `sample_rate = 16000` | 16,000 samples per second |
    | `(2, 16000)` | two channels, one second of audio |
    | `float32` | floating point amplitudes, here between −1 and 1 |
    | `(batch, channels, samples)` | layout when we stack equal length clips |

    We generate 440 Hz in the left channel and 660 Hz in the right. Use `arange / sample_rate` so samples are exactly one sampling interval apart. `linspace(0, 1, 16000)` includes both endpoints and gives a different spacing.
    """)
    return


@app.cell
def _(torch):
    sample_rate = 16000
    time = torch.arange(sample_rate) / sample_rate
    waveform = 0.25 * torch.stack(
        [
            torch.sin(2 * torch.pi * 440 * time),
            torch.sin(2 * torch.pi * 660 * time),
        ]
    )
    mono = waveform.mean(dim=0, keepdim=True)
    print("stereo", tuple(waveform.shape), "mono", tuple(mono.shape))
    print("duration", waveform.shape[-1] / sample_rate, "seconds")
    return mono, sample_rate, time, waveform


@app.cell
def _(plt, time, waveform):
    _fig, _axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    for _axis, _channel, _name in zip(
        _axes, waveform, ["Left: 440 Hz", "Right: 660 Hz"]
    ):
        _axis.plot(time[:160].numpy(), _channel[:160].numpy())
        _axis.set(ylabel="Amplitude", title=_name, ylim=(-0.3, 0.3))
    _axes[-1].set_xlabel("Time (s)")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## WAV files and playback

    **WAV** is an audio file format. We use **PCM (pulse-code modulation)**, which stores regularly spaced sample values. In 16-bit PCM each sample is rounded to one of 65,536 representable levels. This rounding is **quantisation**, which explains the small round-trip error below. **Encoding** writes samples into a file representation; **decoding** reads them back.

    The [SoundFile API](https://python-soundfile.readthedocs.io/) uses `(samples, channels)`, so we **transpose** at this boundary: we swap the channel and sample axes without changing the sound. We write 16-bit PCM to a temporary directory and read it back as `float32`. Quantisation means the decoded values are close to, but not exactly, the originals. This conversion does not normalise the loudness.

    [TorchAudio 2.9 and later](https://docs.pytorch.org/audio/main/torchaudio) use TorchCodec for `torchaudio.load` and `torchaudio.save`. Older releases use audio backends, the libraries which do the file reading and writing. TorchCodec and [FFmpeg](https://ffmpeg.org/about.html) are tools for encoding and decoding media. [CUDA](https://developer.nvidia.com/cuda-zone) allows PyTorch to use supported NVIDIA graphics processors. I use SoundFile here so this example also works with the repository's older CUDA installation, without requiring FFmpeg. The processing in the rest of the series uses torchaudio.

    With a compatible TorchCodec and FFmpeg installation, the equivalent torchaudio calls are:

    ```python
    waveform, sample_rate = torchaudio.load("clip.wav")
    torchaudio.save("copy.wav", waveform, sample_rate)
    ```

    Check the [TorchCodec installation instructions](https://github.com/pytorch/torchcodec#installing-torchcodec) for the versions required by your PyTorch installation.
    """)
    return


@app.cell
def _(mo, sample_rate, torch, waveform):
    import io
    import tempfile
    from pathlib import Path
    import soundfile as sf

    def audio_player(samples: torch.Tensor, rate: int) -> mo.Html:
        buffer = io.BytesIO()
        sf.write(
            buffer,
            samples.detach().cpu().T.numpy(),
            rate,
            format="WAV",
            subtype="PCM_16",
        )
        return mo.audio(buffer.getvalue())

    with tempfile.TemporaryDirectory() as _directory:
        _path = Path(_directory) / "stereo.wav"
        sf.write(_path, waveform.T.numpy(), sample_rate, subtype="PCM_16")
        _data, loaded_rate = sf.read(_path, dtype="float32", always_2d=True)
    loaded = torch.from_numpy(_data.T.copy())
    roundtrip_error = (loaded - waveform).abs().max().item()
    print("loaded", tuple(loaded.shape), loaded_rate, "Hz")
    print("largest quantisation error", roundtrip_error)
    audio_player(loaded, loaded_rate)
    return (audio_player,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [Resample](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.Resample.html)

    ```python
    T.Resample(orig_freq=16000, new_freq=8000)(waveform)
    ```

    Resampling changes the number of samples whilst preserving duration and pitch. Just changing the playback sample rate changes both duration and pitch. A **low-pass filter** reduces high frequencies whilst retaining lower ones. Before **downsampling** (reducing the sample rate), we filter frequencies which the new rate cannot represent. Otherwise they can appear as false lower frequencies, an effect called **aliasing**. Imagine a filmed wheel appearing to turn backwards because the camera captures too few positions.

    Half the sample rate is called the **Nyquist frequency**. At 8 kHz it is 4 kHz; frequencies above that limit cannot be represented correctly. Practical filters also need a transition region near the limit. The [audio resampling tutorial](https://docs.pytorch.org/audio/main/tutorials/audio_resampling_tutorial.html) shows this with plots.

    `transforms.Resample` caches its filter for repeated use: it keeps the calculated filter instead of rebuilding it each time. `torchaudio.functional.resample` is useful for a single call. Averaging channels is a separate operation: **phase** describes the position within a repeating cycle. If one channel is exactly the negative of the other, adding them gives zero, so averaging them into mono produces silence.
    """)
    return


@app.cell
def _(T, mono):
    resampled = T.Resample(orig_freq=16000, new_freq=8000)(mono)
    print("resampled shape", tuple(resampled.shape))
    print("duration", resampled.shape[-1] / 8000, "seconds")
    return (resampled,)


@app.cell
def _(mo):
    playback_rate = mo.ui.dropdown(
        [8000, 16000, 24000], value=16000, label="Playback sample rate (Hz)"
    )
    playback_rate
    return (playback_rate,)


@app.cell
def _(audio_player, mo, mono, playback_rate, resampled):
    mo.vstack(
        [
            mo.md("**Same samples, changed playback rate**"),
            audio_player(mono, playback_rate.value),
            mo.md("**Resampled to 8 kHz, played at 8 kHz**"),
            audio_player(resampled, 8000),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cropping and level

    Slice the last axis to select time. For half a second starting at 0.25 seconds, use `waveform[:, 4000:12000]` at 16 kHz. A crop can start or end partway through a cycle, causing a click at playback boundaries. We will add fades in Part 3.

    **RMS (root mean square)** squares the samples, averages those squares, then takes the square root. This measures typical signal size without positive and negative values cancelling; peak amplitude tells us how close the waveform is to the representable limit when saving PCM. Floating point processing can exceed ±1, but this can **clip** when encoded: values outside the allowed range are cut off, distorting the waveform. Peak level is the largest absolute sample value; it is different from RMS.
    """)
    return


@app.cell
def _(sample_rate, torch, waveform):
    crop = waveform[:, int(0.25 * sample_rate) : int(0.75 * sample_rate)]
    print("crop", tuple(crop.shape))
    print("RMS per channel", torch.sqrt(waveform.square().mean(dim=-1)))
    print("peak per channel", waveform.abs().amax(dim=-1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Generate two seconds of audio. Check the shape and duration before playback.
    2. Put the same tone in both channels, then negate one channel. What happens when we average them?
    3. Add a 6 kHz tone before resampling to 8 kHz. Explain what the filter should remove.
    4. Compare a quiet WAV with a loud one. Does loading either file change its peak to 1?

    Next we will look at which frequencies are present in the waveform.
    """)
    return


if __name__ == "__main__":
    app.run()

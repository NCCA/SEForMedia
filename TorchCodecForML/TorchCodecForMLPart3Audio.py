import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TorchCodec for Machine Learning, Part 3: audio I/O

    We will encode two tones, decode them, request a time interval and change the output sample rate. This follows the waveform work in [TorchAudioForML](../TorchAudioForML/). TorchCodec handles file I/O here; feature extraction and augmentation can happen afterwards using the tensors.


    The tone used is generated, make sure the volume is not too loud!
    """)
    return


@app.cell
def _():
    import io
    import tempfile
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torchcodec
    from torchcodec.decoders import AudioDecoder
    from torchcodec.encoders import AudioEncoder

    print("torch", torch.__version__, "torchcodec", torchcodec.__version__)
    return AudioDecoder, AudioEncoder, Path, io, mo, plt, tempfile, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A waveform and its sample rate

    A sample is one measurement of a signal. The sample rate tells us how many occur per second; it is separate from pitch. We generate two seconds at 16,000 samples per second, with a different tone in each channel.

    | Quantity | Value |
    | --- | --- |
    | Waveform shape | `(2, 32000)`: channels then samples |
    | Sample rate | 16,000 Hz |
    | Tone frequencies | 440 Hz left, 660 Hz right |
    | Data type | `float32`, with amplitude 0.2 |

    The time coordinate is `arange / sample_rate`. We avoid including a duplicate endpoint at exactly two seconds.
    """)
    return


@app.cell
def _(torch):
    sample_rate = 16000
    time = torch.arange(2 * sample_rate, dtype=torch.float32) / sample_rate
    waveform = 0.2 * torch.stack(
        [
            torch.sin(2 * torch.pi * 440 * time),
            torch.sin(2 * torch.pi * 660 * time),
        ]
    )
    print(tuple(waveform.shape), waveform.dtype)
    return sample_rate, waveform


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [AudioEncoder](https://meta-pytorch.org/torchcodec/stable/generated/torchcodec.encoders.AudioEncoder.html)

    ```python
    AudioEncoder(waveform, sample_rate=16000).to_file("tones.wav")
    ```

    The filename selects the output format. WAV is useful for inspecting a round trip; PCM storage may quantise floating point amplitudes to integer levels. We therefore measure the error rather than assuming bit-for-bit equality with the original floats.

    We retain the temporary directory object whilst the notebook is running. That keeps the file available to downstream cells and removes it when the object is cleaned up.
    """)
    return


@app.cell
def _(AudioDecoder, AudioEncoder, Path, sample_rate, tempfile, waveform):
    media_directory = tempfile.TemporaryDirectory(prefix="torchcodec-audio-")
    audio_path = Path(media_directory.name) / "tones.wav"
    AudioEncoder(waveform, sample_rate=sample_rate).to_file(audio_path)
    audio_decoder = AudioDecoder(audio_path)
    samples = audio_decoder.get_all_samples()
    roundtrip_error = (samples.data - waveform).abs().max().item()
    print(audio_decoder.metadata)
    print("decoded:", tuple(samples.data.shape), samples.sample_rate, "Hz")
    print("largest round-trip error:", roundtrip_error)
    return audio_decoder, audio_path, samples


@app.cell
def _(audio_path, mo):
    mo.audio(audio_path.read_bytes())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [AudioDecoder](https://meta-pytorch.org/torchcodec/stable/generated/torchcodec.decoders.AudioDecoder.html) and time ranges

    `get_all_samples()` gives us an `AudioSamples` object. Its `data` holds the channel-first waveform, whilst `sample_rate` and `pts_seconds` describe its timing.

    We request `[0.5, 1.0)` seconds. At 16 kHz this is 8,000 samples per channel. The graph uses the returned start time, so the excerpt stays on the source timeline. For this generated WAV, the excerpt should match a tensor slice of the full decoding.

    The same decoder can read an audio stream from a video container. `stream_index` refers to the container's stream numbering, including video streams; it is not an audio-channel number.
    """)
    return


@app.cell
def _(audio_decoder, plt, samples, torch):
    segment = audio_decoder.get_samples_played_in_range(
        start_seconds=0.5, stop_seconds=1.0
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
    _axes[0].axvspan(0.5, 1.0, color="orange", alpha=0.3, label="Requested excerpt")
    _axes[0].set(xlabel="Source time (s)", ylabel="Amplitude", title="Left channel")
    _axes[0].legend()
    _axes[1].plot(_segment_time[:160].numpy(), segment.data[0, :160].numpy(), ".-")
    _axes[1].set(
        xlabel="Source time (s)", ylabel="Amplitude", title="First 10 ms of the excerpt"
    )
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Resampling during decoding

    A model may expect mono audio at a particular sample rate. Set these requirements on the decoder and inspect the returned samples, rather than assuming the source metadata describes the converted output.

    Resampling changes the sample spacing whilst preserving elapsed time. It is different from playing the same samples at a different rate, which changes duration and pitch. Channel mixing is separate again: opposite-phase signals can cancel when mixed to mono.
    """)
    return


@app.cell
def _(AudioDecoder, audio_path, samples):
    resampled = AudioDecoder(
        audio_path, sample_rate=8000, num_channels=1
    ).get_all_samples()
    print("source duration:", samples.data.shape[-1] / samples.sample_rate)
    print("output:", tuple(resampled.data.shape), resampled.sample_rate, "Hz")
    print("output duration:", resampled.data.shape[-1] / resampled.sample_rate)
    return (resampled,)


@app.cell
def _(AudioEncoder, io, mo, resampled):
    _playback = io.BytesIO()
    AudioEncoder(resampled.data, sample_rate=resampled.sample_rate).to_file_like(
        _playback, format="wav"
    )
    mo.audio(_playback.getvalue())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compressed bytes in a tensor

    FLAC compresses audio losslessly relative to its stored sample representation. Converting our floating point source to that representation may still introduce quantisation. A lossy format such as MP3 adds a different source of error.

    `to_tensor` returns encoded file bytes as a one-dimensional `uint8` tensor. That tensor is neither a waveform nor normalised model input. We decode it before comparing amplitudes.
    """)
    return


@app.cell
def _(AudioDecoder, AudioEncoder, audio_path, sample_rate, waveform):
    encoded_flac = AudioEncoder(waveform, sample_rate=sample_rate).to_tensor(
        format="flac"
    )
    flac_samples = AudioDecoder(encoded_flac).get_all_samples()
    flac_error = (flac_samples.data - waveform).abs().max().item()
    print("encoded FLAC:", tuple(encoded_flac.shape), encoded_flac.dtype)
    print("WAV bytes:", audio_path.stat().st_size, "FLAC bytes:", encoded_flac.numel())
    print("largest FLAC round-trip error:", flac_error)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try it

    1. Decode at 12 kHz. Predict the sample count for two seconds before running it.
    2. Change the excerpt to `[0.25, 0.75)`. Check both the number of samples and the returned timestamp.
    3. Make the second channel the negative of the first, then request mono. Explain the result.
    4. Encode to MP3 and compare file size and error. Check decoded length before subtracting tensors: lossy codecs can introduce delay and padding.

    Next: [datasets and model inputs](TorchCodecForMLPart4Datasets.py).
    """)
    return


if __name__ == "__main__":
    app.run()

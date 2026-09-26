#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # torchaudio for Machine Learning, Part 3: augmentation

    Audio **augmentation** creates altered versions of training examples whilst keeping the intended **label**, the answer we want a model to predict. For example, adding background hiss to a recording of “stop” should not change its word label. I will compare adding noise, changing level, filtering and masking spectrograms. Whether these preserve a label depends on the task: changing pitch might be reasonable for some speech tasks and wrong for musical note recognition.
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
    ## Signal, noise and repeatable experiments

    The **signal** is the sound we want to learn from; **noise** is the unwanted sound mixed with it. Our clean signal contains 440 Hz and 2200 Hz tones, where Hz means cycles per second. A higher-frequency pure tone has a higher perceived pitch. `torch.randn` generates random values from a normal, or bell-shaped, distribution; in this example they produce a hiss rather than a recorded background sound.

    A **random seed** chooses a repeatable starting point for a random-number generator. Reusing the seed lets us compare transforms on the same noise. It does not make every generated value identical.

    **SNR** means **signal-to-noise ratio**, a comparison of signal and noise energy. Here energy means the sum of squared sample values. **Decibels (dB)** express the ratio logarithmically: multiplying the ratio by ten adds 10 dB. We measure the added noise as `noisy - clean` when checking the result below. Part 2 explains [frequency features and logarithmic scales](TorchAudioForMLPart2Features.py).
    """)
    return


@app.cell
def _(torch):
    sample_rate = 16000
    time = torch.arange(sample_rate) / sample_rate
    clean = (
        0.25 * torch.sin(2 * torch.pi * 440 * time)
        + 0.1 * torch.sin(2 * torch.pi * 2200 * time)
    ).unsqueeze(0)
    _generator = torch.Generator().manual_seed(42)
    noise = torch.randn(clean.shape, generator=_generator)
    return clean, noise, sample_rate, time


@app.cell
def _(mo):
    snr_control = mo.ui.slider(
        start=0, stop=30, step=5, value=15, label="Signal-to-noise ratio (dB)"
    )
    snr_control
    return (snr_control,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [add_noise](https://docs.pytorch.org/audio/stable/generated/torchaudio.functional.add_noise.html)

    ```python
    torchaudio.functional.add_noise(waveform, noise, snr)
    ```

    The function scales the noise to the requested signal-to-noise ratio before adding it. Here 0 dB means equal signal and noise energy; 20 dB means the signal has 100 times the noise energy. A larger value gives less noise. Our fixed noise tensor makes the slider comparisons repeatable.
    """)
    return


@app.cell
def _(clean, noise, snr_control, torch):
    from torchaudio import functional as AF

    noisy = AF.add_noise(clean, noise, torch.tensor([float(snr_control.value)]))
    measured_snr = (
        10 * torch.log10(clean.square().sum() / (noisy - clean).square().sum())
    ).item()
    print("measured SNR", measured_snr, "dB")
    return AF, noisy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gain, fades and filters

    **Gain** multiplies sample amplitudes. [Vol](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.Vol.html) with −6 dB makes the amplitude about half as large; this does not guarantee that we perceive it as half as loud. PCM is the sample encoding used for our WAV files. Values beyond its allowed range can be cut off, or **clipped**, causing distortion.

    A **fade** changes gain gradually. [Fade](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.Fade.html) brings our sound in and out over 800 samples, or 50 milliseconds at 16 kHz. A linear fade changes the amplitude multiplier by equal steps.

    A **low-pass filter** reduces higher frequencies whilst retaining lower ones. Its **cutoff** marks the transition region, not a brick wall: a 1 kHz cutoff does not abruptly remove everything above 1000 Hz. [lowpass_biquad](https://docs.pytorch.org/audio/stable/generated/torchaudio.functional.lowpass_biquad.html) uses a small recursive filter; *biquad* is the name of that filter structure. Listen for the reduction of our higher tone and compare the waveform plots.
    """)
    return


@app.cell
def _(AF, T, clean, sample_rate):
    quieter = T.Vol(gain=-6, gain_type="db")(clean)
    faded = T.Fade(fade_in_len=800, fade_out_len=800, fade_shape="linear")(clean)
    filtered = AF.lowpass_biquad(clean, sample_rate, cutoff_freq=1000)
    return faded, filtered, quieter


@app.cell
def _(clean, faded, filtered, noisy, plt, quieter, time):
    _fig, _axes = plt.subplots(5, 1, figsize=(10, 8), sharex=True, sharey=True)
    for _axis, _values, _name in zip(
        _axes,
        [clean, noisy, quieter, faded, filtered],
        ["Original", "Added noise", "Gain: −6 dB", "50 ms fades", "Low-pass: 1 kHz"],
    ):
        _axis.plot(time[:1600].numpy(), _values[0, :1600].numpy(), linewidth=0.7)
        _axis.set(title=_name, ylabel="Amplitude", ylim=(-0.7, 0.7))
    _axes[-1].set_xlabel("Time (s)")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell
def _(clean, faded, filtered, mo, noisy, sample_rate):
    import io
    import soundfile as sf

    _players = []
    for _name, _samples in [
        ("Original", clean),
        ("Noise", noisy),
        ("Fades", faded),
        ("Low-pass", filtered),
    ]:
        _buffer = io.BytesIO()
        sf.write(
            _buffer, _samples.T.numpy(), sample_rate, format="WAV", subtype="PCM_16"
        )
        _players.append(mo.vstack([mo.md(_name), mo.audio(_buffer.getvalue())]))
    mo.hstack(_players, wrap=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SpecAugment masks

    **SpecAugment** is a method of augmenting spectrogram features. A **mask** hides a selected part by replacing its values, here with zeros. A spectrogram frame describes a short time interval, and a mel band groups nearby frequencies using overlapping weights. [Part 2](TorchAudioForMLPart2Features.py) introduces both. *Time warping* moves positions along the time axis, stretching or compressing local timing.

    [FrequencyMasking and TimeMasking](https://docs.pytorch.org/audio/stable/transforms.html#augmentations) hide bands or frames. Their parameters specify an upper bound for the randomly sampled mask width, so a call can draw a zero-width mask. Here we mask linear mel power before converting it to dB. Setting log features to zero would have a different meaning.

    I have chosen a seed which puts the frequency mask across the upper tone. The copies below let us compare the original with the result on the same colour scale. Masking does not create a new playable waveform. These two operations demonstrate the masking part of SpecAugment; we are not implementing its time-warping step.

    Calling `.eval()` on a masking transform does not disable its randomness. **Training** uses examples to adjust the model. **Validation** uses separate examples to help choose settings; **testing** checks the final model on examples held back from those choices. Apply augmentation explicitly in the training path and keep it out of validation and test preprocessing.
    """)
    return


@app.cell
def _(T, clean, sample_rate, torch):
    clean_power = T.MelSpectrogram(
        sample_rate=sample_rate, n_fft=512, hop_length=128, n_mels=64
    )(clean)
    original_power = clean_power.clone()
    with torch.random.fork_rng():
        torch.manual_seed(4)
        masked_power = T.TimeMasking(time_mask_param=20)(
            T.FrequencyMasking(freq_mask_param=16)(clean_power.clone())
        )
    print("original unchanged", torch.equal(clean_power, original_power))
    return clean_power, masked_power


@app.cell
def _(T, clean_power, masked_power, plt):
    _original_db = T.AmplitudeToDB(stype="power", top_db=80)(clean_power)[0]
    _masked_db = T.AmplitudeToDB(stype="power", top_db=80)(masked_power)[0]
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for _axis, _values, _name in zip(
        _axes, [_original_db, _masked_db], ["Original", "Frequency and time masks"]
    ):
        _image = _axis.imshow(
            _values.numpy(),
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=float(_original_db.max()) - 80,
            vmax=float(_original_db.max()),
        )
        _axis.set(title=_name, xlabel="Frame index", ylabel="Mel band index")
    _fig.colorbar(_image, ax=_axes, label="dB (power reference 1)")
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choosing augmentation

    Use a fixed seed when explaining a transform, then allow examples to vary during training. Split recordings before making augmented copies; otherwise versions of the same recording can appear in both training and validation. This is **data leakage**: evaluation gets access to information it should not have, making results look better than performance on new recordings.

    A low-pass filter suppresses the 2.2 kHz component in our signal. It also removes potentially useful information. A gain change could erase the distinction between classes defined by loudness. Listen to examples and inspect feature plots before adding a transform to a training pipeline.

    ## Exercises

    1. Compare 0, 15 and 30 dB SNR by listening and by checking the measured ratio.
    2. Increase the gain above 0 dB. Check the peak before encoding it as PCM.
    3. Change the random seed for masking. Find a seed which produces a smaller mask.
    4. Decide which operations you would use for speech commands, instrument identification and pitch classification. Explain each choice.

    Part 4 puts features into a dataset and trains a classifier.
    """)
    return


if __name__ == "__main__":
    app.run()

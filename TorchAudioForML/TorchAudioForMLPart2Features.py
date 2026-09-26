#!/usr/bin/env -S uv run marimo edit

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # torchaudio for Machine Learning, Part 2: spectrograms and features

    A waveform shows amplitude (the signed sample value) over time. Frequency is the rate of repetition, measured in hertz (Hz, cycles per second). A **feature** is a numerical description we extract from audio to help a model work with it. A spectrogram shows how the frequency content changes over time. We will build both, then compare linear frequency bins, mel bands and MFCCs. These are different representations of the same input, not interchangeable image formats.
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
    ## What is an FFT?

    Think of a musical chord: we hear one combined sound, but several notes contribute to it. Fourier analysis describes a sampled signal as a sum of simple repeating sine and cosine waves. It tells us how strongly different frequencies contribute, although it does not automatically identify instruments or separate voices.

    **DFT** means **discrete Fourier transform**: the calculation for a finite list of samples. **FFT** means **fast Fourier transform**: an efficient way to calculate that same result. We do not need to implement the algorithm to use it. The [NumPy Fourier background](https://numpy.org/doc/stable/reference/routines.fft.html#background-information) gives the mathematical details.

    | View | What its horizontal axis means | What we look for |
    | --- | --- | --- |
    | Waveform, or time domain | Time in seconds | How sample values change |
    | Spectrum, or frequency domain | Frequency in Hz | Which repeating components contribute |

    A **frequency bin** is one position in the FFT output, associated with a particular frequency. For our default 512-sample FFT at 16 kHz, neighbouring bins are `16000 / 512 = 31.25 Hz` apart. This spacing is a grid for the calculation; a real sound can lie between the grid points.

    The FFT output contains **magnitude**, describing the strength of each component, and **phase**, describing where its cycle starts relative to our time reference. A **complex number** stores the pair of values needed to represent both. In code we can obtain magnitude with `abs()`. Squaring that magnitude produces the **power spectrum** used here. Its values depend on the transform's scaling and are not measurements of electrical watts.

    A single FFT over the entire clip does not directly give a timeline of when each tone occurs. For our changing sound we need to analyse short sections in order.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A changing signal

    A **chirp** is a tone whose frequency changes over time. I have used one whose frequency rises from 300 Hz to 3 kHz, with an 880 Hz tone added halfway through. To generate it, we keep track of how far the wave has progressed through its cycles. This accumulated progress is its phase; mathematically, we obtain it by integrating frequency over time. You do not need calculus for the example: the expression below makes the tone rise at a steady rate. This gives us a diagonal line and a horizontal line to find in the spectrogram.
    """)
    return


@app.cell
def _(torch):
    sample_rate = 16000
    time = torch.arange(sample_rate * 2) / sample_rate
    _phase = 2 * torch.pi * (300 * time + 0.5 * 1350 * time.square())
    signal = (
        0.3 * torch.sin(_phase)
        + 0.15 * torch.sin(2 * torch.pi * 880 * time) * (time >= 1)
    ).unsqueeze(0)
    return sample_rate, signal, time


@app.cell
def _(mo, torch):
    import io
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

    return (audio_player,)


@app.cell
def _(audio_player, mo, sample_rate, signal):
    mo.vstack(
        [
            mo.md(
                "Listen to the rising chirp. The steady 880 Hz tone joins halfway through."
            ),
            audio_player(signal, sample_rate),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From an FFT to a spectrogram

    We divide the recording into short overlapping sections called **frames**. We calculate an FFT for each frame, then arrange the results in time order. This is the **short-time Fourier transform (STFT)**. A **spectrogram** displays the magnitude or power of these results as an image: time goes across, frequency goes up, and colour represents strength.

    Before each FFT we apply a **window function**, a set of weights that gently reduces the samples near the frame's edges. The default **Hann window** has a smooth raised-and-lowered shape. This reduces **spectral leakage**, where cutting a finite section of a wave spreads its energy across neighbouring bins. Windowing reduces leakage but does not make each tone occupy exactly one bin.

    The **hop** is how many samples we move forward before analysing the next frame. At the default settings, each frame covers `512 / 16000 = 0.032` seconds, or 32 milliseconds. A hop of 128 samples starts a frame every 8 milliseconds, so neighbouring frames overlap.

    **Padding** adds values beyond a recording's boundaries so we can analyse frames near its ends. With `center=True`, TorchAudio centres the frames and uses reflected samples at the boundaries by default. The [STFT reference](https://docs.pytorch.org/docs/stable/generated/torch.stft.html) explains these parameters.

    Try the dropdown and compare the plots below. Here the FFT size also sets the actual window length. A longer window observes more cycles, helping distinguish nearby frequencies, but gives a less precise view of sudden changes. Increasing FFT size by adding zeros alone would only make the frequency grid denser; it would not add more observations.
    """)
    return


@app.cell
def _(mo):
    fft_size = mo.ui.dropdown([256, 512, 1024], value=512, label="FFT window (samples)")
    fft_size
    return (fft_size,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [Spectrogram](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.Spectrogram.html)

    | Argument | Choice here | Meaning |
    | --- | --- | --- |
    | `n_fft` | dropdown selection | FFT size, also the default window length |
    | `hop_length` | `n_fft // 4` | samples between successive windows |
    | `power` | `2.0` | squared magnitude, giving power |
    | `center` | `True` | pad the input to centre the windows |

    Our audio samples are real numbers. Their FFT has a symmetry, so the one-sided spectrum only needs the non-negative frequencies. For our even FFT sizes, this gives `n_fft // 2 + 1` bins, including 0 Hz and half the sample rate. Larger windows separate nearby frequencies better, but spread short events across more time. A smaller hop places windows closer together; it does not improve the window's underlying frequency resolution.
    """)
    return


@app.cell
def _(T, fft_size, sample_rate, signal, torch):
    n_fft = fft_size.value
    hop = n_fft // 4
    power = T.Spectrogram(n_fft=n_fft, hop_length=hop, power=2.0)(signal)
    power_db = T.AmplitudeToDB(stype="power", top_db=80)(power)
    frame_times = torch.arange(power.shape[-1]) * hop / sample_rate
    frequencies = torch.arange(power.shape[-2]) * sample_rate / n_fft
    print("(channels, frequency bins, frames)", tuple(power.shape))
    print("bin spacing", sample_rate / n_fft, "Hz; hop", hop / sample_rate, "s")
    return frame_times, frequencies, hop, n_fft, power_db


@app.cell
def _(frame_times, frequencies, plt, power_db, signal, time):
    _fig, _axes = plt.subplots(2, 1, figsize=(10, 6))
    _axes[0].plot(time.numpy(), signal[0].numpy(), linewidth=0.5)
    _axes[0].set(xlabel="Time (s)", ylabel="Amplitude", title="Waveform")
    _image = _axes[1].pcolormesh(
        frame_times.numpy(),
        frequencies.numpy(),
        power_db[0].numpy(),
        shading="auto",
        cmap="magma",
    )
    _axes[1].set(xlabel="Time (s)", ylabel="Frequency (Hz)", title="Power spectrogram")
    _fig.colorbar(_image, ax=_axes[1], label="dB (power reference 1)")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mel bands: grouping frequencies

    An FFT gives us evenly spaced frequency bins. With our default `n_fft=512` and sample rate of 16 kHz, the bins are 31.25 Hz apart. The power spectrum tells us how much energy each bin contains during one short frame.

    Our perception of pitch does not follow an evenly spaced Hz scale. The mel scale is an approximation to perceived pitch spacing. Equal steps in mel correspond to smaller frequency differences at low frequencies and larger differences at high frequencies. It is useful for audio features, but it is not a complete model of hearing.

    HTK (Hidden Markov Model Toolkit) and Slaney name two conventions for calculating a mel scale; their settings are documented in [MelScale](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.MelScale.html). We use the default HTK mel scale in this notebook. Its conversion from frequency $f$ in Hz to mel is:

    $$m(f) = 2595\log_{10}\left(1 + \frac{f}{700}\right)$$

    For example, 0 Hz maps to 0 mel, 1 kHz to about 1000 mel, and 8 kHz to about 2840 mel. Doubling a frequency does not double its mel value.

    ### From FFT bins to mel bands

    A mel band is the output of a weighted filter covering a range of frequencies. We place overlapping triangular filters across the spectrum, spacing their boundary and centre points equally in mel. On a Hz axis, the filters become wider towards the high frequencies. A bin near a triangle's centre gets a large weight; a bin near its edges gets a small one.

    A **weight** is a multiplier that controls a value's contribution. A **filter bank** is a collection of these frequency-selective filters. For each time frame we multiply the power in each FFT bin by its filter weight, then sum over the bins:

    $$E_{b,t} = \sum_k H_{b,k}P_{k,t}$$

    Here $P_{k,t}$ is power in FFT bin $k$ at frame $t$, $H_{b,k}$ is the weight assigned by mel filter $b$, and $E_{b,t}$ is the resulting band energy. This is a weighted sum, not a selection of one FFT bin or an average over time. Adjacent filters overlap, so one tone can contribute to several bands.

    The [filter-bank API](https://docs.pytorch.org/audio/stable/generated/torchaudio.functional.melscale_fbanks.html) exposes these weights. In this notebook we use 64 bands over 0–8 kHz. The default `norm=None` leaves the triangles without area normalisation (rescaling each filter to compensate for its width); `norm="slaney"` changes their weighting. Keep the scale and normalisation consistent between training and inference.

    [MelSpectrogram](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.MelSpectrogram.html) performs the STFT, power calculation and filter-bank projection for us. With the default dropdown setting, the shapes are:

    | Representation | Shape | Meaning of one column |
    | --- | --- | --- |
    | Power spectrogram | `(1, 257, 251)` | 257 FFT-bin powers for one frame |
    | Mel spectrogram | `(1, 64, 251)` | 64 weighted band energies for the same frame |

    The first dimension is our single audio channel. The last is time, and it stays unchanged when we apply the filters. Changing the FFT dropdown also changes the hop and therefore the frame count.

    More mel bands retain finer spectral detail, but cannot recover detail absent from the FFT. Too many bands for a small FFT can produce filters with no supporting bins. Set `sample_rate` to the actual waveform rate: passing a different value changes the filter interpretation, not the audio's sample rate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why take the logarithm?

    Band energies can span a large range. **Logarithms** describe numbers in terms of powers: `log10(10) = 1`, `log10(100) = 2` and `log10(1000) = 3`. Each tenfold increase becomes one extra step. Taking the logarithm compresses that range so quieter structure is easier to inspect. It also turns a multiplicative change in energy into an additive change in the log features.

    A **decibel (dB)** expresses a ratio on a logarithmic scale. For power, a tenfold increase is +10 dB and a hundredfold increase is +20 dB. Our plot uses a numerical power reference of 1; 0 dB does not mean silence. The **natural logarithm**, used later by `torch.log`, uses the mathematical constant e (about 2.718) as its base instead of 10.

    For the mel plot below, we use `AmplitudeToDB(stype="power", top_db=80)`. Power uses $10\log_{10}(E)$; amplitude uses $20\log_{10}(A)$. Numerical protection is needed at zero. The `top_db` setting limits the displayed range to 80 dB below the maximum. These are not calibrated sound pressure levels.

    Read the mel plot as follows: left to right is time, bottom to top is increasing mel band index, and colour indicates band energy in dB. Band 20 does not mean 20 Hz or 20 mel. Its centre frequency depends on the sample rate, band count, frequency limits and chosen mel scale.

    The rising chirp moves through the bands. The steady 880 Hz tone forms a horizontal feature after one second. Mel filtering spreads their energy over neighbouring bands and discards some of the original frequency detail.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MFCCs: describing the shape of the spectrum

    **MFCC** stands for **mel-frequency cepstral coefficient**. A **coefficient** is a numerical weight in a combination of patterns. *Cepstral* names this family of features obtained by transforming a logarithmic spectrum; you do not need to calculate a cepstrum separately here. We start with the mel energies for one frame, take their logarithms, then apply a discrete cosine transform (DCT) across the mel-band axis. The time frames are processed independently.

    The DCT expresses the log mel spectrum as a combination of cosine patterns. The first pattern is constant across all bands, the next varies slowly across them, and later patterns alternate more rapidly. Each coefficient says how much of one pattern is needed to describe the frame.

    You can read the equation as “multiply each log band value by a cosine weight, then add the results”. The symbol $\sum$ means “sum”, and $\propto$ means “proportional to”. The type-II label specifies the particular [DCT convention](https://docs.pytorch.org/audio/stable/generated/torchaudio.functional.create_dct.html). Ignoring normalisation factors, the type-II DCT used here has the form:

    $$c_{q,t} \propto \sum_{b=0}^{B-1} L_{b,t}\cos\left[\frac{\pi q}{B}\left(b+\frac12\right)\right]$$

    $L_{b,t}$ is the log energy in band $b$, $B$ is the number of mel bands, and $q$ is the coefficient index. Notice that the sum runs over bands, not time. Each coefficient combines information from the whole mel spectrum.

    - Coefficient 0 is proportional to the sum of log band energies. It is sensitive to overall signal level, but is not literally the waveform's total energy.
    - The next few coefficients describe broad changes in spectral shape, such as the balance between lower and higher bands.
    - Higher-order coefficients describe finer variations across bands. Their index does not identify an audio frequency.

    This broad shape is often called the *spectral envelope*. For speech it can describe resonances associated with the vocal tract (frequencies reinforced by the shape of the throat and mouth); it is different from following every individual harmonic (a component at a whole-number multiple of a fundamental frequency). MFCCs provide a compact description, but they are not guaranteed to retain the information needed for every task.

    ### Our settings: 64 bands, 13 coefficients

    We keep coefficients 0–12 using `n_mfcc=13`, giving shape `(1, 13, 251)` at the default FFT setting. Retaining only 13 of the 64 possible DCT coefficients discards finer spectral detail. Thirteen is a teaching choice and a common small feature size, not a required value.

    Here **orthonormal** means that the cosine patterns are independent and scaled to unit length, avoiding arbitrary differences in their size. An orthonormal DCT retaining all 64 coefficients could recover the log mel values. Keeping only 13 cannot generally do so. Even all 64 would not restore the original waveform: the earlier power spectrum discarded phase, and mel filtering grouped frequency bins together.

    [TorchAudio's MFCC transform](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.MFCC.html) defaults to a decibel mel representation. We explicitly use `log_mels=True`, which instead takes the natural logarithm with numerical protection at zero. Consequently, the MFCCs below are not computed directly from the displayed, range-limited `mel_db` tensor. They start from the waveform using the same mel settings.

    On the MFCC plot, the vertical axis is **coefficient index**, and the colour is a signed coefficient value. A negative value is a valid cosine weight; it does not mean negative physical energy. We cannot read this plot as a frequency map in the way we read the mel plot.

    `ComputeDeltas` estimates how each coefficient changes across neighbouring time frames. These delta features describe local change. They are separate from the DCT, which operates across bands within each frame.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### MFCCs versus averaging log mel features

    In Part 4 we take a different route: we average each log mel band over time to obtain one vector (a one-dimensional list of numbers) per clip. For a tensor with shape `(bands, frames)`, that operation is:

    ```python
    features = torch.log(mel.clamp_min(1e-6)).mean(dim=-1)
    ```

    With 32 bands it returns 32 numbers. The logarithm happens **before** the mean; `log(mean(mel))` is a different calculation. Time averaging discards the order of events, so low-then-high tones can look much like high-then-low tones.

    | Operation | Axis combined | What remains |
    | --- | --- | --- |
    | Mel filter bank | FFT frequency bins | Band energies for every frame |
    | MFCC DCT and coefficient selection | Log mel bands | A smaller spectral description for every frame |
    | Mean of log mel features over time | Time frames | One value per band for the whole clip |

    We could also average MFCCs over time, but computing MFCCs does not itself perform that averaging. Here we retain the frame sequence so we can inspect changes in the sound.
    """)
    return


@app.cell
def _(T, hop, n_fft, sample_rate, signal):
    mel_power = T.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop, n_mels=64
    )(signal)
    mel_db = T.AmplitudeToDB(stype="power", top_db=80)(mel_power)
    mfcc = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,
        log_mels=True,
        melkwargs={"n_fft": n_fft, "hop_length": hop, "n_mels": 64},
    )(signal)
    deltas = T.ComputeDeltas()(mfcc)
    print(
        "mel",
        tuple(mel_db.shape),
        "MFCC",
        tuple(mfcc.shape),
        "deltas",
        tuple(deltas.shape),
    )
    return mel_db, mfcc


@app.cell
def _(frame_times, mel_db, mfcc, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    for _axis, _values, _title, _label in zip(
        _axes,
        [mel_db[0], mfcc[0]],
        ["Log mel spectrogram", "MFCCs"],
        ["Mel band index", "Coefficient index"],
    ):
        _image = _axis.imshow(
            _values.numpy(),
            origin="lower",
            aspect="auto",
            extent=[
                float(frame_times[0]),
                float(frame_times[-1]),
                -0.5,
                _values.shape[0] - 0.5,
            ],
            cmap="magma",
        )
        _axis.set(title=_title, xlabel="Time (s)", ylabel=_label)
        _fig.colorbar(
            _image,
            ax=_axis,
            label="dB" if _label == "Mel band index" else "Coefficient value",
        )
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Keeping phase for reconstruction

    An **inverse transform** reverses the change of representation: here it takes frequency components back to waveform samples. A power spectrogram discards phase. We cannot pass it straight into an inverse STFT and expect the original waveform. Set `power=None` to retain the complex STFT, then use matching window and hop settings in [InverseSpectrogram](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.InverseSpectrogram.html). Supplying the original length avoids an ambiguous output length.
    """)
    return


@app.cell
def _(T, hop, n_fft, signal):
    complex_spectrum = T.Spectrogram(n_fft=n_fft, hop_length=hop, power=None)(signal)
    reconstructed = T.InverseSpectrogram(n_fft=n_fft, hop_length=hop)(
        complex_spectrum, length=signal.shape[-1]
    )
    reconstruction_error = (reconstructed - signal).abs().max().item()
    print("largest reconstruction error", reconstruction_error)
    return (reconstructed,)


@app.cell
def _(audio_player, mo, reconstructed, sample_rate, signal):
    mo.vstack(
        [
            mo.md(
                "Compare the original with the reconstruction. Keeping the complex spectrum lets us recover the waveform, so these should sound the same."
            ),
            mo.hstack(
                [
                    mo.vstack(
                        [mo.md("**Original**"), audio_player(signal, sample_rate)]
                    ),
                    mo.vstack(
                        [
                            mo.md("**Reconstructed**"),
                            audio_player(reconstructed, sample_rate),
                        ]
                    ),
                ],
                wrap=True,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Explain the difference between a waveform, a spectrum and a spectrogram using the plots above. Then change the FFT size and locate the 880 Hz tone. Compare its width and the time at which it appears.
    2. Try 32 and 80 mel bands. Keep enough FFT bins to support the filters and inspect any warnings.
    3. Compare `log_mels=True` and `False` in MFCC. Why do the numbers change?
    4. Remove the phase from the reconstruction example. Explain what information is missing.

    In Part 3 we will alter the waveform and its features for training.
    """)
    return


if __name__ == "__main__":
    app.run()

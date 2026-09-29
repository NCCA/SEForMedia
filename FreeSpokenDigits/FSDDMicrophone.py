#!/usr/bin/env -S uv run marimo run
import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torchaudio
    from torchcodec.decoders import AudioDecoder

    from DigitCNN import DigitCNN

    SAMPLE_RATE = 8000
    # these must match the MelSpectrogram in FSDDataLoader.py or the model sees
    # features it was never trained on
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=256,
        hop_length=80,
        n_mels=64,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Free Spoken Digits Live

    This uses the model trained in [part 3](FSDDMarimoPt3.py) to guess which digit you say. Press the microphone button, say a single digit (zero to nine) in English, then press it again to stop. Your browser will ask for permission to use the microphone the first time.

    The model was trained on the [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset), which is only six speakers recorded in fairly quiet rooms, so don't be surprised if it gets things wrong. Speak clearly, keep the background quiet and leave a small gap before and after the word.
    """)
    return


@app.cell
def _():
    # part 3 saves torch.__version__, which is a TorchVersion object rather than
    # a str, so we allow that one class and keep the safe weights_only loader
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(
            mo.notebook_dir() / "digits.pt", map_location="cpu", weights_only=True
        )
    model = DigitCNN()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return (model,)


@app.cell
def _():
    microphone = mo.ui.microphone(label="Say a digit")
    trim_threshold = mo.ui.slider(
        start=0.01,
        stop=0.2,
        step=0.01,
        value=0.05,
        label="Silence threshold",
        show_value=True,
    )
    mo.hstack([microphone, trim_threshold], justify="start", gap=2)
    return microphone, trim_threshold


@app.function
def decode_audio(data: bytes) -> torch.Tensor:
    """
    Decode a recording into mono audio at 8 kHz.

    The browser gives us a compressed webm / ogg blob rather than a wav, so
    torchcodec (via FFmpeg) does the decoding and resampling for us.

    Parameters
    ----------
        data : bytes
            the raw bytes of the recording in any format FFmpeg understands

    Returns
    -------
        torch.Tensor
            audio samples with shape [1, samples]
    """
    decoder = AudioDecoder(data, sample_rate=SAMPLE_RATE, num_channels=1)
    return decoder.get_all_samples().data


@app.function
def trim_silence(
    audio: torch.Tensor, frame: int = 160, threshold: float = 0.05
) -> torch.Tensor:
    """
    Cut the quiet parts from the start and end of a recording.

    The FSDD recordings are trimmed so the digit starts straight away, but a
    browser recording will have silence (and the click of the mouse) either
    side. Without trimming, the first second the model sees may be mostly
    silence.

    Parameters
    ----------
        audio : torch.Tensor
            samples with shape [1, samples]
        frame : int
            samples per energy frame, 160 is 20 ms at 8 kHz
        threshold : float
            fraction of the loudest frame's RMS a frame needs to count as speech

    Returns
    -------
        torch.Tensor
            the trimmed samples with shape [1, samples]
    """
    if audio.shape[-1] < frame:
        return audio
    frames = audio[0, : audio.shape[-1] // frame * frame].reshape(-1, frame)
    rms = frames.pow(2).mean(dim=1).sqrt()
    loud = torch.nonzero(rms >= rms.max() * threshold).flatten()
    if loud.numel() == 0:
        return audio
    start, end = loud[0].item() * frame, (loud[-1].item() + 1) * frame
    return audio[:, start:end]


@app.function
def audio_to_features(audio: torch.Tensor) -> torch.Tensor:
    """
    Turn audio into the normalised log-mel features the model expects.

    This is the same as SpokenDigits.__getitem__ in FSDDataLoader.py: take the
    first second, pad with zeros if short, then log and normalise.

    Parameters
    ----------
        audio : torch.Tensor
            samples at 8 kHz with shape [1, samples]

    Returns
    -------
        torch.Tensor
            features with shape [1, 64, 101]
    """
    audio = audio[:, :SAMPLE_RATE]
    audio = torch.nn.functional.pad(audio, (0, SAMPLE_RATE - audio.shape[-1]))
    feature = mel_transform(audio).clamp_min(1e-10).log()
    return (feature - feature.mean()) / feature.std().clamp_min(1e-6)


@app.cell
def _(microphone, trim_threshold):
    _data = microphone.value.getvalue()
    mo.stop(len(_data) == 0, mo.md("Record a digit above to see a prediction."))
    recording = trim_silence(decode_audio(_data), threshold=trim_threshold.value)
    features = audio_to_features(recording)
    return features, recording


@app.cell
def _(features, model):
    with torch.inference_mode():
        probabilities = model(features.unsqueeze(0)).softmax(1)[0]
    prediction = int(probabilities.argmax())
    return prediction, probabilities


@app.cell
def _(features, prediction, probabilities, recording):
    _fig, (_bars, _mel) = plt.subplots(1, 2, figsize=(11, 3), layout="constrained")
    _colours = ["tab:blue"] * 10
    _colours[prediction] = "tab:orange"
    _bars.bar(range(10), probabilities.numpy(), color=_colours)
    _bars.set(xlabel="Digit", ylabel="Softmax score", xticks=range(10), ylim=(0, 1))
    _mel.imshow(features[0].numpy(), origin="lower", aspect="auto", cmap="magma")
    _mel.set(xlabel="Time frame (10 ms)", ylabel="Mel band", title="Model input")
    plt.close(_fig)

    mo.vstack(
        [
            mo.md(
                f"# I heard **{prediction}** ({probabilities[prediction]:.0%} softmax score)"
            ),
            mo.md("This is the trimmed audio the model actually used:"),
            mo.audio(recording.numpy(), rate=SAMPLE_RATE, normalize=True),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What happens to the recording

    The steps are the same as training, with one extra. The browser sends a compressed recording, which torchcodec decodes to mono at 8 kHz. I then trim the quiet frames from each end, because the dataset recordings start as soon as the speaker does and the model has never seen a second of silence before a word. Any 20 ms frame quieter than the silence threshold (a fraction of the loudest frame) is treated as silence. When I tested this on dataset recordings padded with noise and encoded like a browser recording, 0.05 gave clips about the same length as the originals. Too high and quiet sounds like the "s" in seven get cut off, too low and the background noise stays in. If your room is noisy try moving the slider and listen to the trimmed audio to see what the model gets. After that it's exactly what `SpokenDigits` does: take the first second, pad it, and turn it into a normalised log-mel spectrogram of shape [1, 64, 101].

    The softmax score is not a calibrated probability. A confident wrong answer is quite common, especially with a voice or microphone unlike those in the training data.

    To run this as an app use

    ```bash
    uv run marimo run FSDDMicrophone.py
    ```

    `digits.pt` is created by the save button at the end of part 3 and needs to be in the same folder as this notebook.
    """)
    return


if __name__ == "__main__":
    app.run()

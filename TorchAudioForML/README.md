# torchaudio for Machine Learning

Four Marimo notebooks for the SE for Media machine learning classes, following [PyTorchForML](../PyTorchForML/) and [TorchVisionForML](../TorchVisionForML/). I use generated audio so we can run the examples without downloading recordings or model weights.

| Notebook | Topics |
| --- | --- |
| [Part 1: waveforms](TorchAudioForMLPart1Waveforms.py) | Channels, sample rates, WAV files, playback, resampling and cropping |
| [Part 2: features](TorchAudioForMLPart2Features.py) | STFT, power, decibels, mel bands, MFCCs and reconstruction |
| [Part 3: augmentation](TorchAudioForMLPart3Augmentation.py) | Noise, gain, fades, filtering and spectrogram masks |
| [Part 4: datasets](TorchAudioForMLPart4Datasets.py) | Variable lengths, DataLoader, classification and checkpoints |

Run these commands from the repository root. Each notebook runs independently; change the filename to open another part.

```sh
uv sync
uv run marimo edit TorchAudioForML/TorchAudioForMLPart1Waveforms.py
```

The notebooks include plots, audio controls and exercises. Start with a comfortable playback volume. Part 4 trains a small CPU model on generated tones; its accuracy only measures this synthetic task.

The project lockfile selects torchaudio alongside the existing PyTorch installation, including its CUDA 12.4 index on Linux and Windows. The WAV example uses SoundFile. Recent [torchaudio file I/O](https://docs.pytorch.org/audio/main/torchaudio) uses TorchCodec; Part 1 explains the difference and links to its installation requirements. TorchCodec and FFmpeg are not needed for these runnable examples.

For the full API, see the [torchaudio documentation](https://docs.pytorch.org/audio/stable/index.html).

## Developer notes

The tests execute all four notebooks and check their numerical results. HTML export executes the cells and builds a viewable notebook.

```sh
MPLBACKEND=Agg OMP_NUM_THREADS=1 uv run python -m unittest discover -s TorchAudioForML/tests
uv run marimo check TorchAudioForML/*.py
uv run --with ruff ruff check TorchAudioForML
uv run marimo export html TorchAudioForML/TorchAudioForMLPart1Waveforms.py -o /tmp/torchaudio-part1.html
```

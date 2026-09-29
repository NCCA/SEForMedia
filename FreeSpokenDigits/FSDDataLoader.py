from pathlib import Path

import torch
import torchaudio
from torchcodec.decoders import AudioDecoder


def split_recordings(
    paths: list[Path],
) -> tuple[list[Path], list[Path], list[Path]]:
    """Separate takes into training, validation and test recordings."""
    train, validation, test = [], [], []
    for path in paths:
        digit, _speaker, take = path.stem.split("_")
        if not 0 <= int(digit) <= 9 or int(take) < 0:
            raise ValueError(f"Invalid digit or take in {path.name}")
        target = test if int(take) < 5 else validation if int(take) < 10 else train
        target.append(path)
    if not all((train, validation, test)):
        raise ValueError(
            "We need recordings in all three take ranges: 0–4, 5–9 and 10 onwards."
        )
    return train, validation, test


class SpokenDigits(torch.utils.data.Dataset):
    """Decode and cache one-second log-mel features on the CPU.

    Attributes
    ----------
    paths : list[Path]
        Recording filenames, which also supply the digit labels.
    """

    def __init__(self, paths: list[Path]) -> None:
        self.paths = list(paths)
        self.cache = {}
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=8000,
            n_fft=256,
            hop_length=80,
            n_mels=64,
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if index not in self.cache:
            samples = AudioDecoder(
                str(self.paths[index]),
                sample_rate=8000,
                num_channels=1,
            ).get_all_samples()
            audio = samples.data[:, :8000]
            audio = torch.nn.functional.pad(audio, (0, 8000 - audio.shape[-1]))
            feature = self.mel(audio).clamp_min(1e-10).log()
            feature = (feature - feature.mean()) / feature.std().clamp_min(1e-6)
            label = int(self.paths[index].stem.split("_")[0])
            self.cache[index] = (feature, label)
        return self.cache[index]

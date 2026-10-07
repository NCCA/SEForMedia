import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF


def double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(),
    )


class UNet(nn.Module):
    """Four resolutions, with encoder features concatenated into the decoder."""

    def __init__(self, out_channels: int = 3, upsampling: str = "transpose") -> None:
        super().__init__()
        if upsampling not in ("transpose", "resize"):
            raise ValueError("Choose transpose or resize")
        self.encoders = nn.ModuleList(
            [
                double_conv(3, 32),
                double_conv(32, 64),
                double_conv(64, 128),
                double_conv(128, 256),
            ]
        )
        self.pool = nn.MaxPool2d(2)
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for source, dest in ((256, 128), (128, 64), (64, 32)):
            if upsampling == "transpose":
                up = nn.ConvTranspose2d(source, dest, kernel_size=2, stride=2)
            else:
                up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(source, dest, 3, padding=1),
                )
            self.ups.append(up)
            self.decoders.append(double_conv(2 * dest, dest))
        self.head = nn.Conv2d(32, out_channels, 1)

    def forward(self, x: torch.Tensor, use_skips: bool = True) -> torch.Tensor:
        height, width = x.shape[-2:]
        x = F.pad(x, (0, (-width) % 8, 0, (-height) % 8), mode="replicate")
        skips = []
        for encoder in self.encoders[:-1]:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.encoders[-1](x)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            skip = skip if use_skips else torch.zeros_like(skip)
            x = decoder(torch.cat((x, skip), dim=1))
        return self.head(x)[..., :height, :width]


class SuperResolutionUNet(nn.Module):
    def __init__(self, upsampling: str = "transpose") -> None:
        super().__init__()
        self.unet = UNet(3, upsampling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        return self.unet(x)


class ESPCN(nn.Module):
    """An RGB teaching variant of the sub-pixel convolution network."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2),
            nn.Tanh(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 3 * 2**2, 3, padding=1),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def make_patch(
    image: torch.Tensor,
    trimap: torch.Tensor,
    task: str,
    generator: torch.Generator,
    sigma: float = 25 / 255,
    size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Crop once so the photograph and mask stay aligned.

    Parameters
    ----------
    image : torch.Tensor
        RGB floats in [0, 1], in CHW order.
    trimap : torch.Tensor
        Labels 1 (pet), 2 (background), 3 (unlabelled boundary).
    task : str
        denoise, sr or segment.
    generator : torch.Generator
        Supplies both crop coordinates and Gaussian noise.
    """
    if task not in ("denoise", "sr", "segment") or size < 8 or size % 2:
        raise ValueError("Use denoise, sr or segment with an even patch size >= 8")
    height, width = image.shape[-2:]
    pad = (0, max(0, size - width), 0, max(0, size - height))
    image = F.pad(image, pad, mode="replicate")
    trimap = F.pad(trimap, pad, mode="constant", value=3)
    y = int(torch.randint(image.shape[-2] - size + 1, (), generator=generator))
    x = int(torch.randint(image.shape[-1] - size + 1, (), generator=generator))
    clean = image[:, y : y + size, x : x + size]
    mask = trimap[:, y : y + size, x : x + size]
    valid = (mask != 3).float()
    if task == "denoise":
        noise = torch.randn(clean.shape, generator=generator) * sigma
        return clean + noise, noise, torch.ones_like(valid)
    if task == "sr":
        low = F.interpolate(clean[None], scale_factor=0.5, mode="area")[0]
        return low, clean, torch.ones_like(valid)
    return clean, (mask == 1).float(), valid


class PetPatches(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        indices: list[int],
        task: str,
        samples: int = 128,
        seed: int = 42,
        fixed: bool = False,
    ) -> None:
        self.dataset, self.indices, self.task = dataset, indices, task
        self.samples, self.seed, self.fixed = samples, seed, fixed
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = (
            torch.Generator().manual_seed(self.seed + index)
            if self.fixed
            else self.generator
        )
        chosen = (
            self.indices[index % len(self.indices)]
            if self.fixed
            else self.indices[
                int(torch.randint(len(self.indices), (), generator=generator))
            ]
        )
        image, mask = self.dataset[chosen]
        return make_patch(
            TF.to_tensor(image.convert("RGB")),
            TF.pil_to_tensor(mask),
            self.task,
            generator,
        )


def segmentation_loss(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    bce = nn.BCEWithLogitsLoss(reduction="none")(logits, target)
    bce = (bce * valid).sum() / valid.sum().clamp_min(1)
    probability = logits.sigmoid() * valid
    target = target * valid
    axes = (1, 2, 3)
    dice = (2 * (probability * target).sum(axes) + 1e-6) / (
        probability.sum(axes) + target.sum(axes) + 1e-6
    )
    return bce + (1 - dice).mean()


def image_scores(
    reference: torch.Tensor, prediction: torch.Tensor
) -> tuple[float, float]:
    if reference.shape != prediction.shape or min(reference.shape[-2:]) <= 4:
        raise ValueError(
            "Matching CHW images larger than the four-pixel scoring crop are required"
        )
    difference = (
        reference.double()[..., 2:-2, 2:-2]
        - prediction.clamp(0, 1).double()[..., 2:-2, 2:-2]
    )
    mse = difference.square().mean().item()
    psnr = (
        float("inf")
        if mse == 0
        else -10 * torch.log10(torch.tensor(mse, dtype=torch.float64)).item()
    )
    return mse, psnr


def mask_scores(
    probability: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> tuple[float, float]:
    prediction = (probability >= 0.5) & valid.bool()
    target = target.bool() & valid.bool()
    intersection = (prediction & target).sum().item()
    total = prediction.sum().item() + target.sum().item()
    union = (prediction | target).sum().item()
    return (
        2 * intersection / total if total else 1.0,
        intersection / union if union else 1.0,
    )


def tiled_predict(
    model: nn.Module,
    image: torch.Tensor,
    tile: int = 128,
    overlap: int = 32,
    scale: int = 1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Blend overlapping predictions whilst keeping the full frame on the CPU.

    Parameters
    ----------
    model : nn.Module
        Maps a batch of input tiles to outputs at the requested scale.
    image : torch.Tensor
        A CPU CHW image. Only one tile at a time is sent to the device.
    tile : int
        Input tile width and height.
    overlap : int
        Shared input pixels between adjacent tiles.
    scale : int
        Output pixels per input pixel (1 or 2).
    device : torch.device
        The device holding the model.
    """
    if not 0 <= overlap < tile or scale not in (1, 2) or image.device.type != "cpu":
        raise ValueError("Require a CPU image, 0 <= overlap < tile and scale 1 or 2")
    height, width = image.shape[-2:]
    padded = F.pad(
        image, (0, max(0, tile - width), 0, max(0, tile - height)), mode="replicate"
    )
    ph, pw = padded.shape[-2:]

    def starts(length: int) -> list[int]:
        return sorted(
            set(range(0, length - tile + 1, tile - overlap)) | {length - tile}
        )

    ramp = torch.hann_window(tile * scale, periodic=False).clamp_min(0.05)
    weight = ramp[:, None] * ramp[None, :]
    normaliser = torch.zeros(1, ph * scale, pw * scale)
    output = None
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for y in starts(ph):
                for x in starts(pw):
                    patch = padded[:, y : y + tile, x : x + tile][None].to(device)
                    prediction = model(patch)[0].cpu()
                    if prediction.shape[-2:] != (tile * scale, tile * scale):
                        raise ValueError("Model output size does not match scale")
                    if output is None:
                        output = torch.zeros(
                            prediction.shape[0], ph * scale, pw * scale
                        )
                    ys = slice(y * scale, (y + tile) * scale)
                    xs = slice(x * scale, (x + tile) * scale)
                    output[:, ys, xs] += prediction * weight
                    normaliser[:, ys, xs] += weight
    finally:
        model.train(was_training)
    return (output / normaliser)[:, : height * scale, : width * scale]

import torch


class DigitCNN(torch.nn.Module):
    """Classify a batch of single-channel mel spectrograms."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.AdaptiveAvgPool2d((4, 5)),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 4 * 5, 10),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)

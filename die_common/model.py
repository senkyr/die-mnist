"""SimpleCNN – jediná definice sítě pro DigitLearner, DigitTeaser, DigitTester i dispatch.

Názvy a pořadí vrstev (features.0 … features.10, classifier.1, classifier.4)
jsou klíče ve state dictu uložených digit_cnn.pt. Jakákoli změna struktury
znamená, že dřív natrénované modely nepůjdou načíst – test
tests/test_model.py proto hlídá seznam klíčů.
"""

from __future__ import annotations

import torch
from torch import nn

from die_common.data import NUM_CLASSES


class SimpleCNN(nn.Module):
    """Malá konvoluční síť vhodná pro 32x32 číslice."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.2) -> None:
        super().__init__()
        self.dropout = dropout
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_model(model_path, num_classes: int = NUM_CLASSES, dropout: float = 0.2,
               device: torch.device | str = "cpu") -> SimpleCNN:
    """Vytvoří SimpleCNN, načte váhy (jen tenzory, weights_only) a přepne do eval()."""
    model = SimpleCNN(num_classes=num_classes, dropout=dropout)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

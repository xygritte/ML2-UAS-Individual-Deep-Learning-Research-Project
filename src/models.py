from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class CNNBaseline(nn.Module):
    """CNN sederhana sebagai baseline."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_resnet18(num_classes: int = 10, pretrained: bool = True):
    """Model utama transfer learning berbasis ResNet-18."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

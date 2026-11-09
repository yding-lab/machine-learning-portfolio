"""Models: small CNN from scratch, and ResNet-18 (ImageNet weights) for probing / fine-tuning."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from .config import PRETRAINED


class SmallCNN(nn.Module):
    """Four conv blocks + global pooling; about 0.4M parameters."""

    def __init__(self, n_classes: int):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 192))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(192, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def resnet18(n_classes: int | None = None, pretrained: bool = PRETRAINED) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=weights)
    if n_classes is None:
        m.fc = nn.Identity()  # 512-d feature extractor
    else:
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def freeze_except_last_block(m: nn.Module) -> nn.Module:
    for name, p in m.named_parameters():
        p.requires_grad = name.startswith(("layer4", "fc"))
    return m


@torch.no_grad()
def predict_logits(model: nn.Module, loader) -> torch.Tensor:
    model.eval()
    return torch.cat([model(x) for x, _ in loader])

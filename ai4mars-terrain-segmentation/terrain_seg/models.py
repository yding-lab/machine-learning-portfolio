from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from .config import PRETRAINED


def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """A small U-Net trained from scratch (4 levels, 16-128 channels)."""

    def __init__(self, n_classes: int, base: int = 16):
        super().__init__()
        ch = [base * 2 ** i for i in range(4)]
        self.down = nn.ModuleList([_block(3 if i == 0 else ch[i - 1], c) for i, c in enumerate(ch)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i], ch[i - 1], 2, stride=2) for i in range(3, 0, -1)])
        self.dec = nn.ModuleList([_block(ch[i - 1] * 2, ch[i - 1]) for i in range(3, 0, -1)])
        self.head = nn.Conv2d(ch[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.down):
            x = d(x)
            if i < 3:
                skips.append(x)
                x = nn.functional.max_pool2d(x, 2)
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        return self.head(x)


class LRASPP(nn.Module):
    """torchvision LR-ASPP with a MobileNetV3-Large backbone (ImageNet weights when PRETRAINED)."""

    def __init__(self, n_classes: int, pretrained: bool = PRETRAINED):
        super().__init__()
        w = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = lraspp_mobilenet_v3_large(weights=None, weights_backbone=w, num_classes=n_classes)

    def forward(self, x):
        return self.net(x)["out"]


MODELS = {"unet": UNet, "lraspp_mobilenet_v3": LRASPP}

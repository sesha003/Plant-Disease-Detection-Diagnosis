"""Model zoo: our own small CNN + pretrained torchvision models for transfer learning.

| name                | what it is                                  | params (38 classes) |
|---------------------|---------------------------------------------|---------------------|
| simple_cnn          | 4 conv blocks we build ourselves            | ~1.2 M              |
| resnet18 / resnet50 | residual networks pretrained on ImageNet    | 11.2 M / 23.6 M     |
| efficientnet_b0     | compound-scaled MBConv network              | 4.0 M               |
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models as tvm

SUPPORTED_MODELS = (
    "simple_cnn",
    "resnet18",
    "resnet50",
    "efficientnet_b0",
)


# --------------------------------------------------------------------------- #
# Our own CNN
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Sequential):
    """[Conv3x3 -> BatchNorm -> ReLU] x 2 -> MaxPool2x2.

    - Conv 3x3, padding 1, stride 1 keeps H and W the same.
    - Two stacked 3x3 convs see a 5x5 area with fewer weights than one 5x5 conv.
    - bias=False because BatchNorm adds its own learnable shift (beta).
    - MaxPool 2x2 halves H and W.
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


class SimpleCNN(nn.Module):
    """Input (N,3,224,224)
    block1 -> (N, 32,112,112)
    block2 -> (N, 64, 56, 56)
    block3 -> (N,128, 28, 28)
    block4 -> (N,256, 14, 14)
    global average pool -> (N,256) -> dropout -> linear -> (N, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        widths: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.3,
    ):
        super().__init__()
        blocks, c = [], in_channels
        for w in widths:
            blocks.append(ConvBlock(c, w))
            c = w
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, num_classes))
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.3,
    widths: tuple[int, ...] | list[int] = (32, 64, 128, 256),
) -> nn.Module:
    """Create a model and replace its last layer with a `num_classes` output."""
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model {name!r}. Choose from {SUPPORTED_MODELS}")
    weights = "DEFAULT" if pretrained else None

    if name == "simple_cnn":
        return SimpleCNN(num_classes, widths=tuple(widths), dropout=dropout)

    if name in ("resnet18", "resnet50"):
        m = getattr(tvm, name)(weights=weights)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
        return m

    if name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m

    raise ValueError(f"Unknown model {name!r}. Choose from {SUPPORTED_MODELS}")


def head_module(model: nn.Module, name: str) -> nn.Module:
    """The newly created, task-specific part of the network."""
    return model.fc if name.startswith("resnet") else model.classifier


def set_backbone_trainable(model: nn.Module, name: str, trainable: bool) -> None:
    """Freeze/unfreeze everything except the head (transfer-learning warm-up)."""
    for p in model.parameters():
        p.requires_grad = trainable
    for p in head_module(model, name).parameters():
        p.requires_grad = True


def gradcam_layer(model: nn.Module, name: str) -> nn.Module:
    """Last convolutional stage: where Grad-CAM reads activations."""
    if name.startswith("resnet"):
        return model.layer4
    return model.features[-1]


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)

"""Perakitan model: backbone MobileNetV3-Small + modul attention + classifier."""

import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from src.attention import SEBlock, CBAM, CoordAttention

FEATURE_CHANNELS = 576


def _attention_module(variant):
    if variant == "baseline":
        return nn.Identity()
    if variant == "se":
        return SEBlock(FEATURE_CHANNELS)
    if variant == "cbam":
        return CBAM(FEATURE_CHANNELS)
    if variant == "ca":
        return CoordAttention(FEATURE_CHANNELS)
    raise ValueError(f"varian tidak dikenal: {variant}")


def build_model(variant, num_classes=2, pretrained=True):
    """Bangun satu varian model. Kembalikan model dan backbone-nya.

    Backbone dipakai terpisah supaya fase 1 training bisa dibekukan
    tanpa mematikan modul attention dan classifier.
    """
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = mobilenet_v3_small(weights=weights).features
    head = nn.Sequential(
        _attention_module(variant),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),
        nn.Dropout(0.2),
        nn.Linear(FEATURE_CHANNELS, num_classes),
    )
    return nn.Sequential(backbone, head), backbone


def set_backbone_frozen(backbone, frozen):
    for p in backbone.parameters():
        p.requires_grad = not frozen

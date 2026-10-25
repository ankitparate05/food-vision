"""Model factory. Every architecture exposes a backbone (``features``) and a head
(``classifier``) so the two-phase training loop works the same for all of them."""
import torch.nn as nn
from torchvision import models

SUPPORTED_MODELS = ("efficientnet_b0", "convnext_tiny")


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=weights)
        in_features = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
        return m

    if name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.convnext_tiny(weights=weights)
        # classifier = [LayerNorm2d, Flatten, Linear]; replace only the final layer
        in_features = m.classifier[2].in_features
        m.classifier[2] = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
        return m

    raise ValueError(f"Unknown model {name!r}. Supported: {', '.join(SUPPORTED_MODELS)}")


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in model.features.parameters():
        p.requires_grad = trainable


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

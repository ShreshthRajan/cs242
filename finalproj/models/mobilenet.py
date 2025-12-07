"""
MobileNetV2 for CIFAR-10
Uses pretrained ImageNet weights and adapts for 10 classes
"""

import torch
import torch.nn as nn
import torchvision.models as models


def MobileNetV2_CIFAR10(pretrained=True):
    """
    Load MobileNetV2 adapted for CIFAR-10

    Args:
        pretrained: Use ImageNet pretrained weights

    Returns:
        MobileNetV2 model with 10 output classes
    """
    # Load pretrained MobileNetV2
    if pretrained:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        model = models.mobilenet_v2(weights=None)

    # Modify classifier for CIFAR-10 (10 classes instead of 1000)
    model.classifier[1] = nn.Linear(
        in_features=1280,
        out_features=10
    )

    return model


if __name__ == '__main__':
    # Test model creation
    model = MobileNetV2_CIFAR10(pretrained=True)

    x = torch.randn(2, 3, 32, 32)
    y = model(x)

    total_params = sum(p.numel() for p in model.parameters())

    print(f"MobileNetV2 for CIFAR-10:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    assert y.shape == (2, 10), f"Output shape mismatch: {y.shape}"
    print("\n✅ MobileNetV2 architecture verified!")

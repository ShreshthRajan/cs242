"""
ResNet-20 for CIFAR-10
Based on: "Deep Residual Learning for Image Recognition" (He et al., 2015)
Architecture: 3 stages with 3 blocks each, total 20 layers (3×3×2 + 2 = 20)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-20
    Two 3x3 conv layers with skip connection
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # First conv: 3x3, stride may downsample
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # Second conv: 3x3, stride=1
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            # Projection shortcut for dimension matching
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Add skip connection
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet for CIFAR-10 (32x32 images, 10 classes)
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        # Initial conv: 3x3, 16 filters, stride=1 (no downsampling)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Three stages with increasing channels: 16, 32, 64
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        # Final classifier
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """Create a stage with num_blocks residual blocks"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial conv
        out = F.relu(self.bn1(self.conv1(x)))

        # Three stages
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # Global average pooling + classifier
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def ResNet20(num_classes=10):
    """
    ResNet-20 for CIFAR-10
    Architecture: [3, 3, 3] blocks per stage
    Total layers: 1 + 2×3×3 + 1 = 20
    Parameters: ~0.27M
    """
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)


def ResNet32(num_classes=10):
    """ResNet-32 for CIFAR-10 (if needed for comparison)"""
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)


def ResNet44(num_classes=10):
    """ResNet-44 for CIFAR-10 (if needed for comparison)"""
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes)


if __name__ == '__main__':
    # Test model creation
    net = ResNet20()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)

    # Count parameters
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print(f"ResNet-20 Architecture:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Expected: ~270,000 parameters")

    assert y.shape == (2, 10), f"Output shape mismatch: {y.shape}"
    print("\n✅ Model architecture verified!")

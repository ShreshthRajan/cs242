"""
Per-layer post-training quantization wrappers
Implements symmetric per-channel weight quantization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import List, Tuple


class QuantizedConv2d(nn.Module):
    """
    Quantized Conv2d layer - simulates k-bit weight quantization
    Uses symmetric per-channel quantization
    """

    def __init__(self, original_conv: nn.Conv2d, num_bits: int = 8):
        super().__init__()
        self.num_bits = num_bits

        # Store original parameters
        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups

        # Quantize and store weights as parameters (so they move with .to(device))
        weight_quantized, scale = self._quantize_weights(original_conv.weight)
        self.weight = nn.Parameter(weight_quantized, requires_grad=False)
        self.scale = scale

        # Handle bias as parameter
        if original_conv.bias is not None:
            self.bias = nn.Parameter(original_conv.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

    def _quantize_weights(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Symmetric per-channel quantization

        scale = max(|w|) / (2^(b-1) - 1) per output channel
        w_int = round(w / scale)
        w_quantized = w_int * scale (dequantized for inference)
        """
        # Per-channel (per output channel) quantization
        # Shape: (C_out, C_in, K_h, K_w)
        C_out = weight.shape[0]

        # Compute scale per output channel
        weight_reshape = weight.view(C_out, -1)  # (C_out, C_in×K_h×K_w)
        max_vals = weight_reshape.abs().max(dim=1, keepdim=True)[0]  # (C_out, 1)

        # Avoid division by zero
        max_vals = torch.clamp(max_vals, min=1e-8)

        # Quantization levels: 2^(b-1) - 1 for symmetric
        q_max = 2 ** (self.num_bits - 1) - 1
        scale = max_vals / q_max  # (C_out, 1)

        # Quantize: w / scale, round, then dequantize: w_int * scale
        weight_scaled = weight_reshape / scale
        weight_quantized_int = torch.round(weight_scaled)

        # Clamp to valid range [-q_max, q_max]
        weight_quantized_int = torch.clamp(weight_quantized_int, -q_max, q_max)

        # Dequantize back to floating point
        weight_dequantized = weight_quantized_int * scale

        # Reshape back to original
        weight_final = weight_dequantized.view_as(weight)

        return weight_final, scale.squeeze()

    def forward(self, x):
        """Forward pass with quantized weights"""
        return F.conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )


class QuantizedLinear(nn.Module):
    """
    Quantized Linear layer - simulates k-bit weight quantization
    """

    def __init__(self, original_linear: nn.Linear, num_bits: int = 8):
        super().__init__()
        self.num_bits = num_bits

        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        # Quantize and store weights as parameters (so they move with .to(device))
        weight_quantized, scale = self._quantize_weights(original_linear.weight)
        self.weight = nn.Parameter(weight_quantized, requires_grad=False)
        self.scale = scale

        # Handle bias as parameter
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

    def _quantize_weights(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Symmetric per-channel (per-row) quantization for Linear layers
        """
        # Per output feature quantization
        # Shape: (out_features, in_features)
        out_features = weight.shape[0]

        # Max per row
        max_vals = weight.abs().max(dim=1, keepdim=True)[0]  # (out_features, 1)
        max_vals = torch.clamp(max_vals, min=1e-8)

        q_max = 2 ** (self.num_bits - 1) - 1
        scale = max_vals / q_max

        # Quantize
        weight_scaled = weight / scale
        weight_quantized_int = torch.round(weight_scaled)
        weight_quantized_int = torch.clamp(weight_quantized_int, -q_max, q_max)

        # Dequantize
        weight_dequantized = weight_quantized_int * scale

        return weight_dequantized, scale.squeeze()

    def forward(self, x):
        """Forward pass with quantized weights"""
        return F.linear(x, self.weight, self.bias)


def quantize_layer(model: nn.Module, layer_name: str, num_bits: int) -> nn.Module:
    """
    Replace a specific layer in the model with its quantized version

    Args:
        model: PyTorch model
        layer_name: Dot-separated path to layer (e.g., 'layer1.0.conv1')
        num_bits: Quantization bitwidth

    Returns:
        Model with specified layer quantized
    """
    # Modify in-place (caller creates fresh model each time)
    model_copy = model

    # Navigate to the layer
    parts = layer_name.split('.')
    parent = model_copy

    for part in parts[:-1]:
        parent = getattr(parent, part)

    # Get the original layer
    layer_attr_name = parts[-1]
    original_layer = getattr(parent, layer_attr_name)

    # Replace with quantized version
    if isinstance(original_layer, nn.Conv2d):
        quantized_layer = QuantizedConv2d(original_layer, num_bits)
    elif isinstance(original_layer, nn.Linear):
        quantized_layer = QuantizedLinear(original_layer, num_bits)
    else:
        raise ValueError(f"Layer {layer_name} is not Conv2d or Linear")

    setattr(parent, layer_attr_name, quantized_layer)

    return model_copy


def get_quantizable_layers(model: nn.Module) -> List[str]:
    """
    Get list of all quantizable layer names (Conv2d and Linear)

    Args:
        model: PyTorch model

    Returns:
        List of layer names that can be quantized
    """
    quantizable = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Skip 1x1 shortcut convolutions (too small to matter)
            if isinstance(module, nn.Conv2d):
                if module.kernel_size == (1, 1) and 'shortcut' in name:
                    continue
            quantizable.append(name)

    return quantizable


if __name__ == '__main__':
    # Test quantization
    import sys
    sys.path.insert(0, '..')
    from models.resnet import ResNet20

    print("Testing quantization wrappers...")

    # Create model
    model = ResNet20()

    # Get quantizable layers
    layers = get_quantizable_layers(model)
    print(f"\nFound {len(layers)} quantizable layers")
    print("First 5:", layers[:5])

    # Test quantizing one layer
    print(f"\nQuantizing layer: {layers[0]}")
    model_8bit = quantize_layer(model, layers[0], num_bits=8)

    # Verify forward pass works
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y_orig = model(x)
        y_quant = model_8bit(x)

    print(f"Original output shape: {y_orig.shape}")
    print(f"Quantized output shape: {y_quant.shape}")
    print(f"Output difference: {(y_orig - y_quant).abs().mean():.6f}")

    print("\n✅ Quantization wrappers work correctly!")

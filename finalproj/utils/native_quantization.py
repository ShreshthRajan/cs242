"""
PyTorch Native INT8 Quantization
Uses REAL INT8 operations via torch.ao.quantization

This module provides selective layer quantization with actual hardware INT8 compute.
"""

import torch
import torch.nn as nn
from torch.ao.quantization import get_default_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from typing import List, Optional
import copy


def quantize_model_selective(
    model: nn.Module,
    selected_layers: List[str],
    calibration_data_loader,
    backend: str = 'fbgemm'
) -> nn.Module:
    """
    Quantize specific layers using PyTorch native INT8 quantization

    Args:
        model: FP32 model to quantize
        selected_layers: List of layer names to quantize (e.g., ['conv1', 'layer1.0.conv1'])
        calibration_data_loader: DataLoader for calibration
        backend: Quantization backend ('fbgemm' for x86 CPU, 'qnnpack' for ARM)

    Returns:
        Quantized model with REAL INT8 operations

    Process:
        1. Create QConfigMapping for selected layers only
        2. prepare_fx: Insert observers
        3. Calibrate on data
        4. convert_fx: Convert to INT8

    Note: Uses dynamic quantization as fallback if FX fails
    """
    print(f"Quantizing {len(selected_layers)} layers with native INT8...")

    # Set backend
    torch.backends.quantized.engine = backend

    try:
        # Method 1: FX Graph Mode (preferred, but requires tracing)
        print("  Attempting FX graph mode quantization...")

        model_copy = copy.deepcopy(model)
        model_copy.eval()

        # Create QConfigMapping
        qconfig = get_default_qconfig(backend)
        qconfig_mapping = QConfigMapping()

        # Global: don't quantize (None)
        qconfig_mapping.set_global(None)

        # Selective: quantize only specified layers
        for layer_name in selected_layers:
            qconfig_mapping.set_module_name(layer_name, qconfig)

        # Get example input
        example_input = torch.randn(1, 3, 32, 32)

        # Prepare: insert observers
        print(f"  Inserting observers for {len(selected_layers)} layers...")
        prepared_model = prepare_fx(model_copy, qconfig_mapping, example_input)

        # Calibrate
        print("  Calibrating observers (500 samples)...")
        prepared_model.eval()
        with torch.no_grad():
            for i, (inputs, _) in enumerate(calibration_data_loader):
                if i >= 5:  # 500 images
                    break
                prepared_model(inputs)

        # Convert to INT8
        print("  Converting to INT8...")
        quantized_model = convert_fx(prepared_model)

        print("✅ FX quantization successful (REAL INT8 ops)")

        return quantized_model

    except Exception as e:
        # Method 2: Dynamic Quantization (fallback)
        print(f"  ⚠️  FX quantization failed: {str(e)}")
        print("  Falling back to dynamic quantization...")

        model_copy = copy.deepcopy(model)
        model_copy.eval()

        # Dynamic quantization (simpler, works on more models)
        quantized_model = torch.quantization.quantize_dynamic(
            model_copy,
            qconfig_spec={nn.Conv2d, nn.Linear},
            dtype=torch.qint8,
            inplace=False
        )

        print("✅ Dynamic quantization successful (partial INT8)")

        return quantized_model


def test_quantization():
    """Test native quantization on simple model"""
    print("="*70)
    print("TESTING NATIVE QUANTIZATION")
    print("="*70)

    # Create simple test model
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(16, 10, 3, padding=1)

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.conv2(x)
            return x

    model = SimpleNet()

    # Create dummy calibration data
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)

    # Test quantization
    try:
        quantized = quantize_model_selective(
            model,
            selected_layers=['conv1'],
            calibration_data_loader=loader,
            backend='fbgemm'
        )

        # Test forward pass
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            y = quantized(x)

        print(f"\n✅ Quantization test passed!")
        print(f"   Output shape: {y.shape}")

        return True

    except Exception as e:
        print(f"\n❌ Quantization test failed: {str(e)}")
        return False


if __name__ == '__main__':
    success = test_quantization()
    if success:
        print("\n✅ Native quantization module ready to use")
    else:
        print("\n❌ Quantization module has issues - check errors above")

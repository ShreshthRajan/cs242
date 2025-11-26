"""
Arithmetic Intensity Calculator for Neural Networks
Computes FLOPs and Bytes for each layer to calculate AI = FLOPs/Bytes
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class AICalculator:
    """
    Calculates arithmetic intensity for each layer in a model
    AI = Total FLOPs / Total Bytes (from roofline model)
    """

    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...],
                 batch_size: int = 1):
        """
        Args:
            model: PyTorch model to analyze
            input_shape: Input tensor shape (C, H, W) for images
            batch_size: Batch size for inference (default=1 for single-image AI)
        """
        self.model = model
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.layer_info = {}
        self.total_flops = 0
        self.total_bytes = 0

        # Profile the model
        self._profile_model()

    def _profile_model(self):
        """Run forward pass with hooks to capture layer information"""
        # Register hooks to capture activations
        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                # Store layer metadata
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    self.layer_info[name] = {
                        'module': module,
                        'input_shape': input[0].shape,
                        'output_shape': output.shape,
                        'type': 'conv' if isinstance(module, nn.Conv2d) else 'linear'
                    }
            return hook

        # Register hooks for all Conv2d and Linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(make_hook(name)))

        # Run dummy forward pass
        dummy_input = torch.randn(self.batch_size, *self.input_shape)
        with torch.no_grad():
            self.model.eval()
            _ = self.model(dummy_input)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Compute FLOPs and Bytes for each layer
        for name, info in self.layer_info.items():
            module = info['module']

            if info['type'] == 'conv':
                flops, bytes_val = self._compute_conv_metrics(module, info)
            else:  # linear
                flops, bytes_val = self._compute_linear_metrics(module, info)

            info['flops'] = flops
            info['bytes'] = bytes_val
            info['ai'] = flops / bytes_val if bytes_val > 0 else 0

            self.total_flops += flops
            self.total_bytes += bytes_val

    def _compute_conv_metrics(self, module: nn.Conv2d, info: Dict) -> Tuple[int, int]:
        """
        Compute FLOPs and Bytes for Conv2d layer

        FLOPs = 2 × C_out × H_out × W_out × C_in × K_h × K_w
        Bytes = Weight_bytes + Activation_bytes
        """
        # Extract dimensions
        N, C_in, H_in, W_in = info['input_shape']
        N, C_out, H_out, W_out = info['output_shape']
        K_h, K_w = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)

        # FLOPs: 2 multiply-accumulates per output element
        # Each output pixel requires C_in × K_h × K_w MACs
        flops = 2 * C_out * H_out * W_out * C_in * K_h * K_w

        # Bytes: Weights + Input activations + Output activations
        num_params = module.weight.numel()
        if module.bias is not None:
            num_params += module.bias.numel()

        weight_bytes = num_params * 4  # FP32 = 4 bytes per param
        input_bytes = N * C_in * H_in * W_in * 4
        output_bytes = N * C_out * H_out * W_out * 4

        total_bytes = weight_bytes + input_bytes + output_bytes

        return flops, total_bytes

    def _compute_linear_metrics(self, module: nn.Linear, info: Dict) -> Tuple[int, int]:
        """
        Compute FLOPs and Bytes for Linear layer

        FLOPs = 2 × in_features × out_features
        Bytes = Weight_bytes + Activation_bytes
        """
        in_features = module.in_features
        out_features = module.out_features
        N = info['input_shape'][0]

        # FLOPs: matrix multiplication
        flops = 2 * in_features * out_features

        # Bytes: Weights + Inputs + Outputs
        num_params = module.weight.numel()
        if module.bias is not None:
            num_params += module.bias.numel()

        weight_bytes = num_params * 4  # FP32

        # Activations (assuming flattened input for FC)
        input_bytes = N * in_features * 4
        output_bytes = N * out_features * 4

        total_bytes = weight_bytes + input_bytes + output_bytes

        return flops, total_bytes

    def get_model_ai(self) -> float:
        """Get overall model arithmetic intensity"""
        return self.total_flops / self.total_bytes if self.total_bytes > 0 else 0

    def get_layer_ai(self, layer_name: str) -> float:
        """Get arithmetic intensity for specific layer"""
        if layer_name in self.layer_info:
            return self.layer_info[layer_name]['ai']
        return 0

    def compute_ai_with_quantized_layer(self, layer_name: str,
                                        num_bits: int) -> float:
        """
        Compute model AI if a specific layer is quantized to num_bits

        Args:
            layer_name: Name of layer to quantize
            num_bits: Bitwidth (e.g., 8 or 4)

        Returns:
            New model-level AI with that layer quantized
        """
        if layer_name not in self.layer_info:
            return self.get_model_ai()

        layer = self.layer_info[layer_name]
        module = layer['module']

        # Original bytes for this layer
        original_bytes = layer['bytes']

        # Compute new bytes with quantized weights
        num_params = module.weight.numel()
        if module.bias is not None:
            num_params += module.bias.numel()

        # Quantized weight bytes (FP32 → k-bit)
        quantized_weight_bytes = num_params * (num_bits / 8)

        # Activation bytes stay the same (only quantizing weights)
        if layer['type'] == 'conv':
            N, C_in, H_in, W_in = layer['input_shape']
            N, C_out, H_out, W_out = layer['output_shape']
            input_bytes = N * C_in * H_in * W_in * 4
            output_bytes = N * C_out * H_out * W_out * 4
        else:  # linear
            # Linear layer has (N, features) shape
            input_shape = layer['input_shape']
            output_shape = layer['output_shape']
            input_bytes = input_shape[0] * module.in_features * 4
            output_bytes = output_shape[0] * module.out_features * 4

        new_layer_bytes = quantized_weight_bytes + input_bytes + output_bytes

        # New total bytes
        new_total_bytes = self.total_bytes - original_bytes + new_layer_bytes

        # New model AI (FLOPs unchanged)
        new_ai = self.total_flops / new_total_bytes

        return new_ai

    def get_summary(self) -> Dict:
        """Get summary statistics"""
        return {
            'total_flops': self.total_flops,
            'total_bytes': self.total_bytes,
            'model_ai': self.get_model_ai(),
            'num_layers': len(self.layer_info),
            'layer_details': {
                name: {
                    'type': info['type'],
                    'flops': info['flops'],
                    'bytes': info['bytes'],
                    'ai': info['ai'],
                    'memory_share': info['bytes'] / self.total_bytes
                }
                for name, info in self.layer_info.items()
            }
        }

    def print_summary(self):
        """Print formatted summary"""
        summary = self.get_summary()

        print("="*70)
        print("ARITHMETIC INTENSITY ANALYSIS")
        print("="*70)
        print(f"Model: ResNet-20")
        print(f"Input shape: {self.input_shape}")
        print(f"Batch size: {self.batch_size}")
        print(f"\nModel-level metrics:")
        print(f"  Total FLOPs: {summary['total_flops']:,}")
        print(f"  Total Bytes: {summary['total_bytes']:,}")
        print(f"  Model AI: {summary['model_ai']:.4f} FLOPs/Byte")
        print(f"  Number of quantizable layers: {summary['num_layers']}")

        print(f"\nPer-layer breakdown:")
        print(f"{'Layer':<30} {'Type':<8} {'AI':<10} {'Mem %':<8}")
        print("-"*70)

        for name, details in summary['layer_details'].items():
            print(f"{name:<30} {details['type']:<8} "
                  f"{details['ai']:<10.2f} {details['memory_share']*100:<7.1f}%")


if __name__ == '__main__':
    # Test with ResNet-20
    import sys
    sys.path.insert(0, '..')
    from models.resnet import ResNet20

    model = ResNet20()
    calculator = AICalculator(model, input_shape=(3, 32, 32), batch_size=1)

    calculator.print_summary()

    # Test quantization AI calculation
    print("\n" + "="*70)
    print("EXAMPLE: Quantizing first conv layer")
    print("="*70)

    first_layer = list(calculator.layer_info.keys())[0]
    print(f"\nLayer: {first_layer}")
    print(f"Baseline AI: {calculator.get_model_ai():.4f}")

    for bits in [8, 4]:
        new_ai = calculator.compute_ai_with_quantized_layer(first_layer, bits)
        ai_increase = (new_ai / calculator.get_model_ai() - 1) * 100
        print(f"{bits}-bit quantization: AI = {new_ai:.4f} (+{ai_increase:.1f}%)")

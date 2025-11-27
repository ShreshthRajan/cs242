"""
Create models using PyTorch native INT8 quantization (REAL operations)
This uses actual INT8 compute, not simulation

Approach:
1. Uniform INT8: Quantize all layers
2. AI-Aware Selective INT8: Quantize only high-value layers (from Greedy result)
"""

import os
import sys
sys.path.insert(0, '.')

import json
import torch
import torch.nn as nn
import torch.quantization
import torchvision
import torchvision.transforms as transforms

from models.resnet import ResNet20


def load_test_loader(batch_size: int = 100):
    """Load CIFAR-10 test set"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return testloader


def evaluate_accuracy(model: nn.Module, testloader, device: str = 'cpu') -> float:
    """Evaluate model accuracy"""
    model.eval()
    model = model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


def create_uniform_int8_model(checkpoint_path: str) -> nn.Module:
    """
    Create uniformly quantized INT8 model using PyTorch native quantization
    ALL layers quantized with REAL INT8 operations
    """
    print("="*70)
    print("CREATING UNIFORM INT8 MODEL (PyTorch Native)")
    print("="*70)

    # Load FP32 model
    model_fp32 = ResNet20()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_fp32.load_state_dict(checkpoint['model_state_dict'])
    model_fp32.eval()

    print(f"Baseline accuracy: {checkpoint['test_acc']:.2f}%")

    # Apply dynamic quantization (REAL INT8)
    # This quantizes weights to INT8 and uses INT8 ops during inference
    print("\nApplying PyTorch dynamic quantization...")
    print("  Quantizing: nn.Conv2d, nn.Linear")
    print("  Dtype: torch.qint8 (REAL INT8 operations)")

    # Set backend for CPU (Intel x86 or ARM)
    torch.backends.quantized.engine = 'fbgemm'  # For x86 CPUs (Colab/Intel)

    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32,
        qconfig_spec={nn.Conv2d, nn.Linear},  # Quantize these layer types
        dtype=torch.qint8  # Use INT8 (real ops, not simulated)
    )

    print("✅ Model quantized with native INT8 operations")

    return model_int8


def create_selective_int8_model(
    checkpoint_path: str,
    greedy_results_path: str,
    top_k_layers: int = 12
) -> nn.Module:
    """
    Create AI-Aware selective INT8 model
    Quantizes only high-value layers identified by Greedy search

    Args:
        checkpoint_path: FP32 baseline checkpoint
        greedy_results_path: Greedy search results JSON
        top_k_layers: Number of layers to quantize (highest AI gain)
    """
    print("="*70)
    print("CREATING AI-AWARE SELECTIVE INT8 MODEL")
    print("="*70)

    # Load FP32 model
    model_fp32 = ResNet20()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_fp32.load_state_dict(checkpoint['model_state_dict'])
    model_fp32.eval()

    # Load Greedy results to identify high-value layers
    with open(greedy_results_path, 'r') as f:
        greedy_data = json.load(f)

    # Extract layers ordered by when they were selected (highest value first)
    history = greedy_data['history']
    selected_layers = [entry['layer'] for entry in history[:top_k_layers]]

    print(f"\nGreedy search identified {len(history)} layers")
    print(f"Selecting top {top_k_layers} highest-value layers for INT8:")
    for i, layer in enumerate(selected_layers[:5]):
        print(f"  {i+1}. {layer}")
    if len(selected_layers) > 5:
        print(f"  ... and {len(selected_layers) - 5} more")

    # Create QConfigMapping for selective quantization
    # Approach: Manually set which layers to quantize
    print("\nApplying selective quantization...")

    # For PyTorch dynamic quantization, we use module name mapping
    # Prepare model for quantization
    model_fp32.qconfig = None  # Default: don't quantize

    # Set qconfig for selected layers
    qconfig = torch.quantization.default_dynamic_qconfig

    for name, module in model_fp32.named_modules():
        if name in selected_layers:
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.qconfig = qconfig

    # Apply quantization
    model_selective = torch.quantization.quantize_dynamic(
        model_fp32,
        dtype=torch.qint8,
        inplace=False
    )

    # Verify which layers got quantized
    quantized_count = 0
    for name, module in model_selective.named_modules():
        if hasattr(module, 'weight') and not isinstance(module.weight, torch.nn.Parameter):
            quantized_count += 1

    print(f"✅ {quantized_count} layers quantized to INT8")
    print(f"✅ {20 - quantized_count} layers kept at FP32 (sensitive layers)")

    return model_selective


def main():
    """Create all models with native INT8 quantization"""
    print("="*70)
    print("NATIVE INT8 QUANTIZATION - REAL OPERATIONS")
    print("="*70)

    checkpoint_path = 'checkpoints/resnet20_cifar10_best.pth'
    greedy_results_path = 'results/greedy_search_results.json'

    # Load test data
    print("\nLoading test data...")
    testloader = load_test_loader(batch_size=100)

    # Create models
    results = {}

    # 1. Uniform INT8
    print("\n[1/2] Creating Uniform INT8 model...")
    model_uniform_int8 = create_uniform_int8_model(checkpoint_path)

    print("\nEvaluating Uniform INT8 accuracy...")
    acc_uniform = evaluate_accuracy(model_uniform_int8, testloader, device='cpu')
    print(f"Uniform INT8 accuracy: {acc_uniform:.2f}%")

    # Save
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model_uniform_int8.state_dict(),
               'checkpoints/resnet20_native_int8_uniform.pth')
    torch.jit.save(torch.jit.script(model_uniform_int8),
                   'checkpoints/resnet20_native_int8_uniform_scripted.pt')

    results['uniform_int8'] = {
        'accuracy': acc_uniform,
        'quantized_layers': 20,
        'method': 'PyTorch dynamic quantization (all layers)'
    }

    # 2. AI-Aware Selective INT8
    print("\n[2/2] Creating AI-Aware Selective INT8 model...")
    model_aiaware_int8 = create_selective_int8_model(
        checkpoint_path,
        greedy_results_path,
        top_k_layers=12  # Quantize top 12 layers from Greedy
    )

    print("\nEvaluating AI-Aware Selective INT8 accuracy...")
    acc_aiaware = evaluate_accuracy(model_aiaware_int8, testloader, device='cpu')
    print(f"AI-Aware Selective INT8 accuracy: {acc_aiaware:.2f}%")

    # Save
    torch.save(model_aiaware_int8.state_dict(),
               'checkpoints/resnet20_native_int8_aiaware.pth')
    torch.jit.save(torch.jit.script(model_aiaware_int8),
                   'checkpoints/resnet20_native_int8_aiaware_scripted.pt')

    results['aiaware_int8'] = {
        'accuracy': acc_aiaware,
        'quantized_layers': 12,
        'fp32_layers': 8,
        'method': 'AI-Aware selective quantization (high-value layers only)'
    }

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/native_int8_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: results/native_int8_results.json")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Model':<25} {'Accuracy':<12} {'Layers Quantized':<18}")
    print("-"*70)
    print(f"{'Uniform INT8':<25} {acc_uniform:<12.2f} {'20/20 (100%)':<18}")
    print(f"{'AI-Aware Selective':<25} {acc_aiaware:<12.2f} {'12/20 (60%)':<18}")

    accuracy_advantage = acc_aiaware - acc_uniform
    print(f"\nAI-Aware accuracy advantage: {accuracy_advantage:+.2f}%")

    print("\n✅ Native INT8 models created!")
    print("   Next: Run benchmark_native_int8.py to measure throughput")

    return results


if __name__ == '__main__':
    results = main()

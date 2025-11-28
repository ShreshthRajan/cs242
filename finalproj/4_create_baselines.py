"""
Create uniform quantization baseline models
Quantizes all layers to the same bitwidth for comparison
"""

import os
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from models.resnet import ResNet20
from utils.quantization import quantize_layer, get_quantizable_layers
from utils.ai_calculator import AICalculator


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


def evaluate_accuracy(model: nn.Module, testloader, device: str) -> float:
    """Evaluate model accuracy"""
    model.eval()
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


def create_uniform_quantized_model(
    checkpoint_path: str,
    num_bits: int,
    device: str = 'cuda'
) -> nn.Module:
    """
    Create model with all layers uniformly quantized to num_bits

    Args:
        checkpoint_path: Path to FP32 baseline checkpoint
        num_bits: Bitwidth (4 or 8)
        device: Device to use

    Returns:
        Uniformly quantized model
    """
    print(f"\n{'='*70}")
    print(f"CREATING UNIFORM {num_bits}-BIT MODEL")
    print(f"{'='*70}")

    # Load base model
    model = ResNet20()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Baseline accuracy: {checkpoint['test_acc']:.2f}%")

    # Get all quantizable layers
    layers = get_quantizable_layers(model)
    print(f"Quantizing {len(layers)} layers to {num_bits}-bit...")

    # Quantize all layers
    for i, layer_name in enumerate(layers):
        model = quantize_layer(model, layer_name, num_bits)
        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{len(layers)} layers quantized")

    model = model.to(device)
    print(f"✅ All {len(layers)} layers quantized to {num_bits}-bit")

    return model


def main():
    """Create all baseline models"""
    print("="*70)
    print("CREATING UNIFORM QUANTIZATION BASELINES")
    print("="*70)

    checkpoint_path = 'checkpoints/resnet20_cifar10_best.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Load test data
    print("\nLoading test data...")
    testloader = load_test_loader()

    # Load baseline for AI calculation
    base_model = ResNet20()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    base_model.load_state_dict(checkpoint['model_state_dict'])

    baseline_acc = checkpoint['test_acc']
    print(f"FP32 Baseline: Accuracy={baseline_acc:.2f}%")

    # Calculate baseline AI
    ai_calc = AICalculator(base_model, input_shape=(3, 32, 32), batch_size=1)
    baseline_ai = ai_calc.get_model_ai()
    baseline_bytes = ai_calc.total_bytes
    baseline_flops = ai_calc.total_flops

    print(f"FP32 Baseline: AI={baseline_ai:.4f}, Bytes={baseline_bytes:,}, FLOPs={baseline_flops:,}")

    # Results storage
    results = {
        'fp32': {
            'accuracy': baseline_acc,
            'ai': baseline_ai,
            'bytes': baseline_bytes,
            'flops': baseline_flops,
            'model_size_mb': baseline_bytes / (1024 * 1024)
        }
    }

    # Create uniform baselines
    for num_bits in [8, 4]:
        # Create model
        model = create_uniform_quantized_model(checkpoint_path, num_bits, device)

        # Evaluate accuracy
        print(f"\nEvaluating {num_bits}-bit model accuracy...")
        accuracy = evaluate_accuracy(model, testloader, device)

        # Calculate AI (all layers at num_bits)
        # Compute total bytes with all layers quantized
        total_bytes = baseline_bytes
        for layer_name in get_quantizable_layers(base_model):
            layer_info = ai_calc.layer_info[layer_name]
            num_params = layer_info['module'].weight.numel()
            if layer_info['module'].bias is not None:
                num_params += layer_info['module'].bias.numel()

            # Subtract FP32 bytes, add quantized bytes
            fp32_bytes = num_params * 4
            quant_bytes = num_params * (num_bits / 8)
            total_bytes = total_bytes - fp32_bytes + quant_bytes

        model_ai = baseline_flops / total_bytes
        model_size_mb = total_bytes / (1024 * 1024)

        print(f"\n{num_bits}-bit Model:")
        print(f"  Accuracy: {accuracy:.2f}% (Δ{accuracy - baseline_acc:+.2f}%)")
        print(f"  AI: {model_ai:.4f} (Δ{(model_ai/baseline_ai - 1)*100:+.1f}%)")
        print(f"  Model size: {model_size_mb:.2f} MB (Δ{(model_size_mb/(baseline_bytes/(1024*1024)) - 1)*100:+.1f}%)")

        # Save model
        save_path = f'checkpoints/resnet20_uniform_int{num_bits}.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'accuracy': accuracy,
            'ai': model_ai,
            'num_bits': num_bits,
            'config': f'uniform_{num_bits}bit'
        }, save_path)

        print(f"✅ Saved: {save_path}")

        # Store results
        results[f'int{num_bits}'] = {
            'accuracy': accuracy,
            'ai': model_ai,
            'bytes': total_bytes,
            'flops': baseline_flops,
            'model_size_mb': model_size_mb
        }

    # Load Greedy result for comparison (if available)
    import json
    try:
        print(f"\n{'='*70}")
        print("LOADING GREEDY SEARCH RESULT")
        print(f"{'='*70}")

        with open('results/greedy_search_results.json', 'r') as f:
            greedy_data = json.load(f)

        results['ai_aware'] = {
            'accuracy': greedy_data['final_accuracy'],
            'ai': greedy_data['final_ai'],
            'config': greedy_data['final_config']
        }

        print(f"AI-Aware (Greedy):")
        print(f"  Accuracy: {greedy_data['final_accuracy']:.2f}%")
        print(f"  AI: {greedy_data['final_ai']:.4f}")
        print(f"  Config: {greedy_data['config_summary']}")
    except FileNotFoundError:
        print("\n⚠️  Greedy search results not found (run greedy_search.py first)")
        print("   Skipping AI-Aware comparison for now")

    # Save comparison results
    os.makedirs('results', exist_ok=True)
    with open('results/baseline_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: results/baseline_comparison.json")

    # Summary table
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Accuracy':<12} {'AI':<12} {'Size (MB)':<12}")
    print("-"*70)

    for name, data in results.items():
        if name == 'ai_aware':
            continue
        acc = data['accuracy']
        ai = data['ai']
        size = data.get('model_size_mb', 0)
        print(f"{name.upper():<20} {acc:<12.2f} {ai:<12.4f} {size:<12.2f}")

    if 'ai_aware' in results:
        print(f"{'AI-AWARE (Greedy)':<20} {results['ai_aware']['accuracy']:<12.2f} "
              f"{results['ai_aware']['ai']:<12.4f} {'TBD':<12}")

    print("\n✅ Baseline models created successfully!")


if __name__ == '__main__':
    main()

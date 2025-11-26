"""
Run per-layer quantization experiments
For each layer, quantize individually and measure (AI, Accuracy)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from models.resnet import ResNet20
from utils.ai_calculator import AICalculator
from utils.quantization import quantize_layer, get_quantizable_layers


def get_test_loader(batch_size=100):
    """Load CIFAR-10 test set"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return testloader


def evaluate_model(model, testloader, device):
    """
    Evaluate model accuracy on test set

    Returns:
        test_accuracy (float): Percentage accuracy
    """
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

    accuracy = 100.0 * correct / total
    return accuracy


def run_experiments(checkpoint_path: str, output_path: str):
    """
    Run per-layer quantization experiments

    For each layer:
        - Quantize only that layer to 8-bit and 4-bit
        - Measure accuracy and AI
        - Save results

    Args:
        checkpoint_path: Path to trained baseline checkpoint
        output_path: Path to save results JSON
    """
    print("="*70)
    print("PER-LAYER QUANTIZATION EXPERIMENTS")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Load baseline model
    print("\nLoading baseline model...")
    model = ResNet20(num_classes=10)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    baseline_acc = checkpoint['test_acc']
    print(f"Baseline accuracy: {baseline_acc:.2f}%")

    # Load test data
    print("\nLoading CIFAR-10 test set...")
    testloader = get_test_loader(batch_size=100)

    # Verify baseline accuracy
    print("\nVerifying baseline accuracy...")
    verified_acc = evaluate_model(model, testloader, device)
    print(f"Verified baseline accuracy: {verified_acc:.2f}%")

    # Compute baseline AI
    print("\nComputing arithmetic intensity...")
    ai_calculator = AICalculator(model.cpu(), input_shape=(3, 32, 32), batch_size=1)
    baseline_ai = ai_calculator.get_model_ai()

    print(f"\nBaseline metrics:")
    print(f"  AI: {baseline_ai:.4f} FLOPs/Byte")
    print(f"  Total FLOPs: {ai_calculator.total_flops:,}")
    print(f"  Total Bytes: {ai_calculator.total_bytes:,}")

    # Get quantizable layers
    quantizable_layers = get_quantizable_layers(model)
    print(f"\nFound {len(quantizable_layers)} quantizable layers:")
    for i, layer in enumerate(quantizable_layers[:5]):
        print(f"  {i+1}. {layer}")
    if len(quantizable_layers) > 5:
        print(f"  ... and {len(quantizable_layers) - 5} more")

    # Results storage
    results = {
        'baseline': {
            'ai': baseline_ai,
            'accuracy': verified_acc,
            'flops': ai_calculator.total_flops,
            'bytes': ai_calculator.total_bytes
        },
        'per_layer': []
    }

    # Run experiments for each layer
    print(f"\n{'='*70}")
    print(f"Running {len(quantizable_layers)} layers × 2 bitwidths = {len(quantizable_layers)*2} experiments")
    print(f"{'='*70}\n")

    model = model.to(device)  # Move back to GPU for evaluation

    for layer_idx, layer_name in enumerate(tqdm(quantizable_layers, desc="Layers")):
        layer_results = {
            'layer_name': layer_name,
            'layer_index': layer_idx,
            'experiments': []
        }

        for num_bits in [8, 4]:
            # Quantize this layer only (on CPU to avoid memory issues)
            model_cpu = model.cpu()
            quantized_model = quantize_layer(model_cpu, layer_name, num_bits)
            quantized_model = quantized_model.to(device)
            model = model.to(device)  # Move original back to GPU

            # Evaluate accuracy
            quant_acc = evaluate_model(quantized_model, testloader, device)

            # Compute new AI
            new_ai = ai_calculator.compute_ai_with_quantized_layer(layer_name, num_bits)

            # Accuracy drop
            acc_drop = verified_acc - quant_acc

            # AI increase
            ai_increase_pct = (new_ai / baseline_ai - 1) * 100

            layer_results['experiments'].append({
                'num_bits': num_bits,
                'accuracy': quant_acc,
                'accuracy_drop': acc_drop,
                'ai': new_ai,
                'ai_increase_pct': ai_increase_pct
            })

            tqdm.write(
                f"  {layer_name} ({num_bits}-bit): "
                f"Acc={quant_acc:.2f}% (Δ{acc_drop:+.2f}%), "
                f"AI={new_ai:.4f} (Δ{ai_increase_pct:+.1f}%)"
            )

            # Cleanup
            del quantized_model
            torch.cuda.empty_cache() if device == 'cuda' else None

        results['per_layer'].append(layer_results)

    # Save results
    print(f"\n{'='*70}")
    print("Saving results...")
    os.makedirs('results', exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Results saved to: {output_path}")

    # Print summary statistics
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")

    all_8bit_accs = [exp['experiments'][0]['accuracy'] for exp in results['per_layer']]
    all_4bit_accs = [exp['experiments'][1]['accuracy'] for exp in results['per_layer']]

    print(f"Baseline: AI={baseline_ai:.4f}, Acc={verified_acc:.2f}%")
    print(f"\n8-bit quantization (per layer):")
    print(f"  Best accuracy: {max(all_8bit_accs):.2f}%")
    print(f"  Worst accuracy: {min(all_8bit_accs):.2f}%")
    print(f"  Mean accuracy: {sum(all_8bit_accs)/len(all_8bit_accs):.2f}%")

    print(f"\n4-bit quantization (per layer):")
    print(f"  Best accuracy: {max(all_4bit_accs):.2f}%")
    print(f"  Worst accuracy: {min(all_4bit_accs):.2f}%")
    print(f"  Mean accuracy: {sum(all_4bit_accs)/len(all_4bit_accs):.2f}%")

    print(f"\n✅ All experiments complete!")

    return results


if __name__ == '__main__':
    checkpoint_path = 'checkpoints/resnet20_cifar10_best.pth'
    output_path = 'results/per_layer_results.json'

    results = run_experiments(checkpoint_path, output_path)

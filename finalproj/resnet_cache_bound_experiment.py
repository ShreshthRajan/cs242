"""
ResNet-20 Cache-Bound Speedup Experiment (Batch=1)
Demonstrates AI-aware layer selection reduces computation time

Comparison:
1. FP32 baseline
2. Random 12-layer INT8 selection
3. AI-Aware 12-layer INT8 selection (from Greedy)

With batch=1, model is cache-limited on CPU.
Smaller model (more quantization of heavy layers) = better cache utilization = faster.
"""

import os
import sys
import json
import time
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

from models.resnet import ResNet20
from utils.quantization import quantize_layer, get_quantizable_layers
from utils.ai_calculator import AICalculator


# Configuration
NUM_LAYERS_TO_QUANTIZE = 12
BITWIDTH = 8  # INT8 for all experiments
BATCH_SIZE = 1  # Single image inference (cache-bound)
WARMUP_ITERS = 100
BENCHMARK_ITERS = 1000
NUM_RUNS = 5


def load_test_data():
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
        testset, batch_size=100, shuffle=False, num_workers=0
    )

    return testloader


def evaluate_accuracy(model, testloader):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


def benchmark_latency(model, batch_size=BATCH_SIZE):
    """
    Benchmark single-image inference latency on CPU
    Cache-limited regime where model size matters
    """
    model.eval()
    model = model.cpu()

    dummy_input = torch.randn(batch_size, 3, 32, 32)

    # Warmup (warm CPU cache)
    print(f"  Warming up ({WARMUP_ITERS} iterations)...")
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = model(dummy_input)

    # Benchmark
    print(f"  Benchmarking ({NUM_RUNS} runs × {BENCHMARK_ITERS} iterations)...")
    latencies = []

    for run in range(NUM_RUNS):
        start = time.time()

        with torch.no_grad():
            for _ in range(BENCHMARK_ITERS):
                _ = model(dummy_input)

        elapsed = time.time() - start
        latency_ms = (elapsed * 1000) / BENCHMARK_ITERS  # ms per image
        latencies.append(latency_ms)

    latencies = np.array(latencies)
    mean_latency = latencies.mean()
    std_latency = latencies.std()

    throughput = 1000.0 / mean_latency  # images/sec

    return mean_latency, std_latency, throughput


def main():
    """
    Run cache-bound experiment on ResNet-20
    Demonstrates AI-aware selection beats random selection
    """
    print("="*70)
    print("RESNET-20 CACHE-BOUND EXPERIMENT (Batch=1)")
    print("AI-Aware vs Random Layer Selection")
    print("="*70)

    print(f"\nConfiguration:")
    print(f"  Model: ResNet-20")
    print(f"  Batch size: {BATCH_SIZE} (single-image inference)")
    print(f"  Layers to quantize: {NUM_LAYERS_TO_QUANTIZE}")
    print(f"  Bitwidth: INT{BITWIDTH}")
    print(f"  Device: CPU (cache-limited)")

    checkpoint_path = 'experiments/resnet20/checkpoints/resnet20_cifar10_best.pth'
    testloader = load_test_data()

    results = {}

    # 1. FP32 Baseline
    print("\n" + "="*70)
    print("[1/3] FP32 BASELINE")
    print("="*70)

    model_fp32 = ResNet20()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_fp32.load_state_dict(checkpoint['model_state_dict'])

    print("Evaluating accuracy...")
    acc_fp32 = evaluate_accuracy(model_fp32, testloader)

    print("Benchmarking latency...")
    latency_fp32, std_fp32, throughput_fp32 = benchmark_latency(model_fp32)

    print(f"\n📊 FP32 Results:")
    print(f"  Accuracy: {acc_fp32:.2f}%")
    print(f"  Latency: {latency_fp32:.3f} ± {std_fp32:.3f} ms/image")
    print(f"  Throughput: {throughput_fp32:.1f} img/sec")

    results['fp32'] = {
        'accuracy': acc_fp32,
        'latency_ms': latency_fp32,
        'throughput': throughput_fp32
    }

    # 2. Random Layer Selection
    print("\n" + "="*70)
    print("[2/3] RANDOM LAYER SELECTION")
    print("="*70)

    # Get all layers
    all_layers = get_quantizable_layers(model_fp32)

    # Random selection (fixed seed)
    random.seed(42)
    random_layers = random.sample(all_layers, NUM_LAYERS_TO_QUANTIZE)

    print(f"Randomly selected {len(random_layers)} layers:")
    for layer in random_layers[:5]:
        print(f"  - {layer}")
    if len(random_layers) > 5:
        print(f"  ... and {len(random_layers) - 5} more")

    # Create model
    model_random = ResNet20()
    model_random.load_state_dict(checkpoint['model_state_dict'])

    # Quantize random layers
    for layer_name in random_layers:
        model_random = quantize_layer(model_random, layer_name, BITWIDTH)

    print("Evaluating accuracy...")
    acc_random = evaluate_accuracy(model_random, testloader)

    print("Benchmarking latency...")
    latency_random, std_random, throughput_random = benchmark_latency(model_random)

    speedup_random = throughput_random / throughput_fp32

    print(f"\n📊 Random Selection Results:")
    print(f"  Accuracy: {acc_random:.2f}%")
    print(f"  Latency: {latency_random:.3f} ± {std_random:.3f} ms/image")
    print(f"  Throughput: {throughput_random:.1f} img/sec")
    print(f"  Speedup: {speedup_random:.2f}×")

    results['random'] = {
        'accuracy': acc_random,
        'latency_ms': latency_random,
        'throughput': throughput_random,
        'speedup': speedup_random,
        'selected_layers': random_layers
    }

    # 3. AI-Aware Selection (from Greedy)
    print("\n" + "="*70)
    print("[3/3] AI-AWARE LAYER SELECTION (Greedy)")
    print("="*70)

    # Load Greedy results
    greedy_path = 'experiments/resnet20/results/greedy_search_results.json'
    with open(greedy_path, 'r') as f:
        greedy_data = json.load(f)

    # Get top 12 layers
    history = greedy_data['history']
    aiaware_layers = [entry['layer'] for entry in history[:NUM_LAYERS_TO_QUANTIZE]]

    print(f"AI-Aware selected {len(aiaware_layers)} layers (from Greedy):")
    for i, layer in enumerate(aiaware_layers[:5]):
        print(f"  {i+1}. {layer}")
    if len(aiaware_layers) > 5:
        print(f"  ... and {len(aiaware_layers) - 5} more")

    # Create model
    model_aiaware = ResNet20()
    model_aiaware.load_state_dict(checkpoint['model_state_dict'])

    # Quantize AI-aware layers
    for layer_name in aiaware_layers:
        model_aiaware = quantize_layer(model_aiaware, layer_name, BITWIDTH)

    print("Evaluating accuracy...")
    acc_aiaware = evaluate_accuracy(model_aiaware, testloader)

    print("Benchmarking latency...")
    latency_aiaware, std_aiaware, throughput_aiaware = benchmark_latency(model_aiaware)

    speedup_aiaware = throughput_aiaware / throughput_fp32

    print(f"\n📊 AI-Aware Selection Results:")
    print(f"  Accuracy: {acc_aiaware:.2f}%")
    print(f"  Latency: {latency_aiaware:.3f} ± {std_aiaware:.3f} ms/image")
    print(f"  Throughput: {throughput_aiaware:.1f} img/sec")
    print(f"  Speedup: {speedup_aiaware:.2f}×")

    results['aiaware'] = {
        'accuracy': acc_aiaware,
        'latency_ms': latency_aiaware,
        'throughput': throughput_aiaware,
        'speedup': speedup_aiaware,
        'selected_layers': aiaware_layers
    }

    # Save results
    os.makedirs('experiments/resnet20/results', exist_ok=True)
    with open('experiments/resnet20/results/cache_bound_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Final comparison
    print("\n" + "="*70)
    print("PROFESSOR'S QUESTION - FINAL ANSWER")
    print("="*70)

    print(f"\n{'Config':<25} {'Latency (ms)':<15} {'Speedup':<10} {'Accuracy':<10}")
    print("-"*70)
    print(f"{'FP32':<25} {latency_fp32:<15.3f} {'1.00×':<10} {acc_fp32:<10.2f}%")
    print(f"{'Random 12-layer INT8':<25} {latency_random:<15.3f} {speedup_random:<10.2f}× {acc_random:<10.2f}%")
    print(f"{'AI-Aware 12-layer INT8':<25} {latency_aiaware:<15.3f} {speedup_aiaware:<10.2f}× {acc_aiaware:<10.2f}%")

    # Key finding
    print("\n" + "="*70)
    print("KEY FINDING")
    print("="*70)

    if speedup_aiaware > speedup_random * 1.05:
        advantage = (speedup_aiaware / speedup_random - 1) * 100
        print(f"\n✅ AI-Aware is {advantage:.1f}% FASTER than Random!")
        print(f"   Proof: Selecting specific layers by AI metric reduces computation time")
    else:
        print(f"\n⚠️  Speedups similar:")
        print(f"   Random: {speedup_random:.2f}×")
        print(f"   AI-Aware: {speedup_aiaware:.2f}×")

    if speedup_aiaware > 1.1:
        print(f"\n✅ PROFESSOR'S QUESTION ANSWERED:")
        print(f"   For cache-bound ResNet-20 (batch=1, CPU), AIQ reduces")
        print(f"   computation time by {speedup_aiaware:.2f}× via intelligent")
        print(f"   quantization of specific high-AI layers")

    print(f"\n✅ Results saved: experiments/resnet20/results/cache_bound_results.json")

    return results


if __name__ == '__main__':
    results = main()

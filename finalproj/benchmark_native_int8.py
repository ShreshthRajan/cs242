"""
Benchmark PyTorch native INT8 models (REAL operations)
Measures actual INT8 compute performance on CPU
"""

import os
import sys
sys.path.insert(0, '.')

import time
import json
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision

from models.resnet import ResNet20


def benchmark_model(model: nn.Module, name: str, num_warmup: int = 50,
                   num_iterations: int = 500, num_runs: int = 5,
                   batch_size: int = 128, device: str = 'cpu'):
    """
    Benchmark model with statistical rigor

    Args:
        model: Model to benchmark
        name: Model name
        num_warmup: Warmup iterations
        num_iterations: Benchmark iterations per run
        num_runs: Number of independent runs
        batch_size: Batch size for inference
        device: 'cpu' (INT8 optimized for CPU)
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARKING: {name}")
    print(f"{'='*70}")

    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Benchmark runs
    print(f"Running {num_runs} benchmark runs ({num_iterations} iterations each)...")
    throughputs = []

    for run in range(num_runs):
        start = time.time()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)

        elapsed = time.time() - start

        total_images = num_iterations * batch_size
        throughput = total_images / elapsed
        throughputs.append(throughput)

        print(f"  Run {run+1}/{num_runs}: {throughput:.2f} images/sec")

    # Statistics
    throughputs = np.array(throughputs)
    mean_throughput = throughputs.mean()
    std_throughput = throughputs.std()
    median_throughput = np.median(throughputs)
    latency_ms = 1000.0 / mean_throughput

    print(f"\n📊 Results:")
    print(f"  Throughput: {mean_throughput:.2f} ± {std_throughput:.2f} images/sec")
    print(f"  Latency: {latency_ms:.3f} ms/image")
    print(f"  Median: {median_throughput:.2f} images/sec")
    print(f"  Range: [{throughputs.min():.2f}, {throughputs.max():.2f}]")

    return {
        'throughput_mean': mean_throughput,
        'throughput_std': std_throughput,
        'throughput_median': median_throughput,
        'latency_ms': latency_ms,
        'device': device
    }


def main():
    """Benchmark all native INT8 models"""
    print("="*70)
    print("NATIVE INT8 THROUGHPUT BENCHMARKING")
    print("="*70)
    print("\nUsing PyTorch native INT8 quantization")
    print("Device: CPU (INT8 ops optimized for CPU)\n")

    checkpoint_path = 'checkpoints/resnet20_cifar10_best.pth'
    device = 'cpu'  # INT8 optimization is CPU-focused in PyTorch

    # Load test data
    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root='./data', train=False, download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                   (0.2023, 0.1994, 0.2010))
            ])
        ),
        batch_size=100, shuffle=False
    )

    # Results storage
    benchmark_results = {}

    # 1. FP32 Baseline
    print("\n[1/3] Benchmarking FP32 Baseline...")
    model_fp32 = ResNet20()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_fp32.load_state_dict(checkpoint['model_state_dict'])

    acc_fp32 = checkpoint['test_acc']
    bench_fp32 = benchmark_model(model_fp32, "FP32 Baseline", device=device)
    benchmark_results['fp32'] = {
        'accuracy': acc_fp32,
        **bench_fp32
    }

    # 2. Uniform INT8
    print("\n[2/3] Benchmarking Uniform INT8...")

    # Check if model exists, otherwise create it
    if not os.path.exists('checkpoints/resnet20_native_int8_uniform_scripted.pt'):
        print("Creating Uniform INT8 model...")
        from create_native_int8_models import create_uniform_int8_model
        model_uniform = create_uniform_int8_model(checkpoint_path)
        torch.jit.save(torch.jit.script(model_uniform),
                      'checkpoints/resnet20_native_int8_uniform_scripted.pt')
    else:
        model_uniform = torch.jit.load('checkpoints/resnet20_native_int8_uniform_scripted.pt')

    acc_uniform = evaluate_accuracy(model_uniform, testloader, device)
    bench_uniform = benchmark_model(model_uniform, "Uniform INT8", device=device)
    benchmark_results['uniform_int8'] = {
        'accuracy': acc_uniform,
        **bench_uniform
    }

    # 3. AI-Aware Selective INT8
    print("\n[3/3] Benchmarking AI-Aware Selective INT8...")

    if not os.path.exists('checkpoints/resnet20_native_int8_aiaware_scripted.pt'):
        print("Creating AI-Aware Selective INT8 model...")
        from create_native_int8_models import create_selective_int8_model
        model_aiaware = create_selective_int8_model(
            checkpoint_path,
            'results/greedy_search_results.json',
            top_k_layers=12
        )
        torch.jit.save(torch.jit.script(model_aiaware),
                      'checkpoints/resnet20_native_int8_aiaware_scripted.pt')
    else:
        model_aiaware = torch.jit.load('checkpoints/resnet20_native_int8_aiaware_scripted.pt')

    acc_aiaware = evaluate_accuracy(model_aiaware, testloader, device)
    bench_aiaware = benchmark_model(model_aiaware, "AI-Aware Selective INT8", device=device)
    benchmark_results['aiaware_int8'] = {
        'accuracy': acc_aiaware,
        **bench_aiaware
    }

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/native_int8_benchmark.json', 'w') as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"\n✅ Results saved to: results/native_int8_benchmark.json")

    # Final comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON - NATIVE INT8")
    print("="*70)
    print(f"{'Model':<25} {'Accuracy':<12} {'Throughput':<20} {'Speedup':<10}")
    print("-"*70)

    fp32_throughput = benchmark_results['fp32']['throughput_mean']

    for name, data in benchmark_results.items():
        acc = data['accuracy']
        throughput = data['throughput_mean']
        speedup = throughput / fp32_throughput

        display_name = {
            'fp32': 'FP32 Baseline',
            'uniform_int8': 'Uniform INT8',
            'aiaware_int8': 'AI-Aware Selective'
        }[name]

        print(f"{display_name:<25} {acc:<12.2f} {throughput:<20.1f} {speedup:<10.2f}x")

    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    uniform_speedup = benchmark_results['uniform_int8']['throughput_mean'] / fp32_throughput
    aiaware_speedup = benchmark_results['aiaware_int8']['throughput_mean'] / fp32_throughput

    uniform_acc = benchmark_results['uniform_int8']['accuracy']
    aiaware_acc = benchmark_results['aiaware_int8']['accuracy']

    print(f"\nSpeedups over FP32:")
    print(f"  Uniform INT8: {uniform_speedup:.2f}× speedup, {uniform_acc:.2f}% accuracy")
    print(f"  AI-Aware: {aiaware_speedup:.2f}× speedup, {aiaware_acc:.2f}% accuracy")

    print(f"\nAI-Aware vs Uniform INT8:")
    print(f"  Accuracy: {aiaware_acc - uniform_acc:+.2f}% (better if positive)")
    print(f"  Relative speedup: {aiaware_speedup / uniform_speedup:.2f}× (>1 = faster)")

    if aiaware_acc > uniform_acc and aiaware_speedup >= uniform_speedup * 0.95:
        print("\n✅ AI-AWARE WINS: Better accuracy at similar/better speedup!")
    elif aiaware_acc > uniform_acc:
        print("\n✅ AI-AWARE ADVANTAGE: Better accuracy (slight speedup tradeoff)")
    else:
        print("\n⚠️  Results analysis needed")

    print("\n✅ Benchmarking complete with REAL INT8 operations!")

    return benchmark_results


if __name__ == '__main__':
    results = main()

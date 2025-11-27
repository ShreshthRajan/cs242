"""
Benchmark inference throughput for all quantization configurations
Measures images/second and ms/image on GPU with proper warm-up and statistics

ICML-grade benchmarking with:
- Proper GPU warm-up
- Multiple runs for statistical significance
- Memory tracking
- Synchronized CUDA timing
"""

import os
import sys
sys.path.insert(0, '.')

import time
import json
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms

from models.resnet import ResNet20


class ThroughputBenchmark:
    """
    Production-grade throughput benchmarking with statistical rigor
    """

    def __init__(self, device: str = 'cuda', batch_size: int = 128):
        """
        Args:
            device: 'cuda' or 'cpu'
            batch_size: Inference batch size (larger = better throughput)
        """
        self.device = device
        self.batch_size = batch_size

        # Benchmark parameters (ICML-grade)
        self.warmup_iterations = 50   # Warm up GPU, compile kernels
        self.benchmark_iterations = 200  # Actual measurement iterations
        self.num_runs = 5  # Multiple runs for statistics

        print(f"Benchmark configuration:")
        print(f"  Device: {device}")
        print(f"  Batch size: {batch_size}")
        print(f"  Warmup iterations: {self.warmup_iterations}")
        print(f"  Benchmark iterations: {self.benchmark_iterations}")
        print(f"  Number of runs: {self.num_runs}")

        # Create dummy input for benchmarking
        self.dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)

        # CUDA synchronization for accurate timing
        self.use_cuda_events = device == 'cuda'

    def _warmup(self, model: nn.Module):
        """Warm up GPU and compile kernels"""
        model.eval()
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = model(self.dummy_input)

        # Synchronize after warmup
        if self.use_cuda_events:
            torch.cuda.synchronize()

    def _benchmark_single_run(self, model: nn.Module) -> float:
        """
        Single benchmark run with CUDA events for accurate timing

        Returns:
            throughput: Images per second
        """
        model.eval()

        if self.use_cuda_events:
            # GPU timing with CUDA events (most accurate)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()

            with torch.no_grad():
                for _ in range(self.benchmark_iterations):
                    _ = model(self.dummy_input)

            end_event.record()
            torch.cuda.synchronize()

            # Time in milliseconds
            elapsed_ms = start_event.elapsed_time(end_event)
            elapsed_s = elapsed_ms / 1000.0

        else:
            # CPU timing
            start = time.time()

            with torch.no_grad():
                for _ in range(self.benchmark_iterations):
                    _ = model(self.dummy_input)

            elapsed_s = time.time() - start

        # Calculate throughput
        total_images = self.benchmark_iterations * self.batch_size
        throughput = total_images / elapsed_s  # images/second

        return throughput

    def benchmark_model(self, model: nn.Module, name: str) -> dict:
        """
        Benchmark model with multiple runs and statistical analysis

        Args:
            model: Model to benchmark
            name: Model name for logging

        Returns:
            dict with throughput statistics
        """
        print(f"\n{'='*70}")
        print(f"BENCHMARKING: {name}")
        print(f"{'='*70}")

        model = model.to(self.device)

        # Warmup
        print("Warming up...")
        self._warmup(model)

        # Multiple benchmark runs
        print(f"Running {self.num_runs} benchmark runs...")
        throughputs = []

        for run in range(self.num_runs):
            throughput = self._benchmark_single_run(model)
            throughputs.append(throughput)
            print(f"  Run {run+1}/{self.num_runs}: {throughput:.2f} images/sec")

        # Statistics
        throughputs = np.array(throughputs)
        mean_throughput = throughputs.mean()
        std_throughput = throughputs.std()
        median_throughput = np.median(throughputs)

        # Latency (ms per image)
        mean_latency = 1000.0 / mean_throughput  # ms per image

        results = {
            'throughput_mean': mean_throughput,
            'throughput_std': std_throughput,
            'throughput_median': median_throughput,
            'throughput_min': throughputs.min(),
            'throughput_max': throughputs.max(),
            'latency_ms': mean_latency,
            'batch_size': self.batch_size,
            'device': self.device
        }

        print(f"\n📊 Results:")
        print(f"  Throughput: {mean_throughput:.2f} ± {std_throughput:.2f} images/sec")
        print(f"  Latency: {mean_latency:.3f} ms/image")
        print(f"  Median: {median_throughput:.2f} images/sec")

        # GPU memory usage
        if self.device == 'cuda':
            mem_allocated = torch.cuda.max_memory_allocated() / (1024**2)
            print(f"  Peak GPU memory: {mem_allocated:.2f} MB")
            results['gpu_memory_mb'] = mem_allocated

        return results


def load_model_from_checkpoint(checkpoint_path: str, device: str = 'cuda') -> nn.Module:
    """Load model from checkpoint"""
    model = ResNet20()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    return model


def main():
    """Benchmark all 4 configurations"""
    print("="*70)
    print("THROUGHPUT BENCHMARKING - ALL CONFIGURATIONS")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cpu':
        print("\n⚠️  WARNING: Running on CPU (slow)")
        print("   For accurate benchmarks, use GPU in Colab")

    # Initialize benchmarker
    benchmarker = ThroughputBenchmark(device=device, batch_size=128)

    # Model configurations
    configs = {
        'FP32': 'checkpoints/resnet20_cifar10_best.pth',
        'INT8': 'checkpoints/resnet20_uniform_int8.pth',
        'INT4': 'checkpoints/resnet20_uniform_int4.pth',
        'AI-Aware': 'checkpoints/resnet20_ai_aware_greedy.pth'
    }

    # Benchmark each model
    benchmark_results = {}

    for name, checkpoint_path in configs.items():
        print(f"\n{'='*70}")
        print(f"Loading {name} model...")
        print(f"{'='*70}")

        model = load_model_from_checkpoint(checkpoint_path, device)

        # Benchmark
        results = benchmarker.benchmark_model(model, name)
        benchmark_results[name] = results

        # Cleanup
        del model
        torch.cuda.empty_cache() if device == 'cuda' else None

    # Load accuracy and AI data
    print(f"\n{'='*70}")
    print("LOADING ACCURACY AND AI DATA")
    print(f"{'='*70}")

    with open('results/baseline_comparison.json', 'r') as f:
        baseline_data = json.load(f)

    with open('results/greedy_search_results.json', 'r') as f:
        greedy_data = json.load(f)

    # Combine all metrics
    final_results = {
        'FP32': {
            'accuracy': baseline_data['fp32']['accuracy'],
            'ai': baseline_data['fp32']['ai'],
            'model_size_mb': baseline_data['fp32']['model_size_mb'],
            **benchmark_results['FP32']
        },
        'INT8': {
            'accuracy': baseline_data['int8']['accuracy'],
            'ai': baseline_data['int8']['ai'],
            'model_size_mb': baseline_data['int8']['model_size_mb'],
            **benchmark_results['INT8']
        },
        'INT4': {
            'accuracy': baseline_data['int4']['accuracy'],
            'ai': baseline_data['int4']['ai'],
            'model_size_mb': baseline_data['int4']['model_size_mb'],
            **benchmark_results['INT4']
        },
        'AI-Aware': {
            'accuracy': greedy_data['final_accuracy'],
            'ai': greedy_data['final_ai'],
            'config': greedy_data['config_summary'],
            **benchmark_results['AI-Aware']
        }
    }

    # Save complete results
    os.makedirs('results', exist_ok=True)
    with open('results/complete_benchmark.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n✅ Results saved to: results/complete_benchmark.json")

    # Print comparison table
    print(f"\n{'='*70}")
    print("COMPLETE COMPARISON")
    print(f"{'='*70}")
    print(f"{'Model':<12} {'Acc%':<8} {'AI':<10} {'Throughput':<15} {'Latency':<12} {'Size':<8}")
    print("-"*70)

    for name, data in final_results.items():
        acc = data['accuracy']
        ai = data['ai']
        throughput = data['throughput_mean']
        latency = data['latency_ms']
        size = data.get('model_size_mb', 0)

        print(f"{name:<12} {acc:<8.2f} {ai:<10.2f} "
              f"{throughput:<15.1f} {latency:<12.3f} {size:<8.2f}")

    # Key findings
    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}")

    fp32_throughput = final_results['FP32']['throughput_mean']
    int8_throughput = final_results['INT8']['throughput_mean']
    int4_throughput = final_results['INT4']['throughput_mean']
    aiaware_throughput = final_results['AI-Aware']['throughput_mean']

    print(f"\nThroughput improvements over FP32:")
    print(f"  INT8: {(int8_throughput/fp32_throughput - 1)*100:+.1f}%")
    print(f"  INT4: {(int4_throughput/fp32_throughput - 1)*100:+.1f}%")
    print(f"  AI-Aware: {(aiaware_throughput/fp32_throughput - 1)*100:+.1f}%")

    print(f"\nAI-Aware vs Uniform baselines:")
    print(f"  vs INT8: {(aiaware_throughput/int8_throughput - 1)*100:+.1f}% throughput, "
          f"{final_results['AI-Aware']['ai'] - final_results['INT8']['ai']:+.1f} AI")
    print(f"  vs INT4: {(aiaware_throughput/int4_throughput - 1)*100:+.1f}% throughput, "
          f"{final_results['AI-Aware']['accuracy'] - final_results['INT4']['accuracy']:+.2f}% accuracy")

    print("\n✅ Benchmarking complete!")

    return final_results


if __name__ == '__main__':
    results = main()

"""
Greedy Search for AI-Aware Quantization
Iteratively selects layers to quantize based on loss function J(q)

Based on formulation:
    J(q) = λ·AI(q) - (1-λ)·AccLoss(q)

Where:
    - AI(q): Arithmetic intensity of configuration q
    - AccLoss(q): Accuracy loss from baseline (Err(q) - Err(q^FP))
    - λ: Tradeoff parameter (0=only accuracy, 1=only AI)
"""

import os
import sys
import json
import copy
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Set

from models.resnet import ResNet20
from utils.ai_calculator import AICalculator
from utils.quantization import quantize_layer, get_quantizable_layers


class GreedyQuantizationSearch:
    """
    Greedy search for optimal per-layer quantization policy
    Maximizes J(q) = λ·AI(q) - (1-λ)·AccLoss(q)
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_path: str,
        testloader: torch.utils.data.DataLoader,
        device: str = 'cuda',
        lambda_param: float = 0.5,
        bitwidth_options: List[int] = [8, 4]
    ):
        """
        Args:
            model: Base FP32 model
            checkpoint_path: Path to trained checkpoint
            testloader: CIFAR-10 test data loader
            device: 'cuda' or 'cpu'
            lambda_param: Tradeoff between AI (λ=1) and accuracy (λ=0)
            bitwidth_options: Candidate bitwidths to try (default: [8, 4])
        """
        self.device = device
        self.lambda_param = lambda_param
        self.bitwidth_options = sorted(bitwidth_options, reverse=True)  # Try 8-bit before 4-bit

        # Load trained model
        self.base_model = model.to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.base_model.load_state_dict(checkpoint['model_state_dict'])
        self.base_model.eval()

        self.testloader = testloader

        # Baseline metrics
        print("Computing baseline metrics...")
        self.baseline_accuracy = self._evaluate_accuracy(self.base_model)
        self.ai_calculator = AICalculator(
            self.base_model.cpu(),
            input_shape=(3, 32, 32),
            batch_size=1
        )
        self.baseline_ai = self.ai_calculator.get_model_ai()

        print(f"Baseline: Accuracy={self.baseline_accuracy:.2f}%, AI={self.baseline_ai:.4f}")

        # Get quantizable layers
        self.quantizable_layers = get_quantizable_layers(self.base_model)
        print(f"Found {len(self.quantizable_layers)} quantizable layers")

        # Current configuration: layer_name → bitwidth (32=FP32)
        self.current_config = {layer: 32 for layer in self.quantizable_layers}
        self.current_model = copy.deepcopy(self.base_model).to(device)
        self.current_accuracy = self.baseline_accuracy
        self.current_ai = self.baseline_ai

        # Search history
        self.history = []

    def _evaluate_accuracy(self, model: nn.Module) -> float:
        """Evaluate model accuracy on test set"""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100.0 * correct / total
        return accuracy

    def _compute_loss(self, ai: float, accuracy: float) -> float:
        """
        Compute loss function J(q) = λ·AI(q) - (1-λ)·AccLoss(q)

        We want to MAXIMIZE this (higher AI, lower AccLoss)
        So we MINIMIZE the negative: -J(q)
        """
        # Normalize AI to [0, 1] scale (divide by reasonable max ~60)
        ai_normalized = ai / 60.0

        # Accuracy loss from baseline (want to minimize this)
        acc_loss = max(0, self.baseline_accuracy - accuracy)  # Clamp to positive

        # Loss function (minimize -J)
        loss = -self.lambda_param * ai_normalized + (1 - self.lambda_param) * (acc_loss / 10.0)

        return loss

    def _try_quantize_layer(
        self,
        layer_name: str,
        num_bits: int
    ) -> Tuple[nn.Module, float, float, float]:
        """
        Try quantizing a specific layer and return metrics

        Returns:
            (model, accuracy, ai, loss)
        """
        # Create model with this layer quantized
        candidate_model = copy.deepcopy(self.current_model).cpu()

        # Quantize the layer
        candidate_model = quantize_layer(candidate_model, layer_name, num_bits)
        candidate_model = candidate_model.to(self.device)

        # Evaluate accuracy
        candidate_accuracy = self._evaluate_accuracy(candidate_model)

        # Calculate new AI
        candidate_ai = self.ai_calculator.compute_ai_with_quantized_layer(
            layer_name, num_bits
        )

        # Compute loss
        candidate_loss = self._compute_loss(candidate_ai, candidate_accuracy)

        return candidate_model, candidate_accuracy, candidate_ai, candidate_loss

    def search(self, max_iterations: int = 20, min_accuracy: float = 91.0) -> Dict:
        """
        Run greedy search to find optimal quantization configuration

        Args:
            max_iterations: Maximum number of layers to quantize
            min_accuracy: Minimum acceptable accuracy (hard constraint)

        Returns:
            Dict with final configuration and search history
        """
        print("="*70)
        print("GREEDY SEARCH FOR AI-AWARE QUANTIZATION")
        print("="*70)
        print(f"Lambda: {self.lambda_param} (0=accuracy only, 1=AI only)")
        print(f"Bitwidth options: {self.bitwidth_options}")
        print(f"Minimum accuracy: {min_accuracy}%")
        print(f"Max iterations: {max_iterations}")
        print("="*70)

        current_loss = self._compute_loss(self.current_ai, self.current_accuracy)

        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            print(f"Current config: {sum(1 for b in self.current_config.values() if b < 32)} layers quantized")
            print(f"Current: Acc={self.current_accuracy:.2f}%, AI={self.current_ai:.4f}, Loss={current_loss:.4f}")

            # Find best move
            best_move = None
            best_model = None
            best_loss = current_loss
            best_metrics = None

            # Try quantizing each FP32 layer
            unquantized_layers = [
                layer for layer, bits in self.current_config.items()
                if bits == 32
            ]

            if not unquantized_layers:
                print("All layers quantized. Search complete.")
                break

            print(f"Trying {len(unquantized_layers)} candidate layers...")

            for layer_name in unquantized_layers:
                for num_bits in self.bitwidth_options:
                    # Try this move
                    candidate_model, cand_acc, cand_ai, cand_loss = self._try_quantize_layer(
                        layer_name, num_bits
                    )

                    # Check hard constraint
                    if cand_acc < min_accuracy:
                        continue  # Reject moves that violate accuracy constraint

                    # Track best move (minimize loss = maximize J)
                    if cand_loss < best_loss:
                        best_loss = cand_loss
                        best_move = (layer_name, num_bits)
                        best_model = candidate_model
                        best_metrics = (cand_acc, cand_ai)

                    # Cleanup
                    del candidate_model
                    torch.cuda.empty_cache() if self.device == 'cuda' else None

            # Check for improvement
            delta_loss = best_loss - current_loss

            if best_move is None:
                print("No valid moves found (all violate accuracy constraint).")
                break

            if delta_loss >= 0:
                print(f"No improvement found (ΔLoss={delta_loss:+.4f} ≥ 0). Search converged.")
                break

            # Accept best move
            layer_name, num_bits = best_move
            best_acc, best_ai = best_metrics

            print(f"✓ Selected: {layer_name} → {num_bits}-bit")
            print(f"  New: Acc={best_acc:.2f}%, AI={best_ai:.4f}, Loss={best_loss:.4f}")
            print(f"  Improvement: ΔLoss={delta_loss:.4f}, ΔAI={best_ai - self.current_ai:+.4f}")

            # Update current state
            self.current_config[layer_name] = num_bits
            self.current_model = best_model
            self.current_accuracy = best_acc
            self.current_ai = best_ai
            current_loss = best_loss

            # Record history
            self.history.append({
                'iteration': iteration + 1,
                'layer': layer_name,
                'bitwidth': num_bits,
                'accuracy': best_acc,
                'ai': best_ai,
                'loss': best_loss,
                'delta_loss': delta_loss
            })

        # Final results
        print("\n" + "="*70)
        print("SEARCH COMPLETE")
        print("="*70)
        print(f"Iterations: {len(self.history)}")
        print(f"Layers quantized: {sum(1 for b in self.current_config.values() if b < 32)}")
        print(f"Final accuracy: {self.current_accuracy:.2f}%")
        print(f"Final AI: {self.current_ai:.4f}")
        print(f"AI improvement: {(self.current_ai/self.baseline_ai - 1)*100:.1f}%")

        # Configuration summary
        config_summary = {}
        for bits in [4, 8, 32]:
            layers_at_bits = [l for l, b in self.current_config.items() if b == bits]
            config_summary[f'{bits}-bit'] = len(layers_at_bits)
            if layers_at_bits and bits < 32:
                print(f"\n{bits}-bit layers ({len(layers_at_bits)}):")
                for layer in layers_at_bits[:5]:
                    print(f"  - {layer}")
                if len(layers_at_bits) > 5:
                    print(f"  ... and {len(layers_at_bits) - 5} more")

        # Save results
        results = {
            'final_config': self.current_config,
            'final_accuracy': self.current_accuracy,
            'final_ai': self.current_ai,
            'baseline_accuracy': self.baseline_accuracy,
            'baseline_ai': self.baseline_ai,
            'lambda': self.lambda_param,
            'config_summary': config_summary,
            'history': self.history
        }

        return results


def load_test_data(batch_size: int = 100):
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


def main():
    """Run greedy search for AI-aware quantization"""
    print("="*70)
    print("AI-AWARE QUANTIZATION - GREEDY SEARCH")
    print("="*70)

    # Configuration
    checkpoint_path = 'checkpoints/resnet20_cifar10_best.pth'
    output_path = 'results/greedy_search_results.json'
    lambda_param = 0.5  # Balance AI and accuracy equally
    min_accuracy = 91.0  # Don't drop below 91%

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Load model and data
    print("\nLoading model and data...")
    model = ResNet20()
    testloader = load_test_data(batch_size=100)

    # Run greedy search
    searcher = GreedyQuantizationSearch(
        model=model,
        checkpoint_path=checkpoint_path,
        testloader=testloader,
        device=device,
        lambda_param=lambda_param,
        bitwidth_options=[8, 4]  # Try 8-bit first, then 4-bit
    )

    start_time = time.time()
    results = searcher.search(
        max_iterations=20,
        min_accuracy=min_accuracy
    )
    elapsed = time.time() - start_time

    print(f"\nSearch completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")

    # Save results
    os.makedirs('results', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")

    # Save final model
    model_path = 'checkpoints/resnet20_ai_aware_greedy.pth'
    torch.save({
        'model_state_dict': searcher.current_model.state_dict(),
        'config': results['final_config'],
        'accuracy': results['final_accuracy'],
        'ai': results['final_ai'],
    }, model_path)

    print(f"✅ Model saved to: {model_path}")

    # Print final summary
    print("\n" + "="*70)
    print("FINAL CONFIGURATION")
    print("="*70)
    print(f"Baseline → AI-Aware:")
    print(f"  Accuracy: {results['baseline_accuracy']:.2f}% → {results['final_accuracy']:.2f}%")
    print(f"  AI: {results['baseline_ai']:.4f} → {results['final_ai']:.4f}")
    print(f"  Improvement: {(results['final_ai']/results['baseline_ai'] - 1)*100:.1f}% AI gain")
    print(f"  Accuracy change: {results['final_accuracy'] - results['baseline_accuracy']:+.2f}%")

    return results


if __name__ == '__main__':
    results = main()

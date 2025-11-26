"""
Generate scatter plot: Arithmetic Intensity vs Accuracy
Shows tradeoffs for per-layer quantization
"""

import json
import matplotlib.pyplot as plt
import numpy as np


def plot_ai_vs_accuracy(results_path: str, output_path: str):
    """
    Create scatter plot of AI vs Accuracy for per-layer quantization

    Args:
        results_path: Path to results JSON from run_per_layer_experiments.py
        output_path: Path to save plot image
    """
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)

    baseline = results['baseline']
    per_layer = results['per_layer']

    # Extract data points
    ai_8bit = []
    acc_8bit = []
    ai_4bit = []
    acc_4bit = []
    layer_names = []

    for layer_result in per_layer:
        layer_names.append(layer_result['layer_name'])

        # 8-bit
        exp_8bit = layer_result['experiments'][0]
        ai_8bit.append(exp_8bit['ai'])
        acc_8bit.append(exp_8bit['accuracy'])

        # 4-bit
        exp_4bit = layer_result['experiments'][1]
        ai_4bit.append(exp_4bit['ai'])
        acc_4bit.append(exp_4bit['accuracy'])

    # Create plot
    plt.figure(figsize=(12, 8))

    # Baseline point (large, starred)
    plt.scatter(baseline['ai'], baseline['accuracy'],
               s=300, c='black', marker='*', edgecolors='gold',
               linewidths=2, label='Baseline (FP32)', zorder=10)

    # 8-bit points
    plt.scatter(ai_8bit, acc_8bit,
               s=100, c='blue', alpha=0.6, edgecolors='darkblue',
               label='8-bit (per layer)', zorder=5)

    # 4-bit points
    plt.scatter(ai_4bit, acc_4bit,
               s=100, c='red', alpha=0.6, edgecolors='darkred',
               label='4-bit (per layer)', zorder=5)

    # Annotate interesting points (highest AI, lowest accuracy drop)
    # Find best 8-bit layer (high AI, minimal acc drop)
    ai_increase_8bit = [(ai - baseline['ai'], acc, name)
                        for ai, acc, name in zip(ai_8bit, acc_8bit, layer_names)]
    ai_increase_8bit.sort(reverse=True)

    # Annotate top 3 highest AI gains for 8-bit
    for i in range(min(3, len(ai_increase_8bit))):
        ai_delta, acc, name = ai_increase_8bit[i]
        ai_val = baseline['ai'] + ai_delta
        acc_drop = baseline['accuracy'] - acc

        # Only annotate if accuracy drop is reasonable (< 5%)
        if acc_drop < 5:
            plt.annotate(
                name.split('.')[-1],  # Just show layer number
                xy=(ai_val, acc),
                xytext=(10, -10),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )

    # Labels and formatting
    plt.xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=14, fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title('Per-Layer Quantization: AI vs Accuracy Tradeoff\n'
             'ResNet-20 on CIFAR-10',
             fontsize=16, fontweight='bold')

    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3, linestyle='--')

    # Set reasonable axis limits
    ai_all = ai_8bit + ai_4bit + [baseline['ai']]
    acc_all = acc_8bit + acc_4bit + [baseline['accuracy']]

    ai_min, ai_max = min(ai_all), max(ai_all)
    acc_min, acc_max = min(acc_all), max(acc_all)

    plt.xlim(ai_min * 0.98, ai_max * 1.02)
    plt.ylim(acc_min - 2, acc_max + 1)

    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")

    plt.show()

    # Print key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)

    # Find layer with highest AI gain and minimal accuracy loss
    best_tradeoff_8bit = None
    best_score = -float('inf')

    for i, name in enumerate(layer_names):
        ai_gain = (ai_8bit[i] - baseline['ai']) / baseline['ai']
        acc_drop = baseline['accuracy'] - acc_8bit[i]

        # Score: maximize AI gain, minimize accuracy drop
        # Penalize heavily for accuracy drops > 2%
        score = ai_gain - (acc_drop / 2.0)

        if score > best_score and acc_drop < 5:
            best_score = score
            best_tradeoff_8bit = {
                'name': name,
                'ai': ai_8bit[i],
                'acc': acc_8bit[i],
                'ai_gain_pct': ai_gain * 100,
                'acc_drop': acc_drop
            }

    if best_tradeoff_8bit:
        print(f"\nBest 8-bit tradeoff layer: {best_tradeoff_8bit['name']}")
        print(f"  AI: {best_tradeoff_8bit['ai']:.4f} "
              f"(+{best_tradeoff_8bit['ai_gain_pct']:.1f}%)")
        print(f"  Accuracy: {best_tradeoff_8bit['acc']:.2f}% "
              f"(Δ{best_tradeoff_8bit['acc_drop']:+.2f}%)")

    # Worst layer (highest accuracy drop)
    worst_layer_idx = acc_8bit.index(min(acc_8bit))
    worst_layer = layer_names[worst_layer_idx]
    print(f"\nMost sensitive layer: {worst_layer}")
    print(f"  8-bit accuracy: {acc_8bit[worst_layer_idx]:.2f}% "
          f"(Δ{acc_8bit[worst_layer_idx] - baseline['accuracy']:+.2f}%)")

    # Statistics
    print(f"\n8-bit quantization range:")
    print(f"  AI: {min(ai_8bit):.4f} to {max(ai_8bit):.4f}")
    print(f"  Accuracy: {min(acc_8bit):.2f}% to {max(acc_8bit):.2f}%")


if __name__ == '__main__':
    results_path = 'results/per_layer_results.json'
    plot_path = 'results/ai_vs_accuracy_scatter.png'

    plot_ai_vs_accuracy(results_path, plot_path)

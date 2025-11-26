"""
Generate improved plots: Per-layer line plots with dual Y-axes
Separate plots for 8-bit and 4-bit quantization
"""

import json
import matplotlib.pyplot as plt
import numpy as np


def plot_per_layer_line_plots(results_path: str):
    """
    Create two separate plots (8-bit and 4-bit) with dual Y-axes:
    - X-axis: Layer index
    - Left Y-axis: Test Accuracy (%)
    - Right Y-axis: Arithmetic Intensity (FLOPs/Byte)
    """
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)

    baseline = results['baseline']
    per_layer = results['per_layer']

    # Extract data
    layer_names = [l['layer_name'] for l in per_layer]
    layer_indices = list(range(len(layer_names)))

    # 8-bit data
    acc_8bit = [l['experiments'][0]['accuracy'] for l in per_layer]
    ai_8bit = [l['experiments'][0]['ai'] for l in per_layer]

    # 4-bit data
    acc_4bit = [l['experiments'][1]['accuracy'] for l in per_layer]
    ai_4bit = [l['experiments'][1]['ai'] for l in per_layer]

    # Create 8-bit plot
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Accuracy (left Y-axis)
    color_acc = 'tab:blue'
    ax1.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold', color=color_acc)
    ax1.plot(layer_indices, acc_8bit, 'o-', color=color_acc, linewidth=2,
             markersize=6, label='Accuracy (8-bit)')
    ax1.axhline(y=baseline['accuracy'], color=color_acc, linestyle='--',
                linewidth=2, alpha=0.5, label='Baseline Accuracy')
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_ylim(min(acc_8bit) - 0.5, max(acc_8bit) + 0.5)
    ax1.grid(True, alpha=0.3, axis='y')

    # AI (right Y-axis)
    ax2 = ax1.twinx()
    color_ai = 'tab:red'
    ax2.set_ylabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=12,
                   fontweight='bold', color=color_ai)
    ax2.plot(layer_indices, ai_8bit, 's-', color=color_ai, linewidth=2,
             markersize=6, label='AI (8-bit)')
    ax2.axhline(y=baseline['ai'], color=color_ai, linestyle='--',
                linewidth=2, alpha=0.5, label='Baseline AI')
    ax2.tick_params(axis='y', labelcolor=color_ai)
    ax2.set_ylim(min(ai_8bit) * 0.995, max(ai_8bit) * 1.005)

    # Title and formatting
    plt.title('8-bit Per-Layer Quantization: Accuracy & AI Tradeoffs\n'
             'ResNet-20 on CIFAR-10',
             fontsize=14, fontweight='bold', pad=20)

    # X-axis labels (show every 2nd layer)
    ax1.set_xticks(layer_indices[::2])
    ax1.set_xticklabels([f"L{i}" for i in layer_indices[::2]], rotation=0)

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=10)

    plt.tight_layout()
    plt.savefig('results/8bit_per_layer.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/8bit_per_layer.png")

    # Create 4-bit plot
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Accuracy (left Y-axis)
    color_acc = 'tab:blue'
    ax1.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold', color=color_acc)
    ax1.plot(layer_indices, acc_4bit, 'o-', color=color_acc, linewidth=2,
             markersize=6, label='Accuracy (4-bit)')
    ax1.axhline(y=baseline['accuracy'], color=color_acc, linestyle='--',
                linewidth=2, alpha=0.5, label='Baseline Accuracy')
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_ylim(min(acc_4bit) - 0.5, max(acc_4bit) + 0.5)
    ax1.grid(True, alpha=0.3, axis='y')

    # AI (right Y-axis)
    ax2 = ax1.twinx()
    color_ai = 'tab:red'
    ax2.set_ylabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=12,
                   fontweight='bold', color=color_ai)
    ax2.plot(layer_indices, ai_4bit, 's-', color=color_ai, linewidth=2,
             markersize=6, label='AI (4-bit)')
    ax2.axhline(y=baseline['ai'], color=color_ai, linestyle='--',
                linewidth=2, alpha=0.5, label='Baseline AI')
    ax2.tick_params(axis='y', labelcolor=color_ai)
    ax2.set_ylim(min(ai_4bit) * 0.995, max(ai_4bit) * 1.005)

    # Title and formatting
    plt.title('4-bit Per-Layer Quantization: Accuracy & AI Tradeoffs\n'
             'ResNet-20 on CIFAR-10',
             fontsize=14, fontweight='bold', pad=20)

    # X-axis labels
    ax1.set_xticks(layer_indices[::2])
    ax1.set_xticklabels([f"L{i}" for i in layer_indices[::2]], rotation=0)

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=10)

    plt.tight_layout()
    plt.savefig('results/4bit_per_layer.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/4bit_per_layer.png")

    # Print insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)

    # Find best layers (high AI gain, minimal accuracy loss)
    for bitwidth, ai_vals, acc_vals in [('8-bit', ai_8bit, acc_8bit),
                                         ('4-bit', ai_4bit, acc_4bit)]:
        print(f"\n{bitwidth} Quantization:")

        # Best AI gain
        ai_gains = [(ai - baseline['ai']) / baseline['ai'] * 100 for ai in ai_vals]
        best_ai_idx = ai_gains.index(max(ai_gains))

        print(f"  Highest AI gain: Layer {best_ai_idx} ({layer_names[best_ai_idx]})")
        print(f"    AI: {ai_vals[best_ai_idx]:.4f} (+{ai_gains[best_ai_idx]:.1f}%)")
        print(f"    Accuracy: {acc_vals[best_ai_idx]:.2f}% "
              f"(Δ{acc_vals[best_ai_idx] - baseline['accuracy']:+.2f}%)")

        # Worst accuracy
        worst_acc_idx = acc_vals.index(min(acc_vals))
        print(f"  Largest accuracy drop: Layer {worst_acc_idx} ({layer_names[worst_acc_idx]})")
        print(f"    Accuracy: {acc_vals[worst_acc_idx]:.2f}% "
              f"(Δ{acc_vals[worst_acc_idx] - baseline['accuracy']:+.2f}%)")

    print("\n" + "="*70)


if __name__ == '__main__':
    results_path = 'results/per_layer_results.json'
    plot_per_layer_line_plots(results_path)

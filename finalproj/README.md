# AI-QS: Arithmetic-Intensity-Aware Quantization Search

**CS2420 Final Project - Fall 2025**

## Project Overview

An automated tool that optimizes per-layer bitwidth allocation to maximize arithmetic intensity (AI) on GPU hardware, bridging roofline analysis with neural network quantization.

**Novelty:** First work to use arithmetic intensity (FLOPs/Bytes) as the primary optimization objective for quantization policy.

## MVP Scope

1. Train ResNet-20 baseline on CIFAR-10 (~91-92% accuracy)
2. Quantize layers individually (8-bit, 4-bit)
3. Measure (AI, Accuracy) tradeoffs per layer
4. Generate scatter plot showing heterogeneous layer sensitivities

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Baseline Model
```bash
cd finalproj
python train_baseline.py
```

Expected:
- Training time: ~2-3 hours on GPU
- Final accuracy: ~91-92%
- Checkpoint: `checkpoints/resnet20_cifar10_best.pth`

### 3. Run Per-Layer Quantization Experiments
```bash
python run_per_layer_experiments.py
```

Expected:
- Time: ~20-30 minutes
- Output: `results/per_layer_ai_accuracy.json`

### 4. Generate Plot
```bash
python plot_results.py
```

Output: `results/ai_vs_accuracy_scatter.png`

## Project Structure

```
finalproj/
├── models/
│   ├── __init__.py
│   └── resnet.py          # ResNet-20 architecture
├── utils/
│   ├── ai_calculator.py   # Arithmetic intensity computation
│   └── quantization.py    # Per-layer quantization wrappers
├── data/                  # CIFAR-10 dataset (auto-downloaded)
├── checkpoints/           # Saved model checkpoints
├── results/               # Experimental results and plots
├── train_baseline.py      # Step 1: Train FP32 baseline
├── run_per_layer_experiments.py  # Step 2: Per-layer quantization sweep
├── plot_results.py        # Step 3: Visualization
└── requirements.txt

```

## Team

- Shreshth Rajan
- Nikhil Jain
- Taig Singh

## References

1. He et al., "Deep Residual Learning for Image Recognition" (2015)
2. Williams et al., "Roofline: An Insightful Visual Performance Model" (2009)
3. DoReFa-Net: Training Low Bitwidth CNNs (2016)

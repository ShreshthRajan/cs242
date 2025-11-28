# AI-QS: Arithmetic-Intensity-Aware Quantization Search

**CS2420 Final Project - Fall 2025**

## Overview

A framework for optimizing per-layer neural network quantization to maximize arithmetic intensity (roofline metric) while maintaining accuracy. Demonstrates Pareto-optimal quantization configurations on ResNet-20/CIFAR-10.

**Key Contribution:** First work to use arithmetic intensity (FLOPs/Bytes) as the primary optimization objective for automated quantization policy search.

## Results

- **51.6% AI improvement** over FP32 baseline (29.85 → 45.25 FLOPs/Byte)
- **<1% accuracy degradation** (91.92% → 91.02%)
- **Pareto optimal** vs uniform INT8 and INT4 quantization
- **Smart layer allocation:** Heavy layers (layer3) → 4-bit, Light layers (layer1) → 8-bit

## Execution Pipeline

Run scripts in numerical order:

```bash
# Step 1: Train baseline ResNet-20 (~2 hours on GPU)
python 1_train_baseline.py

# Step 2: Per-layer sensitivity analysis (~20 mins on GPU)
python 2_per_layer_analysis.py

# Step 3: Greedy search for optimal config (~20 mins on GPU)
python 3_greedy_search.py

# Step 4: Create uniform baselines (~5 mins)
python 4_create_baselines.py

# Step 5: Generate publication plots (~10 secs)
python 5_generate_plots.py
```

## Project Structure

```
finalproj/
├── 1_train_baseline.py          # Train ResNet-20 on CIFAR-10
├── 2_per_layer_analysis.py      # Per-layer quantization sensitivity
├── 3_greedy_search.py           # Multi-layer optimization (Greedy)
├── 4_create_baselines.py        # Uniform INT8/INT4 baselines
├── 5_generate_plots.py          # Final visualization
├── models/resnet.py             # ResNet-20 architecture
├── utils/
│   ├── ai_calculator.py         # Arithmetic intensity computation
│   └── quantization.py          # Per-layer quantization wrappers
├── tests/                       # Testing utilities
├── checkpoints/                 # Trained models
└── results/                     # Plots and metrics
```

## Key Findings

1. AI-Aware achieves **45.25 FLOPs/Byte** (highest among >91% accuracy configs)
2. Maintains **91.02% accuracy** (<1% drop, within research standards)
3. Outperforms Uniform INT8: +7% AI at similar accuracy
4. Outperforms Uniform INT4: +1% accuracy at similar AI
5. Greedy search identifies optimal layer allocation in 20 iterations

## Team

Shreshth Rajan, Nikhil Jain, Taig Singh

## References

1. Williams et al., "Roofline: An Insightful Visual Performance Model" (2009)
2. He et al., "Deep Residual Learning for Image Recognition" (2015)
3. Zhou et al., "DoReFa-Net: Training Low Bitwidth CNNs" (2016)

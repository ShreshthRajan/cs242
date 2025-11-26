#!/bin/bash

# Complete MVP Pipeline Runner
# Runs all steps for AI-QS MVP

echo "============================================================"
echo "AI-QS MVP PIPELINE"
echo "============================================================"

# Step 0: Verify checkpoint exists
if [ ! -f "checkpoints/resnet20_cifar10_best.pth" ]; then
    echo "❌ Error: Baseline checkpoint not found!"
    echo "   Please run: python train_baseline.py first"
    exit 1
fi

echo "✅ Baseline checkpoint found"

# Step 1: Test pipeline
echo ""
echo "[1/3] Testing pipeline..."
python test_pipeline.py
if [ $? -ne 0 ]; then
    echo "❌ Pipeline test failed!"
    exit 1
fi

# Step 2: Run per-layer experiments
echo ""
echo "[2/3] Running per-layer quantization experiments..."
echo "      This will take ~20-30 minutes..."
python run_per_layer_experiments.py
if [ $? -ne 0 ]; then
    echo "❌ Experiments failed!"
    exit 1
fi

# Step 3: Generate plot
echo ""
echo "[3/3] Generating visualization..."
python plot_results.py
if [ $? -ne 0 ]; then
    echo "❌ Plotting failed!"
    exit 1
fi

echo ""
echo "============================================================"
echo "✅ MVP COMPLETE!"
echo "============================================================"
echo ""
echo "Results:"
echo "  - Data: results/per_layer_results.json"
echo "  - Plot: results/ai_vs_accuracy_scatter.png"
echo ""
echo "Ready for presentation!"

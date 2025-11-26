"""
Test complete Step 2 pipeline before running full experiments
"""

import sys
sys.path.insert(0, '.')

import torch
from models.resnet import ResNet20
from utils.ai_calculator import AICalculator
from utils.quantization import get_quantizable_layers, quantize_layer

print("="*70)
print("TESTING STEP 2 PIPELINE")
print("="*70)

# Test 1: Model loading
print("\n[1/5] Testing model loading...")
model = ResNet20()
checkpoint = torch.load('checkpoints/resnet20_cifar10_best.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
print(f"   Baseline accuracy: {checkpoint['test_acc']:.2f}%")

# Test 2: AI Calculator
print("\n[2/5] Testing AI calculator...")
calculator = AICalculator(model, input_shape=(3, 32, 32), batch_size=1)
baseline_ai = calculator.get_model_ai()
print(f"✅ AI Calculator initialized")
print(f"   Model AI: {baseline_ai:.4f} FLOPs/Byte")
print(f"   Total FLOPs: {calculator.total_flops:,}")
print(f"   Total Bytes: {calculator.total_bytes:,}")
print(f"   Layers found: {len(calculator.layer_info)}")

# Test 3: Layer enumeration
print("\n[3/5] Testing layer enumeration...")
layers = get_quantizable_layers(model)
print(f"✅ Found {len(layers)} quantizable layers:")
for i, layer in enumerate(layers[:5]):
    ai_8bit = calculator.compute_ai_with_quantized_layer(layer, 8)
    ai_4bit = calculator.compute_ai_with_quantized_layer(layer, 4)
    print(f"   {i+1}. {layer:<25} AI(8bit)={ai_8bit:.4f}, AI(4bit)={ai_4bit:.4f}")
if len(layers) > 5:
    print(f"   ... and {len(layers)-5} more")

# Test 4: Quantization
print("\n[4/5] Testing per-layer quantization...")
test_layer = layers[0]
model_8bit = quantize_layer(model, test_layer, num_bits=8)
model_4bit = quantize_layer(model, test_layer, num_bits=4)

# Test forward pass
x = torch.randn(2, 3, 32, 32)
with torch.no_grad():
    y_orig = model(x)
    y_8bit = model_8bit(x)
    y_4bit = model_4bit(x)

diff_8bit = (y_orig - y_8bit).abs().mean()
diff_4bit = (y_orig - y_4bit).abs().mean()

print(f"✅ Quantization successful for layer: {test_layer}")
print(f"   Output diff (8-bit): {diff_8bit:.6f}")
print(f"   Output diff (4-bit): {diff_4bit:.6f}")
print(f"   4-bit > 8-bit: {diff_4bit > diff_8bit} (expected)")

# Test 5: Full experiment workflow simulation
print("\n[5/5] Testing experiment workflow...")
print(f"   Will run {len(layers)} layers × 2 bitwidths = {len(layers)*2} experiments")
print(f"   Estimated time: ~{len(layers)*2*1.5:.0f} seconds = ~{len(layers)*2*1.5/60:.0f} minutes")

# Simulate one complete experiment
print(f"\n   Simulating experiment for layer: {test_layer}")
for bits in [8, 4]:
    quant_model = quantize_layer(model, test_layer, num_bits=bits)
    new_ai = calculator.compute_ai_with_quantized_layer(test_layer, bits)
    ai_increase = (new_ai / baseline_ai - 1) * 100

    # Quick forward pass test
    with torch.no_grad():
        _ = quant_model(x)

    print(f"   {bits}-bit: AI = {new_ai:.4f} (+{ai_increase:.2f}%), forward pass OK ✓")

print("\n" + "="*70)
print("✅ ALL TESTS PASSED - PIPELINE READY!")
print("="*70)
print("\nNext steps:")
print("  1. Run: python run_per_layer_experiments.py  (~20-30 mins)")
print("  2. Run: python plot_results.py")
print("\nReady for MVP!")

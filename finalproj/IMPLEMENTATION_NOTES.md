# Greedy Search Implementation Notes

## Algorithm Overview

### Loss Function
```
J(q) = λ·AI(q) - (1-λ)·AccLoss(q)

Where:
- AI(q): Model arithmetic intensity (FLOPs/Bytes)
- AccLoss(q): Accuracy loss from FP32 baseline
- λ ∈ [0,1]: Tradeoff parameter
  - λ=0: Optimize accuracy only
  - λ=1: Optimize AI only
  - λ=0.5: Balance both equally
```

### Greedy Algorithm
```
1. Start: q^(0) = all FP32
2. Loop:
   a. For each unquantized layer ℓ:
      - Try quantizing to 8-bit, compute ΔJ_8
      - Try quantizing to 4-bit, compute ΔJ_4
   b. Select: layer m* with best (minimum) ΔJ
   c. If ΔJ_m* < 0: accept move, q^(t+1) = q_m*
   d. Else: stop (no improvement)
3. Return: final configuration q*
```

## Implementation Details

### Key Design Decisions

**1. Bitwidth Ordering**
- Try 8-bit before 4-bit (conservative → aggressive)
- Allows graceful degradation if 4-bit hurts accuracy

**2. Accuracy Constraint**
- Hard constraint: min_accuracy = 91.0%
- Reject any move that drops below threshold
- Prevents catastrophic accuracy loss

**3. AI Calculation**
- Analytical (not measured) for speed
- Based on layer parameter counts and activation sizes
- Updated incrementally as layers quantized

**4. Model Management**
- Deep copy current model for each candidate
- Evaluate accuracy on full test set (10K images)
- GPU memory cleanup after each evaluation

### Performance Optimizations

**1. Lazy Evaluation**
- Only evaluate candidates that might improve loss
- Skip candidates that obviously violate constraints

**2. GPU Efficiency**
- Batch evaluation where possible
- Aggressive memory cleanup (torch.cuda.empty_cache())
- Move models to CPU during quantization to avoid OOM

**3. Early Stopping**
- Stop when no valid moves exist
- Stop when improvement plateaus (ΔJ ≥ 0)

## Expected Behavior

### Typical Search Trajectory

**Iteration 1:**
- Tries all 20 layers × 2 bitwidths = 40 candidates
- Likely selects: layer3.X.conv2 at 4-bit (big layer, high AI gain)

**Iteration 2-5:**
- Continues quantizing layer3 layers (biggest AI gains)
- Accuracy stays above 91.5%

**Iteration 6-10:**
- Starts quantizing layer2 layers to 8-bit
- More conservative due to diminishing returns

**Convergence:**
- Stops when all high-value layers quantized
- Final config: ~5-8 layers at 4-bit, ~10-12 at 8-bit, ~2-5 at FP32

### Success Criteria

✅ **Accuracy maintained:** Final acc ≥ 91.0%
✅ **AI improved:** Final AI > Baseline AI
✅ **Converges:** Terminates in < 20 iterations
✅ **Reproducible:** Same results with same λ

## Troubleshooting

**Issue: Search stops after 1 iteration**
→ ΔJ ≥ 0 for all candidates (loss function too aggressive on accuracy)
→ Fix: Increase λ to favor AI more

**Issue: All candidates violate min_accuracy**
→ min_accuracy threshold too high
→ Fix: Lower to 90.5% or 90.0%

**Issue: CUDA OOM**
→ Too many models in GPU memory
→ Fix: More aggressive cleanup (already implemented)

**Issue: Search is slow (>1 hour)**
→ Each iteration evaluates 40 candidates
→ Expected: ~3-5 min per iteration on GPU

## Testing Checklist

Before full run:
- [ ] Test 1: Loss function behaves correctly
- [ ] Test 2: Single iteration completes
- [ ] Test 3: Search converges properly
- [ ] Full run: Complete search in reasonable time

## Usage

```bash
# Run tests first
python test_greedy_search.py

# Run full search (if tests pass)
python greedy_search.py
```

"""
Unit and integration tests for Greedy Search
Validates algorithm correctness before full execution
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from models.resnet import ResNet20
from greedy_search import GreedyQuantizationSearch, load_test_data


def test_loss_function():
    """Test loss function computation"""
    print("="*70)
    print("[TEST 1] Loss Function")
    print("="*70)

    # Create dummy searcher
    model = ResNet20()
    testloader = load_test_data()

    searcher = GreedyQuantizationSearch(
        model=model,
        checkpoint_path='checkpoints/resnet20_cifar10_best.pth',
        testloader=testloader,
        device='cpu',
        lambda_param=0.5
    )

    # Test loss computation
    baseline_loss = searcher._compute_loss(searcher.baseline_ai, searcher.baseline_accuracy)
    print(f"Baseline loss: {baseline_loss:.4f}")

    # Higher AI, same accuracy = better (lower loss)
    better_loss = searcher._compute_loss(searcher.baseline_ai * 1.1, searcher.baseline_accuracy)
    print(f"10% higher AI, same acc: {better_loss:.4f}")
    assert better_loss < baseline_loss, "Higher AI should give lower loss!"
    print("✅ Higher AI → lower loss (correct)")

    # Same AI, lower accuracy = worse (higher loss)
    worse_loss = searcher._compute_loss(searcher.baseline_ai, searcher.baseline_accuracy - 1)
    print(f"Same AI, 1% lower acc: {worse_loss:.4f}")
    assert worse_loss > baseline_loss, "Lower accuracy should give higher loss!"
    print("✅ Lower accuracy → higher loss (correct)")

    print("\n✅ TEST 1 PASSED\n")


def test_single_iteration():
    """Test one iteration of greedy search"""
    print("="*70)
    print("[TEST 2] Single Greedy Iteration")
    print("="*70)

    model = ResNet20()
    testloader = load_test_data()

    searcher = GreedyQuantizationSearch(
        model=model,
        checkpoint_path='checkpoints/resnet20_cifar10_best.pth',
        testloader=testloader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        lambda_param=0.5
    )

    print("\nRunning 1 iteration (will evaluate all 20 layers × 2 bitwidths = 40 candidates)...")
    print("This may take 2-3 minutes...\n")

    # Run just 1 iteration
    results = searcher.search(max_iterations=1, min_accuracy=90.0)

    assert len(searcher.history) == 1, "Should have 1 move in history"
    print(f"\n✅ Greedy made 1 move:")
    print(f"   Layer: {searcher.history[0]['layer']}")
    print(f"   Bitwidth: {searcher.history[0]['bitwidth']}")
    print(f"   Accuracy: {searcher.history[0]['accuracy']:.2f}%")
    print(f"   AI: {searcher.history[0]['ai']:.4f}")

    print("\n✅ TEST 2 PASSED\n")


def test_convergence():
    """Test that search stops when no improvement"""
    print("="*70)
    print("[TEST 3] Convergence Behavior")
    print("="*70)

    model = ResNet20()
    testloader = load_test_data()

    # Use high lambda (prioritize AI heavily)
    searcher = GreedyQuantizationSearch(
        model=model,
        checkpoint_path='checkpoints/resnet20_cifar10_best.pth',
        testloader=testloader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        lambda_param=0.9,  # Heavily favor AI
        bitwidth_options=[8]  # Only try 8-bit (less aggressive)
    )

    print("\nRunning search with λ=0.9 (favor AI over accuracy)...")
    results = searcher.search(max_iterations=20, min_accuracy=90.0)

    print(f"\n✅ Search converged after {len(searcher.history)} iterations")
    print(f"   Final accuracy: {results['final_accuracy']:.2f}%")
    print(f"   Final AI: {results['final_ai']:.4f}")
    print(f"   AI improvement: {(results['final_ai']/results['baseline_ai'] - 1)*100:.1f}%")

    assert results['final_accuracy'] >= 90.0, "Should maintain min accuracy"
    print("✅ Accuracy constraint satisfied")

    assert results['final_ai'] > results['baseline_ai'], "Should improve AI"
    print("✅ AI improved from baseline")

    print("\n✅ TEST 3 PASSED\n")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("GREEDY SEARCH TEST SUITE")
    print("="*70 + "\n")

    try:
        # Run tests
        test_loss_function()
        test_single_iteration()
        test_convergence()

        print("="*70)
        print("✅ ALL TESTS PASSED - GREEDY SEARCH IS CORRECT")
        print("="*70)
        print("\nReady to run full search with: python greedy_search.py")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

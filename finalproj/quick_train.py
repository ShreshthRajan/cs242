"""
Quick training script with reduced epochs for fast testing
Use this to verify everything works before full 200-epoch run
"""

import sys
sys.path.insert(0, '.')

from train_baseline import *

def quick_train():
    """
    Quick training with 10 epochs to verify pipeline works
    Use this before starting the full 200-epoch run
    """
    print("="*60)
    print("QUICK TRAINING TEST (10 EPOCHS)")
    print("="*60)

    # Reduced hyperparameters for testing
    epochs = 10
    batch_size = 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load data
    trainloader, testloader = get_cifar10_loaders(batch_size=batch_size, num_workers=2)

    # Model
    model = ResNet20(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 8], gamma=0.1)
    scaler = GradScaler()

    # Quick training
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, trainloader, criterion, optimizer, scaler, device, epoch
        )

        if epoch % 2 == 0 or epoch == epochs - 1:
            test_loss, test_acc = evaluate(model, testloader, criterion, device)

        scheduler.step()

    print("\n✅ Quick training test passed!")
    print("   Ready to run full 200-epoch training with: python train_baseline.py")

if __name__ == '__main__':
    quick_train()

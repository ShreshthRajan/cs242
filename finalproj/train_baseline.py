"""
Train ResNet-20 baseline on CIFAR-10
Optimized for fast, accurate training with state-of-the-art techniques
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler

from models.resnet import ResNet20


def get_cifar10_loaders(batch_size=128, num_workers=4):
    """
    Load CIFAR-10 with standard data augmentation
    """
    # Training transforms: augmentation for better generalization
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010)),
    ])

    # Test transforms: no augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010)),
    ])

    # Load datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True,
        transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True,
        transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return trainloader, testloader


def train_epoch(model, trainloader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch with mixed precision"""
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Mixed precision training for speed
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Backward pass with gradient scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Statistics
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 50 == 0:
            acc = 100. * correct / total
            sys.stdout.write(
                f'\rEpoch {epoch}: [{batch_idx}/{len(trainloader)}] '
                f'Loss: {train_loss/(batch_idx+1):.3f} | '
                f'Acc: {acc:.2f}%'
            )
            sys.stdout.flush()

    # Final epoch stats
    epoch_loss = train_loss / len(trainloader)
    epoch_acc = 100. * correct / total
    print(f' | Final - Loss: {epoch_loss:.3f} | Acc: {epoch_acc:.2f}%')

    return epoch_loss, epoch_acc


def evaluate(model, testloader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Compute metrics
    test_loss = test_loss / len(testloader)
    test_acc = 100. * correct / total

    print(f'Test - Loss: {test_loss:.3f} | Acc: {test_acc:.2f}%')

    return test_loss, test_acc


def main():
    """
    Train ResNet-20 baseline on CIFAR-10
    Target: ~91-92% test accuracy
    """
    print("="*60)
    print("TRAINING RESNET-20 BASELINE ON CIFAR-10")
    print("="*60)

    # Hyperparameters (following original paper + modern optimizations)
    epochs = 200
    batch_size = 128
    lr_init = 0.1
    momentum = 0.9
    weight_decay = 1e-4
    milestones = [100, 150]  # LR decay at these epochs
    gamma = 0.1

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Data loaders
    print("\nLoading CIFAR-10 dataset...")
    trainloader, testloader = get_cifar10_loaders(
        batch_size=batch_size,
        num_workers=4 if device == 'cuda' else 2
    )
    print(f"Train samples: {len(trainloader.dataset)}")
    print(f"Test samples: {len(testloader.dataset)}")

    # Model
    print("\nInitializing ResNet-20...")
    model = ResNet20(num_classes=10).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr_init,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # Learning rate scheduler (MultiStepLR)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=gamma
    )

    # Mixed precision scaler for faster training
    scaler = GradScaler()

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    print(f"LR schedule: {lr_init} → decay by {gamma}× at epochs {milestones}")
    print("-"*60)

    best_acc = 0
    train_history = {'loss': [], 'acc': []}
    test_history = {'loss': [], 'acc': []}

    start_time = time.time()

    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, trainloader, criterion, optimizer,
            scaler, device, epoch
        )

        # Evaluate every 10 epochs (save time)
        if epoch % 10 == 0 or epoch == epochs - 1:
            test_loss, test_acc = evaluate(model, testloader, criterion, device)

            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                print(f'💾 Saving checkpoint (best acc: {best_acc:.2f}%)...')
                state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'test_acc': test_acc,
                    'test_loss': test_loss,
                }
                os.makedirs('checkpoints', exist_ok=True)
                torch.save(state, 'checkpoints/resnet20_cifar10_best.pth')

            # Record history
            test_history['loss'].append(test_loss)
            test_history['acc'].append(test_acc)

        train_history['loss'].append(train_loss)
        train_history['acc'].append(train_acc)

        # Update learning rate
        scheduler.step()

        # Print LR changes
        if epoch + 1 in milestones:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n📉 Learning rate reduced to {current_lr}")

    # Training complete
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Final test accuracy: {test_history['acc'][-1]:.2f}%")

    # Save final model
    print("\n💾 Saving final checkpoint...")
    final_state = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'test_acc': test_history['acc'][-1],
        'best_acc': best_acc,
        'train_history': train_history,
        'test_history': test_history,
    }
    torch.save(final_state, 'checkpoints/resnet20_cifar10_final.pth')

    # Save training history
    torch.save({
        'train_history': train_history,
        'test_history': test_history,
    }, 'checkpoints/training_history.pth')

    print("\n✅ Baseline training complete!")
    print(f"   Best checkpoint: checkpoints/resnet20_cifar10_best.pth")
    print(f"   Final checkpoint: checkpoints/resnet20_cifar10_final.pth")

    return model, best_acc


if __name__ == '__main__':
    model, best_acc = main()

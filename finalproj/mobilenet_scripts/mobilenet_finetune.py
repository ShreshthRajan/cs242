"""
Fine-tune pretrained MobileNetV2 on CIFAR-10
Quick adaptation for our quantization experiments
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models.mobilenet import MobileNetV2_CIFAR10


def get_cifar10_loaders(batch_size=128):
    """Load CIFAR-10 with standard augmentation"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2
    )

    return trainloader, testloader


def evaluate(model, testloader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def main():
    """Fine-tune MobileNetV2 on CIFAR-10"""
    print("="*70)
    print("FINE-TUNING MOBILENETV2 FOR CIFAR-10")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Load pretrained model
    print("\nLoading pretrained MobileNetV2...")
    model = MobileNetV2_CIFAR10(pretrained=True).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # Data
    print("\nLoading CIFAR-10...")
    trainloader, testloader = get_cifar10_loaders(batch_size=128)

    # Quick fine-tuning (10 epochs, enough for adaptation)
    print("\nFine-tuning for 10 epochs...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    for epoch in range(10):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 50 == 0:
                acc = 100. * correct / total
                sys.stdout.write(
                    f'\rEpoch {epoch}: [{batch_idx}/{len(trainloader)}] '
                    f'Loss: {train_loss/(batch_idx+1):.3f} | Acc: {acc:.2f}%'
                )
                sys.stdout.flush()

        # Evaluate
        test_acc = evaluate(model, testloader, device)
        print(f' | Test Acc: {test_acc:.2f}%')

    # Save
    os.makedirs('experiments/mobilenet/checkpoints', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_acc': test_acc
    }, 'experiments/mobilenet/checkpoints/mobilenet_cifar10.pth')

    print(f"\n✅ Fine-tuned model saved")
    print(f"   Accuracy: {test_acc:.2f}%")
    print(f"   Checkpoint: experiments/mobilenet/checkpoints/mobilenet_cifar10.pth")

    return model, test_acc


if __name__ == '__main__':
    model, acc = main()

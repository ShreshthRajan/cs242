"""
Debug: Test quantizing multiple layers together
To understand why Greedy stops after 1 iteration
"""

import sys
sys.path.insert(0, '.')

import torch
from models.resnet import ResNet20
from utils.quantization import quantize_layer
import torchvision.transforms as transforms
import torchvision

# Load model
model = ResNet20()
checkpoint = torch.load('checkpoints/resnet20_cifar10_best.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# Load test data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

def eval_acc(model):
    model.eval()
    correct = 0
    total = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total

print("Testing multi-layer quantization...")
print(f"Baseline accuracy: {checkpoint['test_acc']:.2f}%\n")

# Test 1: Single layer (from Greedy result)
model1 = ResNet20()
model1.load_state_dict(checkpoint['model_state_dict'])
model1 = quantize_layer(model1, 'layer3.1.conv2', 4)
acc1 = eval_acc(model1)
print(f"1. Only layer3.1.conv2 at 4-bit: {acc1:.2f}%")

# Test 2: Two layers
model2 = ResNet20()
model2.load_state_dict(checkpoint['model_state_dict'])
model2 = quantize_layer(model2, 'layer3.1.conv2', 4)
model2 = quantize_layer(model2, 'layer3.0.conv2', 4)
acc2 = eval_acc(model2)
print(f"2. layer3.1.conv2 + layer3.0.conv2 at 4-bit: {acc2:.2f}%")

# Test 3: Three layers
model3 = ResNet20()
model3.load_state_dict(checkpoint['model_state_dict'])
model3 = quantize_layer(model3, 'layer3.1.conv2', 4)
model3 = quantize_layer(model3, 'layer3.0.conv2', 4)
model3 = quantize_layer(model3, 'layer3.2.conv2', 4)
acc3 = eval_acc(model3)
print(f"3. Three layer3.X.conv2 at 4-bit: {acc3:.2f}%")

# Test 4: Mix 8-bit and 4-bit
model4 = ResNet20()
model4.load_state_dict(checkpoint['model_state_dict'])
model4 = quantize_layer(model4, 'layer3.1.conv2', 4)
model4 = quantize_layer(model4, 'layer3.0.conv2', 8)
model4 = quantize_layer(model4, 'layer2.0.conv2', 8)
acc4 = eval_acc(model4)
print(f"4. Mixed (1×4bit, 2×8bit): {acc4:.2f}%")

print("\n✅ Debug complete - see if multi-layer works!")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from model import LightweightCNN, count_parameters

# Set random seed for reproducibility
torch.manual_seed(42)

# Data augmentation and normalization
train_transform = transforms.Compose([
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def load_data():
    # Load MNIST dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    return train_loader, test_loader

def train_model(model, train_loader, device, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f'Batch: {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    accuracy = 100. * correct / total
    print(f"Total trainable parameters: {count_parameters(model)}")
    print(f'\nTraining Accuracy after 1 Epoch: {accuracy:.2f}%')
    print(f'Training Loss: {total_loss/len(train_loader):.4f}')
    return accuracy

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

def save_augmented_samples(train_dataset, num_samples=100):
    indices = torch.randperm(len(train_dataset))[:num_samples]
    samples = torch.stack([train_dataset[i][0] for i in indices])
    
    samples = samples * 0.3081 + 0.1307
    
    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        plt.subplot(10, 10, i + 1)
        plt.imshow(samples[i].squeeze(), cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('augmented_samples.png')
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightweightCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_loader, test_loader = load_data()
    
    print(f"Total trainable parameters: {count_parameters(model)}")
    
    # Train for one epoch
    print("Starting training...")
    train_accuracy = train_model(model, train_loader, device, optimizer, criterion)
    
    # Save augmented samples
    save_augmented_samples(train_loader.dataset)
    
    # Evaluate the model
    print("\nEvaluating model...")
    test_accuracy = test(model, test_loader, device)

if __name__ == "__main__":
    main() 
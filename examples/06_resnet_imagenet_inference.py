# -*- coding:utf-8 -*-
# @File  : 06_resnet_imagenet_inference.py
# @Author: ZZW
# @Date  : 2025/2/20
"""Memintelli example 6: ResNet in Imagenet dataset using Memintelli.
This example demonstrates the usage of Memintelli with vgg to load pre-trained model.
"""
import os
import sys
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.nn import functional as F
# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from NN_models import ResNet_zoo
from pimpy.memmat_tensor import DPETensor

def load_dataset(data_root, batch_size=256):
    """Load dataset with normalization."""
    # Create dataset directories if not exist
    os.makedirs(data_root, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_set = datasets.ImageNet(root=data_root, split='train', transform=transform)
    test_set = datasets.ImageNet(root=data_root, split='val', transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, device, 
                epochs=10, lr=0.001, mem_enabled=True):
    """Train the model with progress tracking and validation.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        test_loader: Test data loader
        device: Computation device
        epochs: Number of training epochs
        lr: Learning rate
        mem_enabled: If mem_enabled is True, the model will use the memristive engine for memristive weight updates
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # Training phase
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                if mem_enabled:
                    model.update_weights()
                
                epoch_loss += loss.item() * images.size(0)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Validation phase
        avg_loss = epoch_loss / len(train_loader.dataset)
        val_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1} - Avg loss: {avg_loss:.4f}, Val accuracy: {val_acc:.2%}")

def evaluate(model, test_loader, device):
    """Evaluate model performance on test set.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Computation device
        
    Returns:
        Classification accuracy
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total
    
def main():
    # Configuration
    data_root = "D:/dataset/imagenet"   # Change this to your dataset directory
    batch_size = 16
    learning_rate = 0.001
    # Slicing configuration and INT/FP mode settings
    input_slice = (1, 1, 2)
    weight_slice = (1, 1, 2)
    bw_e = None

    model_name = 'resnet18'     # Select the model name
    mem_enabled = False          # Select the memrsitive mode or software mode

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_dataset(data_root, batch_size)

    mem_engine = DPETensor(
        var=0.05,
        rdac=2**2,
        g_level=2**2,
        radc=2**10,
        weight_quant_gran=(256, 1),
        input_quant_gran=(1, 256),
        weight_paral_size=(64, 1),
        input_paral_size=(1, 64),
        device=device
    )

    model = ResNet_zoo(model_name=model_name, pretrained=True, mem_enabled=mem_enabled, 
    engine = mem_engine, input_slice=input_slice, weight_slice=weight_slice, device=device, bw_e=bw_e).to(device)
    model.update_weight()
    final_acc = evaluate(model, test_loader, device)
    print(f"\nFinal test accuracy of {model_name} in Imagenet: {final_acc:.2%}")

if __name__ == "__main__":
    main()
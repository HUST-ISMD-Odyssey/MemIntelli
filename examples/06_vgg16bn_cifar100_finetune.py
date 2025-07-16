# -*- coding:utf-8 -*-
# @File  : 06_vgg16bn_cifar100_finetune.py
# @Author: ZZW
# @Date  : 2025/2/20
"""Memintelli example 6: VGG16_bn on CIFAR-100 data set finetuning using Memintelli.
This example demonstrates the usage of Memintelli with vgg to finetune pre-trained model.
"""
import os
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.nn import functional as F

from memintelli.NN_models import vgg_cifar_zoo
from memintelli.pimpy.memmat_tensor import DPETensor

def load_dataset(data_root, batch_size=256, num_classes=10):
    """Load dataset with normalization."""
    # Create dataset directories if not exist
    os.makedirs(data_root, exist_ok=True)
    if num_classes == 10:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
        ])
        train_set = datasets.CIFAR10(data_root, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(data_root, train=False, download=True, transform=transform)
    elif num_classes == 100:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])
        ])
        train_set = datasets.CIFAR100(data_root, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR100(data_root, train=False, download=True, transform=transform)

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * epochs, eta_min=1e-7)  # T_max=总epoch数
    epoch_accuracies = []
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
                    model.update_weight()
                
                epoch_loss += loss.item() * images.size(0)
                pbar.set_postfix({"loss": f"{loss.item():.6f}", "lr": f"{scheduler.get_last_lr()[0]:.8f}"})
                scheduler.step()

        # Validation phase
        avg_loss = epoch_loss / len(train_loader.dataset)
        val_acc = evaluate(model, test_loader, device)
        epoch_accuracies.append(val_acc)  # 记录当前epoch准确率
        print(f"Epoch {epoch+1} - Avg loss: {avg_loss:.6f}, Val accuracy: {val_acc:.2%}")
    return  epoch_accuracies

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
        progress_bar = tqdm(test_loader, desc="Evaluating", unit="batch")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Update progress bar with current accuracy
            accuracy = 100 * correct / total
            progress_bar.set_postfix(accuracy=f'{accuracy:.2f}%')
    
    return correct / total
    
def main():
    # Configuration
    data_root = "/dataset/"   # Change this to your dataset directory
    batch_size = 32
    epochs = 10
    learning_rate = 1e-4
    # Slicing configuration and INT/FP mode settings
    input_slice = (1, 1, 1, 1)
    weight_slice = (1, 1, 1, 1)
    bw_e = None

    mem_enabled = True       # Select the memrsitive mode or software mode
    model_name = "vgg16_bn"   # Select the model name
    num_classes = 100          # if 10, it is cifar10, if 100, it is cifar100

    # Initialize components
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_dataset(data_root, batch_size, num_classes=num_classes)

    mem_engine = DPETensor(
        HGS=1e-5,                       # High conductance state
        LGS=1e-8,                       # Low conductance state
        write_variation=0.0,          # Write variation
        rate_stuck_HGS=0.005,          # Rate of stuck at HGS
        rate_stuck_LGS=0.00,          # Rate of stuck at LGS
        read_variation=0.05,           # Read variation
        vnoise=0.0,                   # Random Gaussian noise of voltage
        rdac=2**1,                      # Number of DAC resolution 
        g_level=2**1,                   # Number of conductance levels
        radc=2**6
        )

    model = vgg_cifar_zoo(model_name=model_name, num_classes=num_classes, pretrained=True, mem_enabled=mem_enabled, 
    engine=mem_engine, input_slice=input_slice, weight_slice=weight_slice, device=device, bw_e=bw_e,
    input_paral_size=(1, 32), weight_paral_size=(32, 32), input_quant_gran=(1, 64), weight_quant_gran=(64, 64)).to(device)
    model.update_weight()
    initial_acc = evaluate(model, test_loader, device)
    print(f"Initial test accuracy {model_name} in CIFAR-{num_classes}: {initial_acc:.2%}")
    train_accuracies=train_model(model, train_loader, test_loader, device, epochs=epochs, lr=learning_rate, mem_enabled=mem_enabled)
    all_accuracies = [initial_acc] + train_accuracies
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(all_accuracies)), all_accuracies, marker='o', linestyle='-', color='#2c7fb8')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.title('Fine-tuning Accuracy Progression', fontsize=14)
    plt.xticks(range(0, len(all_accuracies)))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
# -*- coding:utf-8 -*-
# @File  : 02_mlp_inference.py
# @Author: ZZW
# @Date  : 2025/2/20
"""Memintelli example 2: MLP inference using Memintelli.
This example demonstrates the usage of Memintelli with a simple MLP classifier that has been trained in software.
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
from NN_layers.linear import LinearMem
from pimpy.memmat_tensor import DPETensor

class MNISTClassifier(nn.Module):
    """MLP classifier for MNIST with configurable layers.
    Args:
        engine: Memristive simulation engine
        input_slice: Input tensor slicing configuration
        weight_slice: Weight tensor slicing configuration
        device: Computation device (CPU/GPU)
        layer_dims: List of layer dimensions [input_dim, hidden_dims..., output_dim]
        bw_e: if bw_e is None, the memristive engine is INT mode, otherwise, the memristive engine is FP mode (bw_e is the bitwidth of the exponent)
        mem_enabled: If mem_enabled is True, the model will use the memristive engine for memristive weight updates
    """
    def __init__(self, engine, input_slice, weight_slice, device, 
                 layer_dims=[784, 512, 128, 10], bw_e=None, mem_enabled=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.flatten = nn.Flatten()
        self.engine = engine
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            if mem_enabled is True:
                self.layers.append(
                    LinearMem(engine, in_dim, out_dim, input_slice, weight_slice,
                             device=device, bw_e=bw_e)
                )
            else:
                self.layers.append(nn.Linear(in_dim, out_dim))

    def forward(self, x):
        """Forward pass with ReLU activation and final softmax."""
        x = self.flatten(x)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return F.softmax(x, dim=1)

    def update_weights(self):
        """Convert the model weights (FP32) to PIM sliced_weights. 
        ***This function is very important for loading as well as updating pre-training weights in inference or training.***            """
        if self.mem_enabled:
            for layer in self.layers:
                layer.update_weight()

def load_mnist(data_root, batch_size=256):
    """Load MNIST dataset with normalization."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Create dataset directories if not exist
    os.makedirs(data_root, exist_ok=True)

    train_set = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_root, train=False, download=True, transform=transform)

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
    config = {
        "data_root": "/dataset/",   # Change this to your dataset directory
        "batch_size": 256,
        "epochs": 10,
        "learning_rate": 0.001,
        "layer_dims": [784, 512, 128, 10],
        "input_slice": (1, 1, 2),
        "weight_slice": (1, 1, 2),
        "bw_e": 8
    }
    
    # Initialize components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_mnist(config["data_root"], config["batch_size"])
    
    # Initialize the software model with mem_enabled=False
    model = MNISTClassifier(
        engine=None,
        input_slice=config["input_slice"],
        weight_slice=config["weight_slice"],
        device=device,
        layer_dims=config["layer_dims"],
        bw_e=config["bw_e"],
        mem_enabled=False,      # Set mem_enabled=False for software model
    ).to(device)

    # Train the software model
    train_model(
        model,
        train_loader,
        test_loader,
        device,
        epochs=config["epochs"],
        lr=config["learning_rate"],
        mem_enabled=False
    )

    # Initialize memristive engine and model
    mem_engine = DPETensor(
        var=0.02,
        rdac=2**2,
        g_level=2**2,
        radc=2**12,
        weight_quant_gran=(128, 128),
        input_quant_gran=(1, 128),
        weight_paral_size=(64, 64),
        input_paral_size=(1, 64)
    )

    mdoel_mem = MNISTClassifier(
        engine=mem_engine,
        input_slice=config["input_slice"],
        weight_slice=config["weight_slice"],
        device=device,
        layer_dims=config["layer_dims"],
        mem_enabled=True
    ).to(device)
    # Load the pre-trained weights from the software model and use update_weights() to convert them to memristive sliced_weights
    mdoel_mem.load_state_dict(model.state_dict())
    mdoel_mem.update_weights()
    
    final_acc_mem = evaluate(mdoel_mem, test_loader, device)
    print(f"\nFinal test accuracy in memristive mode: {final_acc_mem:.2%}")

if __name__ == "__main__":
    main()
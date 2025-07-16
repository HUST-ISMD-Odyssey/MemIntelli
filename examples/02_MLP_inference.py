# -*- coding:utf-8 -*-
# @File  : 02_mlp_inference.py
# @Author: ZZW
# @Date  : 2025/2/20
"""Memintelli example 2: MLP inference using Memintelli.
This example demonstrates the usage of Memintelli with a simple MLP classifier that has been trained in software.
"""
import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.nn import functional as F

from memintelli.NN_layers.linear import LinearMem
from memintelli.pimpy.memmat_tensor import DPETensor

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
        input_paral_size: The size of the input data used for parallel computation, where (1, 32) here indicates that the input matrix is divided into 1×32 sub-inputs for parallel computation
        weight_paral_size: The size of the crossbar array used for parallel computation, where (32, 32) here indicates that the weight matrix is divided into 32x32 sub-arrays for parallel computation
        input_quant_gran: The quantization granularity of the input data
        weight_quant_gran: The quantization granularity of the weight data
    """
    def __init__(self, engine, input_slice, weight_slice, device, 
                 layer_dims=[784, 512, 128, 10], bw_e=None, mem_enabled=True, 
                 input_paral_size=(1, 32), weight_paral_size=(32, 32), 
                 input_quant_gran=(1, 64), weight_quant_gran=(64, 64)):
        super().__init__()
        self.layers = nn.ModuleList()
        self.flatten = nn.Flatten()
        self.engine = engine
        self.mem_enabled = mem_enabled
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            if mem_enabled is True:
                self.layers.append(
                    LinearMem(engine, in_dim, out_dim, input_slice, weight_slice,
                             device=device, bw_e=bw_e, input_paral_size=input_paral_size, weight_paral_size=weight_paral_size, 
                             input_quant_gran=input_quant_gran, weight_quant_gran=weight_quant_gran)
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

    def update_weight(self):
        """Convert the model weights (FP32) to PIM sliced_weights. 
        This function is very important for loading as well as updating pre-training weights in inference or training."""
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
                    model.update_weight()
                
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
    data_root = "/dataset/"   # Change this to your dataset directory
    batch_size = 256
    epochs = 5
    learning_rate = 0.001
    layer_dims = [784, 512, 128, 10]
    # Slicing configuration and INT/FP mode settings
    input_slice = (1, 1, 2)
    weight_slice = (1, 1, 2)
    bw_e = 8
    
    # Initialize components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_mnist(data_root, batch_size)
    
    # Initialize the software model with mem_enabled=False
    model = MNISTClassifier(
        engine=None,
        input_slice=input_slice,
        weight_slice=weight_slice,
        device=device,
        layer_dims=layer_dims,
        bw_e=bw_e,
        mem_enabled=False,      # Set mem_enabled=False for software model
    ).to(device)

    # Train the software model
    train_model(
        model,
        train_loader,
        test_loader,
        device,
        epochs=epochs,
        lr=learning_rate,
        mem_enabled=False
    )

    # Initialize memristive engine and model
    mem_engine = DPETensor(
        HGS=1e-5,                       # High conductance state
        LGS=1e-8,                       # Low conductance state
        write_variation=0.05,          # Write variation
        rate_stuck_HGS=0.005,          # Rate of stuck at HGS
        rate_stuck_LGS=0.005,          # Rate of stuck at LGS
        read_variation={0:0.05, 1:0.05, 2:0.05, 3:0.05},           # Read variation
        vnoise=0.05,                   # Random Gaussian noise of voltage
        rdac=2**2,                      # Number of DAC resolution 
        g_level=2**2,                   # Number of conductance levels
        radc=2**12
        )

    mdoel_mem = MNISTClassifier(
        engine=mem_engine,
        input_slice=input_slice,
        weight_slice=weight_slice,
        device=device,
        layer_dims=layer_dims,
        mem_enabled=True,
        input_paral_size=(1, 32), weight_paral_size=(32, 32), 
        input_quant_gran=(1, 64), weight_quant_gran=(64, 64)
    ).to(device)
    # Load the pre-trained weights from the software model and use update_weight() to convert them to memristive sliced_weights
    mdoel_mem.load_state_dict(model.state_dict())
    mdoel_mem.update_weight()
    
    final_acc_mem = evaluate(mdoel_mem, test_loader, device)
    print(f"\nFinal test accuracy in memristive mode: {final_acc_mem:.2%}")

if __name__ == "__main__":
    main()
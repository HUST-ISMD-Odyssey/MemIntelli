import torch
from torchvision import datasets
import torchvision.transforms as transforms

from NN_layers.linear import LinearMem
from pimpy.memmat_tensor import DPETensor
from tqdm import tqdm
import os
from time import time
import torch.nn as nn
import torch.nn.functional as F

#PATH_DATASET = os.path.join("data", "DATASET")
PATH_DATASET = os.path.join("D:/dataset/", "MNIST")

# Training parameters.
EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 0.001

def load_dataset():
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    trainset = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    testset = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

    return trainloader, testloader

class MLP_mem(torch.nn.Module):
    def __init__(self, engine, input_slice, weight_slice, device, bw_e=None, input_en=False):
        super(MLP_mem, self).__init__()
        self.fc1 = LinearMem(engine, 784, 512, input_slice, weight_slice, device=device, bw_e=bw_e,input_en=input_en)
        self.fc2 = LinearMem(engine, 512, 128, input_slice, weight_slice, device=device, bw_e=bw_e,input_en=input_en)
        self.fc3 = LinearMem(engine, 128, 10, input_slice, weight_slice, device=device, bw_e=bw_e,input_en=input_en)
        self.engine = engine

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

    def update_weight(self):
        self.fc1.update_weight()
        self.fc2.update_weight()
        self.fc3.update_weight()

def train(model, n_epochs, train_loader, test_loader, device,  mem_en=True):
    lossfunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in tqdm(range(n_epochs)):
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1)
            optimizer.zero_grad()
            output = model(images)
            loss = lossfunc(output, labels)
            loss.backward()
            optimizer.step()
            if  mem_en:
                model.update_weight()
            train_loss += loss.item() * images.size(0)
        train_loss = train_loss / len(train_loader.dataset)
        print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
        #print('Inference time: {:.6f}, Backward time: {:.6f}'.format(inference_time_sum, backward_time_sum))
        # 每遍历一遍数据集，测试一下准确率
        acc = test(model, test_loader, device)

def test(model,  test_loader=None, device=None):
    model.eval()
    correct = 0
    total = 0
    count = 0
    progress_bar = tqdm(test_loader, desc="Testing")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        images = images.view(images.size(0), -1)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        count += labels.size(0)

        current_accuracy = 100.0 * correct / total
        progress_bar.set_description(f"Testing (Current Accuracy: {current_accuracy:.2f}%)")
        
    final_accuracy =  correct / total
    return final_accuracy

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_dataset()
    engine = DPETensor(var=0.02,rdac=2**2,g_level=2**2,radc=2**12,quant_array_gran=(128,1),quant_input_gran=(1,128),paral_array_size=(64,1),paral_input_size=(1,64))
    in_slice_method = (1, 1, 2)
    weight_slice_method =  (1, 1, 2)
    model = MLP_mem(engine, in_slice_method, weight_slice_method, device, bw_e=None, input_en=True)
    model.to(device)
    train(model, EPOCHS, train_loader, test_loader, device, mem_en=True)
    test(model,  test_loader, device)
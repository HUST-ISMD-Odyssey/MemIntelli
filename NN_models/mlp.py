# -*- coding:utf-8 -*-
# @File  : mlp.py
# @Author: Zhou
# @Date  : 2024/5/8

import torch.nn as nn
import torch.nn.functional as F
from NN_layers import LinearMem

class MLP_2(nn.Module):
    def __init__(self):
        super(MLP_2, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

class MLP_3(nn.Module):
    def __init__(self):
        super(MLP_3, self).__init__()
        self.linear1 = nn.Linear(28*28, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        x = F.softmax(x, dim=1)
        return x

class MLP_2_mem(nn.Module):
    def __init__(self, engine, input_slice, weight_slice, device):
        super(MLP_2_mem, self).__init__()
        self.fc1 = LinearMem(engine, 28*28, 512, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device)
        self.fc2 = LinearMem(engine, 512, 10, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

    def update_weight(self):
        self.fc1.update_weight()
        self.fc2.update_weight()

class MLP_3_mem(nn.Module):
    def __init__(self, engine, input_slice, weight_slice, device):
        super(MLP_3_mem, self).__init__()
        self.fc1 = LinearMem(engine, 784, 512, input_slice, weight_slice, device=device)
        self.fc2 = LinearMem(engine, 512, 128, input_slice, weight_slice, device=device)
        self.fc3 = LinearMem(engine, 128, 10, input_slice, weight_slice, device=device)
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
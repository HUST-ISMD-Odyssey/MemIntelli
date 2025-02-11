# -*- coding:utf-8 -*-
# @File  : alexnet.py
# @Author: Zhou
# @Date  : 2024/5/8

import torch.nn as nn
import torch.nn.functional as F
from NN_layers import Conv2dMem, LinearMem

class AlexNet_ImageNet(nn.Module):
    def __init__(self):
        super(AlexNet_ImageNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, 5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.fc1 = nn.Linear(256*6*6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 3, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 3, stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 3, stride=2)
        x = x.view(-1, 256*6*6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

class AlexNet_ImageNet_mem(nn.Module):
    def __init__(self, engine, input_slice, weight_slice, device):
        super(AlexNet_ImageNet_mem, self).__init__()
        self.conv1 = Conv2dMem(engine, 3, 64, 11, stride=4, padding=2, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device)
        self.conv2 = Conv2dMem(engine, 64, 192, 5, padding=2, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device)
        self.conv3 = Conv2dMem(engine, 192, 384, 3, padding=1, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device)
        self.conv4 = Conv2dMem(engine, 384, 256, 3, padding=1, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device)
        self.conv5 = Conv2dMem(engine, 256, 256, 3, padding=1, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device)
        self.fc1 = LinearMem(engine, 256*6*6, 4096, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device)
        self.fc2 = LinearMem(engine, 4096, 4096, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device)
        self.fc3 = LinearMem(engine, 4096, 1000, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device)
        self.engine = engine

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 3, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 3, stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 3, stride=2)
        x = x.view(-1, 256*6*6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

    def update_weight(self):
        for m in self.modules():
            if isinstance(m, LinearMem) or isinstance(m, Conv2dMem):
                m.update_weight()


class AlexNet_Cifar(nn.Module):
    def __init__(self):
        super(AlexNet_Cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 7, stride=2, padding=2)
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.fc1 = nn.Linear(256*3*3, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 3, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 3, stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        # x = F.max_pool2d(x, 3, stride=2)
        x = x.view(-1, 256*3*3)
        x = F.dropout(F.relu(self.fc1(x)), 0.5, training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), 0.5, training=self.training)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

class AlexNet_Cifar_mem(nn.Module):
    def __init__(self, engine, input_slice, weight_slice, device):
        super(AlexNet_Cifar_mem, self).__init__()
        self.conv1 = Conv2dMem(engine, 3, 96, 7, stride=2, padding=2, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device)
        self.conv2 = Conv2dMem(engine, 96, 256, 5, padding=2, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device)
        self.conv3 = Conv2dMem(engine, 256, 384, 3, padding=1, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device)
        self.conv4 = Conv2dMem(engine, 384, 256, 3, padding=1, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device)
        self.conv5 = Conv2dMem(engine, 256, 256, 3, padding=1, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device)
        self.fc1 = LinearMem(engine, 256*2*2, 4096, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device)
        self.fc2 = LinearMem(engine, 4096, 4096, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device)
        self.fc3 = LinearMem(engine, 4096, 10, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 3, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 3, stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        # x = F.max_pool2d(x, 3, stride=2)
        x = x.view(-1, 256*2*2)
        x = F.dropout(F.relu(self.fc1(x)), 0.5, training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), 0.5, training=self.training)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

    def update_weight(self):
        for m in self.modules():
            if isinstance(m, LinearMem) or isinstance(m, Conv2dMem):
                m.update_weight()
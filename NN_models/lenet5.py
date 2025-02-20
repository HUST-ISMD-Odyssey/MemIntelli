# -*- coding:utf-8 -*-
# @File  : lenet5.py
# @Author: Zhou
# @Date  : 2024/5/8

import torch.nn as nn
import torch.nn.functional as F
from NN_layers import Conv2dMem, LinearMem

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x=self.conv1(x)
        x=F.relu(x)
        #x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16*4*4)
        x = self.fc1(x)
        x = F.relu(x)
        #x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

class LeNet5_mem(nn.Module):
    def __init__(self, engine, input_slice, weight_slice, device, bw_e=None, input_en=False):
        super(LeNet5_mem, self).__init__()
        self.conv1 = Conv2dMem(engine, 1, 6, 5, input_sli_med=input_slice,
                               weight_sli_med=weight_slice, device=device, bw_e=bw_e,input_en=input_en)
        self.conv2 = Conv2dMem(engine, 6, 16, 5, input_sli_med=input_slice,
                               weight_sli_med=weight_slice, device=device, bw_e=bw_e,input_en=input_en)
        self.fc1 = LinearMem(engine, 16*4*4, 120, input_sli_med=input_slice,
                             weight_sli_med=weight_slice, device=device, bw_e=bw_e,input_en=input_en)
        self.fc2 = LinearMem(engine, 120, 84, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device, bw_e=bw_e,input_en=input_en)
        self.fc3 = LinearMem(engine, 84, 10, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device, bw_e=bw_e,input_en=input_en)

    def forward(self, x):
        x=self.conv1(x)
        x=F.relu(x)
        #x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

    def update_weight(self):
        self.conv1.update_weight()
        self.conv2.update_weight()
        self.fc1.update_weight()
        self.fc2.update_weight()
        self.fc3.update_weight()
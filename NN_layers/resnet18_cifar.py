# -*- coding:utf-8 -*-
# @File  : resnet18.py
# @Author: Zhou
# @Date  : 2024/5/8

import torch.nn as nn
import torch.nn.functional as F
from NN_layers import Conv2dMem, LinearMem, SliceMethod

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,act='relu'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        if act == 'gelu':
            self.act = F.gelu
        else:
            self.act = F.relu

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            x = self.downsample(x)
        out += x
        out = self.act(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,act='relu'):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        if act == 'gelu':
            self.act = F.gelu
        else:
            self.act = F.relu
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, act=act)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, act=act)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, act=act)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, act=act)

        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, act):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, act))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18_cifar(block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=10,act='relu'):
    return ResNet(block, num_blocks, num_classes,act)

class BasicBlockMem(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes,  input_sli_med:SliceMethod, weight_sli_med:SliceMethod, engine, device, stride=1, bw_e=None, input_en=False,act='relu'):
        super(BasicBlockMem, self).__init__()
        self.conv1 = Conv2dMem(engine, in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                               input_sli_med=input_sli_med, weight_sli_med=weight_sli_med, device=device,bw_e=bw_e,input_en=input_en)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2dMem(engine, planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
                               input_sli_med=input_sli_med, weight_sli_med=weight_sli_med, device=device,bw_e=bw_e,input_en=input_en)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                Conv2dMem(engine, in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False,
                          input_sli_med=input_sli_med, weight_sli_med=weight_sli_med, device=device,bw_e=bw_e,input_en=input_en),
                nn.BatchNorm2d(self.expansion*planes)
            )
            # self.shortcut = True
            # self.conv3 = Conv2dMem(engine, in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False,
            #               input_sli_med=input_sli_med, weight_sli_med=weight_sli_med, device=device,bw_e=bw_e,input_en=input_en)
            # self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        if act == 'relu':
            self.act = F.relu
        else:
            self.act = F.gelu

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            x = self.downsample(x)
        out += x
        out = self.act(out)
        return out

class ResNetMem(nn.Module):
    def __init__(self, engine, block, num_blocks, input_slice, weight_slice, device, bw_e=None, input_en=False, act='relu', num_classes=10):
        super(ResNetMem, self).__init__()
        self.in_planes = 64

        self.conv1 = Conv2dMem(engine, 3, 64, kernel_size=3, input_sli_med=input_slice, weight_sli_med=weight_slice,
                               stride=1, padding=1, bias=False, device=device, bw_e=bw_e,input_en=input_en)
        self.bn1 = nn.BatchNorm2d(64)
        if act == 'gelu':
            self.act = F.gelu
        else:
            self.act = F.relu
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,
                                       input_sli_med=input_slice, weight_sli_med=weight_slice,
                                       engine=engine, device=device, bw_e=bw_e,input_en=input_en, act=act)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,
                                       input_sli_med=input_slice, weight_sli_med=weight_slice,
                                       engine=engine, device=device, bw_e=bw_e,input_en=input_en, act=act)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,
                                       input_sli_med=input_slice, weight_sli_med=weight_slice,
                                       engine=engine, device=device, bw_e=bw_e,input_en=input_en, act=act)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,
                                       input_sli_med=input_slice, weight_sli_med=weight_slice,
                                       engine=engine, device=device, bw_e=bw_e,input_en=input_en, act=act)
        self.fc = LinearMem(engine, 512 * block.expansion, num_classes,
                                input_sli_med=input_slice, weight_sli_med=weight_slice, device=device, bw_e=bw_e,input_en=input_en)

    def _make_layer(self, block, planes, num_blocks, stride, input_sli_med, weight_sli_med, engine, device, bw_e=None, input_en=False, act='relu'):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, input_sli_med, weight_sli_med, engine, device, stride, bw_e=bw_e, input_en=input_en, act=act))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def update_weight(self):
        for m in self.modules():
            if isinstance(m, LinearMem) or isinstance(m, Conv2dMem):
                m.update_weight()


def ResNet18_cifar_mem(engine, input_slice, weight_slice, device, bw_e=None, input_en=True,act='relu', num_classes=10):
    return ResNetMem(engine, BasicBlockMem, [2, 2, 2, 2],  input_slice, weight_slice, device, bw_e=bw_e, input_en=input_en ,act=act,num_classes=num_classes)
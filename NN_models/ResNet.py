# -*- coding:utf-8 -*-
# @File  : ResNet.py
# @Author: Zhou
# @Date  : 2024/4/1

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from NN_layers import Conv2dMem, LinearMem, SliceMethod

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, input):
        residual = input
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample:
            residual = self.downsample(residual)
        #print("x",x.shape,"residual",residual.shape)
        x += residual
        x = self.relu(x)
        return x

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, input):
        residual = input
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.downsample:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    # 224*224
    def __init__(self, block, num_layer, n_classes=1000, input_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_layer[0])
        self.layer2 = self._make_layer(block, 128, num_layer[1], 2)
        self.layer3 = self._make_layer(block, 256, num_layer[2], 2)
        self.layer4 = self._make_layer(block, 512, num_layer[3], 2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(block.expansion * 512, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block, out_channels, num_block, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet_zoo(pretrained=False, model_name='resnet18',**kwargs):
    if model_name == 'resnet18':
        model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    elif model_name == 'resnet34':
        model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    elif model_name == 'resnet50':
        model = ResNet(BottleNeck, [3, 4, 6, 3], **kwargs)
    elif model_name == 'resnet101':
        model = ResNet(BottleNeck, [3, 4, 23, 3], **kwargs)
    elif model_name == 'resnet152':
        model = ResNet(BottleNeck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls[model_name]))
    return model

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

class BasicBlockMem(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, input_sli_med:SliceMethod, weight_sli_med:SliceMethod, engine, device, stride=1, bw_e=None, input_en=False,downsample=None):
        super(BasicBlockMem, self).__init__()
        self.conv1 = Conv2dMem(engine, in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False,
                               input_sli_med=input_sli_med, weight_sli_med=weight_sli_med, device=device,bw_e=bw_e,input_en=input_en)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = Conv2dMem(engine, out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
                                 input_sli_med=input_sli_med, weight_sli_med=weight_sli_med, device=device,bw_e=bw_e,input_en=input_en)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, input):
        residual = input
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x

class BottleNeckMem(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, input_sli_med:SliceMethod, weight_sli_med:SliceMethod, engine, device, stride=1, bw_e=None, input_en=False,downsample=None):
        super(BottleNeckMem, self).__init__()
        self.conv1 = Conv2dMem(engine, in_channels, out_channels, kernel_size=1, stride=1, bias=False,
                              input_sli_med=input_sli_med, weight_sli_med=weight_sli_med,device=device, bw_e=bw_e, input_en=input_en)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = Conv2dMem(engine, out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False,
                              input_sli_med=input_sli_med, weight_sli_med=weight_sli_med, device=device, bw_e=bw_e, input_en=input_en)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = Conv2dMem(engine, out_channels, out_channels*self.expansion, kernel_size=1, stride=1, bias=False,
                              input_sli_med=input_sli_med, weight_sli_med=weight_sli_med, device=device, bw_e=bw_e, input_en=input_en)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, engine, input):
        residual = input
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x

class ResNetMem(nn.Module):
    def __init__(self, engine,block, num_layer,input_slice, weight_slice, device=None, bw_e=None, input_en=False,  n_classes=1000, input_channels=3):
        super(ResNetMem,self).__init__()
        self.in_channels = 64
        self.conv1 = Conv2dMem(engine, 3, 64, kernel_size=7, input_sli_med=input_slice, weight_sli_med=weight_slice,
                               stride=2, padding=3, bias=False, device=device, bw_e=bw_e,input_en=input_en)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_layer[0], stride=1,
                                       input_sli_med=input_slice, weight_sli_med=weight_slice,
                                       engine=engine, device=device, bw_e=bw_e,input_en=input_en)
        self.layer2 = self._make_layer(block, 128, num_layer[1], stride=2,
                                       input_sli_med=input_slice, weight_sli_med=weight_slice,
                                       engine=engine, device=device, bw_e=bw_e,input_en=input_en)
        self.layer3 = self._make_layer(block, 256, num_layer[2], stride=2,
                                        input_sli_med=input_slice, weight_sli_med=weight_slice,
                                        engine=engine, device=device, bw_e=bw_e,input_en=input_en)
        self.layer4 = self._make_layer(block, 512, num_layer[3], stride=2,
                                        input_sli_med=input_slice, weight_sli_med=weight_slice,
                                        engine=engine, device=device, bw_e=bw_e,input_en=input_en)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = LinearMem(engine, 512 * block.expansion, n_classes,
                                input_sli_med=input_slice, weight_sli_med=weight_slice, device=device, bw_e=bw_e,input_en=input_en)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block, planes, num_blocks, stride, input_sli_med, weight_sli_med, engine, device, bw_e=None, input_en=False):
        downsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2dMem(engine, self.in_channels, planes * block.expansion, kernel_size=1, stride=stride, bias=False,
                          input_sli_med=input_sli_med, weight_sli_med=weight_sli_med, device=device, bw_e=bw_e,input_en=input_en),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, planes, input_sli_med, weight_sli_med, engine, device, stride, bw_e=bw_e, input_en=input_en,downsample=downsample))
        self.in_channels = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, planes, input_sli_med, weight_sli_med, engine, device, stride=1, bw_e=bw_e, input_en=input_en))
        return nn.Sequential(*layers)
    
    def forward(self,input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def update_weight(self):
        for m in self.named_children():
            if isinstance(m, LinearMem) or isinstance(m, Conv2dMem):
                m.update_weight()

def resnet_zoo_mem(engine, input_slice, weight_slice, device, bw_e=None, input_en=True,pretrained=False, model_name='resnet18'):
    if model_name == 'resnet18':
        model = ResNetMem(engine,BasicBlockMem, [2, 2, 2, 2],input_slice, weight_slice, device, bw_e=bw_e, input_en=input_en)
    elif model_name == 'resnet34':
        model = ResNetMem(engine,BasicBlockMem, [3, 4, 6, 3],input_slice, weight_slice, device, bw_e=bw_e, input_en=input_en)
    elif model_name == 'resnet50':
        model = ResNetMem(engine,BottleNeckMem, [3, 4, 6, 3],input_slice, weight_slice, device, bw_e=bw_e, input_en=input_en)
    elif model_name == 'resnet101':
        model = ResNetMem(engine,BottleNeckMem, [3, 4, 23, 3],input_slice, weight_slice, device, bw_e=bw_e, input_en=input_en)
    elif model_name == 'resnet152':
        model = ResNetMem(engine,BottleNeckMem, [3, 8, 36, 3],input_slice, weight_slice, device, bw_e=bw_e, input_en=input_en)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls[model_name]))
    return model
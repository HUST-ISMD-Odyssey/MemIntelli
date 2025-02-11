# -*- coding:utf-8 -*-
# @File  : VGGNet.py
# @Author: Zhou
# @Date  : 2024/4/1

from typing import Any, cast, Dict, List, Optional, Union
import torch.nn as nn
import torch.nn.functional as F
from .convolution import Conv2dMem
from .linear import LinearMem
import torch
import torch.utils.model_zoo as model_zoo
from torchvision.models import vgg16_bn

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

cfgs: Dict[str, List[Union[str, int]]] = {
    'vgg11_bn': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13_bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16_bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19_bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, cfg='vgg16_bn', num_classes=1000):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfgs[cfg], batch_norm=True)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def _make_layers(self,cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x

def vgg_zoo(pretrained=False,model_name='vgg16_bn', num_classes=1000):
    model = VGG(cfg=model_name, num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls[model_name]))
    return model

def vgg_zoo_mem(engine, input_slice, weight_slice, device, bw_e=None, input_en=True, pretrained=False,model_name='vgg16_bn', num_classes=1000):
    model = VGG_mem(engine, input_slice, weight_slice, device,  bw_e=bw_e, input_en=input_en, cfg=model_name,num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls[model_name]))
    return model

class VGG_mem(nn.Module):
    def __init__(self, engine, input_slice, weight_slice, device, bw_e=None, input_en=False, cfg='vgg16_bn', num_classes=1000):
        super(VGG_mem, self).__init__()
        self.features = self._make_layers(engine=engine, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device,bw_e=bw_e,input_en=input_en,
                                          cfg=cfgs[cfg], batch_norm=True)
        self.classifier = nn.Sequential(
            LinearMem(engine, 512 * 7 * 7, 4096,
                      input_sli_med=input_slice, weight_sli_med=weight_slice, device=device, bw_e=bw_e,input_en=input_en),
            nn.ReLU(),
            nn.Dropout(),
            LinearMem(engine, 4096, 4096,
                      input_sli_med=input_slice, weight_sli_med=weight_slice, device=device, bw_e=bw_e,input_en=input_en),
            nn.ReLU(),
            nn.Dropout(),
            LinearMem(engine, 4096, num_classes,
                      input_sli_med=input_slice, weight_sli_med=weight_slice, device=device, bw_e=bw_e,input_en=input_en),
        )

    def _make_layers(self,engine,input_sli_med, weight_sli_med, device, bw_e=None, input_en=False, cfg='vgg16_bn', batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = Conv2dMem(engine, in_channels, v, kernel_size=3, bias=True,  padding=1,
                                    input_sli_med=input_sli_med, weight_sli_med=weight_sli_med, device=device, bw_e=bw_e,input_en=input_en)
                #nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x

    def update_weight(self):
        for m in self.modules():
            if isinstance(m, LinearMem) or isinstance(m, Conv2dMem):
                m.update_weight()

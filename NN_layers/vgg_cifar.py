# -*- coding:utf-8 -*-
# @File  : vgg16.py
# @Author: Zhou
# @Date  : 2024/5/8
import torch.nn as nn
import torch.nn.functional as F
from NN_layers import Conv2dMem, LinearMem
from typing import Union, List, Dict, Any, cast
from torch.hub import load_state_dict_from_url

cifar10_pretrained_weight_urls = {
    'vgg11_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg11_bn-eaeebf42.pt',
    'vgg13_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg13_bn-c01e4a43.pt',
    'vgg16_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg16_bn-6ee7ea24.pt',
    'vgg19_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg19_bn-57191229.pt',
}

cifar100_pretrained_weight_urls = {
    'vgg11_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg11_bn-57d0759e.pt',
    'vgg13_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg13_bn-5ebe5778.pt',
    'vgg16_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg16_bn-7d8c4031.pt',
    'vgg19_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg19_bn-b98f7bd7.pt',
}

cfgs: Dict[str, List[Union[str, int]]] = {
    'vgg11_bn': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13_bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16_bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19_bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_cifar(nn.Module):
    def __init__(self, cfg='vgg16_bn', num_classes=10):
        super(VGG_cifar, self).__init__()
        #cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.features = self._make_layers(cfgs[cfg], batch_norm=True)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
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

def vgg_cifar_zoo(pretrained=False,model_name='vgg16_bn', num_classes=10):
    model = VGG_cifar(cfg=model_name, num_classes=num_classes)
    if pretrained:
        model_urls = cifar10_pretrained_weight_urls if num_classes == 10 else cifar100_pretrained_weight_urls
        model.load_state_dict(load_state_dict_from_url(model_urls[model_name]))
    return model

def vgg_cifar_zoo_mem(engine, input_slice, weight_slice, device, bw_e=None, input_en=True,pretrained=False,model_name='vgg16_bn', num_classes=10):
    model = VGG_cifar_mem(engine, input_slice, weight_slice, device,  bw_e=bw_e, input_en=input_en, cfg=model_name,num_classes=num_classes)
    if pretrained:
        model_urls = cifar10_pretrained_weight_urls if num_classes == 10 else cifar100_pretrained_weight_urls
        model.load_state_dict(load_state_dict_from_url(model_urls[model_name]))
    return model

class VGG_cifar_mem(nn.Module):
    def __init__(self, engine, input_slice, weight_slice, device, bw_e=None, input_en=False, cfg='vgg16_bn', num_classes=10):
        super(VGG_cifar_mem, self).__init__()
        self.features = self._make_layers(engine=engine, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device,bw_e=bw_e,input_en=input_en,
                                          cfg=cfgs[cfg], batch_norm=True)
        self.classifier = nn.Sequential(
            LinearMem(engine, 512, 512,
                      input_sli_med=input_slice, weight_sli_med=weight_slice, device=device, bw_e=bw_e,input_en=input_en),
            #nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            LinearMem(engine, 512, 512,
                      input_sli_med=input_slice, weight_sli_med=weight_slice, device=device, bw_e=bw_e,input_en=input_en),
            #nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            LinearMem(engine, 512, num_classes,
                      input_sli_med=input_slice, weight_sli_med=weight_slice, device=device, bw_e=bw_e,input_en=input_en),
            #nn.Linear(512, num_classes),
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
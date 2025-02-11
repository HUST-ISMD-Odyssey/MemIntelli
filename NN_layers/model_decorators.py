# -*- coding:utf-8 -*-
# @File  : model_decorators.py
# @Author: Zhou
# @Date  : 2024/5/17

# this file is used to define the decorators for the model
# the decorators are used to follow the structure of the model
# and to make the model more readable

from functools import wraps

def record_linear(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.params[self.__class__.__name__] = {
            'type': 'linear',
            'in_dim': args[0].shape,
            'out_dim': [args[0], self.out_features],
            'in_features': self.in_features,
            'out_features': self.out_features,
            'bias': self.bias,
            'input_sli_med': self.input_sli_med,
            'len_input_sli': len(self.input_sli_med),
            'weight_sli_med': self.weight_sli_med,
            'len_weight_sli': len(self.weight_sli_med)
        }
        return func(self, *args, **kwargs)
    return wrapper

def record_conv2d(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.params[self.__class__.__name__] = {
            'type': 'conv',
            'in_dim': args[0].shape,
            'out_dim': [args[0], self.out_channels],
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'dilation': self.dilation,
            'bias': self.bias,
            'input_sli_med': self.input_sli_med,
            'len_input_sli': len(self.input_sli_med),
            'weight_sli_med': self.weight_sli_med,
            'len_weight_sli': len(self.weight_sli_med)
        }
        return func(self, *args, **kwargs)
    return wrapper

def record_conv1d(func):
    pass

def record_maxpool2d(func):
    pass

def record_activation(func):
    pass

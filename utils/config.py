# -*- coding: utf-8 -*-
# @Time    : 2022/6/29 11:15
# @Author  : Zhou
# @FileName: config.py
# @Software: PyCharm

import yaml

# def get_config(filename, config_level, **kwargs):
#     try:
#         with open(filename, encoding='utf-8') as file:
#             data = yaml.bat.safe_load(file)
#             try:
#                 if config_level == 'device_level':
#                     # kwargs include 'model' and determines the model type of the memristor
#                     assert len(kwargs) == 1
#                     return data['device_level'][kwargs['model']]
#                 elif config_level == 'crossbar_level':
#                     return data['crossbar_level']
#                 elif config_level == 'circuit_level':
#                     # kwargs include the keyword 'circuit_block' and determines the circuit type
#                     base_params = data['circuit_level']['base_params']
#                     try:
#                         circuit_params = data['circuit_level'][kwargs['circuit_block']]
#                         base_params.update(circuit_params)
#                     except:
#                         pass
#                     return base_params
#                 elif config_level == 'system_level':
#                     # return the sub-levels in the system level
#                     if kwargs:
#                         return data['system_level'][kwargs['model']]
#                     else:
#                         return data['system_level']
#             except:
#                 raise ValueError('Do not have config level {}'.format(config_level))
#     except:
#         raise ValueError('Do not have config file {}'.format(filename))

def get_config(filename, block=None):
    try:
        with open(filename, encoding='utf-8') as file:
            data = yaml.safe_load(file)
        if block is None:
            return data
        else:
            try:
                return data[block]
            except KeyError:
                return data

    except:
        raise ValueError('Do not have config file {}'.format(filename))

def set_params(config_path, ):
    pass
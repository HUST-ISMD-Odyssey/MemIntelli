# -*- coding:utf-8 -*-
# @File  : data_formats.py
# @Author: Zhou
# @Date  : 2024/1/19
import copy
import torch
#torch.set_default_dtype(torch.float64)

class DividedMat(object):
    '''
    the class DividedMat is used to store the divided mat
    key properties:
        mat_addr: list, the address of the divided mat (StorageData)
        submat_id_table: list, ids of the divided mat, arranged by the original mat
            for example, the original mat is divided into 2*2, the submat_id_table is
            {0: [0, 1], 1: [2, 3]}
        divided_row_num: the number of the divided row
        divided_col_num: the number of the divided col
    '''
    def __init__(self, row_size, col_size, submat_id):
        self.row_size = row_size
        self.col_size = col_size
        # store the address of the divided mat
        self.mat_addr = []
        self.submat_id_table = submat_id
        self.divided_row_num = len(submat_id[0])
        self.divided_col_num = len(submat_id)

        self._iter_id = -1

    def __getitem__(self, item):
        return self.mat_addr[item]

    def __iter__(self):
        return self

    def __next__(self):
        if self._iter_id + 1 < len(self):
            self._iter_id += 1
            return self.mat_addr[self._iter_id]
        else:
            self._iter_id = -1
            raise StopIteration

    def __repr__(self):
        return 'divided mat'

    def __len__(self):
        return len(self.mat_addr)

    def add_addr(self, submat_id, addr, data, slice_len, used_space=None):
        # use the submat_id to find the location of the divided mat
        """
        :param submat_id: the id of the divided mat
        :param slice_idx: the index of the slice
        :param addr: the mapped address of the divided mat
        :param data: the pseudo mat data
        :param used_space: if partial storage, the used space
        :return:
        """
        if len(self.mat_addr) == submat_id + 1:
            self.mat_addr[submat_id]._update_addr(addr, used_space)
        else:
            self.mat_addr.append(StorageData(submat_id, addr, data, slice_len, used_space))


class PseudoMatrix(object):
    """
    build the fake matrix for the compiler simulation
    """
    def __init__(self, *args):
        if type(args[0]) is list:
            args = args[0]
        if len(args) == 1:
            self.row_size = 1
            self.col_size = args[0]
        elif len(args) == 2:
            self.row_size = args[0]
            self.col_size = args[1]
        elif len(args) == 3:
            self.row_size = args[1]
            self.col_size = args[2]
            self.in_channel = args[0]
        elif len(args) == 4:
            self.row_size = args[2]
            self.col_size = args[3]
            self.in_channel = args[1]
            self.batch_size = args[0]
        else:
            raise ValueError('The input args is not correct')

        # used for the divided matrix
        self.up = None
        self.down = None
        self.left = None
        self.right = None
        self.index = -1
        self.full_flag = False

    def __len__(self):
        return self.row_size

    @property
    def shape(self):
        if hasattr(self, 'batch_size'):
            return self.batch_size, self.in_channel, self.row_size, self.col_size
        elif hasattr(self, 'in_channel'):
            return self.in_channel, self.row_size, self.col_size
        else:
            return self.row_size, self.col_size

    @property
    def size(self):
        if hasattr(self, 'batch_size'):
            return self.row_size * self.col_size * self.in_channel * self.batch_size
        elif hasattr(self, 'in_channel'):
            return self.row_size * self.col_size * self.in_channel
        else:
            return self.row_size * self.col_size

    def __sizeof__(self):
        pass

    def __repr__(self):
        if hasattr(self, 'batch_size'):
            return 'pseudo matrix with shape:{}'.format(( self.batch_size, self.in_channel, self.row_size, self.col_size))
        elif hasattr(self, 'in_channel'):
            return 'pseudo matrix with shape:{}'.format((self.in_channel, self.row_size, self.col_size))
        else:
            return 'pseudo matrix with shape:{}'.format((self.row_size, self.col_size))

    def __getitem__(self, item):
        return PseudoMatrix(item)

    def set_location(self, up, down, left, right):
        # set the location of the divided matrix
        # keeps the id of the divided matrix
        self.up = up
        self.down = down
        self.left = left
        self.right = right

    def set_index(self, index):
        self.index = index

    def add_row(self, row):
        self.row_size += row

    def add_col(self, col):
        self.col_size += col

class PseudoLayer():
    def __init__(self, data:PseudoMatrix, input_sli_method=None, weight_sli_med=None):
        self.data = data
        self.input_slice_method = input_sli_method
        self.weight_slice_method = weight_sli_med
        self.input_slice_method_len = len(input_sli_method)
        self.weight_slice_method_len = len(weight_sli_med)


class PseudoModel():
    '''
    the pseudo model is used to store the model
    '''
    def __init__(self):
        self.model = {}

    def __getitem__(self, item):
        return self.model[item]

    def __setitem__(self, key, value):
        self.model[key] = value

    def __repr__(self):
        return 'pseudo model'

    def __len__(self):
        return len(self.model)

    def layers(self):
        return self.model.keys()

    def items(self):
        return self.model.items()

    def add_linear_layer(self, name, in_features, out_features, bias, input_sli_med, weight_sli_med):
        self.model[name] = {
            'type': 'linear',
            'in_dim': in_features,
            'out_dim': out_features,
            'bias': bias,
            'input_sli_med': input_sli_med,
            'len_input_sli': len(input_sli_med),
            'weight_sli_med': weight_sli_med,
            'len_weight_sli': len(weight_sli_med)
        }

    def add_conv2d_layer(self, name, in_channels, out_channels, kernel_size, input_sli_med, weight_sli_med,
                         stride, padding, dilation, bias):
        self.model[name] = {
            'type':'conv2d',
            'in_dim':None,
            'out_dim':None,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'input_sli_med': input_sli_med,
            'weight_sli_med': weight_sli_med,
            'len_input_sli': len(input_sli_med),
            'len_weight_sli': len(weight_sli_med),
            'stride': stride,
            'padding': padding,
            'dilation': dilation,
            'bias': bias
        }

    def add_pooling_layer(self, name, kernel_size, stride, padding, dilation, pool_type='max'):
        self.model[name] = {
            'type':'pooling',
            'in_dim':None,
            'out_dim':None,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'dilation': dilation,
            'pool_type': pool_type
        }

    def add_activation_layer(self, name, activation_type):
        self.model[name] = {
            'type': 'activation',
            'activation_type': activation_type
        }

    def add_flatten_layer(self, name):
        self.model[name] = {
            'type': 'flatten'
        }

    def add_dropout_layer(self, name, p):
        self.model[name] = {
            'type': 'dropout',
            'p': p
        }

    def add_batchnorm_layer(self, name, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        self.model[name] = {
            'type': 'batchnorm',
            'num_features': num_features,
            'eps': eps,
            'momentum': momentum,
            'affine': affine,
            'track_running_stats': track_running_stats
        }

    def forward(self, x):
        pass


class StorageData():
    """
    是一个数据结构，保存的是存储子矩阵的索引，存储子矩阵的地址，存储子矩阵的数据
    only used in the DividedMat.add_addr
    the stored data format is the submat_id, the address, the data, the slice_len, the used_space
    Example:
        submat_id:0,  data:(128 64)pseudo_matrix, slice_len:2, addr:['tile1_PU0_0', None, 'tile1_PU0_1', None]
    """
    def __init__(self, submat_id, addr, data, slice_len, used_space=None):
        self.submat_id = submat_id
        # address is composed of the block id and the location of the submat
        # if used space is None, the mat use the full address space
        self.addr = [addr, used_space]
        self.slice_len = slice_len
        self.data = data

    def __repr__(self):
        return 'submat_id:{}'.format(self.submat_id)

    def __len__(self):
        return self.slice_len

    def __iter__(self):
        return self

    def __next__(self):
        if self._iter_id + 1 < len(self):
            self._iter_id += 1
            return self.addr[self._iter_id * 2], self.addr[self._iter_id * 2 + 1]
        else:
            self._iter_id = -1
            raise StopIteration

    def __getitem__(self, item):
        return self.addr[item: item + 2]

    # @property
    # def block_id(self):
    #     return int(re.findall(r'B(.*?)_', self.addr)[0])

    def _update_addr(self, addr, used_space):
        self.addr += [addr, used_space]


class SlicedData():
    """
    record the key attributes of the sliced data
    data: the input data with quantization
    max_data: the max data of the input data, (divided_num, 1, 1)
    slice_method: the slice method of the data, tuple
    sliced_data: the sliced data of the input data, (divided_num, len(slice_method), row, col)
    sliced_weights: the weights of each slice for the sliced data, (len(slice_method),)
    sliced_max_weights: the max weights of each slice for the sliced data, (len(slice_method),)
    sliced_data_recalled: the flag to record the sliced data is calculated or not

    """
    def __init__(self, slice_method:torch.Tensor, bw_e=None, slice_data_flag=False,device=None):
        """
        the sliced data for the data slicing method with quantization
        :param data: the input data
        :param slice_method: the data slicing method, bit width of each slice, tuple
        :param bw_e: the bit width of the exponent,
                    if None, the exponent is not used, and the SlicedData is the uint type, the sign is the first bit
                    if not None, the SlicedData is fp type, the exponent is the last several bits
        :param device: use cpu or gpu, default is cpu (None)
        """
        self.bw_e = bw_e
        self.slice_data_flag = slice_data_flag
        self.slice_method = slice_method
        self.device = torch.device('cpu') if device is None else device
        self.shape = None
        self._init_data(slice_method, bw_e, device)

    def _init_data(self, slice_method:torch.Tensor, bw_e, device):
        assert slice_method[0] == 1, 'the first slice should be 1'
        if bw_e is None:
            # optimize the calculation of the sliced_max_weights
            self.sliced_max_weights = torch.zeros(len(slice_method), device=device)
            self.sliced_weights = torch.zeros(len(slice_method), device=device)
            temp_s, i = 0, 0
            for slice in slice_method.flip(0):
                self.sliced_max_weights[i] = 2 ** slice - 1
                self.sliced_weights[i] = 2 ** temp_s
                temp_s += slice
                i += 1
            self.sliced_weights[-1] *= -1
        # fp type
        else:
            self.sliced_max_weights = torch.zeros(len(slice_method), device=device)
            self.sliced_weights = torch.zeros(len(slice_method), device=device)
            temp_s, i = 0, 0
            for slice in slice_method.flip(0):
                self.sliced_max_weights[i] = 2 ** slice - 1
                self.sliced_weights[i] = 2 ** temp_s
                temp_s += slice
                i += 1
            self.sliced_weights[-1] *= -1

    def __repr__(self):
        return 'sliced data with slice_method:{}'.format(self.slice_method)

    def __len__(self):
        return len(self.slice_method)

    # @property
    # def shape(self):
    #     return self.quantized_data.shape

    def t(self):
        copy_ = copy.deepcopy(self)
        copy_.sliced_data = self.sliced_data.transpose(-4, -5)
        copy_.quantized_data = self.quantized_data.T
        copy_.max_data = self.max_data.transpose(0,1)
        return copy_

    def size(self):
        return self.quantized_data.size()

    def slice_data_imp(self, engine, data):
        """
        implement the localized slicing of the data
        :param engine: dot product engine, DPETensor
        :param data: tensor, 2D or 3D, if 2D, the shape is (row, col), if 3D, the shape is (batch, row, col)
        :param transpose: if True, the data is sliced by the column, if False, the data is sliced by the row
        :return:
        """
        #
        data = data.to(engine.device)
        self.sliced_data, self.quantized_data, self.max_data, self.e_bias = engine.slice_data(data,
                                                                                self.slice_method,
                                                                                self.bw_e,
                                                                                self.slice_data_flag
                                                                                 )
        self.shape = self.quantized_data.shape

class SliceMethod():
    def __init__(self, slice_method, bw_e=None):
        self.slice_method = slice_method
        self.bw_e = bw_e

    def __len__(self):
        return len(self.slice_method)

if __name__ == "__main__":
    a = PseudoMatrix(10, 10)
    b = PseudoMatrix(3, 5, 10, 10)
    print(a)
    print(b.shape[0])
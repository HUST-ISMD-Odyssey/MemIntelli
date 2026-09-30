# -*- coding:utf-8 -*-
# @File  : data_formats.py
# @Author: Zhou
# @Date  : 2024/1/19
import copy
import torch
import math
from memintelli.pimpy.utils import quant_map_tensor, bfp_map_tensor

class SlicedData(object):
    """
    record the key attributes of the sliced data
    data: the input data with quantization
    max_data: the max data of the input data, (divided_num, 1, 1)
    slice_method: the slice method of the data, tuple
    sliced_data: the sliced data of the input data, (divided_num, len(slice_method), row, col)
    sliced_weights: the weights of each slice for the sliced data, (len(slice_method),)
    sliced_max_weights: the max weights of each slice for the sliced data, (len(slice_method),)
    """

    def __init__(self, slice_method: torch.Tensor, bw_e=None, is_weight=False, paral_size = (64, 64),
                 quant_gran=None, device=None, inference=False):
        """
        the sliced data for the data slicing method with quantization
        :param slice_method: the data slicing method, bit width of each slice,
        :param bw_e: the bit width of the exponent,
                    if None, the exponent is not used, and the SlicedData is the uint type, the sign is the first bit
                    if not None, the SlicedData is bfp type, the exponent is the last several bits
        :param is_weight: if True, slice the input data, if False, slice the weight data
        :param device: use cpu or gpu, default is cpu (None)
        :param inference: the flag to record the sliced data is used in the inference model or not
        """
        self.bw_e = bw_e
        self.is_weight = is_weight
        self.device = torch.device('cpu') if device is None else torch.device(device)
        self.shape = None
        self.inference = inference
        # if inference is True, the sliced data of weight keeps the conductance of G
        self.G = None
        self.G_indices = None  # compressed uint8 level indices (4x smaller than float32 G)
        self.paral_size = paral_size
        if quant_gran is None:
            self.quant_gran = paral_size
        else:
            self.quant_gran = quant_gran
        if slice_method[0] != 1:
            raise ValueError('The first bit of the slice method should be 1')
        self.slice_method = slice_method
        self.device = torch.device('cpu') if device is None else torch.device(device)
        self.shape = None

        self.sliced_data = None
        self.quantized_data = None
        self.max_data = None
        self.e_bias = None

        self.sliced_max_weights = torch.empty(len(slice_method), device=self.device)
        self.sliced_weights = torch.empty(len(slice_method), device=self.device)
        self._init_data(slice_method, self.device)

    def _init_data(self, slice_method: torch.Tensor, device):
        assert slice_method[0] == 1, 'the first slice should be 1'
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

    def __repr__(self):
        return 'sliced data with slice_method:{}'.format(self.slice_method)

    def __len__(self):
        return len(self.slice_method)

    def t(self):
        copy_ = copy.deepcopy(self)
        copy_.max_data = self.max_data.transpose(0,1)
        if self.is_weight:
            if self.G is not None:
                copy_.G = self.G.transpose(-4, -5)
            if self.G_indices is not None:
                copy_.G_indices = self.G_indices.transpose(-4, -5)
        if self.inference:
            copy_.sliced_data = None
            copy_.quantized_data = None
        else:
            copy_.sliced_data = self.sliced_data.transpose(-4, -5)
            copy_.quantized_data = self.quantized_data.T
        return copy_

    def size(self):
        return self.quantized_data.size()

    def compress_G(self, engine):
        """Compress float32 G to uint8 level indices for memory-efficient inference.

        ~4x memory savings (float32 → uint8). Only valid when engine.write_variation == 0,
        because write variation adds continuous noise that cannot be losslessly compressed.
        Stuck faults are handled correctly (they map to level 0 or g_level-1).

        Uses in-place operations to minimize peak memory during compression.
        Peak = G (float32) + G_indices (uint8) ≈ 1.25x of G.

        Args:
            engine: DPETensor engine with conductance parameters.
        """
        if self.G is None:
            return
        # In-place conversion: G → level indices (reuses G's memory)
        self.G.sub_(engine.LGS)            # G = G - LGS
        self.G.div_(engine.Q_G)            # G = (G - LGS) / Q_G → level index (float)
        self.G.round_()                     # round to nearest integer level
        self.G.clamp_(0, engine.g_level - 1)  # clamp to valid range
        # Convert to compact integer type (new allocation, but much smaller)
        if engine.g_level <= 256:
            self.G_indices = self.G.to(torch.uint8)
        else:
            self.G_indices = self.G.to(torch.int16)
        self.G = None  # free float32 tensor

    def slice_data_imp(self, engine, data):
        """
        implement the localized slicing of the data, and apply mapping
        :param engine: dot product engine, DPETensor
        :param data: tensor, 2D or 3D, if 2D, the shape is (row, col), if 3D, the shape is (batch, row, col)
        :return:
        """
        data = data.to(engine.device)
        # Synchronize all internal tensors to the computation device.
        # This handles the case where the model is on a different device than the engine
        # (e.g., model on cuda:1 but engine on cuda:0).
        compute_device = data.device
        if self.slice_method.device != compute_device:
            self.slice_method = self.slice_method.to(compute_device)
            self.sliced_max_weights = self.sliced_max_weights.to(compute_device)
            self.sliced_weights = self.sliced_weights.to(compute_device)
            self.device = compute_device
        self._slice_data(data)
        self.shape = data.shape
        if self.is_weight:
            max_weights = self.sliced_max_weights.reshape(1, 1, -1, 1, 1)
            self.G = engine._num2G(self.sliced_data, max_weights)
        if self.inference:
            # In inference mode, quantized_data is not needed (only used for backward).
            # For weights: sliced_data was only needed to compute G, so free it.
            # For inputs: sliced_data is still needed for the forward dot product.
            self.quantized_data = None
            if self.is_weight:
                self.sliced_data = None

    def _slice_data(self, mat: torch.Tensor):
        """
        Slices the input or weight data using the specified method.

        Parameters:
            mat (torch.Tensor): Data to be sliced.

        Returns:
            data_int (torch.Tensor): the quantized data, the shape is (num_divide_row_a, num_divide, num_slice ,m , n)
                                    or (batch, num_divide_row_a, num_divide, num_slice ,m , n)
            mat_data (torch.Tensor): the data quantized by the slice method, the shape is the same as the data
            max_mat (torch.Tensor): the max value of the data for each quantization granularity,
                        the shape is (num_divide_row_a, num_divide, 1, 1) or (batch, num_divide_row_a, num_divide, 1, 1)
            e_bias (torch.Tensor): the bias of the exponent for each quantization granularity,
                        the shape is (num_divide_row_a, num_divide, 1, 1) or (batch, num_divide_row_a, num_divide, 1, 1)
        """
        # Convert 2d to 3d makes it easier to follow along with the process
        unsqueezed = False

        if len(mat.shape) == 2:
            mat = mat.unsqueeze(0)
            unsqueezed = True

        # Quantization and parallelization parameters
        # todo decouple the quantization params from the DPETensor class
        quant_gran = self.quant_gran
        paral_size = self.paral_size

        # Decode the quantization granularity
        if quant_gran == "per-matrix":
            quant_gran = mat.shape[1:]
        elif quant_gran == "per-row":
            quant_gran = (1, mat.shape[2])
        elif quant_gran == "per-col":
            quant_gran = (mat.shape[1], 1)
        else:
            quant_gran = quant_gran

        quant_gran = list(quant_gran)
        if not isinstance(self.quant_gran, str) and any(g % p for g, p in zip(quant_gran, paral_size)):
            raise ValueError("Legacy quant_gran cannot be rounded up silently; use SimulationEngine for column scales")
        if self.quant_gran in ("per-row", "per-col") and any(g < p for g, p in zip(quant_gran, paral_size)):
            raise ValueError("Legacy per-row/per-col scales cannot be enlarged silently; use SimulationEngine")
        # extend quant_gran to an integer multiple of paral_size

        quant_gran[0] = math.ceil(quant_gran[0] / paral_size[0]) * paral_size[0]
        quant_gran[1] = math.ceil(quant_gran[1] / paral_size[1]) * paral_size[1]

        # check the number of the quantization granularity
        num_gran_row = math.ceil(mat.shape[1] / quant_gran[0])
        num_gran_col = math.ceil(mat.shape[2] / quant_gran[1])

        num_divide_row = quant_gran[0] // paral_size[0]
        num_divide_col = quant_gran[1] // paral_size[1]

        temp_mat = torch.zeros((mat.shape[0], num_gran_row * quant_gran[0], num_gran_col * quant_gran[1]),
                               device=mat.device)
        temp_mat[:, :mat.shape[1], :mat.shape[2]] = mat
        temp_mat = temp_mat.reshape(mat.shape[0], num_gran_row, quant_gran[0], num_gran_col,
                                    quant_gran[1]).transpose(2, 3)
        max_abs_temp_mat = torch.max(torch.max(torch.abs(temp_mat), dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        max_abs_temp_mat = max_abs_temp_mat.unsqueeze(3).unsqueeze(4).expand(-1, -1, -1, num_divide_row,
                                                                             num_divide_col, -1, -1)
        max_abs_temp_mat = max_abs_temp_mat.transpose(2, 3).reshape(mat.shape[0], num_gran_row * num_divide_row,
                                                                    num_gran_col * num_divide_col, 1, 1)

        # Broadcast max_abs_temp_mat from (mat.shape[0], num_gran_row, num_gran_col, 1, 1) to
        #                           (mat.shape[0], num_gran_row, num_gran_col, num_divide_row, num_divide_col, 1, 1)
        temp_mat = temp_mat.reshape(mat.shape[0], num_gran_row, num_gran_col, num_divide_row, paral_size[0],
                                    num_divide_col, paral_size[1]).transpose(4, 5)
        temp_mat = temp_mat.transpose(2, 3)
        temp_mat = temp_mat.reshape(mat.shape[0], num_gran_row * num_divide_row, num_gran_col * num_divide_col,
                                    paral_size[0], paral_size[1])

        if self.bw_e:  # define the bfp_map_tensor function
            self.sliced_data, self.quantized_data, self.max_data, self.e_bias = bfp_map_tensor(temp_mat,
                                                                                               self.slice_method,
                                                                                               max_abs_temp_mat,
                                                                                               skip_quantized=self.inference)
        else:
            self.sliced_data, self.quantized_data, self.max_data, self.e_bias  = quant_map_tensor(temp_mat,
                                                                                                  self.slice_method,
                                                                                                  max_abs_temp_mat,
                                                                                                  skip_quantized=self.inference)

        if self.quantized_data is not None:
            self.quantized_data = self.quantized_data.transpose(2, 3).reshape(mat.shape[0], num_gran_row * num_divide_row * paral_size[0],
                                        num_gran_col * num_divide_col * paral_size[1])[:, :mat.shape[1], :mat.shape[2]]
        # remove the unsqueezed dimension and assign the values to the class attributes
        if unsqueezed:
            self.sliced_data = self.sliced_data.squeeze(0)
            if self.quantized_data is not None:
                self.quantized_data = self.quantized_data.squeeze(0)
            self.max_data = self.max_data.squeeze(0)
            if self.e_bias is not None:
                self.e_bias = self.e_bias.squeeze(0)


# v3 keeps the public SlicedData name for source compatibility, but routes it
# through the multimode backend. Mode 0 is the original two's-complement bit
# slicing path, so existing code keeps its behavior while future optimization
# work targets one backend.
from memintelli.pimpy.data_formats_multimode import SlicedDataMultiMode as _SlicedDataMultiMode


class SlicedData(_SlicedDataMultiMode):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mode", 0)
        super().__init__(*args, **kwargs)

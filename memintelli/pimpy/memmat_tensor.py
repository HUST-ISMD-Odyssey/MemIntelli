# -*- coding:utf-8 -*-
# @File  : dpe_tensor.py
# @Author: Zhou
# @Date  : 2024/6/27

'''
this is a new version of the dpe_tensor.py
we use the tensor to realize the dot product, and only consider the INT format data
this version is more efficient than the previous version
'''
import math
import itertools

import torch
from matplotlib import pyplot as plt
from memintelli.pimpy.utils import SNR, dot_high_dim
from memintelli.pimpy.data_formats import SlicedData

import time


def wire_resistance(x, y, wire_resistance):
    pass


class DPETensor(object):
    '''
    Implements a dot product engine using bit-sliced tensor operations for matrix multiplication.
    Supports INT and FP data formats with configurable quantization granularity and device settings.
    '''
    def __init__(
            self, HGS=1e-5, LGS=1e-7, g_level=16, write_variation=0.02, read_variation=0.02,
             vnoise=0.05, wire_resistance=0,
            rdac=2 ** 4, radc=2 ** 8, vread=0.2,
            rate_stuck_HGS=0.001, rate_stuck_LGS=0.000,
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            inference_chunk_size=None):
        """
        Parameters:
            HGS (float): High conductance state
            LGS (float): Low conductance state
            g_level (int): Number of conductance levels
            write_variation (float): Write variation - fixed Gaussian noise added to each conductance state
            read_variation (dict or float): Read variation - dynamic noise for each g level
                Can be a dict like {0: 0.05, 1: 0.1, 2: 0.1, 3: 0.15} or a single float applied to all levels
            vnoise (float): Random Gaussian noise of voltage
            wire_resistance (float): Wire resistance
            rdac (int): Number of DAC resolution
            radc (int or list): Number of ADC resolution. Can be:
                - Single integer: Same resolution for all weight slices
                - List/tuple: Different resolution for each weight slice (length must match number of weight slices)
            vread (float): Read voltage
            rate_stuck_HGS (float): Ratio of stuck faults to HGS state, i.e., the probability of stuck on, range: [0, 1]
            rate_stuck_LGS (float): Ratio of stuck faults to LGS state, i.e., the probability of stuck off, range: [0, 1]
            weight_quant_gran (str or tuple): Quantization granularity of the weight matrix
                "per-matrix" -> The whole matrix is quantized together (i.e., the quantization granularity is (m, n)
                                the same as the matrix shape).
                "per-row" -> Each row of the matrix is quantized separately. (i.e., the quantization granularity is (1, n)).
                "per-col" -> Each column of the matrix is quantized separately. (i.e., the quantization granularity is (m, 1)).
                (a, b) -> The quantization granularity is (a, b).
            input_quant_gran (str or tuple): Quantization granularity of the input matrix
            input_paral_size (tuple): The size of the input matrix used for parallel computation
            weight_paral_size (tuple): The size of the weight matrix used for parallel computation
            inference_chunk_size (int or None): Maximum number of elements in intermediate tensor
                during inference dot product. Controls memory-speed tradeoff for large layers.
                - None (default): auto ~32M elements (~128MB float32)
                - Smaller values: less peak memory, more chunks (slower)
                - Larger values: more peak memory, fewer chunks (faster)
                Tip: For 8GB GPU, try 16*1024*1024; for 24GB GPU, try 64*1024*1024.
        """
        self.HGS = HGS
        self.LGS = LGS
        self.g_level = g_level
        self.write_variation = write_variation

        # Handle read variation parameter
        if isinstance(read_variation, dict):
            # Validate the dictionary keys
            if set(read_variation.keys()) != set(range(g_level)):
                raise ValueError(f'read_variation dict must have keys for all g levels 0 to {g_level-1}')
            self.read_variation = read_variation
        elif isinstance(read_variation, (int, float)):
            # Single value applied to all levels
            self.read_variation = {i: read_variation for i in range(g_level)}
        else:
            raise ValueError('read_variation must be a dict, float, or None')

        self.vnoise = vnoise
        self.rdac = rdac
        if isinstance(radc, (list, tuple)):
            self.radc = torch.tensor(radc, device=device)
            self.radc = self.radc.flip(0)
            self.radc_is_list = True
        else:
            self.radc = radc
            self.radc_is_list = False
        self.vread = vread

        # these parameters are optional
        self.wire_resistance = wire_resistance
        self.rate_stuck_HGS = rate_stuck_HGS
        self.rate_stuck_LGS = rate_stuck_LGS

        self.device = device

        if self.radc_is_list:
            if torch.any(self.radc < 2):
                raise ValueError('All ADC resolution values should be larger than 1!')
        else:
            if self.radc < 2:
                raise ValueError('The resolution of the ADC should be larger than 1!')
        if self.rdac < 2:
            raise ValueError('The resolution of the DAC should be larger than 1!')
        if self.g_level < 2:
            raise ValueError('The number of the conductance levels should be larger than 1!')
        if self.LGS >= self.HGS:
            raise ValueError('The low conductance state should be smaller than the high conductance state!')
        if self.rate_stuck_HGS + self.rate_stuck_LGS > 1:
            raise ValueError('The sum of stuck fault rates should not exceed 1!')

        # Pre-compute conductance levels for efficiency (float32 to avoid float64 memory bloat)
        self.Q_G = (self.HGS - self.LGS) / (self.g_level - 1)
        self.conductance_levels = torch.tensor(
            [self.LGS + i * self.Q_G for i in range(self.g_level)],
            device=self.device, dtype=torch.float32)

        # Inference memory optimization: max elements in intermediate tensor before chunking
        # Default ~32M elements ≈ 128MB float32; set lower for smaller GPUs
        self.inference_chunk_size = inference_chunk_size

    def _validate_radc_with_slices(self, mat: SlicedData):
        """
        Validate that radc array length matches the number of weight slices.

        Parameters:
            mat (SlicedData): Weight tensor data.
        """
        if self.radc_is_list:
            num_weight_slices = len(mat)
            if len(self.radc) != num_weight_slices:
                raise ValueError(f'Length of radc array ({len(self.radc)}) must match number of weight slices ({num_weight_slices})')

    def __call__(self, x: SlicedData, mat: SlicedData):
        return self.MapReduceDot(x, mat)

    def MapReduceDot(self, x: SlicedData, mat: SlicedData):
        """
        Implements matrix multiplication using the MapReduce method.

        Parameters:
            x (SlicedData): Input tensor (shape: (m, n) or (batch, m, n)).
            mat (SlicedData): Weight tensor (shape: (n, p)).
            wire_factor (bool): Consider wire resistance (not implemented).

        Returns:
            torch.Tensor: Result of the matrix multiplication.
        """
        if mat.device.type != x.device.type:
            raise ValueError('The input data and weight data should be in the same device!')
        # check the quantization shape of the input data and weight data
        if x.shape[-1] != mat.shape[-2]:
            raise ValueError('The input data mismatches the shape of weight data!')
        self._validate_radc_with_slices(mat)
        if self.wire_resistance > 0:
            raise NotImplementedError('The wire_factor is not supported in the training version!')
        else:
            # Use memory-efficient inference path when either input or weight is in inference mode
            if getattr(x, 'inference', False) or getattr(mat, 'inference', False):
                result = self._dot_inference(x, mat)
            else:
                result = self._dot(x, mat, self._num2V, self._gen_read_noise)
        return result

    def _num2G(self, data, max_weights):
        """
        Converts weight data to static resistance (conductance).

        Memory optimization: uses in-place float32 arithmetic instead of int64 lookup.
        For lm_head (622M elements): saves ~5GB peak memory (no int64 intermediate).

        Parameters:
            data (torch.Tensor): Weight data (uint8/int16 sliced data).
            max_weights (torch.Tensor): Maximum weight values per quantization group.

        Returns:
            torch.Tensor: float32 conductance values.
        """
        # Step 1: Quantize to conductance via in-place float32 arithmetic.
        # Avoids creating large int64 lookup indices (saves ~2x memory).
        # G = round(data/max * (g_level-1)) * Q_G + LGS
        Q_G_t = torch.tensor(self.Q_G, dtype=torch.float32, device=data.device)
        LGS_t = torch.tensor(self.LGS, dtype=torch.float32, device=data.device)

        G = data.to(torch.float32)          # uint8 → float32 (new tensor, only copy)
        G.div_(max_weights)                  # in-place: data / max_weights
        G.mul_(self.g_level - 1)             # in-place: * (g_level - 1)
        G.round_()                           # in-place: round to nearest level
        G.clamp_(0, self.g_level - 1)        # in-place: clamp to valid range
        G.mul_(Q_G_t)                        # in-place: level_index * Q_G
        G.add_(LGS_t)                        # in-place: + LGS → final conductance

        # Step 2: Add write variation (fixed noise per device)
        if self.write_variation > 0:
            generator = torch.Generator(device=G.device)
            generator.manual_seed(42)  # fixed seed for reproducibility
            write_noise = torch.normal(0, self.write_variation, G.shape,
                                       generator=generator, device=G.device)
            G.addcmul_(G, write_noise, value=1.0)  # G += G * noise, in-place
            del write_noise

        # Step 3: Add stuck at fault (applied after write variation)
        if self.rate_stuck_HGS > 0 or self.rate_stuck_LGS > 0:
            generator_stuck = torch.Generator(device=G.device)
            generator_stuck.manual_seed(123)  # fixed seed
            random_vals = torch.rand(G.shape, generator=generator_stuck, device=G.device)

            if self.rate_stuck_HGS > 0:
                stuck_hgs_mask = random_vals < self.rate_stuck_HGS
                G[stuck_hgs_mask] = self.HGS

            if self.rate_stuck_LGS > 0:
                stuck_lgs_mask = ((random_vals >= self.rate_stuck_HGS) &
                                  (random_vals < self.rate_stuck_HGS + self.rate_stuck_LGS))
                G[stuck_lgs_mask] = self.LGS
            del random_vals

        # add the wire resistance
        if self.wire_resistance > 0:
            pass
        return G

    def _gen_read_noise(self, mat: SlicedData):
        """
        Converts weight data to resistance with added normal noise.
        Optimized version: avoids expanding G to g_level dimensions.

        Parameters:
            mat (SlicedData): Weight data.

        Returns:
            torch.Tensor: Resistance values (new tensor, does not modify mat.G).
        """
        G = mat.G

        # Apply read variation based on g level
        if any(var > 0 for var in self.read_variation.values()):
            variations = list(self.read_variation.values())
            all_same = all(v == variations[0] for v in variations)

            if all_same and variations[0] > 0:
                # All levels have the same variation - simple case, no level detection needed
                noise = torch.randn_like(G) * variations[0]
                G = G * torch.exp(noise)
            elif not all_same:
                # Per-level variation: compute level indices by rounding (memory-efficient)
                level_indices = torch.round((G - self.LGS) / self.Q_G).long().clamp(0, self.g_level - 1)
                var_tensor = torch.tensor([self.read_variation[i] for i in range(self.g_level)],
                                          device=G.device, dtype=G.dtype)
                std_per_element = var_tensor[level_indices]
                noise = torch.randn_like(G) * std_per_element
                G = G * torch.exp(noise)
                del level_indices, var_tensor, std_per_element  # free memory immediately

        return G

    def _num2V(self, x: SlicedData):
        """
        Converts input data to voltage (scaled by read voltage).

        Parameters:
            x (SlicedData): Input data.

        Returns:
            torch.Tensor: Voltage values.
        """
        xmax = x.sliced_max_weights
        if len(x.shape) == 2:  # without batch, the shape is (num_divide_row_x, num_divide_col_x, num_slice_x, m, n)
            xmax = xmax.reshape(1, 1, -1, 1, 1)
        elif len(x.shape) == 3:  # with batch, the shape is (batch, num_divide_row_x, num_divide_col_x, num_slice_x, m, n)
            xmax = xmax.reshape(1, 1, 1, -1, 1, 1)
        else:
            raise ValueError('The input data dimension is not supported!')
        V_in = self.vread * torch.round(x.sliced_data / xmax * (self.rdac - 1)) / (self.rdac - 1)
        if self.vnoise > 0:
            V_in = V_in * (1 + torch.randn_like(V_in) * self.vnoise)
        return V_in

    def _gen_read_noise_shifted_chunk(self, mat: SlicedData, c_start: int, c_end: int):
        """
        Generate (G_noisy - LGS) for weight column blocks [c_start:c_end] only.
        Memory-efficient: uses in-place ops, avoids int64 intermediates.

        Supports both full G (float32) and compressed G_indices (uint8) storage.
        When G_indices is used, reconstructs via float32 arithmetic (no int64 lookup).

        Parameters:
            mat (SlicedData): Weight data with pre-computed G or G_indices.
            c_start (int): Start index of weight column blocks.
            c_end (int): End index of weight column blocks (exclusive).

        Returns:
            torch.Tensor: (G_noisy - LGS) chunk, float32, shape (nr_y, chunk_cols, ns_y, k, p).
        """
        Q_G_t = torch.tensor(self.Q_G, dtype=torch.float32, device=mat.G_indices.device
                              if mat.G_indices is not None else mat.G.device)
        LGS_t = torch.tensor(self.LGS, dtype=torch.float32, device=Q_G_t.device)

        # Reconstruct G chunk: float32 only, no int64
        is_view = False  # True if G_chunk is a view (can't do in-place on views of stored data)
        if mat.G is not None:
            G_chunk = mat.G[:, c_start:c_end, :, :, :]  # view, no copy
            is_view = True
        elif mat.G_indices is not None:
            # Arithmetic reconstruction: uint8 → float32, then index * Q_G + LGS
            G_chunk = mat.G_indices[:, c_start:c_end, :, :, :].to(torch.float32)
            G_chunk.mul_(Q_G_t).add_(LGS_t)  # in-place: G = level * Q_G + LGS
            is_view = False
        else:
            raise ValueError('No conductance data available. Call update_weight() first.')

        has_noise = any(var > 0 for var in self.read_variation.values())
        if not has_noise:
            if is_view:
                return G_chunk - LGS_t  # new tensor (can't modify view)
            else:
                G_chunk.sub_(LGS_t)
                return G_chunk

        variations = list(self.read_variation.values())
        all_same = all(v == variations[0] for v in variations)

        if all_same and variations[0] > 0:
            # All conductance levels have the same read variation
            noise = torch.randn(G_chunk.shape, device=G_chunk.device, dtype=torch.float32)
            noise.mul_(variations[0])   # in-place: noise * var
            noise.exp_()               # in-place: exp(noise * var)
            if is_view:
                G_shifted = G_chunk * noise  # new tensor
                del noise
                G_shifted.sub_(LGS_t)
                return G_shifted
            else:
                G_chunk.mul_(noise)     # in-place: G * exp(noise)
                del noise
                G_chunk.sub_(LGS_t)     # in-place: - LGS
                return G_chunk
        elif not all_same:
            # Per-level variation: need level indices for per-element variance
            if mat.G_indices is not None:
                level_idx = mat.G_indices[:, c_start:c_end, :, :, :]  # uint8 view
            else:
                level_idx = torch.round((G_chunk - LGS_t) / Q_G_t).clamp_(0, self.g_level - 1).to(torch.uint8)
            var_tensor = torch.tensor([self.read_variation[i] for i in range(self.g_level)],
                                      device=G_chunk.device, dtype=torch.float32)
            # Use int32 indexing for per-level variance (avoids int64)
            noise = torch.randn(G_chunk.shape, device=G_chunk.device, dtype=torch.float32)
            noise.mul_(var_tensor[level_idx.long()])  # long() only on small var_tensor lookup
            noise.exp_()
            del level_idx
            if is_view:
                G_shifted = G_chunk * noise
                del noise
                G_shifted.sub_(LGS_t)
                return G_shifted
            else:
                G_chunk.mul_(noise)
                del noise
                G_chunk.sub_(LGS_t)
                return G_chunk
        else:
            if is_view:
                return G_chunk - LGS_t
            else:
                G_chunk.sub_(LGS_t)
                return G_chunk

    @staticmethod
    def _get_G_shape(mat: SlicedData):
        """Get the shape of the conductance tensor, whether stored as G or G_indices."""
        if mat.G is not None:
            return mat.G.shape
        elif mat.G_indices is not None:
            return mat.G_indices.shape
        else:
            raise ValueError('No conductance data available. Call update_weight() first.')

    def _dot_inference(self, x: SlicedData, mat: SlicedData):
        """
        Memory-efficient inference dot product with automatic chunking for large matrices.

        Key optimizations over the previous version:
        1. Chunked computation along weight column blocks (nc_y) to cap peak memory.
           For lm_head (1024→151936), this reduces intermediate from ~2.5GB to ~100MB.
        2. Per-chunk noise generation avoids materializing full G_shifted tensor.
        3. Uses torch.matmul (broadcasting) instead of einsum for better torch.compile support.
        4. All computations in float32 (avoids float64 promotion from Python scalars).

        Parameters:
            x (SlicedData): Input tensor with shape (m, n) or (batch, m, n).
            mat (SlicedData): Weight tensor with shape (n, p).

        Returns:
            torch.Tensor: Result of the dot product.
        """
        ns_x = len(x.slice_method)
        ns_y = len(mat.slice_method)

        if max(mat.sliced_max_weights) > self.g_level - 1:
            raise ValueError('The weight data is out of the range!')

        if len(x.shape) == 3:
            return self._dot_inference_batch(x, mat, ns_x, ns_y)
        elif len(x.shape) == 2:
            return self._dot_inference_nobatch(x, mat, ns_x, ns_y)
        else:
            raise ValueError('The input data dimension is not supported!')

    def _dot_inference_batch(self, x: SlicedData, mat: SlicedData, ns_x: int, ns_y: int):
        """Batch-mode chunked inference dot product. Input shape: (batch, m, n).

        Key optimization: vectorized (i,j) slice loop.
        Instead of ns_x * ns_y separate matmuls per block c, performs ONE batched matmul
        over all slice combinations. For ns_x=ns_y=4: 16x fewer kernel launches.
        """
        _hgs, _lgs, _vread = float(self.HGS), float(self.LGS), float(self.vread)
        k_dim = x.sliced_data.shape[-1]
        adcRef = (_hgs - _lgs) * _vread * k_dim
        scale_base = adcRef / ((_hgs - _lgs) / (self.g_level - 1)) / _vread / (self.g_level - 1)

        G_shape = self._get_G_shape(mat)
        nc_y = G_shape[1]
        nr_y = G_shape[0]
        k_g, p_g = G_shape[3], G_shape[4]
        B, nr_x, nc_x = x.sliced_data.shape[0], x.sliced_data.shape[1], x.sliced_data.shape[2]
        m_rows, p_cols = x.sliced_data.shape[-2], p_g
        dev = x.sliced_data.device

        # Pre-compute slice scale as a tensor (ns_x, ns_y) — avoids Python float() in loop
        xmax = x.sliced_max_weights.float()   # (ns_x,)
        mmax = mat.sliced_max_weights.float()  # (ns_y,)
        xw = x.sliced_weights.float()          # (ns_x,)
        mw = mat.sliced_weights.float()         # (ns_y,)
        scale_t = (xmax.unsqueeze(1) * mmax.unsqueeze(0) * scale_base *
                   xw.unsqueeze(1) * mw.unsqueeze(0))  # (ns_x, ns_y)

        # Chunk sizing
        output_per_col = B * nr_x * nc_x * m_rows * p_cols
        g_per_col = nr_y * ns_y * k_g * p_g * 2
        total_per_col = output_per_col + g_per_col
        max_numel = self.inference_chunk_size if self.inference_chunk_size else (32 * 1024 * 1024)
        cols_per_chunk = max(1, max_numel // max(total_per_col, 1))
        cols_per_chunk = min(cols_per_chunk, nc_y)

        # Pre-compute Vin for each input slice
        Vin_list = []
        for i in range(ns_x):
            xmax_i = float(xmax[i])
            x_slice_i = x.sliced_data[:, :, :, i, :, :].float()
            Vin_i = _vread * torch.round(x_slice_i / xmax_i * (self.rdac - 1)) / (self.rdac - 1)
            if self.vnoise > 0:
                Vin_i = Vin_i * (1 + torch.randn_like(Vin_i) * self.vnoise)
            Vin_list.append(Vin_i)
            del x_slice_i

        x_block_scale = x.max_data.float() if x.bw_e is None else (2. ** x.e_bias.float())
        slice_denom = ((2 ** (sum(x.slice_method) - 1) - 1) * (2 ** (sum(mat.slice_method) - 1) - 1)
                       if x.bw_e is None else None)
        bfp_factor = (2. ** (4 - sum(x.slice_method) - sum(mat.slice_method))
                      if x.bw_e is not None else None)

        # ADC: pre-compute per-j radc values as tensor for vectorized ADC
        if self.radc_is_list:
            radc_t = self.radc.float()  # (ns_y,) already flipped in __init__
        else:
            radc_val = float(self.radc)

        # Reshape scale for broadcasting: (ns_x, ns_y, 1, 1)
        scale_bcast = scale_t.reshape(ns_x, ns_y, 1, 1)
        BNR = B * nr_x

        output_chunks = []
        for c_start in range(0, nc_y, cols_per_chunk):
            c_end = min(c_start + cols_per_chunk, nc_y)
            chunk_cols = c_end - c_start

            G_shifted_chunk = self._gen_read_noise_shifted_chunk(mat, c_start, c_end)

            if x.bw_e is None:
                mat_max_chunk = mat.max_data[:, c_start:c_end, :, :].float()
            else:
                mat_e_chunk = mat.e_bias[:, c_start:c_end, :, :].float()

            accumulated = None
            for c in range(nc_x):
                if x.bw_e is None:
                    block_sc = (x_block_scale[:, :, c, :, :].unsqueeze(2) *
                                mat_max_chunk[c, :, :, :].unsqueeze(0).unsqueeze(0))
                    block_sc = block_sc / slice_denom
                else:
                    block_sc = (x_block_scale[:, :, c, :, :].unsqueeze(2) *
                                (2. ** mat_e_chunk[c, :, :, :]).unsqueeze(0).unsqueeze(0))
                    block_sc = block_sc * bfp_factor

                if m_rows == 1 and p_cols == 1:
                    # === VECTORIZED FAST PATH ===
                    # One batched matmul replaces ns_x * ns_y individual matmuls.
                    # G: (chunk, ns_y, n) → (ns_y, n, chunk)
                    G_c_all = G_shifted_chunk[c, :, :, :, 0].permute(1, 2, 0)  # (ns_y, n, chunk)
                    # Vin: stack all i → (ns_x, B*nr_x, n)
                    Vin_flat = torch.stack(
                        [Vin_list[i][:, :, c, 0, :].reshape(BNR, k_dim) for i in range(ns_x)]
                    )  # (ns_x, BNR, n)
                    # Batched matmul: (ns_x,1,BNR,n) @ (1,ns_y,n,chunk) → (ns_x,ns_y,BNR,chunk)
                    all_inner = torch.matmul(Vin_flat.unsqueeze(1), G_c_all.unsqueeze(0))
                    del Vin_flat, G_c_all
                    # ADC quantization (vectorized over all slices)
                    if self.radc_is_list:
                        for j in range(ns_y):
                            rj = float(radc_t[j])
                            all_inner[:, j] = torch.round(all_inner[:, j] / adcRef * (rj - 1)) / (rj - 1)
                    else:
                        all_inner = torch.round(all_inner / adcRef * (radc_val - 1)) / (radc_val - 1)
                    # Scale and sum all (i,j) → (BNR, chunk)
                    sum_c = (all_inner * scale_bcast).sum(dim=(0, 1))
                    del all_inner
                    sum_c = sum_c.reshape(B, nr_x, chunk_cols, 1, 1) * block_sc
                else:
                    # === GENERAL PATH (unchanged loop for m_rows>1 or p_cols>1) ===
                    sum_c = None
                    for si, (i, j) in enumerate(itertools.product(range(ns_x), range(ns_y))):
                        G_jc = G_shifted_chunk[c, :, j, :, :]
                        inner = torch.einsum('bxmn,ynp->bxymp',
                                             Vin_list[i][:, :, c, :, :], G_jc)
                        radc_j = float(self.radc[j]) if self.radc_is_list else float(self.radc)
                        inner = torch.round(inner / adcRef * (radc_j - 1)) / (radc_j - 1)
                        inner *= float(scale_t[i, j])
                        if sum_c is None:
                            sum_c = inner
                        else:
                            sum_c += inner
                            del inner
                    sum_c = sum_c * block_sc

                del block_sc
                if accumulated is None:
                    accumulated = sum_c
                else:
                    accumulated += sum_c
                    del sum_c

            del G_shifted_chunk
            output_chunks.append(accumulated)

        if len(output_chunks) == 1:
            out = output_chunks[0]
        else:
            out = torch.cat(output_chunks, dim=2)
        del output_chunks, Vin_list

        out = out.permute(0, 1, 3, 2, 4)
        out = out.reshape(out.shape[0], out.shape[1] * out.shape[2], out.shape[3] * out.shape[4])
        out = out[:out.shape[0], :x.shape[1], :mat.shape[1]]
        return out

    def _dot_inference_nobatch(self, x: SlicedData, mat: SlicedData, ns_x: int, ns_y: int):
        """Non-batch chunked inference dot product. Input shape: (m, n).

        Same vectorized optimization as batch version: ONE batched matmul per block c.
        """
        _hgs, _lgs, _vread = float(self.HGS), float(self.LGS), float(self.vread)
        k_dim = x.sliced_data.shape[-1]
        adcRef = (_hgs - _lgs) * _vread * k_dim
        scale_base = adcRef / ((_hgs - _lgs) / (self.g_level - 1)) / _vread / (self.g_level - 1)

        G_shape = self._get_G_shape(mat)
        nc_y = G_shape[1]
        nr_y = G_shape[0]
        k_g, p_g = G_shape[3], G_shape[4]
        nr_x, nc_x = x.sliced_data.shape[0], x.sliced_data.shape[1]
        m_rows, p_cols = x.sliced_data.shape[-2], p_g
        dev = x.sliced_data.device

        # Pre-compute slice scale as tensor
        xmax = x.sliced_max_weights.float()
        mmax = mat.sliced_max_weights.float()
        xw = x.sliced_weights.float()
        mw = mat.sliced_weights.float()
        scale_t = (xmax.unsqueeze(1) * mmax.unsqueeze(0) * scale_base *
                   xw.unsqueeze(1) * mw.unsqueeze(0))

        output_per_col = nr_x * nc_x * m_rows * p_cols
        g_per_col = nr_y * ns_y * k_g * p_g * 2
        total_per_col = output_per_col + g_per_col
        max_numel = self.inference_chunk_size if self.inference_chunk_size else (32 * 1024 * 1024)
        cols_per_chunk = max(1, max_numel // max(total_per_col, 1))
        cols_per_chunk = min(cols_per_chunk, nc_y)

        Vin_list = []
        for i in range(ns_x):
            xmax_i = float(xmax[i])
            x_slice_i = x.sliced_data[:, :, i, :, :].float()
            Vin_i = _vread * torch.round(x_slice_i / xmax_i * (self.rdac - 1)) / (self.rdac - 1)
            if self.vnoise > 0:
                Vin_i = Vin_i * (1 + torch.randn_like(Vin_i) * self.vnoise)
            Vin_list.append(Vin_i)
            del x_slice_i

        x_block_scale = x.max_data.float() if x.bw_e is None else (2. ** x.e_bias.float())
        slice_denom = ((2 ** (sum(x.slice_method) - 1) - 1) * (2 ** (sum(mat.slice_method) - 1) - 1)
                       if x.bw_e is None else None)
        bfp_factor = (2. ** (4 - sum(x.slice_method) - sum(mat.slice_method))
                      if x.bw_e is not None else None)

        if self.radc_is_list:
            radc_t = self.radc.float()
        else:
            radc_val = float(self.radc)

        scale_bcast = scale_t.reshape(ns_x, ns_y, 1, 1)

        output_chunks = []
        for c_start in range(0, nc_y, cols_per_chunk):
            c_end = min(c_start + cols_per_chunk, nc_y)
            chunk_cols = c_end - c_start

            G_shifted_chunk = self._gen_read_noise_shifted_chunk(mat, c_start, c_end)

            if x.bw_e is None:
                mat_max_chunk = mat.max_data[:, c_start:c_end, :, :].float()
            else:
                mat_e_chunk = mat.e_bias[:, c_start:c_end, :, :].float()

            accumulated = None
            for c in range(nc_x):
                if x.bw_e is None:
                    block_sc = (x_block_scale[:, c, :, :].unsqueeze(1) *
                                mat_max_chunk[c, :, :, :].unsqueeze(0))
                    block_sc = block_sc / slice_denom
                else:
                    block_sc = (x_block_scale[:, c, :, :].unsqueeze(1) *
                                (2. ** mat_e_chunk[c, :, :, :]).unsqueeze(0))
                    block_sc = block_sc * bfp_factor

                if m_rows == 1 and p_cols == 1:
                    # === VECTORIZED FAST PATH ===
                    G_c_all = G_shifted_chunk[c, :, :, :, 0].permute(1, 2, 0)  # (ns_y, n, chunk)
                    Vin_flat = torch.stack(
                        [Vin_list[i][:, c, 0, :] for i in range(ns_x)]
                    )  # (ns_x, nr_x, n)
                    all_inner = torch.matmul(Vin_flat.unsqueeze(1), G_c_all.unsqueeze(0))
                    del Vin_flat, G_c_all
                    if self.radc_is_list:
                        for j in range(ns_y):
                            rj = float(radc_t[j])
                            all_inner[:, j] = torch.round(all_inner[:, j] / adcRef * (rj - 1)) / (rj - 1)
                    else:
                        all_inner = torch.round(all_inner / adcRef * (radc_val - 1)) / (radc_val - 1)
                    sum_c = (all_inner * scale_bcast).sum(dim=(0, 1))
                    del all_inner
                    sum_c = sum_c.reshape(nr_x, chunk_cols, 1, 1) * block_sc
                else:
                    sum_c = None
                    for si, (i, j) in enumerate(itertools.product(range(ns_x), range(ns_y))):
                        G_jc = G_shifted_chunk[c, :, j, :, :]
                        inner = torch.einsum('xmn,ynp->xymp',
                                             Vin_list[i][:, c, :, :], G_jc)
                        radc_j = float(self.radc[j]) if self.radc_is_list else float(self.radc)
                        inner = torch.round(inner / adcRef * (radc_j - 1)) / (radc_j - 1)
                        inner *= float(scale_t[i, j])
                        if sum_c is None:
                            sum_c = inner
                        else:
                            sum_c += inner
                            del inner
                    sum_c = sum_c * block_sc

                del block_sc
                if accumulated is None:
                    accumulated = sum_c
                else:
                    accumulated += sum_c
                    del sum_c

            del G_shifted_chunk
            output_chunks.append(accumulated)

        if len(output_chunks) == 1:
            out = output_chunks[0]
        else:
            out = torch.cat(output_chunks, dim=1)
        del output_chunks, Vin_list

        out = out.permute(0, 2, 1, 3)
        out = out.reshape(out.shape[0] * out.shape[1], out.shape[2] * out.shape[3])
        out = out[:x.shape[0], :mat.shape[1]]
        return out

    def _dot(self, x: SlicedData, mat: SlicedData, _num2V_func, _num2R_func):
        """
        Computes the dot product of input and weight tensors.

        Parameters:
            x (SlicedData): Input tensor with shape (m, n) or (batch, m, n).
            mat (SlicedData): Weight tensor with shape (n, p).
            _num2V_func (function): Function to convert input data to voltage.
            _num2R_func (function): Function to convert weight data to resistance

        Returns:
            torch.Tensor: Result of the dot product with shape (m, p) or (batch, m, p).
        """
        Vin = _num2V_func(x)
        G = _num2R_func(mat)

        if max(mat.sliced_max_weights) > self.g_level - 1:
            raise ValueError('The weight data is out of the range!')

        if len(x.shape) == 2:
            adcRef = (self.HGS - self.LGS) * self.vread * (Vin.shape[-1])
            QG = (self.HGS - self.LGS) / (self.g_level - 1)
            out = dot_high_dim(Vin, G - self.LGS)
            if self.radc_is_list:
                radc_expanded = self.radc.view(1, 1, 1, 1, -1, 1, 1)  # reshape to broadcast correctly
                out = torch.round(out / adcRef * (radc_expanded - 1)) / (radc_expanded - 1)
            else:
                out = torch.round(out / adcRef * (self.radc - 1)) / (self.radc - 1)

            out = torch.mul(out, x.sliced_max_weights.reshape(1, 1, 1, -1, 1, 1, 1))
            out = (torch.mul(out, mat.sliced_max_weights.reshape(1, 1, 1, 1, -1, 1, 1)) / QG / self.vread / (
                        self.g_level - 1) * adcRef)
            shift_weights = torch.zeros((len(x), len(mat)), device=x.device)

            for i in range(len(x)):
                shift_weights[i] = x.sliced_weights[i] * mat.sliced_weights
            out = torch.mul(out.reshape(out.shape[0], out.shape[1], out.shape[2], -1, out.shape[5], out.shape[6]),
                            shift_weights.reshape(1, 1, 1, -1, 1, 1))
            out = out.sum(dim=3)
            if x.bw_e is None:
                out_block_max = torch.einsum("nmij, mpij->nmpij", x.max_data, mat.max_data)
                out = (out * out_block_max / (2 ** (sum(x.slice_method) - 1) - 1) / (
                            2 ** (sum(mat.slice_method) - 1) - 1))
            else:
                out_block_e_bias = torch.einsum("nmij, mpij->nmpij", 2. ** x.e_bias, 2. ** mat.e_bias)
                out = out * out_block_e_bias * 2. ** (4 - sum(x.slice_method) - sum(mat.slice_method))
            out = out.sum(dim=1)
            out = out.permute(0, 2, 1, 3)
            out = out.reshape(out.shape[0] * out.shape[1], out.shape[2] * out.shape[3])
            out = out[:x.shape[0], :mat.shape[1]]
        elif len(x.shape) == 3:
            adcRef = (self.HGS - self.LGS) * self.vread * Vin.shape[-1]
            QG = (self.HGS - self.LGS) / (self.g_level - 1)
            out = dot_high_dim(Vin, G - self.LGS)

            if self.radc_is_list:
                radc_expanded = self.radc.view(1, 1, 1, 1, 1, -1, 1, 1)  # reshape to broadcast correctly
                out = torch.round(out / adcRef * (radc_expanded - 1)) / (radc_expanded - 1)
            else:
                out = torch.round(out / adcRef * (self.radc - 1)) / (self.radc - 1)
            out = torch.mul(out, x.sliced_max_weights.reshape(1, 1, 1, 1, -1, 1, 1, 1))
            out = (torch.mul(out, mat.sliced_max_weights.reshape(1, 1, 1, 1, 1, -1, 1, 1)) / QG / self.vread / (
                        self.g_level - 1) * adcRef)
            shift_weights = torch.zeros((len(x), len(mat)), device=x.device)

            for i in range(len(x)):
                shift_weights[i] = x.sliced_weights[i] * mat.sliced_weights
            # add the shift weights to the calculated result
            out = torch.mul(out.reshape(out.shape[0], out.shape[1], out.shape[2], out.shape[3], -1, out.shape[6],
                                         out.shape[7]), shift_weights.reshape(1, 1, 1, 1, -1, 1, 1))
            out = out.sum(dim=4)
            if x.bw_e is None:
                out_block_max = torch.einsum("bnmij, mpij->bnmpij", x.max_data, mat.max_data)
                out = (out * out_block_max / (2 ** (sum(x.slice_method) - 1) - 1) / (
                            2 ** (sum(mat.slice_method) - 1) - 1))
            else:
                out_block_e_bias = torch.einsum("bnmij, mpij->bnmpij", 2. ** x.e_bias, 2. ** mat.e_bias)
                out = out * out_block_e_bias * 2. ** (4 - sum(x.slice_method) - sum(mat.slice_method))
            out = out.sum(dim=2)
            out = out.permute(0, 1, 3, 2, 4)
            out = out.reshape(out.shape[0], out.shape[1] * out.shape[2], out.shape[3] * out.shape[4])
            out = out[:out.shape[0], :x.shape[1], :mat.shape[1]]
        else:
            raise ValueError('The input data dimension is not supported!')
        return out


if __name__ == '__main__':
    tb_mode = 1
    device = torch.device('cuda:0')
    if tb_mode == 0:
        torch.manual_seed(42)
        x_data = torch.randn(2, 1000, 1000, dtype=torch.float64, device=device)
        mat_data = torch.randn(1000, 1000, dtype=torch.float64, device=device)
        mblk = torch.tensor([1, 1, 2, 4])
        xblk = torch.tensor([1, 1, 2, 4])
        mat = SlicedData(mblk, device=device, bw_e=None, is_weight=True, quant_gran=(64, 64), paral_size=(64, 64))
        x = SlicedData(xblk, device=device, bw_e=None, quant_gran=(64, 64), paral_size=(64, 64))
        engine = DPETensor(write_variation=0.0, read_variation=0.0, rate_stuck_HGS=0.0, rate_stuck_LGS=0.0, vnoise=0.01,  g_level=16, rdac=16, radc=2 ** 16)
        mat.slice_data_imp(engine, mat_data)
        x.slice_data_imp(engine, x_data)
        start = time.time()
        result = engine(x, mat).cpu().numpy()
        rel_result = torch.matmul(x_data, mat_data).cpu().numpy()
        snr_varlue = SNR(result, rel_result)
        print("SNR(dB)", snr_varlue)
        plt.scatter(rel_result.reshape(-1), result.reshape(-1))
        plt.xlabel('Expected Value of Dot Product')
        plt.ylabel('Measured Value of Dot Product')
        #plt.show()
    elif tb_mode == 1:
        torch.manual_seed(42)
        x_data = torch.randn(1000, 1000, dtype=torch.float64, device=device)
        mat_data = torch.randn(1000, 1000, dtype=torch.float64, device=device)
        mblk = torch.tensor([1, 1, 2, 2])
        xblk = torch.tensor([1, 1, 2, 2])
        for i in range(10):
            mat = SlicedData(mblk, device=device, bw_e=None, is_weight=True, quant_gran=(64, 64), paral_size=(64, 64))
            x = SlicedData(xblk, device=device, bw_e=None, quant_gran=(64, 64), paral_size=(64, 64))
            read_variation = {0: 0.05, 1: 0.05, 2: 0.05, 3: 0.1}
            radc_per_slice = [2**12, 2**11, 2**10, 2**9]  # Different radc for each slice
            engine = DPETensor(write_variation=0.0, read_variation=read_variation, rate_stuck_HGS=0.0, rate_stuck_LGS=0.0, vnoise=0.0,
            g_level=4, rdac=4, radc=radc_per_slice)
            mat.slice_data_imp(engine, mat_data)
            x.slice_data_imp(engine, x_data)
            start = time.time()
            result = engine(x, mat).cpu().numpy()
            rel_result = torch.matmul(x_data, mat_data).cpu().numpy()
            snr_varlue = SNR(result, rel_result)
            print("SNR(dB)", snr_varlue)
            plt.scatter(rel_result.reshape(-1), result.reshape(-1))
            plt.xlabel('Expected Value of Dot Product')
            plt.ylabel('Measured Value of Dot Product')


# v3 canonical backend: keep the historical DPETensor symbol, but implement it
# as DPETensorMultiMode(mode=0). This avoids maintaining two optimization paths.
from memintelli.pimpy.memmat_tensor_multimode import DPETensorMultiMode as _DPETensorMultiMode


class DPETensor(_DPETensorMultiMode):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mode", 0)
        super().__init__(*args, **kwargs)

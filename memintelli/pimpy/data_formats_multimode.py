import copy
import math
import torch
from memintelli.pimpy.utils import quant_map_tensor, bfp_map_tensor


class SlicedDataMultiMode(object):
    """Sliced tensor container for the multi-mode dot product engine."""

    def __init__(
        self,
        slice_method: torch.Tensor,
        bw_e=None,
        is_weight=False,
        paral_size=(64, 64),
        quant_gran=None,
        device=None,
        inference=False,
        mode=0,
    ):
        """
        Initialize a sliced tensor container for one engine mode.

        Parameters:
            slice_method (torch.Tensor): Bit width of each slice. For mode 0 the first element is the sign bit;
                for mode 1 this is kept only for API compatibility and is not used; for mode 2 all bits are
                unsigned magnitude bits.
            bw_e (int or None): Block floating-point exponent width. None selects integer quantization.
            is_weight (bool): True when this object stores weight data; False for activation input data.
            paral_size (tuple): Hardware tile size as (rows, cols).
            quant_gran (str, tuple, or None): Quantization granularity. None uses paral_size.
            device (torch.device or None): Device used for tensors. None selects CPU.
            inference (bool): If True, release tensors that are not needed for forward inference.
            mode (int): Mapping mode. 0 is standard slicing, 1 is linear differential-pair projection, 2 is differential-pair slicing.
        """
        if mode not in (0, 1, 2):
            raise ValueError(f"mode must be 0, 1, or 2; got {mode!r}.")
        if mode == 2 and bw_e is not None:
            raise NotImplementedError(
                "Block floating-point (bw_e is not None) is not supported in Mode 2."
            )

        self.mode = mode
        self.bw_e = bw_e
        self.is_weight = is_weight
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.shape = None
        self.inference = inference

        self.G = None
        self.G_indices = None
        self.G_index_dtype = None
        self.G_is_compressed = False
        self.paral_size = paral_size
        self.quant_gran = paral_size if quant_gran is None else quant_gran

        if mode == 0 and slice_method[0] != 1:
            raise ValueError("First element of slice_method must be 1 (sign bit).")
        self.slice_method = slice_method
        self.total_bits = int(torch.sum(slice_method).item())
        self.is_uniform_1bit_slices = all(int(v.item()) == 1 for v in slice_method.detach().cpu())

        # signed-view fields (all modes)
        self.sliced_data = None
        self.quantized_data = None
        self.max_data = None
        self.precomputed_v_sliced = None
        self.e_bias = None
        self.mode1_w_max = None
        self.activation_slice_fused = False
        self._activation_slice_buffer_key = None
        self._activation_slice_buffer = None
        self._activation_max_buffer = None

        # differential-pair branch fields (mode 2 only)
        self.sliced_data_p = None
        self.sliced_data_n = None
        self.quantized_data_p = None
        self.quantized_data_n = None
        self.max_data_p = None
        self.max_data_n = None
        self.e_bias_p = None
        self.e_bias_n = None

        self.sliced_max_weights = torch.empty(len(slice_method), device=self.device)
        self.sliced_weights = torch.empty(len(slice_method), device=self.device)
        self._init_data(slice_method, self.device)


    def _init_data(self, slice_method: torch.Tensor, device):
        """
        Initialize per-slice reconstruction weights and maximum slice values.

        Parameters:
            slice_method (torch.Tensor): Bit width of each slice.
            device (torch.device): Device where sliced weight metadata is stored.

        Returns:
            None. Updates sliced_max_weights and sliced_weights in place.
        """
        self.sliced_max_weights = torch.zeros(len(slice_method), device=device)
        self.sliced_weights = torch.zeros(len(slice_method), device=device)
        temp_s, i = 0, 0
        for s in slice_method.flip(0):
            self.sliced_max_weights[i] = 2 ** s - 1
            self.sliced_weights[i] = 2 ** temp_s
            temp_s += int(s.item())
            i += 1
        if self.mode == 0:
            self.sliced_weights[-1] *= -1  # two's-complement MSB


    def __repr__(self):
        """
        Return a short string summary of this sliced data object.

        Parameters:
            None.

        Returns:
            str: Human-readable mode, weight/inference flag, and slice method.
        """
        return (
            f"SlicedDataMultiMode[flash](mode={self.mode}, "
            f"is_weight={self.is_weight}, "
            f"inference={self.inference}, "
            f"slice_method={self.slice_method.tolist()})"
        )

    def __len__(self):
        """
        Return the number of bit slices.

        Parameters:
            None.

        Returns:
            int: Number of entries in slice_method.
        """
        return len(self.slice_method)

    def size(self):
        """
        Return the shape of the dequantized data buffer.

        Parameters:
            None.

        Returns:
            torch.Size: Size of quantized_data or the positive branch quantized_data_p.
        """
        for attr in ("quantized_data", "quantized_data_p"):
            v = getattr(self, attr, None)
            if v is not None:
                return v.size()
        raise ValueError("quantized_data has not been populated yet.")

    def t(self):
        """
        Return a transposed copy of the sliced data container.

        Parameters:
            None.

        Returns:
            SlicedDataMultiMode: Deep copy with row/column tensor dimensions transposed.
        """
        c = copy.deepcopy(self)
        if self.max_data is not None:
            c.max_data = self.max_data.transpose(0, 1)
        if self.max_data_p is not None:
            c.max_data_p = self.max_data_p.transpose(0, 1)
        if self.max_data_n is not None:
            c.max_data_n = self.max_data_n.transpose(0, 1)
        if self.mode1_w_max is not None:
            c.mode1_w_max = (
                self.mode1_w_max.T
                if self.mode1_w_max.dim() >= 2
                else self.mode1_w_max.clone()
            )

        if self.is_weight and self.G is not None:
            if isinstance(self.G, tuple):
                c.G = (self.G[0].transpose(-2, -1), self.G[1].transpose(-2, -1))
            else:
                c.G = self.G.transpose(-4, -5)
        if self.is_weight and self.G_indices is not None:
            if isinstance(self.G_indices, tuple):
                c.G_indices = tuple(
                    branch.transpose(-4, -5) for branch in self.G_indices
                )
            else:
                c.G_indices = self.G_indices.transpose(-4, -5)

        if self.inference:
            c.sliced_data = c.quantized_data = None
            c.precomputed_v_sliced = None
            c.quantized_data_p = c.quantized_data_n = None
        else:
            if self.sliced_data is not None:
                c.sliced_data = self.sliced_data.transpose(-4, -5)
            if self.precomputed_v_sliced is not None:
                c.precomputed_v_sliced = self.precomputed_v_sliced.transpose(-4, -5)
            if self.quantized_data is not None:
                c.quantized_data = self.quantized_data.T
            if self.sliced_data_p is not None:
                c.sliced_data_p = self.sliced_data_p.transpose(-4, -5)
            if self.sliced_data_n is not None:
                c.sliced_data_n = self.sliced_data_n.transpose(-4, -5)
            if self.quantized_data_p is not None:
                c.quantized_data_p = self.quantized_data_p.T
            if self.quantized_data_n is not None:
                c.quantized_data_n = self.quantized_data_n.T
        return c


    def slice_data_imp(self, engine, data):
        """
        Quantize, slice, and optionally map data to conductance.

        Parameters:
            engine (DPETensorMultiMode): Dot product engine that provides device and conductance mapping rules.
            data (torch.Tensor): Input tensor to slice. Supports 2-D (rows, cols) or 3-D (batch, rows, cols).

        Returns:
            None. Populates sliced buffers, metadata, and weight conductance fields in place.
        """
        data = data.to(engine.device)
        compute_device = data.device
        if self.slice_method.device != compute_device:
            self.slice_method = self.slice_method.to(compute_device)
            self.sliced_max_weights = self.sliced_max_weights.to(compute_device)
            self.sliced_weights = self.sliced_weights.to(compute_device)
            self.device = compute_device

        # quantized_data is needed at forward only for Mode-1 inputs
        need_qd_forward = (self.mode == 1 and not self.is_weight)
        skip_qd = self.inference and not need_qd_forward

        if self.mode == 1:
            self._prepare_mode1_data(data)
        elif self.mode == 2:
            self._slice_data_diff_pair(
                data,
                skip_quantized=skip_qd,
                keep_signed_view=not self.is_weight,
                use_triton_activation=bool(
                    self.inference
                    and not self.is_weight
                    and getattr(engine, "mode2_input_mode", "signed") == "differential"
                    and getattr(engine, "triton_fuse_activation_slices", False)
                    and getattr(engine, "triton_mode2_diff_activation_slices", False)
                ),
            )
        else:
            self._slice_data_tc(data, skip_quantized=skip_qd, engine=engine)

        self.shape = data.shape

        if self.is_weight:
            engine._prepare_weight_conductance(self)

        if self.inference:
            if self.mode == 1 and not self.is_weight:
                # Mode 1 input: keep quantized_data; sliced_data not used
                self.sliced_data = None
            else:
                # All other cases: quantized_data not needed at forward
                self.quantized_data = None
                if self.is_weight:
                    # G is built; sliced_data no longer needed
                    self.sliced_data = None

            keep_diff_input = (
                self.mode == 2
                and not self.is_weight
                and getattr(engine, "mode2_input_mode", "signed") == "differential"
            )
            # Branch tensors are normally transient. Differential-input Mode 2
            # keeps input branches because the forward pass performs two reads.
            if not keep_diff_input:
                self.sliced_data_p = None
                self.sliced_data_n = None
            self.quantized_data_p = None
            self.quantized_data_n = None

    def compress_G(self, engine):
        """Compress prepared conductance tensors to integer level indices.

        This keeps the legacy inference API working while using the multimode
        backend. Compression is exact only when write variation is disabled.
        """
        if getattr(engine, "mode", 0) == 1:
            return
        if getattr(engine, "write_variation", 0) != 0:
            return
        if self.G is None:
            return
        if getattr(engine, "_g_index_dtype", None) is None:
            return

        def _compress_one(G):
            idx = G.to(torch.float32)
            idx.sub_(engine.LGS)
            idx.div_(engine.Q_G)
            idx.round_()
            idx.clamp_(0, engine.g_level - 1)
            return idx.to(engine._g_index_dtype)

        if isinstance(self.G, tuple):
            self.G_indices = tuple(_compress_one(branch) for branch in self.G)
            self.G_index_dtype = self.G_indices[0].dtype
        else:
            self.G_indices = _compress_one(self.G)
            self.G_index_dtype = self.G_indices.dtype
        self.G = None
        self.G_is_compressed = True


    def _slice_data_tc(self, mat: torch.Tensor, skip_quantized: bool = False, engine=None):
        """
        Slice data using standard two's-complement quantization for Mode 0.

        Parameters:
            mat (torch.Tensor): Tensor to quantize and slice.
            skip_quantized (bool): If True, do not materialize the dequantized quantized_data buffer.

        Returns:
            None. Populates sliced_data, quantized_data, max_data, and e_bias in place.
        """
        """Standard two's-complement quantisation and bit-slicing (Mode 0)."""
        self.activation_slice_fused = False
        if (self.is_weight and getattr(engine, "supports_column_scales", False)
                and self.quant_gran[1] % self.paral_size[1]):
            self._slice_weight_columns(mat, skip_quantized)
            return
        unsqueezed = False
        if mat.dim() == 2:
            mat = mat.unsqueeze(0)
            unsqueezed = True

        qg, ps, ngr, ngc, ndr, ndc = self._geom(mat)
        if self._try_slice_data_tc_triton(mat, qg, ps, ngr, ngc, ndr, ndc, skip_quantized, engine):
            return
        tiled, max_abs = self._tile(mat, ngr, ngc, ndr, ndc, qg, ps)

        if self.bw_e:
            sd, qd, md, eb = bfp_map_tensor(
                tiled, self.slice_method, max_abs,
                skip_quantized=skip_quantized,
            )
        else:
            sd, qd, md, eb = quant_map_tensor(
                tiled, self.slice_method, max_abs,
                skip_quantized=skip_quantized,
            )

        self.sliced_data = sd
        self.max_data = md.to(torch.float32) if self.mode == 0 and md is not None else md
        self.precomputed_v_sliced = None
        self.e_bias = eb

        if qd is not None:
            self.quantized_data = self._untile_quantized(
                qd, mat.shape, ngr, ngc, ndr, ndc, ps
            )
        else:
            self.quantized_data = None

        if unsqueezed:
            self.sliced_data = self.sliced_data.squeeze(0)
            if self.quantized_data is not None:
                self.quantized_data = self.quantized_data.squeeze(0)
            self.max_data = self.max_data.squeeze(0)
            if self.e_bias is not None:
                self.e_bias = self.e_bias.squeeze(0)

    def _slice_weight_columns(self, mat, skip_quantized):
        """Quantize column groups independently without changing array geometry."""
        if mat.ndim != 2 or self.bw_e is not None:
            raise ValueError("Per-column array scales require 2-D integer weights")
        r, c = self.paral_size
        qr, qc = self.quant_gran
        m, p = math.ceil(mat.shape[0] / r), math.ceil(mat.shape[1] / c)
        ngr, ngc = math.ceil(mat.shape[0] / qr), math.ceil(mat.shape[1] / qc)
        grouped = self._pad_to_quant_blocks(mat.unsqueeze(0), ngr, ngc, (qr, qc))
        maxima = grouped.abs().amax(dim=(-2, -1))[0]
        row_group = torch.arange(m, device=mat.device) // (qr // r)
        column_group = (torch.arange(p*c, device=mat.device) // qc).clamp(max=ngc-1)
        scales = maxima[row_group[:, None], column_group[None, :]].reshape(1, m, p, 1, c)
        padded = mat.new_zeros(m*r, p*c)
        padded[:mat.shape[0], :mat.shape[1]] = mat
        tiled = padded.reshape(m, r, p, c).permute(0, 2, 1, 3).unsqueeze(0)
        sd, qd, md, eb = quant_map_tensor(tiled, self.slice_method, scales, skip_quantized)
        self.sliced_data, self.max_data = sd[0], md[0].float()
        self.precomputed_v_sliced, self.e_bias = None, eb
        self.quantized_data = (None if qd is None else
                              qd[0].permute(0, 2, 1, 3).reshape(m*r, p*c)[:mat.shape[0], :mat.shape[1]])

    def _try_slice_data_tc_triton(self, mat, qg, ps, ngr, ngc, ndr, ndc, skip_quantized, engine=None):
        """Try the restricted fused activation slicer for mode-0 2-D inference."""
        if not (
            skip_quantized
            and not self.is_weight
            and self.inference
            and self.mode == 0
            and self.bw_e is None
            and mat.dim() == 3
            and mat.shape[0] == 1
            and mat.is_cuda
            and mat.dtype in (torch.bfloat16, torch.float32)
            and bool(getattr(self, "enable_triton_activation_slicing", False))
            and (
                not bool(getattr(engine, "_mode0_requires_fp32_analog_compute", lambda: False)())
                or bool(getattr(engine, "_mode0_vmm_uses_low_precision_override", lambda: False)())
                or bool(getattr(engine, "_mode0_fast_policy_requested", lambda: False)())
            )
            and not bool(
                getattr(engine, "_mode0_seeded_semantic_audit", lambda: False)()
            )
            and ndr == 1
            and ndc == 1
            and int(ps[0]) == 1
            and int(qg[0]) == int(ps[0])
            and int(qg[1]) == int(ps[1])
            and all(int(v.item()) == 1 for v in self.slice_method)
        ):
            return False
        try:
            from memintelli.pimpy.triton_fast_accumulate import (
                triton_slice_mode0_2d_uniform1,
                triton_slice_mode0_2d_uniform1_with_voltage,
            )
            rows, cols = int(mat.shape[1]), int(mat.shape[2])
            tile_cols = int(ps[1])
            tile_count = math.ceil(cols / tile_cols)
            buffer_key = (
                str(mat.device),
                str(mat.dtype),
                rows,
                tile_count,
                len(self.slice_method),
                tile_cols,
            )
            sliced_out = None
            max_data_out = None
            if (
                bool(getattr(engine, "triton_reuse_activation_slice_buffer", False))
                and self._activation_slice_buffer_key == buffer_key
            ):
                sliced_out = self._activation_slice_buffer
                max_data_out = self._activation_max_buffer
            want_voltage = (
                bool(getattr(engine, "triton_precompute_input_voltage", False))
                and bool(getattr(engine, "triton_direct_final_output", False))
                and bool(getattr(engine, "triton_binary_input_slice_dac", True))
                and getattr(engine, "mode", 0) == 0
            )
            if want_voltage:
                sliced, max_data, precomputed_v = triton_slice_mode0_2d_uniform1_with_voltage(
                    mat.squeeze(0),
                    input_slices=len(self.slice_method),
                    tile_cols=tile_cols,
                    qmax=max(2 ** (self.total_bits - 1) - 1, 1),
                    vread=float(getattr(engine, "vread", 0.2)),
                    voltage_dtype=(
                        torch.float32
                        if bool(getattr(engine, "triton_direct_final_exact_reduce", False))
                        else getattr(engine, "compute_dtype", torch.bfloat16)
                    ),
                    sliced_out=sliced_out,
                    max_data_out=max_data_out,
                )
            else:
                sliced, max_data = triton_slice_mode0_2d_uniform1(
                    mat.squeeze(0),
                    input_slices=len(self.slice_method),
                    tile_cols=tile_cols,
                    qmax=max(2 ** (self.total_bits - 1) - 1, 1),
                    sliced_out=sliced_out,
                    max_data_out=max_data_out,
                )
                precomputed_v = None
        except Exception:
            return False
        if bool(getattr(engine, "triton_reuse_activation_slice_buffer", False)):
            self._activation_slice_buffer_key = buffer_key
            self._activation_slice_buffer = sliced
            self._activation_max_buffer = max_data
        else:
            self._activation_slice_buffer_key = None
            self._activation_slice_buffer = None
            self._activation_max_buffer = None
        self.sliced_data = sliced
        self.max_data = max_data.to(torch.float32)
        self.precomputed_v_sliced = precomputed_v
        self.e_bias = None
        self.quantized_data = None
        self.shape = mat.squeeze(0).shape
        self.activation_slice_fused = True
        return True


    def _prepare_mode1_data(self, mat: torch.Tensor):
        """
        Prepare raw tensor data for Mode 1 without bit-slicing.

        Mode 1 directly projects data to differential-pair conductance states,
        so it must not reuse the Mode 0 two's-complement slicing quantizer.
        """
        if mat.dim() not in (2, 3):
            raise ValueError("Mode 1 data must be 2-D or 3-D.")

        self.quantized_data = mat.to(torch.float32).clone()
        self.sliced_data = None
        self.precomputed_v_sliced = None
        self.max_data = None
        self.e_bias = None


    def _slice_data_diff_pair(
        self,
        mat: torch.Tensor,
        skip_quantized: bool = False,
        keep_signed_view: bool = True,
        use_triton_activation: bool = False,
    ):
        """
        Slice signed data into unsigned positive and negative branches for Mode 2.

        Parameters:
            mat (torch.Tensor): Tensor to decompose, quantize, and slice.
            skip_quantized (bool): If True, skip dequantized branch buffers.
            keep_signed_view (bool): If True, also keep signed sliced_data as positive minus negative branches.

        Returns:
            None. Populates positive/negative branch buffers and shared scale metadata in place.
        """
        self.activation_slice_fused = False
        unsqueezed = False
        if mat.dim() == 2:
            mat = mat.unsqueeze(0)
            unsqueezed = True

        qg, ps, ngr, ngc, ndr, ndc = self._geom(mat)
        if self._try_slice_data_diff_pair_triton(
            mat,
            qg,
            ps,
            ngr,
            ngc,
            ndr,
            ndc,
            skip_quantized,
            keep_signed_view,
            use_triton_activation,
            unsqueezed,
        ):
            return

        padded = self._pad_to_quant_blocks(mat, ngr, ngc, qg)
        max_shared = self._compute_tiled_max_abs(padded, ndr, ndc)
        tiled_signed = self._tile_from_padded(padded, ngr, ngc, ndr, ndc, ps)
        del padded

        sd_p, qd_p, _, eb_p = self._quant_map_tensor_unsigned(
            torch.clamp(tiled_signed, min=0.0),
            max_shared,
            skip_quantized=skip_quantized,
        )
        sd_n, qd_n, _, eb_n = self._quant_map_tensor_unsigned(
            torch.clamp(-tiled_signed, min=0.0),
            max_shared,
            skip_quantized=skip_quantized,
        )
        del tiled_signed

        self.sliced_data_p = sd_p
        self.sliced_data_n = sd_n
        self.max_data_p = self.max_data_n = max_shared
        self.e_bias_p = eb_p
        self.e_bias_n = eb_n

        if keep_signed_view:
            signed_dtype = self._signed_diff_dtype()
            self.sliced_data = sd_p.to(signed_dtype) - sd_n.to(signed_dtype)
        else:
            self.sliced_data = None

        if not skip_quantized and qd_p is not None and qd_n is not None:
            self.quantized_data_p = self._untile_quantized(
                qd_p, mat.shape, ngr, ngc, ndr, ndc, ps
            )
            self.quantized_data_n = self._untile_quantized(
                qd_n, mat.shape, ngr, ngc, ndr, ndc, ps
            )
            self.quantized_data = self.quantized_data_p - self.quantized_data_n
        else:
            self.quantized_data_p = None
            self.quantized_data_n = None
            self.quantized_data = None

        self.max_data = max_shared
        self.e_bias = None

        if unsqueezed:
            if self.sliced_data is not None:
                self.sliced_data = self.sliced_data.squeeze(0)
            self.sliced_data_p = self.sliced_data_p.squeeze(0)
            self.sliced_data_n = self.sliced_data_n.squeeze(0)
            if self.quantized_data is not None:
                self.quantized_data = self.quantized_data.squeeze(0)
                self.quantized_data_p = self.quantized_data_p.squeeze(0)
                self.quantized_data_n = self.quantized_data_n.squeeze(0)
            self.max_data = self.max_data.squeeze(0)
            self.max_data_p = self.max_data_p.squeeze(0)
            self.max_data_n = self.max_data_n.squeeze(0)

    def _try_slice_data_diff_pair_triton(
        self,
        mat,
        qg,
        ps,
        ngr,
        ngc,
        ndr,
        ndc,
        skip_quantized,
        keep_signed_view,
        use_triton_activation,
        unsqueezed,
    ):
        """Try fused activation slicing for Mode-2 differential input."""
        if not (
            use_triton_activation
            and skip_quantized
            and keep_signed_view
            and unsqueezed
            and mat.dim() == 3
            and mat.shape[0] == 1
            and mat.is_cuda
            and mat.dtype in (torch.bfloat16, torch.float32)
            and ndr == 1
            and int(ps[0]) == 1
            and int(qg[0]) == int(ps[0])
            and int(qg[1]) % int(ps[1]) == 0
            and int(self.slice_method.max().item()) <= 8
        ):
            return False
        try:
            from memintelli.pimpy.triton_fast_accumulate import triton_slice_mode2_diff_2d_uniform
            sd_p, sd_n, max_shared = triton_slice_mode2_diff_2d_uniform(
                mat.squeeze(0),
                slice_bits=tuple(int(v.item()) for v in self.slice_method),
                tile_cols=int(ps[1]),
                quant_cols=int(qg[1]),
                qmax=max(2 ** self.total_bits - 1, 1),
            )
        except Exception:
            return False

        self.sliced_data_p = sd_p
        self.sliced_data_n = sd_n
        self.max_data_p = self.max_data_n = max_shared
        self.e_bias_p = self.e_bias_n = None
        signed_dtype = self._signed_diff_dtype()
        self.sliced_data = sd_p.to(signed_dtype) - sd_n.to(signed_dtype)
        self.quantized_data_p = None
        self.quantized_data_n = None
        self.quantized_data = None
        self.max_data = max_shared
        self.e_bias = None
        self.activation_slice_fused = True
        return True

    def _quant_map_tensor_unsigned(
        self,
        mat: torch.Tensor,
        max_abs,
        skip_quantized: bool = False,
    ):
        """
        Quantize and bit-slice a non-negative tensor branch.

        Parameters:
            mat (torch.Tensor): Non-negative tiled tensor with shape (batch, row_blocks, col_blocks, tile_rows, tile_cols).
            max_abs (torch.Tensor): Per-block absolute maximum values used for normalization.
            skip_quantized (bool): If True, return None instead of a dequantized tensor.

        Returns:
            tuple: (data_int, mat_dq_or_None, max_abs, None), matching the signed quantization helper interface.
        """
        qmax = float(max(2 ** self.total_bits - 1, 1))

        safe_max = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
        normalized = torch.clamp(mat / safe_max, min=0.0, max=1.0)
        matq = torch.round(normalized * qmax).int()

        mat_dq = None if skip_quantized else (matq / qmax * max_abs)
        branch_dtype = self._unsigned_branch_dtype()

        data_int = torch.empty(
            (
                mat.shape[0], mat.shape[1], mat.shape[2],
                len(self.slice_method),
                mat.shape[3], mat.shape[4],
            ),
            device=mat.device, dtype=branch_dtype,
        )
        bit_offset = 0
        for idx in range(len(self.slice_method)):
            sw = int(self.slice_method[-1 - idx].item())
            low = 2 ** bit_offset
            high = 2 ** (bit_offset + sw)
            data_int[:, :, :, idx, :, :] = ((matq - matq % low) % high) >> bit_offset
            bit_offset += sw

        del matq

        return data_int, mat_dq, max_abs, None

    def _geom(self, mat):
        """
        Compute quantization and tile geometry for a tensor.

        Parameters:
            mat (torch.Tensor): Tensor with shape (batch, rows, cols).

        Returns:
            tuple: (qg, ps, ngr, ngc, ndr, ndc), where qg is quant granularity, ps is tile size, ngr/ngc are quant block counts, and ndr/ndc are tile counts per quant block.
        """
        ps = self.paral_size
        qg = self.quant_gran

        if qg == "per-matrix":
            qg = mat.shape[1:]
        elif qg == "per-row":
            qg = (1, mat.shape[2])
        elif qg == "per-col":
            qg = (mat.shape[1], 1)

        qg = list(qg)
        if not isinstance(self.quant_gran, str) and any(g % p for g, p in zip(qg, ps)):
            raise ValueError(
                "This legacy path cannot honor sub-array quant_gran. "
                "Use SimulationEngine for independent weight column scales."
            )
        if self.quant_gran in ("per-row", "per-col") and any(g < p for g, p in zip(qg, ps)):
            raise ValueError("Legacy per-row/per-col scales cannot be enlarged silently; use SimulationEngine")
        qg[0] = math.ceil(qg[0] / ps[0]) * ps[0]
        qg[1] = math.ceil(qg[1] / ps[1]) * ps[1]

        ngr = math.ceil(mat.shape[1] / qg[0])
        ngc = math.ceil(mat.shape[2] / qg[1])
        ndr = qg[0] // ps[0]
        ndc = qg[1] // ps[1]
        return qg, ps, ngr, ngc, ndr, ndc

    def _unsigned_branch_dtype(self):
        """
        Choose the integer dtype used for unsigned branch slices.

        Parameters:
            None.

        Returns:
            torch.dtype: uint8 when each slice fits in 8 bits, otherwise int16.
        """
        return torch.uint8 if int(self.slice_method.max().item()) <= 8 else torch.int16

    def _signed_diff_dtype(self):
        """
        Choose the integer dtype used for signed Mode 2 slice differences.

        Parameters:
            None.

        Returns:
            torch.dtype: int16 for small slices, otherwise int32.
        """
        return torch.int16 if int(self.slice_method.max().item()) <= 8 else torch.int32

    def _pad_to_quant_blocks(self, mat, ngr, ngc, qg):
        """
        Pad a tensor so its rows and columns fit complete quantization blocks.

        Parameters:
            mat (torch.Tensor): Tensor with shape (batch, rows, cols).
            ngr (int): Number of quantization block rows.
            ngc (int): Number of quantization block columns.
            qg (tuple): Quantization block size as (rows, cols).

        Returns:
            torch.Tensor: Padded and reshaped tensor grouped by quantization block.
        """
        B, device = mat.shape[0], mat.device
        padded = torch.zeros(
            (B, ngr * qg[0], ngc * qg[1]),
            device=device,
            dtype=mat.dtype,
        )
        padded[:, :mat.shape[1], :mat.shape[2]] = mat
        return padded.reshape(B, ngr, qg[0], ngc, qg[1]).transpose(2, 3)

    def _compute_tiled_max_abs(self, padded, ndr, ndc):
        """
        Compute per-tile maximum absolute values from padded quantization blocks.

        Parameters:
            padded (torch.Tensor): Padded tensor grouped by quantization block.
            ndr (int): Number of tile rows per quantization block.
            ndc (int): Number of tile columns per quantization block.

        Returns:
            torch.Tensor: Per-tile maximum absolute values with shape (batch, tile_block_rows, tile_block_cols, 1, 1).
        """
        B, ngr, ngc = padded.shape[0], padded.shape[1], padded.shape[2]
        padded_abs = padded.abs()
        block_max = padded_abs.amax(dim=-1, keepdim=True).amax(dim=-2, keepdim=True)
        del padded_abs
        return (
            block_max
            .unsqueeze(3).unsqueeze(4)
            .expand(-1, -1, -1, ndr, ndc, -1, -1)
            .transpose(2, 3)
            .reshape(B, ngr * ndr, ngc * ndc, 1, 1)
        )

    def _tile_from_padded(self, padded, ngr, ngc, ndr, ndc, ps):
        """
        Split a padded tensor into hardware-sized tiles.

        Parameters:
            padded (torch.Tensor): Padded tensor grouped by quantization block.
            ngr (int): Number of quantization block rows.
            ngc (int): Number of quantization block columns.
            ndr (int): Number of tile rows per quantization block.
            ndc (int): Number of tile columns per quantization block.
            ps (tuple): Hardware tile size as (rows, cols).

        Returns:
            torch.Tensor: Tiled tensor with shape (batch, tile_block_rows, tile_block_cols, tile_rows, tile_cols).
        """
        B = padded.shape[0]
        return (
            padded
            .reshape(B, ngr, ngc, ndr, ps[0], ndc, ps[1])
            .transpose(4, 5)
            .transpose(2, 3)
            .reshape(B, ngr * ndr, ngc * ndc, ps[0], ps[1])
        )

    def _tile(self, mat, ngr, ngc, ndr, ndc, qg, ps, max_override=None):
        """
        Pad, scale, and tile a tensor for localized quantization.

        Parameters:
            mat (torch.Tensor): Tensor with shape (batch, rows, cols).
            ngr (int): Number of quantization block rows.
            ngc (int): Number of quantization block columns.
            ndr (int): Number of tile rows per quantization block.
            ndc (int): Number of tile columns per quantization block.
            qg (tuple): Quantization block size as (rows, cols).
            ps (tuple): Hardware tile size as (rows, cols).
            max_override (torch.Tensor or None): Optional precomputed per-tile maximum values.

        Returns:
            tuple: (tiled, max_abs), where tiled is the hardware-tiled tensor and max_abs is its scale tensor.
        """
        B = mat.shape[0]
        padded = self._pad_to_quant_blocks(mat, ngr, ngc, qg)

        if max_override is None:
            max_abs = self._compute_tiled_max_abs(padded, ndr, ndc)
        else:
            max_abs = max_override.to(mat.device)
            if max_abs.dim() == 4 and B == 1:
                max_abs = max_abs.unsqueeze(0)
            expected = (B, ngr * ndr, ngc * ndc, 1, 1)
            if tuple(max_abs.shape) != expected:
                raise ValueError(
                    f"max_override shape mismatch: expected {expected}, "
                    f"got {tuple(max_abs.shape)}"
                )

        tiled = self._tile_from_padded(padded, ngr, ngc, ndr, ndc, ps)
        del padded
        return tiled, max_abs

    @staticmethod
    def _untile_quantized(qd, orig_shape, ngr, ngc, ndr, ndc, ps):
        """
        Restore a tiled dequantized tensor to the original matrix layout.

        Parameters:
            qd (torch.Tensor): Dequantized tiled tensor.
            orig_shape (tuple or torch.Size): Original shape before padding and tiling.
            ngr (int): Number of quantization block rows.
            ngc (int): Number of quantization block columns.
            ndr (int): Number of tile rows per quantization block.
            ndc (int): Number of tile columns per quantization block.
            ps (tuple): Hardware tile size as (rows, cols).

        Returns:
            torch.Tensor: Dequantized tensor cropped back to orig_shape.
        """
        B = orig_shape[0]
        qg0 = ndr * ps[0]
        qg1 = ndc * ps[1]
        return (
            qd
            .reshape(B, ngr, ndr, ngc, ndc, ps[0], ps[1])
            .transpose(2, 3)
            .transpose(4, 5)
            .reshape(B, ngr, ngc, qg0, qg1)
            .transpose(2, 3)
            .reshape(B, ngr * qg0, ngc * qg1)
            [:, :orig_shape[1], :orig_shape[2]]
        )

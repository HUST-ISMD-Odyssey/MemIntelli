"""Lossless encoding of four binary conductance-index slices."""
import torch


def pack_indices(indices):
    """Encode uint8 [M,P,4,64,64] indices (0 or 15) as [M,P,1,64,32]."""
    if indices.ndim != 5 or indices.shape[2:] != (4, 64, 64) or indices.dtype != torch.uint8:
        raise ValueError("Expected uint8 indices with shape [M,P,4,64,64]")
    packed = torch.empty((*indices.shape[:2], 1, 64, 32),
                         device=indices.device, dtype=torch.uint8)
    for start in range(0, indices.shape[1], 16):
        slab = indices[:, start:start + 16]
        if not bool(((slab == 0) | (slab == 15)).all()):
            raise ValueError("Compact state supports only indices 0 and 15")
        nibble = torch.zeros((*slab.shape[:2], 64, 64),
                             device=indices.device, dtype=torch.uint8)
        for bit in range(4):
            nibble.bitwise_or_((slab[:, :, bit] == 15).to(torch.uint8) << bit)
        packed[:, start:start + 16, 0] = nibble[..., ::2] | (nibble[..., 1::2] << 4)
    return packed


def unpack_indices(packed):
    """Diagnostic inverse; not used on the inference hot path."""
    if packed.ndim != 5 or packed.shape[2:] != (1, 64, 32) or packed.dtype != torch.uint8:
        raise ValueError("Expected uint8 compact state with shape [M,P,1,64,32]")
    out = torch.empty((*packed.shape[:2], 4, 64, 64),
                      device=packed.device, dtype=torch.uint8)
    for bit in range(4):
        out[:, :, bit, :, ::2] = ((packed[:, :, 0] >> bit) & 1) * 15
        out[:, :, bit, :, 1::2] = ((packed[:, :, 0] >> (bit + 4)) & 1) * 15
    return out

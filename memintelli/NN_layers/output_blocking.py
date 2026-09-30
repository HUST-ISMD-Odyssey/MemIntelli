import concurrent.futures

import torch
import torch.nn as nn


class OutputBlockedLinearMem(nn.Module):
    def __init__(
        self,
        *,
        blocks,
        block_ranges,
        in_features,
        out_features,
        block_devices=None,
        parallel=False,
    ):
        super().__init__()
        if len(blocks) != len(block_ranges):
            raise ValueError("blocks and block_ranges must have the same length")
        if not blocks:
            raise ValueError("at least one output block is required")
        self.blocks = nn.ModuleList(blocks)
        self.block_ranges = [(int(start), int(end)) for start, end in block_ranges]
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.shard_count = len(self.blocks)
        self.parallel = bool(parallel)
        if block_devices is None:
            block_devices = [None] * len(self.blocks)
        if len(block_devices) != len(self.blocks):
            raise ValueError("block_devices and blocks must have the same length")
        self.block_devices = [torch.device(device) if device is not None else None for device in block_devices]

    @property
    def shards(self):
        return self.blocks

    @property
    def shard_ranges(self):
        return self.block_ranges

    @property
    def shard_devices(self):
        return self.block_devices

    @property
    def weight_sliced(self):
        return self.blocks[0].weight_sliced

    @property
    def engine(self):
        return self.blocks[0].engine

    def iter_blocks(self):
        return iter(self.blocks)

    def _run_block(self, block, block_device, input_tensor, return_device):
        if block_device is None:
            block_device = input_tensor.device
        block_input = (
            input_tensor
            if input_tensor.device == block_device
            else input_tensor.to(block_device, non_blocking=True)
        )
        output = block(block_input)
        if output.device != return_device:
            output = output.to(return_device, non_blocking=True)
        return output

    def _run_parallel(self, input_tensor, return_device):
        outputs = [None] * len(self.blocks)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.blocks)) as executor:
            futures = {
                executor.submit(
                    self._run_block,
                    block,
                    device,
                    input_tensor,
                    return_device,
                ): index
                for index, (block, device) in enumerate(zip(self.blocks, self.block_devices))
            }
            for future in concurrent.futures.as_completed(futures):
                outputs[futures[future]] = future.result()
        return outputs

    def forward(self, input_tensor):
        return_device = input_tensor.device
        if self.parallel and len(self.blocks) > 1:
            block_outputs = self._run_parallel(input_tensor, return_device)
            output_shape = (*input_tensor.shape[:-1], self.out_features)
            output = block_outputs[0].new_empty(output_shape, device=return_device)
            for block_output, (start, end) in zip(block_outputs, self.block_ranges):
                output[..., start:end].copy_(block_output)
            return output

        output = None
        for block, device, (start, end) in zip(
            self.blocks,
            self.block_devices,
            self.block_ranges,
        ):
            block_output = self._run_block(block, device, input_tensor, return_device)
            if output is None:
                output_shape = (*input_tensor.shape[:-1], self.out_features)
                output = block_output.new_empty(output_shape, device=return_device)
            output[..., start:end].copy_(block_output)
            del block_output
        return output

    def enable_lazy_inference(self, **kwargs):
        for block in self.blocks:
            if hasattr(block, "enable_lazy_inference"):
                block.enable_lazy_inference(**kwargs)
        return self

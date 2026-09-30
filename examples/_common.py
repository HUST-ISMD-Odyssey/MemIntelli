"""Shared command-line options for the runnable examples."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from memintelli import SimulationEngine


def _nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return result


def _positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--backend", choices=("auto", "triton", "torch"), default="auto")
    p.add_argument("--mode", choices=("speed", "accurate"), default="speed")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--activation-bits", type=int, choices=range(2, 17), default=6)
    p.add_argument("--weight-bits", type=int, choices=range(2, 17), default=6)
    p.add_argument("--input-slice", type=int, choices=range(1, 9), nargs="+")
    p.add_argument("--weight-slice", type=int, choices=range(1, 9), nargs="+")
    p.add_argument("--array-size", type=_positive_int, nargs=2, default=(64, 64))
    p.add_argument("--input-quant-gran", type=_positive_int, nargs=2)
    p.add_argument("--weight-quant-gran", type=_positive_int, nargs=2)
    p.add_argument("--adc-bits", type=int, choices=range(1, 17), default=6)
    p.add_argument("--dac-bits", type=int, choices=range(1, 17), default=1)
    p.add_argument("--adc-clip", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--torch-fuse-adc", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--write-variation", type=float, default=0.0)
    p.add_argument("--read-variation", type=float, default=0.0)
    p.add_argument("--variation-json", type=Path, help="Per-state write_variation/read_variation/drift_coefficient")
    p.add_argument("--drift-coefficient", type=float, default=0.0)
    p.add_argument("--drift-time", type=float, default=0.0)
    p.add_argument("--drift-reference-time", type=float, default=1.0)
    p.add_argument("--hgs", type=float, default=1e-5)
    p.add_argument("--lgs", type=float, default=1e-7)
    p.add_argument("--g-level", type=int, default=16)
    p.add_argument("--vread", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--program-epoch", type=_nonnegative_int, default=0)
    p.add_argument("--input-chunk-rows", type=_positive_int, default=256)
    p.add_argument("--output-chunk-tiles", type=_positive_int, default=8)
    p.add_argument("--batch-size", type=_positive_int, default=8)
    p.add_argument("--limit", type=_nonnegative_int, default=0, help="Limit evaluated samples; 0 means the complete split")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--download", action="store_true")
    p.add_argument("--digital", action="store_true", help="Evaluate the floating-point software model instead")
    p.add_argument("--workers", type=_nonnegative_int, default=0)
    p.add_argument("--output", type=Path, help="Optional JSON summary; no output files are written by default")
    return p


def engine(args):
    if args.limit < 0 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("Invalid sample limit, batch size or worker count")
    torch.manual_seed(args.seed)
    per_state = json.loads(args.variation_json.read_text()) if args.variation_json else {}
    allowed = {"write_variation", "read_variation", "drift_coefficient"}
    if set(per_state)-allowed:
        raise ValueError("Unknown variation configuration keys")
    return SimulationEngine(
        backend=args.backend, mode=args.mode, device=args.device,
        activation_bits=args.activation_bits, weight_bits=args.weight_bits,
        input_slice=args.input_slice, weight_slice=args.weight_slice,
        weight_paral_size=args.array_size, input_quant_gran=args.input_quant_gran,
        weight_quant_gran=args.weight_quant_gran, adc_bits=args.adc_bits, dac_bits=args.dac_bits,
        adc_clip=args.adc_clip, write_variation=per_state.get("write_variation", args.write_variation),
        read_variation=per_state.get("read_variation", args.read_variation),
        drift_coefficient=per_state.get("drift_coefficient", args.drift_coefficient),
        drift_time=args.drift_time, drift_reference_time=args.drift_reference_time,
        HGS=args.hgs, LGS=args.lgs, g_level=args.g_level, vread=args.vread,
        seed=args.seed, program_epoch=args.program_epoch, input_chunk_rows=args.input_chunk_rows,
        output_chunk_tiles=args.output_chunk_tiles, torch_fuse_adc=args.torch_fuse_adc,
    )


def report(args, result, simulator=None):
    if simulator is not None:
        result["simulation"] = simulator.describe()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def loader(dataset, args, shuffle=False):
    if args.limit:
        dataset = torch.utils.data.Subset(dataset, range(min(args.limit, len(dataset))))
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                                       num_workers=args.workers)

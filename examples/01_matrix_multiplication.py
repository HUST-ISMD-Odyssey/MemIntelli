"""Bit-sliced matrix multiplication with configurable arrays and quantization."""
import torch

from _common import engine, parser, report


def main():
    p = parser(__doc__)
    p.add_argument("--shape", type=int, nargs=3, default=(8, 128, 128), metavar=("N", "K", "M"))
    args = p.parse_args()
    simulator = engine(args)
    n, k, m = args.shape
    if min(n, k, m) < 1:
        raise ValueError("Matrix dimensions must be positive")
    x, weight = torch.randn(n, k, device=args.device), torch.randn(k, m, device=args.device)
    mapped = simulator.map_weight(weight)
    with torch.no_grad():
        output = simulator.matmul(x, mapped)
        reference = x @ weight
    report(args, {"shape": args.shape, "rmse_against_fp32": float((output-reference).square().mean().sqrt()),
                  "maximum_absolute_error": float((output-reference).abs().max()),
                  "physical_arrays": int(mapped.G_indices.shape[0]*mapped.G_indices.shape[1]*mapped.G_indices.shape[2])},
           simulator)


if __name__ == "__main__":
    main()

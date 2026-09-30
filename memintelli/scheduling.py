"""Shape-based software tiling; physical arrays and numerical rules stay fixed."""
import math


def plan_chunks(samples, shape, input_slices, budget):
    """Minimize dispatches under a temporary-memory budget, without timed trials."""
    m, p, ns, k, l = shape
    ni = input_slices
    candidates = sorted({1, p, *(2**i for i in range(p.bit_length()) if 2**i <= p)})
    best = None
    for tiles in candidates:
        columns = tiles*l
        # Packed conductance plus the copy needed for a partial last output tile.
        fixed = (6 if samples == 1 else 4)*m*k*ns*columns
        # Sliced inputs, voltages, partial outputs, currents, and input preparation.
        per_row = m*(3*ni*k + 4*columns + 2*ni*ns*columns + 12*k)
        rows = min(samples, max(1, (budget-fixed)//max(1, per_row)))
        if rows >= 32 and rows < samples:
            rows = rows//32*32
        # Balance the tail instead of leaving a tiny, underutilized final GEMM.
        rows = math.ceil(samples/math.ceil(samples/rows))
        estimate = fixed + per_row*rows
        if estimate > budget and tiles != 1:
            continue
        input_calls, output_calls = math.ceil(samples/rows), math.ceil(p/tiles)
        # First reduce dispatches; among ties avoid regenerating weight noise.
        score = (input_calls*output_calls, input_calls, estimate)
        if best is None or score < best[0]:
            best = (score, rows, tiles, estimate)
    _, rows, tiles, estimate = best
    return {"input_rows": rows, "output_tiles": tiles, "estimated_temporary_bytes": estimate,
            "budget_bytes": budget, "minimum_exceeds_budget": estimate > budget}

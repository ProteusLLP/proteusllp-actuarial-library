"""Measure Sobol block memory, generation throughput, and Poisson cost.

Run on the CPU or with PAL_USE_GPU=1 on the PAL GPU runner. The benchmark
measures a complete randomized block, not just extraction of a cached column.
GPU events are synchronized before elapsed-time measurements.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from pal._maths import asnumpy, xp
from pal.qmc import RandomizedQMCGenerator


def _sync() -> None:
    if xp is not np:
        xp.cuda.Device().synchronize()


def profile_size(n_sims: int, dimensions: int) -> dict:
    """Profile complete Sobol block generation and fixed-lambda Poisson."""
    gc.collect()
    tracemalloc.start()
    rng = RandomizedQMCGenerator(seed=1099, dimensions=dimensions)
    _sync()
    start = time.perf_counter()
    first = rng.random(n_sims)
    _sync()
    first_coordinate_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    for _ in range(dimensions - 1):
        rng.random(n_sims)
    _sync()
    remaining_coordinates_ms = (time.perf_counter() - start) * 1000
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    retained_bytes = sum(block.nbytes for block in rng._streams.values())
    first_mean = float(np.mean(asnumpy(first)))
    assert abs(first_mean - 0.5) < 0.005
    assert len(rng._streams) == 1
    del rng, first
    gc.collect()

    poisson_rng = RandomizedQMCGenerator(seed=1099, dimensions=dimensions)
    _sync()
    start = time.perf_counter()
    counts = poisson_rng.poisson(4.0, size=n_sims)
    _sync()
    poisson_ms = (time.perf_counter() - start) * 1000
    poisson_mean = float(np.mean(asnumpy(counts)))
    assert abs(poisson_mean - 4) < 0.03

    return {
        "n_sims": n_sims,
        "dimensions": dimensions,
        "first_coordinate_ms": round(first_coordinate_ms, 3),
        "remaining_coordinates_ms": round(remaining_coordinates_ms, 3),
        "poisson_ms": round(poisson_ms, 3),
        "retained_block_mib": round(retained_bytes / 2**20, 2),
        "cpu_peak_tracemalloc_mib": round(peak_bytes / 2**20, 2),
        "gpu_pool_used_mib": (
            round(xp.get_default_memory_pool().used_bytes() / 2**20, 2)
            if xp is not np
            else None
        ),
        "poisson_mean": round(poisson_mean, 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="131072,1048576")
    parser.add_argument("--dimensions", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("pal-qmc-scaling.json"))
    args = parser.parse_args()
    rows = [profile_size(int(n), args.dimensions) for n in args.sizes.split(",")]
    result = {"backend": "gpu" if xp is not np else "cpu", "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

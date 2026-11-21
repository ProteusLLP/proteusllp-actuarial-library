"""Benchmark compact PAL Parquet against the original long-form Parquet.

Run with:
    python -m benchmarks.benchmark_variable_io --simulations 20000 --variables 20
    python -m benchmarks.benchmark_variable_io --simulations 100000 --variables 100 --compact-only

This is a descriptive benchmark. Timings are wall-clock observations; memory
from tracemalloc covers Python allocations, not all native NumPy/Arrow buffers.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from pal import ProteusVariable, StochasticScalar
from pal.serialization import load, save
from pal.variable_io import to_dataframe as encode_snapshot


def measure(fn: Callable[[], Any]) -> tuple[float, float]:
    """Return wall time and peak traced Python allocation for one operation."""
    tracemalloc.start()
    started = time.perf_counter()
    try:
        fn()
        seconds = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return seconds, peak / (1024 * 1024)


def run(simulations: int, variables: int, compare: bool) -> dict[str, Any]:
    """Compare write/read costs and check round-trip values."""
    import pyarrow

    rng = np.random.default_rng(42)
    losses = ProteusVariable(
        "LoB",
        {f"LoB_{i:03d}": StochasticScalar(rng.lognormal(mean=10, sigma=1.5, size=simulations))
         for i in range(variables)},
    )
    result: dict[str, Any] = {
        "simulations": simulations,
        "variables": variables,
        "values": simulations * variables,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "platform": platform.platform(),
    }

    with tempfile.TemporaryDirectory(prefix="pal-parquet-benchmark-") as directory:
        compact_file = Path(directory) / "compact.parquet"
        write_seconds, write_python_peak = measure(lambda: save(losses, compact_file))
        restored: ProteusVariable[Any] | None = None

        def read_compact() -> None:
            nonlocal restored
            restored = load(compact_file)

        read_seconds, read_python_peak = measure(read_compact)
        assert restored is not None
        for label, value in losses.values.items():
            np.testing.assert_array_equal(restored[label].values, value.values)
        result["compact"] = {
            "write_seconds": write_seconds,
            "read_seconds": read_seconds,
            "python_peak_write_mib": write_python_peak,
            "python_peak_read_mib": read_python_peak,
            "bytes": os.path.getsize(compact_file),
        }

        if compare:
            legacy_file = Path(directory) / "row_form.parquet"
            old_write_seconds, old_write_python_peak = measure(
                lambda: encode_snapshot(losses).to_parquet(legacy_file, index=False)
            )
            old_read_seconds, old_read_python_peak = measure(
                lambda: load(legacy_file)
            )
            result["legacy_row_format"] = {
                "write_seconds": old_write_seconds,
                "read_seconds": old_read_seconds,
                "python_peak_write_mib": old_write_python_peak,
                "python_peak_read_mib": old_read_python_peak,
                "bytes": os.path.getsize(legacy_file),
            }
            result["write_speedup"] = old_write_seconds / write_seconds
            result["read_speedup"] = old_read_seconds / read_seconds
            result["file_size_ratio"] = os.path.getsize(legacy_file) / os.path.getsize(compact_file)
    return result


def main() -> None:
    """Run the benchmark and print machine-readable results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=20000)
    parser.add_argument("--variables", type=int, default=20)
    parser.add_argument("--compact-only", action="store_true", help="Skip the costly legacy row-format baseline")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    if args.simulations <= 0 or args.variables <= 0:
        parser.error("simulations and variables must be positive")
    if not args.compact_only and args.simulations * args.variables > 2_000_000:
        parser.error("Legacy benchmark exceeds 2 million values; use --compact-only")
    result = run(args.simulations, args.variables, not args.compact_only)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

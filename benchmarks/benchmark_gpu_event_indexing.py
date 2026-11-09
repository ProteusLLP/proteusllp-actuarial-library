"""Compare GPU frequency-severity event-index algorithms on a CUDA runner.

Run with PAL_USE_GPU=1. Inputs are constructed outside measured sections.
Every recorded time includes output allocation and device synchronisation.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np

from benchmarks.benchmark_core import environment
from pal._maths import xp
from pal.frequency_severity import _get_sims_of_events

SCENARIOS = ("sparse", "dense", "skewed", "concentrated")
METHODS = ("current", "repeat", "prefix_fill")

_FILL_SOURCE = r"""
extern "C" __global__
void fill_event_indices(const long long* counts, const long long* ends,
                        long long* indices, const long long n_sims) {
    const long long sim = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (sim >= n_sims) return;
    const long long start = ends[sim] - counts[sim];
    const long long count = counts[sim];
    for (long long j = 0; j < count; ++j) indices[start + j] = sim;
}
"""


def make_counts(scenario: str, n_sims: int) -> np.ndarray:
    """Construct reproducible sparse, dense or heavily concentrated counts."""
    if scenario not in SCENARIOS or n_sims < 0:
        raise ValueError("Unknown scenario or negative simulation count")
    rng = np.random.default_rng(133 + n_sims)
    if scenario == "sparse":
        return rng.poisson(0.2, size=n_sims).astype(np.int64)
    if scenario == "dense":
        return rng.poisson(10, size=n_sims).astype(np.int64)
    counts = np.zeros(n_sims, dtype=np.int64)
    if scenario == "concentrated":
        if n_sims:
            counts[0] = min(200000, n_sims * 10)
    else:
        counts[::1000] = 1000
    return counts


@lru_cache(maxsize=1)
def _fill_kernel() -> Any:
    """Compile the experimental CUDA segment-fill kernel once per process."""
    return xp.RawKernel(_FILL_SOURCE, "fill_event_indices")


def _prefix_fill(counts: Any) -> Any:
    """Fill each simulation's segment directly after the prefix sum."""
    if counts.size == 0:
        return xp.empty(0, dtype=xp.int64)
    ends = xp.cumsum(counts, dtype=xp.int64)
    # An output array with a data-dependent shape requires one host scalar.
    total = int(ends[-1])
    result = xp.empty(total, dtype=xp.int64)
    if total:
        threads = 256
        blocks = (counts.size + threads - 1) // threads
        _fill_kernel()((blocks,), (threads,), (counts, ends, result, np.int64(counts.size)))
    return result


def _repeat(counts: Any) -> Any:
    """Evaluate CuPy's device-count repeat path without a CPU count copy."""
    return xp.repeat(xp.arange(counts.size, dtype=xp.int64), counts)


def _method(name: str) -> Callable[[Any], Any]:
    return {"current": _get_sims_of_events, "repeat": _repeat, "prefix_fill": _prefix_fill}[name]


def _sync() -> None:
    xp.cuda.Device().synchronize()


def _timed_call(fn: Callable[[Any], Any], counts: Any) -> tuple[float, int, int]:
    """Measure wall time and pool occupancy with result still live."""
    pool = xp.get_default_memory_pool()
    _sync()
    before = pool.used_bytes()
    start = time.perf_counter_ns()
    result = fn(counts)
    _sync()
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    used = pool.used_bytes() - before
    reserved = pool.total_bytes()
    del result
    return elapsed_ms, used, reserved


def measure_method(
    name: str, counts: Any, expected: np.ndarray, warmup: int, repeat: int
) -> dict[str, Any]:
    """Validate a method before recording warmed and cold-pool timings."""
    fn = _method(name)
    try:
        result = fn(counts)
        _sync()
        if not isinstance(result, xp.ndarray) or result.dtype != xp.int64:
            raise AssertionError("GPU indices must remain a device int64 array")
        np.testing.assert_array_equal(xp.asnumpy(result), expected)
        del result

        for _ in range(warmup):
            value = fn(counts)
            _sync()
            del value

        # Warm compilation has completed. Clear cached allocations to measure
        # the first pool-cold call separately from steady-state timings.
        xp.get_default_memory_pool().free_all_blocks()
        cold_ms, cold_used, cold_reserved = _timed_call(fn, counts)
        timings: list[float] = []
        output_bytes: list[int] = []
        reserved_bytes: list[int] = []
        for _ in range(repeat):
            elapsed, occupied, reserved = _timed_call(fn, counts)
            timings.append(elapsed)
            output_bytes.append(occupied)
            reserved_bytes.append(reserved)
        median = statistics.median(timings)
        return {
            "status": "ok",
            "median_ms": median,
            "mad_ms": statistics.median(abs(t - median) for t in timings),
            "cold_ms": cold_ms,
            "cold_pool_occupied_bytes": cold_used,
            "cold_pool_reserved_bytes": cold_reserved,
            "warm_pool_occupied_bytes": max(output_bytes),
            "warm_pool_reserved_bytes": max(reserved_bytes),
        }
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def run(sizes: list[int], warmup: int, repeat: int) -> dict[str, Any]:
    """Record comparable timings and correctness for every GPU workload."""
    if xp is np:
        raise RuntimeError("CUDA is required: run with PAL_USE_GPU=1")
    if warmup < 0 or repeat < 1 or any(n < 0 for n in sizes):
        raise ValueError("Warmup and sizes must be non-negative; repeat must be positive")
    rows = []
    for n_sims in sizes:
        for scenario in SCENARIOS:
            counts_host = make_counts(scenario, n_sims)
            expected = np.repeat(np.arange(n_sims, dtype=np.int64), counts_host)
            counts_gpu = xp.asarray(counts_host)
            for method in METHODS:
                result = measure_method(method, counts_gpu, expected, warmup, repeat)
                row = {
                    "simulations": n_sims,
                    "scenario": scenario,
                    "events": len(expected),
                    "method": method,
                    **result,
                }
                rows.append(row)
                print(f"{n_sims:>9,} {scenario:>7} {method:>11}: {row.get('median_ms', row.get('error'))}")
            del counts_gpu, expected
    return {
        "schema_version": 1,
        "environment": environment(),
        "warmup": warmup,
        "repeat": repeat,
        "rows": rows,
    }


def significant_gain(baseline: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """Screen for gains larger than noise and a ten-percent practical margin."""
    if baseline["status"] != "ok" or candidate["status"] != "ok":
        return False
    gain = baseline["median_ms"] - candidate["median_ms"]
    noise = 3 * (baseline["mad_ms"] + candidate["mad_ms"])
    return gain > max(noise, baseline["median_ms"] * 0.10)


def markdown_report(data: dict[str, Any]) -> str:
    """Format measurements and tentative regime-by-regime break-even evidence."""
    env = data["environment"]
    lines = [
        "## GPU event-indexing comparison (#133)",
        "",
        f"GPU: {env['gpu']}; CuPy: {env['cupy']}; CUDA runtime: {env['cuda_runtime_version']}",
        f"Commit: {env['commit']}; repetitions: {data['repeat']}; warmup: {data['warmup']}",
        "",
        "| Simulations | Pattern | Events | Method | Warm median (ms) | MAD (ms) | Cold (ms) | "
        "Ratio vs current | Screening result |",
        "| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    groups: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    for row in data["rows"]:
        groups.setdefault((row["simulations"], row["scenario"]), {})[row["method"]] = row
    for (n_sims, scenario), methods in groups.items():
        reference = methods["current"]
        for name in METHODS:
            row = methods[name]
            if row["status"] != "ok":
                lines.append(
                    f"| {n_sims:,} | {scenario} | {row['events']:,} | {name} | - | - | - | - | {row['error']} |"
                )
                continue
            ratio = row["median_ms"] / reference["median_ms"] if reference["status"] == "ok" else float("nan")
            decision = "candidate" if name != "current" and significant_gain(reference, row) else "-"
            lines.append(
                f"| {n_sims:,} | {scenario} | {row['events']:,} | {name} | "
                f"{row['median_ms']:.3f} | {row['mad_ms']:.3f} | {row['cold_ms']:.3f} | "
                f"{ratio:.2f}x | {decision} |"
            )
    lines.extend([
        "",
        "A 'candidate' has a >10% gain exceeding three summed MADs in this run;",
        "it is **not** approval to change the implementation without repeat measurements.",
        "Both cold and warm timings include output allocation and CUDA synchronisation.",
        "GPU inputs, compilation, NumPy expected-output construction and correctness copies are outside timings.",
        "Memory-pool occupied/reserved bytes are in JSON, but do not measure transient peak VRAM.",
        "Nsight Systems traces, when available, reveal kernel launch, allocation and host-sync costs.",
    ])
    return "\n".join(lines) + "\n"


def trace_one() -> None:
    """Emit NVTX ranges for a representative profile with Nsight Systems."""
    if xp is np:
        raise RuntimeError("CUDA is required for event-index tracing")
    from cupyx.profiler import time_range

    for scenario in SCENARIOS:
        counts = xp.asarray(make_counts(scenario, 100000))
        for name in METHODS:
            fn = _method(name)
            # Compile before opening the instrumented range.
            try:
                value = fn(counts)
                _sync()
                del value
                with time_range(f"event-index/{scenario}/{name}", sync=True):
                    value = fn(counts)
                    _sync()
                    del value
            except (TypeError, ValueError, NotImplementedError) as exc:
                print(f"Trace skipped {scenario}/{name}: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """Execute the GPU comparison or NVTX profiling workload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="10000,100000,1000000")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("gpu-event-indexing.json"))
    parser.add_argument("--summary", type=Path, default=Path("gpu-event-indexing.md"))
    parser.add_argument("--trace-only", action="store_true")
    args = parser.parse_args(argv)
    if args.trace_only:
        trace_one()
        return 0
    result = run([int(v) for v in args.sizes.split(",")], args.warmup, args.repeat)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.summary.write_text(markdown_report(result), encoding="utf-8")
    return int(any(row["status"] != "ok" for row in result["rows"] if row["method"] != "repeat"))


if __name__ == "__main__":
    sys.exit(main())

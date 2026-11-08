"""Measure PAL peak memory in isolated processes.

`python -m benchmarks.profile_memory --help` writes JSON and Markdown.
CPU RSS is the OS process high-water mark (including native NumPy allocations).
GPU allocation is CuPy's caching-pool high-water reservation, not total VRAM.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import traceback
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

from pal import set_random_seed
from pal._maths import xp

from benchmarks.benchmark_core import DEFAULT_CASES, build_workload, environment, synchronise
from benchmarks.profile_portfolio import run_portfolio


def _peak_rss_bytes() -> int:
    """Read the OS high-water RSS, respecting Linux and macOS units."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def _pool_reserved_bytes() -> int | None:
    """Report CuPy's reserved default-device memory pool, if available."""
    if xp is np:
        return None
    return int(xp.get_default_memory_pool().total_bytes())


def worker(case: str, n_sims: int, n_lobs: int, n_layers: int) -> dict[str, Any]:
    """Measure one workload in a clean process; includes workload preparation."""
    set_random_seed(20261108)
    tracemalloc.start()
    rss_baseline = _peak_rss_bytes()
    gpu_baseline = _pool_reserved_bytes()

    if case == "portfolio":
        workload = lambda: run_portfolio(n_sims, n_lobs=n_lobs, n_layers=n_layers)
    else:
        workload = build_workload(case, n_sims)
    synchronise()
    rss_after_setup = _peak_rss_bytes()
    gpu_after_setup = _pool_reserved_bytes()

    result = workload()
    synchronise()
    rss_peak = _peak_rss_bytes()
    gpu_after_workload = _pool_reserved_bytes()
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # Retain the result until all high-water readings have been taken.
    del result

    mb = 1024 * 1024
    return {
        "name": case,
        "n_sims": n_sims,
        "n_lobs": n_lobs if case == "portfolio" else None,
        "n_layers": n_layers if case == "portfolio" else None,
        "status": "ok",
        "rss_baseline_mb": round(rss_baseline / mb, 3),
        "rss_after_setup_mb": round(rss_after_setup / mb, 3),
        "rss_peak_mb": round(rss_peak / mb, 3),
        "rss_increment_mb": round(max(0, rss_peak - rss_baseline) / mb, 3),
        "python_tracemalloc_peak_mb": round(python_peak / mb, 3),
        "gpu_pool_baseline_mb": round(gpu_baseline / mb, 3) if gpu_baseline is not None else None,
        "gpu_pool_after_setup_mb": round(gpu_after_setup / mb, 3) if gpu_after_setup is not None else None,
        "gpu_pool_peak_mb": round(gpu_after_workload / mb, 3) if gpu_after_workload is not None else None,
    }


def run_isolated(case: str, n_sims: int, n_lobs: int, n_layers: int) -> dict[str, Any]:
    """Run one sample in its own interpreter to isolate process peak memory."""
    command = [
        sys.executable, "-m", "benchmarks.profile_memory",
        "--worker-case", case, "--worker-size", str(n_sims),
        "--portfolio-lobs", str(n_lobs), "--portfolio-layers", str(n_layers),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("PAL_MEMORY_RESULT:"):
            return json.loads(line.removeprefix("PAL_MEMORY_RESULT:"))
    return {
        "name": case,
        "n_sims": n_sims,
        "status": "error",
        "error": f"Worker exit={completed.returncode}: {completed.stderr[-1200:]}",
    }


def markdown_report(data: dict[str, Any]) -> str:
    """Write a concise report for GitHub Actions and local reviews."""
    env = data["environment"]
    lines = [
        "## PAL peak-memory profile",
        "",
        f"Backend: {env['backend']}; CPU: {env['cpu']}; GPU: {env['gpu'] or 'n/a'}",
        f"Python: {env['python']}; commit: `{env['commit']}`",
        "",
        "| Workload | Simulations | Peak process RSS (MiB) | RSS above import baseline (MiB) | GPU pool reserved (MiB) | Result |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in data["results"]:
        if row["status"] != "ok":
            lines.append(f"| {row['name']} | {row['n_sims']:,} | - | - | - | Error |")
            continue
        pool = row["gpu_pool_peak_mb"]
        lines.append(
            f"| {row['name']} | {row['n_sims']:,} | {row['rss_peak_mb']:.2f} | "
            f"{row['rss_increment_mb']:.2f} | {pool if pool is not None else '-'} | OK |"
        )
    lines.extend([
        "",
        "Each case runs in a fresh process. RSS is the process high-water mark, including imports, setup and",
        "native allocations; the increment is relative to the after-import baseline, not an exact temporary",
        "allocation delta. CuPy pool reservation is the pool's retained device allocation and can differ",
        "from active VRAM usage; non-pool allocations are not included. Python tracemalloc peaks only cover",
        "tracked Python allocations, not raw NumPy/CuPy buffers. Do not compare memory across different",
        "hardware, library versions, or allocator configurations without a controlled baseline.",
    ])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run isolated memory profiles or a single worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="aggregate,occurrence,compound_generation,xol_layer,end_to_end")
    parser.add_argument("--sizes", default="10000,100000")
    parser.add_argument("--portfolio-lobs", type=int, default=10)
    parser.add_argument("--portfolio-layers", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("pal-memory.json"))
    parser.add_argument("--summary", type=Path, default=Path("pal-memory.md"))
    parser.add_argument("--worker-case", help=argparse.SUPPRESS)
    parser.add_argument("--worker-size", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    choices = (*DEFAULT_CASES, "portfolio")
    if args.portfolio_lobs < 1 or args.portfolio_layers < 1:
        parser.error("Portfolio LoBs and layers must be positive")
    if args.worker_case is not None:
        if args.worker_case not in choices or args.worker_size is None or args.worker_size < 1:
            parser.error("Invalid worker case or simulation count")
        record = worker(args.worker_case, args.worker_size, args.portfolio_lobs, args.portfolio_layers)
        print("PAL_MEMORY_RESULT:" + json.dumps(record))
        return 0

    cases = [case.strip() for case in args.cases.split(",")]
    sizes = [int(size) for size in args.sizes.split(",")]
    if any(case not in choices for case in cases) or any(size < 1 for size in sizes):
        parser.error("Invalid case or simulation size")
    results = []
    for size in sizes:
        for case in cases:
            result = run_isolated(case, size, args.portfolio_lobs, args.portfolio_layers)
            results.append(result)
            print(f"{case:22} {size:>10,} : {result['status']}", flush=True)
    data = {"schema_version": 1, "environment": environment(), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    args.summary.write_text(markdown_report(data), encoding="utf-8")
    return 1 if any(row["status"] != "ok" for row in results) else 0


if __name__ == "__main__":
    sys.exit(main())

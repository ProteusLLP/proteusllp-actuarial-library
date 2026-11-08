"""Reproducible, non-gating PAL performance measurements.

Run with `python -m benchmarks.benchmark_core --help`. Time includes GPU
synchronisation but excludes workload setup, warmup and kernel compilation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy

from pal import StochasticScalar, set_random_seed
from pal._maths import xp
from pal.contracts import XoL
from pal.copulas import GaussianCopula
from pal.distributions import LogNormal, Poisson
from pal.frequency_severity import FreqSevSims, FrequencySeverityModel, _get_sims_of_events

DEFAULT_CASES = (
    "event_indexing",
    "stochastic_arithmetic",
    "aggregate",
    "occurrence",
    "percentile_99_5",
    "compound_generation",
    "xol_layer",
    "gaussian_copula",
    "end_to_end",
)


def build_workload(name: str, n_sims: int) -> Callable[[], Any]:
    """Prepare input arrays outside the timed section."""
    if name not in DEFAULT_CASES:
        raise ValueError(f"Unknown benchmark: {name}")
    rng = np.random.default_rng(20261108 + n_sims)
    counts_cpu = rng.poisson(2, size=n_sims).astype(np.int64)
    counts = xp.asarray(counts_cpu)
    if name == "event_indexing":
        return lambda: _get_sims_of_events(counts)

    if name in ("stochastic_arithmetic", "percentile_99_5"):
        scalar = StochasticScalar(xp.asarray(rng.lognormal(9, 0.9, size=n_sims)))
        if name == "percentile_99_5":
            return lambda: scalar.percentile(99.5)
        return lambda: np.exp(scalar * 0.0001) + np.square(scalar / 10000.0)

    model = FrequencySeverityModel(Poisson(mean=2), LogNormal(mu=9, sigma=0.9))
    if name == "compound_generation":
        return lambda: model.generate(n_sims=n_sims)
    if name == "gaussian_copula":
        copula = GaussianCopula(np.array([[1.0, 0.45], [0.45, 1.0]]))
        return lambda: copula.generate(n_sims=n_sims)

    sim_index = _get_sims_of_events(counts)
    event_values = xp.asarray(rng.lognormal(9, 0.9, size=int(counts_cpu.sum())))
    losses = FreqSevSims(sim_index, event_values, n_sims)
    if name == "aggregate":
        return losses.aggregate
    if name == "occurrence":
        return losses.occurrence

    layer = XoL(name="Benchmark layer", limit=25000, excess=10000, premium=1000)
    if name == "xol_layer":
        return lambda: layer.apply(losses)

    def full_model() -> Any:
        sample = model.generate(n_sims=n_sims)
        recoveries = layer.apply(sample).recoveries
        return recoveries.aggregate().percentile(99.5)

    return full_model


def synchronise() -> None:
    """Measure completed GPU work rather than asynchronous kernel dispatch."""
    if xp is not np:
        xp.cuda.Device().synchronize()


def measure(workload: Callable[[], Any], warmup: int, repeat: int) -> dict[str, float]:
    """Return robust descriptive timing statistics in milliseconds."""
    for _ in range(warmup):
        workload()
        synchronise()
    samples = []
    for _ in range(repeat):
        synchronise()
        start = time.perf_counter()
        workload()
        synchronise()
        samples.append((time.perf_counter() - start) * 1000.0)
    median = statistics.median(samples)
    return {
        "median_ms": median,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "mad_ms": statistics.median(abs(x - median) for x in samples),
    }


def environment() -> dict[str, Any]:
    """Record enough hardware and dependency context to compare like with like."""
    hardware = platform.processor() or platform.machine()
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    hardware = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    gpu = None
    cupy_version = None
    cuda_runtime_version = None
    cuda_driver_version = None
    if xp is not np:
        props = xp.cuda.runtime.getDeviceProperties(0)
        gpu = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        cupy_version = xp.__version__
        cuda_runtime_version = int(xp.cuda.runtime.runtimeGetVersion())
        cuda_driver_version = int(xp.cuda.runtime.driverGetVersion())
    return {
        "backend": "cpu" if xp is np else "gpu",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cupy": cupy_version,
        "cuda_runtime_version": cuda_runtime_version,
        "cuda_driver_version": cuda_driver_version,
        "cpu": hardware,
        "cpu_count": os.cpu_count(),
        "gpu": gpu,
        "platform": platform.platform(),
        "commit": os.getenv("GITHUB_SHA", "unknown"),
        "runner": os.getenv("RUNNER_NAME", "local"),
    }


def run_cases(names: list[str], sizes: list[int], warmup: int, repeat: int) -> dict[str, Any]:
    """Run each case at each size, continuing after individual failures."""
    if warmup < 0 or repeat < 1 or any(n < 1 for n in sizes):
        raise ValueError("Sizes/repeat must be positive and warmup non-negative")
    results = []
    for n_sims in sizes:
        for name in names:
            set_random_seed(20261108)
            record: dict[str, Any] = {"name": name, "n_sims": n_sims}
            try:
                workload = build_workload(name, n_sims)
                record.update(measure(workload, warmup=warmup, repeat=repeat))
                record["status"] = "ok"
                print(f"{name:23} n={n_sims:>9,} {record['median_ms']:>12.3f} ms")
            except Exception as exc:
                record["status"] = "error"
                record["error"] = f"{type(exc).__name__}: {exc}"
                print(f"{name:23} n={n_sims:>9,} ERROR: {exc}", file=sys.stderr)
            results.append(record)
    return {"schema_version": 1, "environment": environment(), "warmup": warmup, "repeat": repeat, "results": results}


def comparison(current: dict[str, Any], baseline: dict[str, Any]) -> dict[tuple[str, int], float]:
    """Compare only matching hardware/backend stacks; do not assert speed budgets."""
    a, b = current["environment"], baseline["environment"]
    keys = (
        "backend", "python", "numpy", "scipy", "cupy", "cuda_runtime_version", "cuda_driver_version",
        "cpu", "cpu_count", "platform", "gpu"
    )
    if any(a.get(key) != b.get(key) for key in keys):
        raise ValueError("Baseline environment differs; comparisons require the same hardware and software stack")
    prior = {(row["name"], row["n_sims"]): row["median_ms"] for row in baseline["results"] if row["status"] == "ok"}
    return {
        (row["name"], row["n_sims"]): 100 * (row["median_ms"] / prior[(row["name"], row["n_sims"])] - 1)
        for row in current["results"]
        if row["status"] == "ok" and (row["name"], row["n_sims"]) in prior
    }


def markdown_report(data: dict[str, Any], deltas: dict[tuple[str, int], float]) -> str:
    """Build an Actions-friendly report with failures and comparison deltas."""
    env = data["environment"]
    output = [
        "## PAL performance benchmarks",
        "",
        f"Backend: {env['backend']}; CPU: {env['cpu']}; GPU: {env['gpu'] or 'n/a'}",
        f"Python {env['python']}; NumPy {env['numpy']}; SciPy {env['scipy']}; CuPy {env['cupy'] or 'n/a'}",
        f"CUDA runtime: {env['cuda_runtime_version']}; driver: {env['cuda_driver_version']}",
        f"Commit: `{env['commit']}`; warmed runs: {data['repeat']}",
        "",
        "| Workload | Simulations | Median (ms) | MAD (ms) | Change vs baseline | Result |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in data["results"]:
        key = (row["name"], row["n_sims"])
        if row["status"] != "ok":
            output.append(f"| {row['name']} | {row['n_sims']:,} | - | - | - | Error: {row['error']} |")
            continue
        delta = f"{deltas[key]:+.1f}%" if key in deltas else "-"
        output.append(
            f"| {row['name']} | {row['n_sims']:,} | {row['median_ms']:.3f} | "
            f"{row['mad_ms']:.3f} | {delta} | OK |"
        )
    output.extend([
        "",
        "Descriptive measurements only. Runner contention and hardware changes can affect results.",
        "GPU timings include synchronisation but not JIT compilation; setup is excluded.",
        "A positive change against a compatible baseline indicates slower execution.",
    ])
    return "\n".join(output) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run benchmarks and write stable JSON and Markdown artefacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="10000,100000", help="Comma-separated simulation counts")
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES), help="Comma-separated workload names")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("pal-benchmark.json"))
    parser.add_argument("--summary", type=Path, default=Path("pal-benchmark.md"))
    parser.add_argument("--baseline", type=Path, help="Previous result JSON, from identical hardware/software")
    args = parser.parse_args(argv)
    sizes = [int(v) for v in args.sizes.split(",")]
    names = [v.strip() for v in args.cases.split(",")]
    if any(name not in DEFAULT_CASES for name in names):
        parser.error(f"Unknown workload. Choices: {', '.join(DEFAULT_CASES)}")
    data = run_cases(names, sizes, args.warmup, args.repeat)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    deltas = comparison(data, json.loads(args.baseline.read_text(encoding="utf-8"))) if args.baseline else {}
    args.summary.write_text(markdown_report(data, deltas), encoding="utf-8")
    print(f"Results: {args.output}; report: {args.summary}")
    return 1 if any(row["status"] != "ok" for row in data["results"]) else 0


if __name__ == "__main__":
    sys.exit(main())

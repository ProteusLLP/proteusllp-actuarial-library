"""Benchmark full PAL frequency-severity simulation under MC and RQMC.

Measure generation separately from aggregation on CPU and GPU, with low- and
high-frequency models and both fixed and scenario-dependent severity scales.
No fixed latency thresholds: compare the published measurements over time.

Run:
    python -m benchmarks.benchmark_frequency_severity --sizes 10000,100000
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from pal import StochasticScalar
from pal._maths import asnumpy, xp
from pal.config import SamplingMethod, config, set_random_seed, set_sampling_method
from pal.distributions import Gamma, Poisson
from pal.frequency_severity import FrequencySeverityModel

CASES = ("low_frequency", "high_frequency", "stochastic_scale")


def _sync() -> None:
    """Synchronize asynchronous GPU operations for honest elapsed timings."""
    if xp is not np:
        xp.cuda.Device().synchronize()


def _model(case: str, n_sims: int) -> tuple[FrequencySeverityModel, float]:
    """Build a compound Poisson-Gamma model and its theoretical mean loss."""
    if case == "low_frequency":
        return FrequencySeverityModel(Poisson(1.5), Gamma(alpha=2, theta=40)), 120.0
    if case == "high_frequency":
        return FrequencySeverityModel(Poisson(10), Gamma(alpha=2, theta=40)), 800.0
    if case == "stochastic_scale":
        # One severity scale per simulation, shared across that simulation's claims.
        scale = StochasticScalar(xp.linspace(30, 50, n_sims))
        return FrequencySeverityModel(Poisson(1.5), Gamma(alpha=2, theta=scale)), 120.0
    raise ValueError(f"Unknown frequency-severity benchmark: {case}")


def run_benchmarks(
    sizes: list[int],
    replicates: int = 3,
    cases: tuple[str, ...] = CASES,
) -> dict[str, Any]:
    """Measure compound generation and aggregation, including device work."""
    if not sizes or any(n < 1 for n in sizes):
        raise ValueError("Supply positive simulation sizes")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not cases or any(case not in CASES for case in cases):
        raise ValueError("Supply at least one known benchmark case")

    previous = (config.seed, config.rng, config.sampling_method, config.qmc_dimensions)
    results: list[dict[str, Any]] = []
    try:
        for n_sims in sizes:
            for case in cases:
                for method in SamplingMethod:
                    generation_times: list[float] = []
                    aggregation_times: list[float] = []
                    total_claims: list[int] = []
                    aggregate_means: list[float] = []
                    for repeat in range(replicates):
                        set_sampling_method(method, dimensions=32)
                        set_random_seed(2071 + repeat * 100003)
                        model, target = _model(case, n_sims)
                        _sync()
                        t0 = time.perf_counter()
                        simulations = model.generate(n_sims)
                        _sync()
                        t1 = time.perf_counter()
                        totals = simulations.aggregate()
                        _sync()
                        t2 = time.perf_counter()
                        event_count = len(simulations.values)
                        assert event_count == len(simulations.sim_index)
                        observed_mean = float(np.mean(asnumpy(totals.values)))
                        assert np.isfinite(observed_mean)
                        assert event_count >= 0
                        generation_times.append((t1 - t0) * 1000)
                        aggregation_times.append((t2 - t1) * 1000)
                        total_claims.append(event_count)
                        aggregate_means.append(observed_mean)

                    row = {
                        "case": case,
                        "method": method.name,
                        "n_sims": n_sims,
                        "replicates": replicates,
                        "mean_claims": statistics.mean(total_claims),
                        "expected_aggregate_mean": target,
                        "aggregate_mean": statistics.mean(aggregate_means),
                        "median_generate_ms": statistics.median(generation_times),
                        "median_aggregate_ms": statistics.median(aggregation_times),
                        "median_total_ms": statistics.median(
                            [g + a for g, a in zip(generation_times, aggregation_times)]
                        ),
                        "claims_per_second": 1000 * statistics.mean(total_claims) / statistics.median(
                            generation_times
                        ),
                    }
                    results.append(row)
                    print(
                        f"{case:18s} {method.name:4s} n={n_sims:7,d} "
                        f"claims={row['mean_claims']:,.0f} generate={row['median_generate_ms']:.2f} ms "
                        f"aggregate={row['median_aggregate_ms']:.2f} ms"
                    )
    finally:
        config.seed, config.rng, config.sampling_method, config.qmc_dimensions = previous

    return {"backend": "gpu" if xp is not np else "cpu", "results": results}


def markdown_summary(report: dict[str, Any]) -> str:
    """Format timings for GitHub Actions benchmark summaries."""
    rows = [
        f"### Frequency-severity throughput ({report['backend']})",
        "",
        "| Case | Method | Simulations | Claims | Generate ms | Aggregate ms | Claims/s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["results"]:
        rows.append(
            f"| {item['case']} | {item['method']} | {item['n_sims']:,} | "
            f"{item['mean_claims']:,.0f} | {item['median_generate_ms']:.2f} | "
            f"{item['median_aggregate_ms']:.2f} | {item['claims_per_second']:,.0f} |"
        )
    return "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write machine-readable and human-readable benchmark results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="10000,100000")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--output", type=Path, default=Path("pal-freqsev-performance.json"))
    parser.add_argument("--summary", type=Path, default=Path("pal-freqsev-performance.md"))
    args = parser.parse_args(argv)
    report = run_benchmarks(
        [int(item) for item in args.sizes.split(",")],
        replicates=args.replicates,
        cases=tuple(item.strip() for item in args.cases.split(",")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.summary.write_text(markdown_summary(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

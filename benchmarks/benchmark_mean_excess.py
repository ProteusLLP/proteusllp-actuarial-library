"""Benchmark public PAL mean-excess calculations (CPU deterministic diagnostics).

Run python -m benchmarks.benchmark_mean_excess from an editable installation.
Timings are descriptive and include all public API overhead.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Callable

import numpy as np
import scipy

from pal.distributions import (
    GEV,
    MBBEFD,
    Beta,
    Burr,
    Gamma,
    GeneralizedBetaPrime,
    GeneralizedInverseGaussian,
    InverseGaussian,
    LogNormal,
    NonCentralChiSquared,
    Poisson,
    StudentsT,
    Weibull,
)
from pal.fitting import empirical_mean_excess


def measure(fn: Callable[[], object], repeat: int) -> float:
    """Return the median of warmed wall-clock runs in milliseconds."""
    fn()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def run(sizes: list[int], repeat: int, scalar_limit: int) -> dict:
    """Benchmark ordinary and difficult parameterisations using public APIs."""
    cases = [
        ("gamma", Gamma(2, 1), 0.1, 10),
        ("gamma_tail", Gamma(1000, 1), 2100, 1e7),
        ("lognormal", LogNormal(0, 1), 0.1, 100),
        ("lognormal_narrow", LogNormal(0, 1e-6), 1, 1.0001),
        ("weibull", Weibull(2, 1), 0.1, 100),
        ("burr", Burr(power=2, shape=3, scale=1, loc=0), 0.1, 1000),
        ("gb2", GeneralizedBetaPrime(a=2, b=1, p=2, q=3), 0.1, 10),
        ("gb2_stressed", GeneralizedBetaPrime(a=0.1, b=1, p=10, q=1000), 2, 1e4),
        ("inverse_gaussian", InverseGaussian(1, 4), 0.1, 100),
        ("inverse_gaussian_tail", InverseGaussian(1, 1e4), 1.01, 1e6),
        ("gev", GEV(shape=0.1, scale=1), 0.1, 1e4),
        ("gig", GeneralizedInverseGaussian(p=0.5, chi=4, psi=1), 0.5, 100),
        ("beta_tail", Beta(alpha=2, beta=1000), 0.01, 0.999),
        ("student_t", StudentsT(nu=5), 0.1, 1e4),
        ("ncx2", NonCentralChiSquared(df=4, nonc=3), 0.1, 100),
        ("poisson", Poisson(mean=5), 0.1, 20),
        ("mbbefd", MBBEFD(g=10, b=0.5), 0.01, 0.999),
    ]
    rows = []
    for size in sizes:
        for name, model, lo, hi in cases:
            ds = np.geomspace(lo, hi, size)
            batched = lambda: model.mean_excess(ds)
            # Do not time the scalar loop at large sizes; it is deliberately expensive.
            batch_values = np.asarray(batched())
            if not np.all(np.isfinite(batch_values) & (batch_values > 0)):
                raise ArithmeticError(f"{name} produced unresolved or nonpositive mean excess")
            row = {"case": name, "thresholds": size, "batch_ms": measure(batched, repeat)}
            if size <= scalar_limit:
                def scalar():
                    return np.asarray([model.mean_excess(float(d)) for d in ds])

                np.testing.assert_allclose(batch_values, scalar(), rtol=1e-7, atol=0)
                row["scalar_ms"] = measure(scalar, repeat)
                row["scalar_over_batch"] = row["scalar_ms"] / row["batch_ms"]
            rows.append(row)

    observations = np.random.default_rng(42).lognormal(0, 1, 100_000)
    for size in sizes:
        ds = np.geomspace(0.1, 10, size)
        batched = lambda: empirical_mean_excess(observations, ds)
        baseline = lambda: np.array([np.mean(observations[observations > d] - d) for d in ds])
        row = {"case": "empirical_100k", "thresholds": size, "batch_ms": measure(batched, repeat)}
        if size <= scalar_limit:
            np.testing.assert_allclose(batched(), baseline(), rtol=1e-12, atol=0)
            row["scalar_ms"] = measure(baseline, repeat)
            row["scalar_over_batch"] = row["scalar_ms"] / row["batch_ms"]
        rows.append(row)
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "repeat": repeat,
        "results": rows,
    }


def main() -> None:
    """Run the benchmark and optionally save the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="1000,10000")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--scalar-limit", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run([int(size) for size in args.sizes.split(",")], args.repeat, args.scalar_limit)
    report = json.dumps(result, indent=2)
    print(report)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Benchmark analytical and adaptive limited expected value paths."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pal import distributions as d


def _median_seconds(fn, repeat: int) -> float:
    """Return median warmed execution time."""
    fn()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def benchmark(sizes: list[int], repeat: int) -> list[dict]:
    """Time public LEV methods and verify sampled scalar agreement."""
    cases = [
        ("gamma", d.Gamma(alpha=2.5, theta=3)),
        ("lognormal", d.LogNormal(mu=2, sigma=0.8)),
        ("lognormal-wide", d.LogNormal(mu=0, sigma=100)),
        ("burr", d.Burr(power=2, shape=3, scale=2, loc=0)),
        ("inverse-burr", d.InverseBurr(power=2, shape=3, scale=2, loc=0)),
        ("inverse-burr-stressed", d.InverseBurr(power=2, shape=1e30, scale=1e-15, loc=0)),
        ("gb2", d.GeneralizedBetaPrime(a=1.5, b=2, p=3, q=2)),
        ("gb2-infinite", d.GeneralizedBetaPrime(a=0.7, b=2, p=3, q=0.8)),
        ("inverse-gamma-infinite", d.InverseGamma(alpha=0.7, theta=2)),
        ("inverse-weibull-infinite", d.InverseWeibull(shape=0.7, scale=2)),
        ("inverse-gaussian", d.InverseGaussian(mu=3, lambda_=4)),
        ("gig", d.GeneralizedInverseGaussian(p=0.5, chi=4, psi=1)),
        ("gev", d.GEV(shape=0.2, loc=0, scale=2)),
        ("gev-near-one", d.GEV(shape=1 - 1e-10, loc=0, scale=2)),
        ("gev-infinite", d.GEV(shape=1.2, loc=0, scale=2)),
        ("loggamma", d.LogGamma(alpha=2, beta=1.5)),
        ("beta", d.Beta(alpha=2, beta=3)),
        ("poisson", d.Poisson(mean=4)),
        ("mbbefd", d.MBBEFD(g=3, b=0.2)),
    ]
    rows = []
    for label, model in cases:
        for n in sizes:
            if label == "gig" and n > 10000:
                continue
            limits = np.geomspace(0.1, 1e8, n)
            elapsed = _median_seconds(lambda: model.lev(limits), repeat)  # noqa: B023
            actual = np.asarray(model.lev(limits))
            indices = np.unique(np.linspace(0, n - 1, min(n, 32), dtype=int))
            scalar = np.array([model.lev(float(limits[i])) for i in indices])
            np.testing.assert_allclose(actual[indices], scalar, rtol=1e-8, atol=1e-10)
            row = {"case": label, "limits": n, "batch_ms": 1000 * elapsed}
            if n <= 1000 and label != "gig":
                scalar_time = _median_seconds(lambda: [model.lev(float(x)) for x in limits], 1)  # noqa: B023
                row["scalar_ms"] = 1000 * scalar_time
                row["speedup"] = scalar_time / elapsed
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="1000,10000,100000")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark([int(n) for n in args.sizes.split(",")], args.repeat)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

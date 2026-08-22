"""Benchmark PAL's GPU beta functions against CuPy's implementations."""

from __future__ import annotations

import argparse
import statistics
import time
import typing as t

import cupy as cp
import cupyx.scipy.special as cupy_special

from pal._gpu_beta import betainc, betaincinv


def _time_gpu(function: t.Callable[..., t.Any], *args: t.Any, repeats: int) -> float:
    for _ in range(3):
        function(*args)
    cp.cuda.Device().synchronize()

    timings = []
    for _ in range(repeats):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        function(*args)
        end.record()
        end.synchronize()
        timings.append(cp.cuda.get_elapsed_time(start, end))
    return statistics.median(timings)


def _report(name: str, pal_time: float, cupy_time: float) -> None:
    speedup = cupy_time / pal_time
    print(f"{name:12} PAL {pal_time:9.3f} ms  CuPy {cupy_time:9.3f} ms  speed-up {speedup:6.2f}x")


def main() -> None:
    """Run warmed, device-timed benchmarks over a large mixed-tail array."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    probabilities = cp.linspace(1e-10, 1 - 1e-10, args.size, dtype=cp.float64)
    alpha = 2.5
    beta = 7.0

    compilation_start = time.perf_counter()
    betainc(alpha, beta, probabilities)
    betaincinv(alpha, beta, probabilities)
    cp.cuda.Device().synchronize()
    print(f"PAL first-call compilation and execution: {time.perf_counter() - compilation_start:.3f} s")

    pal_cdf = _time_gpu(betainc, alpha, beta, probabilities, repeats=args.repeats)
    cupy_cdf = _time_gpu(cupy_special.betainc, alpha, beta, probabilities, repeats=args.repeats)
    _report("CDF", pal_cdf, cupy_cdf)

    pal_inverse = _time_gpu(betaincinv, alpha, beta, probabilities, repeats=args.repeats)
    cupy_inverse = _time_gpu(cupy_special.betaincinv, alpha, beta, probabilities, repeats=args.repeats)
    _report("inverse CDF", pal_inverse, cupy_inverse)


if __name__ == "__main__":
    main()

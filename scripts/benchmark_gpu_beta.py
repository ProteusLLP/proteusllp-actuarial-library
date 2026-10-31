"""Benchmark PAL's GPU beta functions against CuPy's implementations."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
import typing as t
from pathlib import Path

import cupy as cp
import cupyx.scipy.special as cupy_special

from pal._gpu_beta import betainc, betaincinv

SHAPES = (
    ("beta_2_5_7", 2.5, 7.0),
    ("beta_general", 2.5, 5.5),
    ("small_shapes", 0.3, 0.7),
    ("student_t_df5", 2.5, 0.5),
    ("student_t_df30", 15.0, 0.5),
    ("nearly_symmetric", 1000.0, 1031.0),
)


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


def _report(result: dict[str, t.Any]) -> None:
    print(
        f"{result['case']:18} {result['layout']:6} n={result['size']:8} {result['function']:7} "
        f"PAL {result['pal_ms']:9.3f} ms  CuPy {result['cupy_ms']:9.3f} ms  "
        f"speed-up {result['speedup']:6.2f}x",
        flush=True,
    )


def _save_report(report: dict[str, t.Any], output: Path | None) -> None:
    if output is not None:
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _append_summary(result: dict[str, t.Any]) -> None:
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as stream:
            stream.write(
                f"| {result['case']} | {result['layout']} | {result['size']} | {result['function']} "
                f"| {result['pal_ms']:.3f} | {result['cupy_ms']:.3f} | {result['speedup']:.2f}x |\n"
            )


def main() -> None:
    """Compare scalar and array shapes at several sizes using warmed CUDA events."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, help="benchmark one size instead of --sizes")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 100_000, 1_000_000])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, help="write timings and GPU stack metadata as JSON")
    args = parser.parse_args()
    sizes = [args.size] if args.size is not None else args.sizes
    if args.repeats < 1 or any(size < 2 for size in sizes):
        parser.error("repeats must be positive and sizes must be at least two")

    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    name = properties["name"]
    metadata = {
        "device": name.decode() if isinstance(name, bytes) else name,
        "compute_capability": f"{properties['major']}.{properties['minor']}",
        "cupy": cp.__version__,
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "python": platform.python_version(),
        "git_sha": os.environ.get("GITHUB_SHA"),
        "repeats": args.repeats,
        "sizes": sizes,
    }
    results: list[dict[str, t.Any]] = []
    report = {"metadata": metadata, "results": results}
    print(json.dumps(metadata), flush=True)
    _save_report(report, args.output)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as stream:
            stream.write(
                f"GPU beta benchmark: {metadata['device']}, CuPy {metadata['cupy']}, "
                f"CUDA runtime {metadata['cuda_runtime']}.\n\n"
                "| Case | Shapes | Size | Function | PAL ms | CuPy ms | Speed-up |\n"
                "| --- | --- | ---: | --- | ---: | ---: | ---: |\n"
            )

    probabilities = cp.linspace(1e-10, 1 - 1e-10, sizes[0], dtype=cp.float64)
    compilation_start = time.perf_counter()
    betainc(2.5, 7.0, probabilities)
    betaincinv(2.5, 7.0, probabilities)
    cp.cuda.Device().synchronize()
    metadata["pal_first_call_seconds"] = time.perf_counter() - compilation_start
    print(f"PAL first-call compilation and execution: {metadata['pal_first_call_seconds']:.3f} s", flush=True)

    for size in sizes:
        probabilities = cp.linspace(1e-10, 1 - 1e-10, size, dtype=cp.float64)
        for case, alpha, beta in SHAPES:
            mean = alpha / (alpha + beta)
            deviation = (alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))) ** 0.5
            # Include the expensive transition region for concentrated shapes.
            cdf_values = cp.linspace(max(1e-10, mean - 8 * deviation), min(1 - 1e-10, mean + 8 * deviation), size)
            for layout in ("scalar", "array"):
                device_alpha = alpha if layout == "scalar" else cp.full(size, alpha, dtype=cp.float64)
                device_beta = beta if layout == "scalar" else cp.full(size, beta, dtype=cp.float64)
                for function_name, pal_function, cupy_function in (
                    ("CDF", betainc, cupy_special.betainc),
                    ("inverse", betaincinv, cupy_special.betaincinv),
                ):
                    values = cdf_values if function_name == "CDF" else probabilities
                    pal_time = _time_gpu(pal_function, device_alpha, device_beta, values, repeats=args.repeats)
                    cupy_time = _time_gpu(cupy_function, device_alpha, device_beta, values, repeats=args.repeats)
                    result = {
                        "case": case,
                        "alpha": alpha,
                        "beta": beta,
                        "layout": layout,
                        "size": size,
                        "function": function_name,
                        "pal_ms": pal_time,
                        "cupy_ms": cupy_time,
                        "speedup": cupy_time / pal_time,
                    }
                    results.append(result)
                    _report(result)
                    _append_summary(result)
                    _save_report(report, args.output)


if __name__ == "__main__":
    main()

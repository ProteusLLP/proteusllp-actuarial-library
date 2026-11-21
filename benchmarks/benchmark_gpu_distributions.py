"""Benchmark native GPU probability functions against equivalent SciPy CPU paths.

Run with PAL_USE_GPU=1 on a CUDA device:
    python -m benchmarks.benchmark_gpu_distributions --sizes 1024,8192,65536

Includes synchronization for GPU timings, but excludes one-time environment
setup. GPU array inputs are preloaded; "CPU bridge" includes host/device copies.
Times include PAL GPU algorithm iterations and RNG operations.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy
from scipy import stats
from scipy.stats import qmc as scipy_qmc

from pal.qmc import RandomizedQMCGenerator

from pal._gpu_distributions import (
    generalized_inverse_gaussian_mc,
    noncentral_chisquare_cdf,
    noncentral_chisquare_ppf,
)
from pal._maths import xp


def _sync() -> None:
    xp.cuda.Device().synchronize()


def _time_call(
    call: Callable[[], Any], *, warmups: int, repeats: int
) -> tuple[float, float]:
    """Median and minimum synchronized wall-clock elapsed time in ms."""
    for _ in range(warmups):
        call()
        _sync()
    durations = []
    for _ in range(repeats):
        _sync()
        started = time.perf_counter()
        result = call()
        _sync()
        durations.append(1000.0 * (time.perf_counter() - started))
        del result
    return statistics.median(durations), min(durations)


def _benchmark_one(
    name: str,
    n_sims: int,
    cpu: Callable[[], Any],
    gpu: Callable[[], Any],
    bridge: Callable[[], Any],
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    cpu_median, cpu_min = _time_call(cpu, warmups=warmups, repeats=repeats)
    gpu_median, gpu_min = _time_call(gpu, warmups=warmups, repeats=repeats)
    bridge_median, bridge_min = _time_call(bridge, warmups=warmups, repeats=repeats)
    result = {
        "method": name,
        "n_sims": n_sims,
        "cpu_scipy_median_ms": round(cpu_median, 4),
        "gpu_native_median_ms": round(gpu_median, 4),
        "cpu_bridge_median_ms": round(bridge_median, 4),
        "cpu_scipy_min_ms": round(cpu_min, 4),
        "gpu_native_min_ms": round(gpu_min, 4),
        "cpu_bridge_min_ms": round(bridge_min, 4),
        "speedup_gpu_vs_cpu": round(cpu_median / gpu_median, 4),
        "speedup_gpu_vs_bridge": round(bridge_median / gpu_median, 4),
    }
    print(
        f"{name:22s} n={n_sims:7,d} CPU={cpu_median:10.3f} ms "
        f"GPU={gpu_median:10.3f} ms bridge={bridge_median:10.3f} ms "
        f"GPU/CPU speedup={cpu_median / gpu_median:8.3f}x",
        flush=True,
    )
    return result


def compare_methods(sizes: list[int], warmups: int, repeats: int) -> dict[str, Any]:
    """Measure probability, RQMC sampling and GIG variants at multiple sizes."""
    if xp is np:
        raise RuntimeError("This benchmark requires PAL_USE_GPU=1 and a CUDA device")
    if repeats < 2 or warmups < 1 or not sizes or any(n <= 0 for n in sizes):
        raise ValueError("Use >=2 repeats, >=1 warmup and positive sample sizes")

    properties = xp.cuda.runtime.getDeviceProperties(0)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    results = []
    for n in sizes:
        host_x = np.linspace(0.1, 25.0, n, dtype=np.float64)
        host_u = np.linspace(0.0005, 0.9995, n, dtype=np.float64)
        gpu_x, gpu_u = xp.asarray(host_x), xp.asarray(host_u)
        df, nonc = 5.0, 3.0
        results.append(
            _benchmark_one(
                "Noncentral Chi2 CDF",
                n,
                lambda: stats.ncx2.cdf(host_x, df, nonc),
                lambda: noncentral_chisquare_cdf(gpu_x, df, nonc),
                lambda: xp.asarray(stats.ncx2.cdf(xp.asnumpy(gpu_x), df, nonc)),
                warmups,
                repeats,
            )
        )
        results.append(
            _benchmark_one(
                "Noncentral Chi2 PPF",
                n,
                lambda: stats.ncx2.ppf(host_u, df, nonc),
                lambda: noncentral_chisquare_ppf(gpu_u, df, nonc),
                lambda: xp.asarray(stats.ncx2.ppf(xp.asnumpy(gpu_u), df, nonc)),
                warmups,
                repeats,
            )
        )

        # A two-coordinate Poisson/gamma construction replaces GPU RQMC
        # 56-fold mixture inversion. Compare against CPU one-coordinate PPF
        # as a throughput baseline, not as a pathwise equivalent QMC design.
        sobol_host = scipy_qmc.Sobol(d=1, scramble=True, seed=146).random(n)[:, 0]
        sobol_gpu = xp.asarray(sobol_host)
        results.append(
            _benchmark_one(
                "Noncentral Chi2 RQMC",
                n,
                lambda: stats.ncx2.ppf(sobol_host, df, nonc),
                lambda: RandomizedQMCGenerator(seed=146, dimensions=32).noncentral_chisquare(df, nonc, size=n),
                lambda: xp.asarray(stats.ncx2.ppf(xp.asnumpy(sobol_gpu), df, nonc)),
                warmups,
                repeats,
            )
        )
        for p, label in ((-0.5, "GIG IG(-1/2) MC"), (0.5, "GIG IG(+1/2) MC"), (2.0, "GIG generic MC")):
            chi, psi = 2.0, 3.0
            b = float(np.sqrt(chi * psi))
            scale = float(np.sqrt(chi / psi))
            cpu_rng = np.random.default_rng(13071984)
            bridge_rng = np.random.default_rng(13071984)
            gpu_rng = xp.random.RandomState(13071984)
            results.append(
                _benchmark_one(
                    label,
                    n,
                    lambda: stats.geninvgauss.rvs(p, b, scale=scale, size=n, random_state=cpu_rng),
                    lambda: generalized_inverse_gaussian_mc(gpu_rng, n, p, b, scale, 0.0),
                    lambda: xp.asarray(stats.geninvgauss.rvs(p, b, scale=scale, size=n, random_state=bridge_rng)),
                    warmups,
                    repeats,
                )
            )

    return {
        "schema_version": 1,
        "hardware": str(device_name),
        "cuda_runtime": xp.cuda.runtime.runtimeGetVersion(),
        "cupy_version": xp.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "warmups": warmups,
        "repeats": repeats,
        "notes": [
            "All timings are synchronized end-to-end wall-clock medians on one GPU-equipped runner.",
            "CPU baseline operates on preloaded host arrays. GPU baseline operates on preloaded device arrays.",
            "CPU bridge transfers input GPU arrays to CPU (CDF/PPF) and output to GPU; GIG bridge transfers output only.",
            "GPU cold-start initialization is excluded through warmup; methods can synchronise scalar loop controls.",
            "Reported speed-up is CPU SciPy median divided by GPU-native median; values below 1 mean GPU is slower.",
            "GIG Monte Carlo is a distributional rather than pathwise comparison.",
            "RQMC GPU uses two coordinates (Poisson mixture); CPU uses one coordinate (SciPy PPF).",
            "Results depend on parameter values, batch size, GPU model and library versions.",
        ],
        "results": results,
    }


def markdown_summary(report: dict[str, Any]) -> str:
    """Format benchmark results for the GitHub job summary and PR."""
    lines = [
        "## GPU-native probability method benchmarks",
        "",
        f"GPU: {report['hardware']}; CuPy {report['cupy_version']}; "
        f"SciPy {report['scipy_version']}. Warmups={report['warmups']}; "
        f"repeats={report['repeats']}.",
        "",
        "| Method | Paths | CPU SciPy ms | Native GPU ms | CPU bridge ms | GPU vs CPU |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        lines.append(
            f"| {row['method']} | {row['n_sims']:,} | "
            f"{row['cpu_scipy_median_ms']:.3f} | {row['gpu_native_median_ms']:.3f} | "
            f"{row['cpu_bridge_median_ms']:.3f} | {row['speedup_gpu_vs_cpu']:.3f}x |"
        )
    lines.extend(
        [
            "",
            "Input arrays are preloaded on their respective devices. "
            "GPU timings include synchronization and algorithm overhead.",
            "CPU bridge timings include host/device transfers. "
            "Speedups below 1 indicate the native GPU approach is slower.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run the benchmarks and write both machine-readable and Markdown reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="1024,8192,65536")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("pal-gpu-distributions.json"))
    parser.add_argument("--summary", type=Path, default=Path("pal-gpu-distributions.md"))
    args = parser.parse_args(argv)
    report = compare_methods([int(item) for item in args.sizes.split(",")], args.warmups, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.summary.write_text(markdown_summary(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

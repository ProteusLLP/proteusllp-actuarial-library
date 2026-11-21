"""Measure GPU sampling against CPU SciPy, including host-bridge costs.

With PAL_USE_GPU=1, run:
    python -m benchmarks.benchmark_gpu_distributions --sizes 1024,8192,65536,262144,1048576

All GPU paths are synchronized, and timing is after one warm-up. RQMC GPU
uses an exact two-coordinate Poisson/gamma construction; CPU SciPy uses a
one-coordinate inverse CDF. Compare distributions and throughput, not paths.
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

from pal._gpu_distributions import generalized_inverse_gaussian_mc
from pal._maths import xp
from pal.distributions import GeneralizedInverseGaussian
from pal.qmc import RandomizedQMCGenerator


def _sync() -> None:
    xp.cuda.Device().synchronize()


def _time_call(call: Callable[[], Any], warmups: int, repeats: int) -> tuple[float, float]:
    """Return synchronized wall-clock median and minimum milliseconds."""
    for _ in range(warmups):
        call()
        _sync()
    durations = []
    for _ in range(repeats):
        _sync()
        start = time.perf_counter()
        result = call()
        _sync()
        durations.append(1000 * (time.perf_counter() - start))
        del result
    return statistics.median(durations), min(durations)


def _compare(
    method: str,
    backend_mode: str,
    n_sims: int,
    cpu: Callable[[], Any],
    backend: Callable[[], Any],
    bridge: Callable[[], Any],
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    cpu_ms, cpu_min = _time_call(cpu, warmups, repeats)
    gpu_ms, gpu_min = _time_call(backend, warmups, repeats)
    bridge_ms, bridge_min = _time_call(bridge, warmups, repeats)
    row = {
        "method": method,
        "backend_mode": backend_mode,
        "n_sims": n_sims,
        "cpu_scipy_median_ms": round(cpu_ms, 4),
        "pal_backend_median_ms": round(gpu_ms, 4),
        "cpu_bridge_median_ms": round(bridge_ms, 4),
        "cpu_min_ms": round(cpu_min, 4),
        "pal_min_ms": round(gpu_min, 4),
        "bridge_min_ms": round(bridge_min, 4),
        "speedup": round(cpu_ms / gpu_ms, 4),
    }
    print(
        f"{method:24s} {n_sims:9,d} {backend_mode:13s} "
        f"CPU={cpu_ms:9.3f} ms PAL={gpu_ms:9.3f} ms "
        f"bridge={bridge_ms:9.3f} ms speedup={cpu_ms / gpu_ms:7.3f}x",
        flush=True,
    )
    return row


def benchmark(sizes: list[int], warmups: int, repeats: int) -> dict[str, Any]:
    """Benchmark production GPU acceleration and deliberate CPU fallbacks."""
    if xp is np:
        raise RuntimeError("Run with PAL_USE_GPU=1 and an available CUDA device")
    if warmups < 1 or repeats < 2 or not sizes or any(n <= 0 for n in sizes):
        raise ValueError("Need >=1 warmup, >=2 repeats and positive batch sizes")
    properties = xp.cuda.runtime.getDeviceProperties(0)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    rows: list[dict[str, Any]] = []
    for n in sizes:
        # CPU reference uses precomputed scrambled Sobol uniforms and SciPy
        # quantiles. The GPU method includes generation of two coordinates.
        host_u = scipy_qmc.Sobol(d=1, scramble=True, seed=146).random(n)[:, 0]
        gpu_u = xp.asarray(host_u)
        df, nonc = 5.0, 3.0
        rows.append(
            _compare(
                "Noncentral chi2 RQMC",
                "native GPU",
                n,
                lambda: stats.ncx2.ppf(host_u, df, nonc),
                lambda: RandomizedQMCGenerator(seed=146, dimensions=32).noncentral_chisquare(df, nonc, size=n),
                lambda: xp.asarray(stats.ncx2.ppf(xp.asnumpy(gpu_u), df, nonc)),
                warmups,
                repeats,
            )
        )
        for p, name in ((-0.5, "GIG MC p=-1/2"), (0.5, "GIG MC p=+1/2"), (2.0, "GIG MC general")):
            b = float(np.sqrt(6.0))
            scale = float(np.sqrt(2.0 / 3.0))
            cpu_rng = np.random.default_rng(146)
            bridge_rng = np.random.default_rng(146)
            gpu_rng = xp.random.RandomState(146)
            if abs(p) == 0.5:
                mode = "native GPU"
                def run_gpu(index: float = p, random: Any = gpu_rng) -> Any:
                    return generalized_inverse_gaussian_mc(random, n, index, b, scale, 0.0)
            else:
                mode = "CPU bridge"
                dist = GeneralizedInverseGaussian(p, 2.0, 3.0)
                def run_gpu(model: GeneralizedInverseGaussian = dist, random: Any = gpu_rng) -> Any:
                    return model.generate(n, rng=random).values
            rows.append(
                _compare(
                    name,
                    mode,
                    n,
                    lambda index=p: stats.geninvgauss.rvs(index, b, scale=scale, size=n, random_state=cpu_rng),
                    run_gpu,
                    lambda index=p: xp.asarray(
                        stats.geninvgauss.rvs(index, b, scale=scale, size=n, random_state=bridge_rng)
                    ),
                    warmups,
                    repeats,
                )
            )
    return {
        "schema_version": 2,
        "device": str(device_name),
        "cupy_version": xp.__version__,
        "scipy_version": scipy.__version__,
        "warmups": warmups,
        "repeats": repeats,
        "notes": [
            "Synchronized wall-clock medians, excluding first warm-up.",
            "All CPU inputs are host arrays; all GPU inputs and outputs are device arrays.",
            "RQMC GPU uses two scrambled dimensions while CPU uses one: not pathwise comparable.",
            "GIG general uses SciPy intentionally, with output transferred to CuPy.",
            "Speedup below one indicates the PAL backend is slower than CPU SciPy.",
        ],
        "results": rows,
    }


def markdown_summary(report: dict[str, Any]) -> str:
    """Return a GitHub-friendly summary of performance."""
    lines = [
        "## PAL CUDA sampling versus SciPy CPU",
        "",
        f"Device: {report['device']}; CuPy {report['cupy_version']}; SciPy {report['scipy_version']}.",
        "",
        "| Method | Paths | Backend | CPU SciPy ms | PAL backend ms | CPU bridge ms | Speedup |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        lines.append(
            f"| {row['method']} | {row['n_sims']:,} | {row['backend_mode']} | "
            f"{row['cpu_scipy_median_ms']:.3f} | {row['pal_backend_median_ms']:.3f} | "
            f"{row['cpu_bridge_median_ms']:.3f} | {row['speedup']:.3f}x |"
        )
    lines.append("")
    lines.append("Faster native sampling is retained; the slow general-index GIG uses CPU SciPy.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and write JSON and GitHub Markdown outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="1024,8192,65536,262144,1048576")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("pal-gpu-distributions.json"))
    parser.add_argument("--summary", type=Path, default=Path("pal-gpu-distributions.md"))
    args = parser.parse_args(argv)
    report = benchmark([int(item) for item in args.sizes.split(",")], args.warmups, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.summary.write_text(markdown_summary(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

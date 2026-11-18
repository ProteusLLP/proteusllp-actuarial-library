"""Compare Monte Carlo and independently scrambled Sobol RQMC in PAL.

Both elapsed time and accuracy are measured. Each replicate uses a separate
seed/scramble. Do not estimate RQMC standard errors from individual paths.
Use: python -m benchmarks.benchmark_sampling --sizes 1024,4096,16384
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from pal._maths import asnumpy, xp
from pal.config import SamplingMethod, config, set_random_seed, set_sampling_method
from pal.copulas import GaussianCopula
from pal.distributions import Gamma, Normal, Poisson
from pal.frequency_severity import FrequencySeverityModel

CASES = ("normal_mean", "poisson_mean", "gaussian_joint_tail", "compound_mean")
RHO = 0.65
TAIL_LEVEL = 0.9


def analytic_joint_tail(rho: float = RHO, probability: float = TAIL_LEVEL) -> float:
    """Return Gaussian copula joint exceedance probability by 1D quadrature."""
    z = norm.ppf(probability)
    denominator = np.sqrt(1 - rho * rho)
    integral, _ = quad(
        lambda x: norm.pdf(x) * norm.cdf((rho * x - z) / denominator),
        z,
        np.inf,
        epsabs=1e-12,
        epsrel=1e-12,
    )
    return float(integral)


def case_value(case: str, n_sims: int) -> float:
    """Generate one PAL estimate using the globally configured RNG."""
    if case == "normal_mean":
        return float(np.mean(asnumpy(Normal(mu=2, sigma=3).generate(n_sims).values)))
    if case == "poisson_mean":
        return float(np.mean(asnumpy(Poisson(mean=4).generate(n_sims).values)))
    if case == "gaussian_joint_tail":
        u, v = (asnumpy(x.values) for x in GaussianCopula(
            [[1.0, RHO], [RHO, 1.0]]
        ).generate(n_sims))
        return float(np.mean((u > TAIL_LEVEL) & (v > TAIL_LEVEL)))
    if case == "compound_mean":
        model = FrequencySeverityModel(Poisson(1.5), Gamma(alpha=2, theta=40))
        return float(np.mean(asnumpy(model.generate(n_sims).aggregate().values)))
    raise ValueError(f"Unknown benchmark case: {case}")


def targets() -> dict[str, float]:
    """Analytical targets provide an independent accuracy reference."""
    return {
        "normal_mean": 2.0,
        "poisson_mean": 4.0,
        "gaussian_joint_tail": analytic_joint_tail(),
        "compound_mean": 120.0,
    }


def _synchronise() -> None:
    if xp is not np:
        xp.cuda.Device().synchronize()


def run_comparison(sizes: list[int], replicates: int = 12, cases: tuple[str, ...] = CASES) -> dict[str, Any]:
    """Compare matched sample sizes, seeds and model operations.

    Time includes QMC block initialization, random-number generation and
    marginal transforms. Device synchronization is included in GPU timings.
    """
    if replicates < 2 or any(size <= 0 for size in sizes):
        raise ValueError("replicates must be >= 2 and sample sizes positive")
    reference = targets()
    results: list[dict[str, Any]] = []
    initial = (config.seed, config.rng, config.sampling_method, config.qmc_dimensions)
    try:
        for n in sizes:
            for case in cases:
                if case not in CASES:
                    raise ValueError(f"Unknown benchmark case: {case}")
                for method in SamplingMethod:
                    draws: list[float] = []
                    milliseconds: list[float] = []
                    for replicate in range(replicates):
                        seed = 20261117 + 65537 * replicate
                        _synchronise()
                        start = time.perf_counter()
                        set_sampling_method(method, dimensions=32)
                        set_random_seed(seed)
                        estimate = case_value(case, n)
                        _synchronise()
                        milliseconds.append(1000 * (time.perf_counter() - start))
                        draws.append(estimate)
                    errors = np.asarray(draws) - reference[case]
                    row = {
                        "case": case,
                        "sampling_method": method.name,
                        "n_sims": n,
                        "replicates": replicates,
                        "target": reference[case],
                        "average_estimate": float(np.mean(draws)),
                        "bias": float(np.mean(errors)),
                        "rmse": float(np.sqrt(np.mean(errors**2))),
                        "estimate_sd": float(np.std(draws, ddof=1)),
                        "median_elapsed_ms": statistics.median(milliseconds),
                        "min_elapsed_ms": min(milliseconds),
                    }
                    results.append(row)
                    print(
                        f"{case:20s} {method.name:4s} n={n:7,d} "
                        f"RMSE={row['rmse']:.6g} median={row['median_elapsed_ms']:.2f} ms"
                    )
    finally:
        config.seed, config.rng, config.sampling_method, config.qmc_dimensions = initial

    return {"schema_version": 1, "backend": "gpu" if xp is not np else "cpu", "results": results}


def markdown_summary(data: dict[str, Any]) -> str:
    """Build a compact report with both methods for each work item."""
    lines = [
        "## MC vs RQMC accuracy and speed",
        "",
        f"Backend: {data['backend']}",
        "",
        "| Case | Paths | Method | RMSE | SD across runs | Median ms |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in data["results"]:
        lines.append(
            f"| {row['case']} | {row['n_sims']:,} | {row['sampling_method']} | "
            f"{row['rmse']:.6g} | {row['estimate_sd']:.6g} | {row['median_elapsed_ms']:.2f} |"
        )
    lines.extend([
        "",
        "Independent runs use different seeds and independent RQMC scrambles. ",
        "RMSE is relative to an analytical reference, not a comparison between",
        "two noisy simulation estimates. Times are descriptive, not CI performance budgets.",
        "Smaller RMSE is better; smaller elapsed milliseconds is faster.",
    ])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run comparisons and emit machine-readable and Markdown reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="1024,4096,16384")
    parser.add_argument("--replicates", type=int, default=12)
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--output", type=Path, default=Path("pal-sampling.json"))
    parser.add_argument("--summary", type=Path, default=Path("pal-sampling.md"))
    args = parser.parse_args(argv)
    cases = tuple(s.strip() for s in args.cases.split(","))
    report = run_comparison([int(x) for x in args.sizes.split(",")], args.replicates, cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.summary.write_text(markdown_summary(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

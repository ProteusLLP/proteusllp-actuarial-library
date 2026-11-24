"""Compare public PAL fitting optimizers on identical deterministic samples."""

from __future__ import annotations

import argparse
import json
import time
import typing as t
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from pal import distributions as d


@dataclass(frozen=True)
class Case:
    name: str
    family: type[d.DistributionBase]
    observations: np.ndarray
    options: dict[str, t.Any]


def _sample(
    name: str,
    family: type[d.DistributionBase],
    reference: t.Any,
    n: int,
    *,
    threshold: float | None = None,
    censor_fraction: float = 0,
    weighted: bool = False,
    transform: t.Callable[[np.ndarray], np.ndarray] | None = None,
) -> Case:
    """Use quantile-spaced observations, conditional on the reporting threshold."""
    probabilities = (np.arange(n, dtype=float) + 0.5) / n
    if threshold is None:
        x = reference.ppf(probabilities)
    else:
        x = reference.isf(reference.sf(threshold) * (1 - probabilities))
    if transform is not None:
        x = transform(x)
    kwargs: dict[str, t.Any] = {}
    if threshold is not None:
        kwargs["threshold"] = threshold
    if censor_fraction:
        cap = float(np.quantile(x, 1 - censor_fraction))
        kwargs["censored"] = x >= cap
        x = np.minimum(x, cap)
    if weighted:
        kwargs["weights"] = 1 + np.arange(n) % 3
    return Case(name, family, np.asarray(x), kwargs)


def cases(n: int) -> list[Case]:
    """Separate well-identified fits from common-data and tail-ridge comparisons."""
    own = [
        (d.Gamma, stats.gamma(2.5, scale=3000)),
        (d.LogNormal, stats.lognorm(0.9, scale=10000)),
        (d.Weibull, stats.weibull_min(1.4, scale=10000)),
        (d.Burr, stats.burr12(1.5, 2, scale=10000)),
        (d.LogLogistic, stats.fisk(2.2, scale=10000)),
        (d.Paralogistic, stats.burr12(1.6, 1.6, scale=10000)),
        (d.InverseBurr, stats.burr(2.3, 1.7, scale=10000)),
        (d.InverseParalogistic, stats.burr(1.8, 1.8, scale=10000)),
        (d.InverseWeibull, stats.invweibull(2.2, scale=10000)),
        (d.InverseExponential, stats.invweibull(1, scale=10000)),
        (d.InverseGamma, stats.invgamma(3, scale=10000)),
        (d.InverseGaussian, stats.invgauss(0.4, scale=25000)),
        (d.GeneralizedBetaPrime, stats.betaprime(2.2, 3.1, scale=10000)),
        (d.GeneralizedInverseGaussian, stats.geninvgauss(0.7, 2.1, scale=10000)),
        (d.GEV, stats.genextreme(-0.15, loc=10000, scale=4000)),
    ]
    result = [_sample(f"{family.__name__}/own", family, rv, min(n, 1000)) for family, rv in own]
    result.extend(
        [
            _sample(
                "GPD/positive",
                d.GPD,
                stats.genpareto(0.3, loc=10000, scale=5000),
                n,
                threshold=10000,
                censor_fraction=0.15,
                weighted=True,
            ),
            _sample(
                "GPD/negative",
                d.GPD,
                stats.genpareto(-0.3, loc=10000, scale=5000),
                n,
                threshold=10000,
                censor_fraction=0.15,
            ),
            _sample("LogGamma/own", d.LogGamma, stats.gamma(2.5, scale=0.4), n, transform=np.exp),
            _sample("Beta/censored", d.Beta, stats.beta(2.5, 4), n, threshold=0.25, censor_fraction=0.15),
            _sample(
                "Normal/truncated", d.Normal, stats.norm(10000, 2500), n, threshold=9000, censor_fraction=0.15
            ),
            _sample("Logistic/own", d.Logistic, stats.logistic(10000, 2500), n),
            _sample("StudentsT/own", d.StudentsT, stats.t(4, loc=10000, scale=2500), n),
            _sample("NonCentralChiSquared/own", d.NonCentralChiSquared, stats.ncx2(4, 3), n),
        ]
    )
    # A common loss sample exposes differences between candidate likelihoods.
    # GIG survival is expensive: reduce the censored tail sample for that family.
    common = stats.lognorm(1.4, scale=10000)
    for family, _ in own:
        if family is not d.GEV:
            result.append(_sample(f"{family.__name__}/common", family, common, n))
            result.append(
                _sample(
                    f"{family.__name__}/tail",
                    family,
                    common,
                    min(n, 250) if family is d.GeneralizedInverseGaussian else min(n, 1000),
                    threshold=100000,
                    censor_fraction=0.1,
                    weighted=True,
                )
            )
    probabilities = (np.arange(n) + 0.5) / n
    mbbefd = d.MBBEFD(g=4, b=0.5)
    result.append(Case("MBBEFD/atom", d.MBBEFD, np.asarray(mbbefd.invcdf(probabilities)), {}))
    return result


def _run(case: Case, optimizer: str) -> dict[str, t.Any]:
    start = time.perf_counter()
    try:
        fit = t.cast(t.Any, case.family).fit(case.observations, optimizer=optimizer, **case.options)
    except (ArithmeticError, RuntimeError, ValueError) as exc:
        return {"seconds": time.perf_counter() - start, "error": str(exc)}
    return {
        "seconds": time.perf_counter() - start,
        "log_likelihood": float(fit.log_likelihood),
        "converged": bool(fit.converged),
        "at_boundary": bool(fit.at_boundary),
        "optimizer": fit.optimizer,
        "fallback": bool(fit.optimizer_fallback),
        "parameters": {key: float(value) for key, value in fit.parameters.items()},
    }


def benchmark(n: int, pattern: str) -> list[dict[str, t.Any]]:
    results = []
    for case in cases(n):
        if pattern and pattern.lower() not in case.name.lower():
            continue
        old = _run(case, "Nelder-Mead")
        new = _run(case, "L-BFGS-B")
        total_weight = float(np.sum(case.options.get("weights", np.ones(len(case.observations)))))
        row = {
            "case": case.name,
            "observations": len(case.observations),
            "weight": total_weight,
            "nelder_mead": old,
            "lbfgsb": new,
            "speedup": old["seconds"] / max(new["seconds"], 1e-12),
            "delta_log_likelihood": (
                new["log_likelihood"] - old["log_likelihood"]
                if "log_likelihood" in old and "log_likelihood" in new
                else None
            ),
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--case", default="", help="Case name substring")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.size < 20:
        parser.error("size must be at least 20")
    results = benchmark(args.size, args.case)
    if args.output is not None:
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

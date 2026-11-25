"""Reproduce the wide mean-excess accuracy sweep with 50-digit references.

Requires mpmath (only for this development diagnostic, not at PAL runtime).
Run python -m benchmarks.check_mean_excess_accuracy.
"""

from __future__ import annotations

import json
from itertools import product

import mpmath as mp
import numpy as np

from pal.distributions import GEV, Beta, Gamma, GeneralizedBetaPrime, InverseGaussian, LogNormal

mp.mp.dps = 50


def beta_reference(a: float, b: float, x: float) -> float:
    """Use the complementary incomplete-beta moment ratio."""
    aa, bb, xx = map(mp.mpf, (a, b, x))
    w = 1 - xx
    return float(w - mp.betainc(bb + 1, aa, 0, w) / mp.betainc(bb, aa, 0, w))


def gb2_reference(a: float, p: float, q: float, x: float) -> float:
    """Use unregularized incomplete-beta moments at unit scale."""
    aa, pp, qq, xx = map(mp.mpf, (a, p, q, x))
    z = 1 / (1 + xx**aa)
    return float(mp.betainc(qq - 1 / aa, pp + 1 / aa, 0, z) / mp.betainc(qq, pp, 0, z) - xx)


def inverse_gaussian_reference(lam: float, x: float) -> float:
    """Use arbitrary-precision normal tail moments with unit mean."""
    ll, xx = map(mp.mpf, (lam, x))
    a = mp.sqrt(ll / xx) * (xx - 1)
    b = mp.sqrt(ll / xx) * (xx + 1)
    s1 = mp.erfc(a / mp.sqrt(2)) / 2
    s2 = mp.exp(2 * ll) * mp.erfc(b / mp.sqrt(2)) / 2
    return float((s1 + s2) / (s1 - s2) - xx)


def gamma_reference(a: float, x: float) -> float:
    """Use the unregularized upper incomplete-gamma ratio."""
    aa, xx = map(mp.mpf, (a, x))
    return float(mp.gammainc(aa + 1, xx) / mp.gammainc(aa, xx) - xx)


def gev_reference(shape: float, x: float) -> float:
    """Evaluate a truncated gamma moment, or the Gumbel limit."""
    xi, xx = map(mp.mpf, (shape, x))
    if xi == 0:
        y = mp.exp(-xx)
        return 1.0 if y < mp.mpf("1e-40") else float((mp.euler + mp.log(y) + mp.e1(y)) / (-mp.expm1(-y)))
    base = 1 + xi * xx
    if base <= 0:
        return float((mp.gamma(1 - xi) - 1) / xi - xx)
    y = base ** (-1 / xi)
    if y < mp.mpf("1e-40"):
        return float(base / (1 - xi))
    return float((mp.gammainc(1 - xi, 0, y) / (-mp.expm1(-y)) - base) / xi)


def lognormal_reference(mu: float, sigma: float, x: float) -> float:
    """Use the arbitrary-precision conditional first moment."""
    mm, ss, xx = map(mp.mpf, (mu, sigma, x))
    z = (mp.log(xx) - mm) / ss
    return float(mp.exp(mm + ss**2 / 2) * mp.erfc((z - ss) / mp.sqrt(2)) / mp.erfc(z / mp.sqrt(2)) - xx)


def run() -> dict:
    """Compare batched public APIs against independent high-precision references."""
    results: dict[str, list[tuple[float, tuple]]] = {}

    def check(name, model, params, thresholds, reference):
        ds = np.asarray(thresholds, dtype=float)
        values = model.mean_excess(ds)
        for d, value in zip(ds, values, strict=True):
            expected = reference(*params, float(d))
            error = abs(float(value) / expected - 1) if np.isfinite(value) else np.inf
            results.setdefault(name, []).append((error, (*params, float(d))))

    for a, b in product([0.1, 1, 2, 10, 100, 1000], repeat=2):
        check("Beta", Beta(a, b), (a, b), [0.01, 0.5, 0.9, 0.999, 1 - 1e-10], beta_reference)
    for a, p in product([0.1, 1, 2, 10, 100], [0.1, 1, 10, 1000]):
        for q in [1.01 / a, 2 / a, max(100, 2 / a), 1000]:
            check("GB2", GeneralizedBetaPrime(a, 1, p, q), (a, p, q), [0.01, 0.5, 1, 2, 10, 1e4], gb2_reference)
    for lam in [1e-4, 0.01, 1, 100, 1e4, 1e6]:
        check("InverseGaussian", InverseGaussian(1, lam), (lam,), [0.01, 0.5, 1, 1.5, 2, 2.01, 10, 100, 1e5, 1e8],
              inverse_gaussian_reference)
    for a in [0.01, 0.1, 2, 10, 100, 1e3, 1e4, 1e6]:
        check("Gamma", Gamma(a, 1), (a,), [max(501, 2.01*a), max(1e3, 3*a), max(1e4, 10*a), max(1e8, 100*a)],
              gamma_reference)
    for xi in [-2, -0.5, -0.1, -0.001, -1e-5, 0, 1e-5, 0.001, 0.1, 0.5, 0.95, 0.999]:
        ds = [d for d in [-5, -1, 0, 1, 5, 10, 100, 1000] if xi >= 0 or d < -1/xi]
        check("GEV", GEV(xi, 1), (xi,), ds, gev_reference)
    for mu, sigma in product([-20, 0, 20], [1e-10, 1e-6, 0.01, 0.1, 1, 5]):
        ds = np.exp(mu + sigma * np.array([-10, -3, 0, 1, 5, 30, 100]))
        check("LogNormal", LogNormal(mu, sigma), (mu, sigma), ds, lognormal_reference)

    report = {}
    for name, rows in results.items():
        finite = [error for error, _ in rows if np.isfinite(error)]
        worst = max(rows, key=lambda row: row[0])
        report[name] = {
            "cases": len(rows),
            "nonfinite": len(rows) - len(finite),
            "max_relative_error": float(max(finite)) if finite else None,
            "p95_relative_error": float(np.percentile(finite, 95)) if finite else None,
            "worst_parameters_and_threshold": worst[1],
        }
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2))
    if any(row["nonfinite"] or row["max_relative_error"] > 1e-7 for row in report.values()):
        raise SystemExit("Mean excess accuracy sweep failed")

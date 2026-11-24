"""Limited expected values of numeric univariate distributions."""

from __future__ import annotations

import datetime as dt
import typing as t

import numpy as np
from scipy import integrate, special, stats

from . import distributions as dist
from ._gig import _gig_log_probabilities
from ._mean_excess import _log_survival, _positive_log_survival
from .fitting import _frozen_distribution, _other_families, _supported_families
from .stochastic_scalar import StochasticScalar

if t.TYPE_CHECKING:
    from .distributions import DistributionBase


def _pareto_lev(d: np.ndarray, shape: float, scale: float) -> np.ndarray:
    log_ratio = np.log(np.maximum(d, scale)) - np.log(scale)
    exponent = 1 - shape
    if exponent == 0:
        return np.minimum(d, scale) + scale * log_ratio
    return np.minimum(d, scale) + scale * np.expm1(exponent * log_ratio) / exponent


def _gpd_lev(d: np.ndarray, shape: float, scale: float) -> np.ndarray:
    x = np.maximum(d, 0)
    if shape < 0:
        x = np.minimum(x, -scale / shape)
    if shape == 0:
        return scale * (-np.expm1(-x / scale))
    log_base = np.log1p(shape * x / scale)
    if shape == 1:
        return scale * log_base
    return scale * (-np.expm1((1 - 1 / shape) * log_base)) / (1 - shape)


def _count_lev(family: type[DistributionBase], p: dict[str, float], d: np.ndarray) -> np.ndarray:
    k = np.floor(d)
    if family is dist.Poisson:
        mean = p["mean"]
        rv = stats.poisson(mean)
        moment = mean * rv.cdf(k - 1)
    elif family in (dist.Binomial, dist.Bernoulli):
        n = 1 if family is dist.Bernoulli else int(p["n"])
        prob = p["p"]
        rv = stats.binom(n, prob)
        moment = n * prob * stats.binom.cdf(k - 1, n - 1, prob) if n > 0 else np.zeros_like(d)
    elif family is dist.NegBinomial:
        n, prob = p["n"], p["p"]
        rv = stats.nbinom(n, prob)
        moment = n * (1 - prob) / prob * stats.nbinom.cdf(k - 1, n + 1, prob)
    else:
        good, bad, draws = int(p["ngood"]), int(p["nbad"]), int(p["n"])
        population = good + bad
        rv = stats.hypergeom(population, good, draws)
        moment = (
            draws * good / population * stats.hypergeom.cdf(k - 1, population - 1, good - 1, draws - 1)
            if good > 0 and draws > 0
            else np.zeros_like(d)
        )
    return np.where(d <= 0, d, moment + d * rv.sf(k))


def _mbbefd_lev(g: float, b: float, d: np.ndarray) -> np.ndarray:
    x = np.clip(d, 0, 1)
    if g == 1 or b == 0:
        return np.minimum(d, 1)
    if b == 1:
        value = np.log1p((g - 1) * x) / (g - 1)
    else:
        log_b = np.log(b)
        log_gb = np.log(g) + log_b
        coefficient = (g - 1 / b) / (1 - 1 / b) if b > 1 else np.expm1(log_gb) / np.expm1(log_b)
        q = coefficient * np.expm1(x * log_b)
        if coefficient == 0:
            value = np.expm1(x * log_b) / log_b
        else:
            value = np.log1p(q) / (coefficient * log_b)
    return np.where(d <= 0, d, value)


def _remaining_lev(family: type[DistributionBase], p: dict[str, float], d: np.ndarray) -> np.ndarray | None:
    if family in (dist.Poisson, dist.Binomial, dist.Bernoulli, dist.NegBinomial, dist.HyperGeometric):
        return _count_lev(family, p, d)
    if family is dist.Uniform:
        a, b = p["a"], p["b"]
        if a == b:
            return np.minimum(d, a)
        x = np.clip(d, a, b) - a
        return np.where(d <= a, d, a + x * (1 - 0.5 * x / (b - a)))
    if family is dist.Beta:
        a, b, scale, loc = p["alpha"], p["beta"], p["scale"], p["loc"]
        x = np.clip(d - loc, 0, scale)
        u = x / scale
        moment = scale * a / (a + b) * special.betainc(a + 1, b, u)
        return np.where(d <= loc, d, loc + moment + x * special.betaincc(a, b, u))
    if family is dist.Normal:
        mu, sigma = p["mu"], p["sigma"]
        z = (d - mu) / sigma
        return mu * special.ndtr(z) - sigma * np.exp(stats.norm.logpdf(z)) + d * special.ndtr(-z)
    if family is dist.Logistic:
        mu, sigma = p["mu"], p["sigma"]
        z = (d - mu) / sigma
        return np.where(z > 0, mu - sigma * np.logaddexp(0, -z), d - sigma * np.logaddexp(0, z))
    if family is dist.StudentsT:
        nu, mu, sigma = p["nu"], p["mu"], p["sigma"]
        if nu <= 1:
            return np.full_like(d, -np.inf)
        z = (d - mu) / sigma
        density_moment = np.exp(stats.t.logpdf(z, nu) + np.logaddexp(np.log(nu), 2 * np.log(np.abs(z))))
        return mu * stats.t.cdf(z, nu) - sigma * density_moment / (nu - 1) + d * stats.t.sf(z, nu)
    if family is dist.NonCentralChiSquared:
        df, nc = p["df"], p["nonc"]
        moment = df * stats.ncx2.cdf(d, df + 2, nc) + nc * stats.ncx2.cdf(d, df + 4, nc)
        return np.where(d <= 0, d, moment + d * stats.ncx2.sf(d, df, nc))
    if family is dist.MBBEFD:
        return _mbbefd_lev(p["g"], p["b"], d)
    return None


def _severity_lev(family: type[DistributionBase], p: dict[str, float], d: np.ndarray) -> np.ndarray:
    loc = p.get("loc", 0.0)
    x = np.maximum(d - loc, 0)
    if family is dist.Pareto:
        return _pareto_lev(d, p["shape"], p["scale"])
    if family is dist.GPD:
        value = _gpd_lev(x, p["shape"], p["scale"])
    elif family is dist.Exponential:
        value = -p["scale"] * np.expm1(-x / p["scale"])
    elif family is dist.Gamma:
        a, theta = p["alpha"], p["theta"]
        z = x / theta
        value = a * theta * special.gammainc(a + 1, z) + x * special.gammaincc(a, z)
    elif family is dist.LogNormal:
        mu, sigma = p["mu"], p["sigma"]
        z = (np.log(x) - mu) / sigma
        value = np.exp(mu + sigma**2 / 2 + special.log_ndtr(z - sigma)) + x * special.ndtr(-z)
    elif family is dist.Weibull:
        a, scale = 1 / p["shape"], p["scale"]
        z = np.power(x / scale, p["shape"])
        value = np.exp(np.log(scale) + special.gammaln(1 + a) + np.log(special.gammainc(a, z)))
    elif family is dist.GeneralizedInverseGaussian:
        b = np.sqrt(p["chi"]) * np.sqrt(p["psi"])
        scale = np.sqrt(p["chi"]) / np.sqrt(p["psi"])
        index = p["p"]
        _, logsf = _gig_log_probabilities(x / scale, index, b)
        logcdf, _ = _gig_log_probabilities(x / scale, index + 1, b)
        logmean = np.log(scale) + np.log(special.kve(index + 1, b)) - np.log(special.kve(index, b))
        value = np.exp(logmean + logcdf) + x * np.exp(logsf)
    elif family is dist.GeneralizedBetaPrime:
        a, b, alpha, beta = (p[name] for name in ("a", "b", "p", "q"))
        power = a * (np.log(x) - np.log(b))
        u = special.expit(power)
        w = special.expit(-power)
        if beta > 1 / a:
            logmean = np.log(b) + special.betaln(alpha + 1 / a, beta - 1 / a) - special.betaln(alpha, beta)
            value = np.exp(logmean + np.log(special.betainc(alpha + 1 / a, beta - 1 / a, u)))
            value += x * special.betainc(beta, alpha, w)
        else:
            value = np.full_like(x, np.nan)
    elif family in (dist.Burr, dist.LogLogistic, dist.Paralogistic):
        power = p["power"] if family is dist.Burr else p["shape"]
        shape = 1.0 if family is dist.LogLogistic else p["shape"]
        scale = p["scale"]
        a = 1 / power
        z = power * (np.log(x) - np.log(scale))
        if shape > a:
            # Integral of survival after the logistic change of variable.
            value = np.exp(
                np.log(scale / power) + special.betaln(a, shape - a)
                + np.log(special.betainc(a, shape - a, special.expit(z)))
            )
        else:
            value = x * special.hyp2f1(a, shape, a + 1, -np.exp(z))
    elif family in (dist.InverseBurr, dist.InverseParalogistic):
        power = p["power"] if family is dist.InverseBurr else p["shape"]
        shape, scale = p["shape"], p["scale"]
        a = 1 / power
        z = power * (np.log(x) - np.log(scale))
        u = special.expit(z)
        if a < 1:
            logmean = np.log(scale) + special.betaln(shape + a, 1 - a) + np.log(shape)
            value = np.exp(logmean + np.log(special.betainc(shape + a, 1 - a, u)))
            logcdf = -shape * np.logaddexp(0, -z)
            value += x * (-np.expm1(logcdf))
        else:
            value = np.full_like(x, np.nan)
    elif family in (dist.InverseGamma, dist.InverseWeibull, dist.InverseExponential):
        if family is dist.InverseGamma:
            a, scale, power = p["alpha"], p["theta"], 1.0
        else:
            a, scale, power = 1.0, p["scale"], p.get("shape", 1.0)
        k = 1 / power
        z = np.power(scale / x, power)
        if a > k:
            logmean = np.log(scale) + special.gammaln(a - k) - special.gammaln(a)
            value = np.exp(logmean + np.log(special.gammaincc(a - k, z))) + x * special.gammainc(a, z)
        elif a == k:
            value = np.exp(np.log(scale) - special.gammaln(a) + np.log(special.exp1(z))) + x * special.gammainc(a, z)
        else:
            value = np.full_like(x, np.nan)
    elif family is dist.LogGamma:
        alpha, beta = p["alpha"], p["beta"]
        z = np.log(np.maximum(x, 1)) / beta
        if beta < 1:
            moment = np.exp(-alpha * np.log1p(-beta) + np.log(special.gammainc(alpha, (1 - beta) * z)))
        elif beta == 1:
            moment = np.exp(alpha * np.log(z) - special.gammaln(alpha + 1))
        else:
            moment = np.exp(alpha * np.log(z) - special.gammaln(alpha + 1)) * special.hyp1f1(
                alpha, alpha + 1, (beta - 1) * z
            )
        return np.where(d <= 1, d, moment + x * special.gammaincc(alpha, z))
    elif family is dist.InverseGaussian:
        mu, lam = p["mu"], p["lambda_"]
        root = np.sqrt(lam / x)
        a = root * (x / mu - 1)
        b = root * (x / mu + 1)
        log_a = special.log_ndtr(a)
        log_b = np.log(0.5 * special.erfcx(b / np.sqrt(2))) - 0.5 * a**2
        moment = mu * np.exp(log_a) * (-np.expm1(np.minimum(log_b - log_a, 0)))
        logsf_a = special.log_ndtr(-a)
        sf = np.exp(logsf_a) * (-np.expm1(np.minimum(log_b - logsf_a, 0)))
        value = moment + x * sf
    else:
        return np.full_like(d, np.nan)
    return np.where(d <= loc, d, loc + value)


def _positive_fallback(family: type[DistributionBase], p: dict[str, float], rv: t.Any, d: float) -> float:
    """Integrate survival on a logarithmic grid from the lower endpoint."""
    loc = p.get("loc", 0.0)
    x = d - loc
    if x <= 0:
        return d
    log_x = float(np.log(x))
    scale = p.get("scale", p.get("b", p.get("theta", 1.0)))
    if family is dist.LogGamma:
        scale = 1.0
    # Normalise by the limit in log coordinates. Split at the characteristic
    # scale so that a wide interval cannot hide the body of the distribution.
    split = max(0.0, log_x - np.log(scale))

    def integrand(u: float) -> float:
        log_y = log_x - u
        log_sf = _positive_log_survival(family, p, log_y)
        if log_sf is None:
            log_sf = _log_survival(family, p, rv, loc + np.exp(log_y))
        return float(np.exp(log_y + log_sf))

    # Use a physical (rather than normalised) integral to avoid loss of
    # a finite mean when the policy limit is huge.
    area = 0.0
    for lo, hi in ((0.0, split), (split, np.inf)):
        if lo == hi:
            continue
        part, error = integrate.quad(integrand, lo, hi, epsabs=0, epsrel=1e-10, limit=300)
        if not np.isfinite(part) or error > 1e-7 * abs(part):
            raise ArithmeticError(f"Limited expected value integration failed for {family.__name__}")
        area += part
    return loc + area


def _gev_lev(p: dict[str, float], d: np.ndarray) -> np.ndarray:
    xi, loc, scale = p["shape"], p["loc"], p["scale"]
    z = (d - loc) / scale
    if xi == 0:
        return np.where(z > 35, loc + scale * np.euler_gamma, d - scale * special.exp1(np.exp(-z)))
    rv = stats.genextreme(-xi, loc=loc, scale=scale)
    out = np.empty_like(d)
    lower, upper = rv.support()
    for i, threshold in enumerate(d):
        if threshold <= lower:
            out[i] = threshold
        elif threshold >= upper:
            out[i] = rv.mean()
        elif xi < 1 and threshold > rv.median():
            # The stop-loss formula is well conditioned above the median.
            excess = float(dist.GEV(shape=xi, loc=loc, scale=scale).mean_excess(float(threshold)))
            out[i] = rv.mean() - np.exp(rv.logsf(threshold)) * excess
        else:
            # Integrate the lower tail directly, in units of scale.
            area, _ = integrate.quad(lambda u: rv.cdf(threshold - scale * u), 0, np.inf, epsabs=0, epsrel=1e-10)
            out[i] = threshold - scale * area
    return out


def limited_expected_value(distribution: DistributionBase, limit: float | np.ndarray) -> float | np.ndarray:
    """Evaluate the expected capped loss, preserving limit shape and order."""
    family = type(distribution)
    p: dict[str, float] = {}
    for name, value in distribution._params.items():
        if isinstance(value, StochasticScalar):
            raise ValueError("lev requires scalar distribution parameters")
        if isinstance(value, (dt.date, np.datetime64)) or family is dist.Categorical:
            raise TypeError("lev requires numeric losses")
        p[name] = float(value)
    inputs = np.asarray(limit, dtype=float)
    result = np.full(inputs.shape, np.nan)
    valid = np.isfinite(inputs)
    d, inverse = np.unique(inputs[valid], return_inverse=True)
    with np.errstate(all="ignore"):
        out = _remaining_lev(family, p, d)
        if out is None:
            if family is dist.GEV:
                out = _gev_lev(p, d)
            elif family in (*_supported_families(), *_other_families()):
                out = _severity_lev(family, p, d)
                loc = p.get("loc", 0.0)
                bad = (~np.isfinite(out)) | (out > d + 1e-12 * np.maximum(1, np.abs(d))) | ((d > loc) & (out <= loc))
                if np.any(bad):
                    rv = (
                        None
                        if family in (dist.LogGamma, dist.GeneralizedBetaPrime)
                        else _frozen_distribution(family, p)
                    )
                    for i in np.flatnonzero(bad):
                        out[i] = _positive_fallback(family, p, rv, float(d[i]))
            else:
                raise ValueError(f"No numeric limited expected value for {family.__name__}")
        result[valid] = out[inverse]
    return float(result) if result.ndim == 0 else result

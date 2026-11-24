"""Deterministic severity mean excess formulas and conditional tail integration."""

from __future__ import annotations

import typing as t

import numpy as np
from scipy import integrate, special

from . import distributions as dist
from .fitting import _frozen_distribution, _gamma_logsf, _gb2_log_regularized_beta, _other_families, _supported_families
from .stochastic_scalar import StochasticScalar

if t.TYPE_CHECKING:
    from .distributions import DistributionBase


def _pareto_excess(ds: np.ndarray, shape: float, scale: float) -> np.ndarray:
    """Evaluate Pareto residual life."""
    return np.maximum(scale - ds, 0) + np.maximum(ds, scale) / (shape - 1)


def _gpd_excess(ds: np.ndarray, shape: float, scale: float, loc: float) -> np.ndarray:
    """Evaluate GPD residual life."""
    base = np.maximum(ds, loc)
    return base - ds + (scale + shape * (base - loc)) / (1 - shape)


def _exponential_excess(ds: np.ndarray, scale: float, loc: float) -> np.ndarray:
    """Evaluate exponential residual life."""
    return np.maximum(loc - ds, 0) + scale


def _gamma_far_excess(alpha: float, z: np.ndarray) -> np.ndarray:
    """Integrate the conditional gamma density after removing its exponential tilt.

    Gauss-Laguerre quadrature is used only for z > 2*alpha and z > 500,
    where the residual integrand is smooth and close to one.
    """
    nodes, weights = special.roots_laguerre(32)
    rate = 1 - (alpha - 1) / z
    u = nodes / rate[:, None]
    ratio = u / z[:, None]
    density = weights * np.exp((alpha - 1) * (np.log1p(ratio) - ratio))
    return np.sum(density * u, axis=1) / np.sum(density, axis=1)


def _gamma_tail_excess(alpha: float, theta: float, d: float) -> float:
    """Integrate conditional gamma survival when the moment ratio is ill-conditioned."""
    z = d / theta
    log_survival = float(_gamma_logsf(alpha, z))
    log_next_survival = float(_gamma_logsf(alpha + 1.0, z))
    excess = alpha * np.exp(log_next_survival - log_survival) - z
    if np.isfinite(excess) and excess > 1e-5 * max(1.0, z):
        return float(theta * excess)
    integral, error = integrate.quad(
        lambda u: np.exp(float(_gamma_logsf(alpha, z + u)) - log_survival),
        0.0,
        np.inf,
        epsabs=1e-11,
        epsrel=1e-10,
        limit=300,
    )
    if not np.isfinite(integral) or integral <= 0 or error > 1e-7 * integral:
        raise ArithmeticError("Gamma conditional survival integration did not converge")
    return float(theta * integral)


def _gamma_excess(ds: np.ndarray, alpha: float, theta: float, loc: float = 0.0) -> np.ndarray:
    """Evaluate shifted Gamma residual life using the incomplete-gamma moment ratio."""
    x = ds - loc
    if alpha == 1:
        return theta - np.minimum(x, 0)
    z = np.maximum(x, 0) / theta
    sf = special.gammaincc(alpha, z)
    excess = alpha * special.gammaincc(alpha + 1, z) / sf - z
    out = np.where((sf > 1e-280) & (excess > 1e-5 * np.maximum(1, z)), theta * excess, np.nan)
    far = z > max(500.0, 2 * alpha)
    out[far] = theta * _gamma_far_excess(alpha, z[far])
    out = np.where(x <= 0, alpha * theta - x, out)
    return _repair_tail(x, out, lambda d: _gamma_tail_excess(alpha, theta, d))


def _positive_log_excess(log_x: np.ndarray, log_ratio: np.ndarray) -> np.ndarray:
    """Compute x*expm1(log_ratio) without intermediate overflow."""
    return np.exp(log_x + log_ratio + np.log(-np.expm1(-log_ratio)))


def _lognormal_tail_excess(mu: float, sigma: float, d: float) -> float:
    """Integrate a conditional normal density when two moments nearly cancel."""
    z = (np.log(d) - mu) / sigma
    width = 1 / max(1.0, z)

    def density(t: float) -> float:
        u = width * t
        return float(np.exp(-z * u - 0.5 * u**2))

    def excess_density(t: float) -> float:
        # expm1 resolves arbitrarily small conditional excesses.
        return float(np.expm1(sigma * width * t) * density(t))

    denominator, den_error = integrate.quad(density, 0, np.inf, epsabs=0, epsrel=1e-10)
    numerator, num_error = integrate.quad(excess_density, 0, np.inf, epsabs=0, epsrel=1e-10)
    if denominator <= 0 or numerator <= 0 or den_error > 1e-7 * denominator or num_error > 1e-7 * numerator:
        raise ArithmeticError("LogNormal conditional moment integration did not converge")
    return float(d * (numerator / denominator))


def _lognormal_excess(ds: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Evaluate the normal-tail moment ratio with a cancellation-safe fallback."""
    log_x = np.log(ds)
    z = (log_x - mu) / sigma
    log_ratio = sigma * (0.5 * sigma - z) + special.log_ndtr(sigma - z) - special.log_ndtr(-z)
    hi = z >= sigma
    den = special.erfcx(z[hi] / np.sqrt(2.0))
    relative = (special.erfcx((z[hi] - sigma) / np.sqrt(2.0)) - den) / den
    log_ratio[hi] = np.log1p(relative)
    out = _positive_log_excess(log_x, log_ratio)
    out = np.where((log_ratio > 1e-5) | (z < -8), out, np.nan)
    out = np.where(ds <= 0, np.exp(mu + 0.5 * sigma**2) - ds, out)
    return _repair_tail(ds, out, lambda d: _lognormal_tail_excess(mu, sigma, d))


def _weibull_excess(ds: np.ndarray, shape: float, scale: float, loc: float) -> np.ndarray:
    """Evaluate the incomplete-gamma survival integral."""
    a = 1 / shape
    if shape == 1:
        return scale + np.maximum(loc - ds, 0)
    z = np.power(np.maximum(ds - loc, 0) / scale, shape)
    log_value = np.log(scale) + special.gammaln(1 + a) + np.log(special.gammaincc(a, z)) + z
    out = np.exp(log_value)
    far = z > 1e7
    out[far] = np.exp(np.log(scale / shape) + (a - 1) * np.log(z[far]) + np.log1p((a - 1) / z[far]))
    out = np.where(ds <= loc, loc + scale * np.exp(special.gammaln(1 + a)) - ds, out)
    unresolved = ~np.isfinite(out) | (out <= 0)
    if np.any(unresolved):
        tail_z = z[unresolved]
        if np.any(~np.isfinite(tail_z)):
            raise ArithmeticError("Weibull attachment exceeds numeric range")
        log_tail = np.log(scale) + special.gammaln(1 + a) + _gamma_logsf(a, tail_z) + tail_z
        out[unresolved] = np.exp(log_tail)
    return out


def _burr_excess(ds: np.ndarray, power: float, shape: float, scale: float, loc: float) -> np.ndarray:
    """Evaluate Burr XII residual life via the complementary incomplete beta."""
    x = ds - loc
    if power == 1:
        return np.maximum(-x, 0) + (scale + np.maximum(x, 0)) / (shape - 1)
    a, b = shape - 1 / power, 1 / power
    log_w = -np.logaddexp(0, power * (np.log(x) - np.log(scale)))
    w = np.exp(log_w)
    log_value = np.log(scale / power) + special.betaln(a, b) + np.log(special.betainc(a, b, w)) - shape * log_w
    far = (w < 1e-12) & (abs(b - 1) * w < 1e-12)
    log_value[far] = np.log(scale / power) - np.log(a) - b * log_w[far]
    out = np.exp(log_value)
    mean = scale * np.exp(special.betaln(a, b + 1) + np.log(shape))
    out = np.where(x <= 0, mean - x, out)
    unresolved = ~np.isfinite(out) | (out <= 0)
    if np.any(unresolved):
        tail_log_w = log_w[unresolved]
        log_beta = special.betaln(a, b) + _gb2_log_regularized_beta(a, b, tail_log_w)
        out[unresolved] = np.exp(np.log(scale / power) + log_beta - shape * tail_log_w)
    return out


def _loglogistic_excess(ds: np.ndarray, shape: float, scale: float, loc: float) -> np.ndarray:
    """Use Burr XII with second shape equal to one."""
    return _burr_excess(ds, shape, 1.0, scale, loc)


def _paralogistic_excess(ds: np.ndarray, shape: float, scale: float, loc: float) -> np.ndarray:
    """Use Burr XII with equal shape parameters."""
    return _burr_excess(ds, shape, shape, scale, loc)


def _beta_prime_excess(x: np.ndarray, a: float, b: float, p: float, q: float) -> np.ndarray:
    """Evaluate the shared GB2 and inverse-Burr truncated first moment."""
    out = np.full(x.shape, np.nan, dtype=float)
    log_mean = np.log(b) + special.betaln(p + 1 / a, q - 1 / a) - special.betaln(p, q)
    positive = x > 0
    log_power = a * (np.log(x[positive]) - np.log(b))
    log_z = -np.logaddexp(0, log_power)
    z = np.exp(log_z)
    complement = np.exp(-np.logaddexp(0, -log_power))
    # Use the hypergeometric ratio in the far tail, cancelling z**q.
    far = log_z < np.log(1e-6)
    log_ratio = np.full(z.shape, np.nan)
    h0 = special.hyp2f1(q, 1 - p, q + 1, z[far])
    tail_shape = q - 1 / a
    h1 = special.hyp2f1(tail_shape, 1 - p - 1 / a, tail_shape + 1, z[far])
    log_ratio[far] = -np.log1p(-1 / (a * q)) - np.log1p(-z[far]) / a + np.log(h1) - np.log(h0)
    central = ~far
    # Evaluate from whichever endpoint is better resolved.
    survival_coordinate, cdf_coordinate = z[central], complement[central]
    numerator = np.where(
        survival_coordinate < 0.5,
        special.betainc(tail_shape, p + 1 / a, survival_coordinate),
        special.betaincc(p + 1 / a, tail_shape, cdf_coordinate),
    )
    denominator = np.where(
        survival_coordinate < 0.5,
        special.betainc(q, p, survival_coordinate),
        special.betaincc(p, q, cdf_coordinate),
    )
    if q == 1:
        denominator = -np.expm1(-p * np.logaddexp(0, -log_power[central]))
    log_ratio[central] = log_mean + np.log(numerator) - np.log(denominator) - np.log(x[positive][central])
    out[positive] = np.where(log_ratio > 1e-8, _positive_log_excess(np.log(x[positive]), log_ratio), np.nan)
    out[~positive] = np.exp(log_mean) - x[~positive]
    return out


def _gb2_tail_excess(pars: dict[str, float], attachment: float) -> float:
    """GB2 mean residual life from complementary incomplete beta moments."""
    a, b, p, q = (pars[name] for name in ("a", "b", "p", "q"))
    if a * q <= 1:
        return np.inf
    if attachment <= 0.0:
        mean = b * np.exp(special.betaln(p + 1 / a, q - 1 / a) - special.betaln(p, q))
        return float(mean - attachment)

    log_ratio = a * (np.log(attachment) - np.log(b))
    log_z = float(-np.logaddexp(0.0, log_ratio))
    z = float(np.exp(log_z))
    if log_z < np.log(1e-6):
        # Exact incomplete-beta identity:
        # B_z(A,B) = z**A/A * 2F1(A, 1-B; A+1; z).
        # Taking the ratio of the two beta moments cancels the tiny
        # probabilities, the beta normalisers and z**(q-1/a).
        tail_shape = q - 1.0 / a
        h_survival = float(special.hyp2f1(q, 1.0 - p, q + 1.0, z))
        h_moment = float(special.hyp2f1(tail_shape, 1.0 - p - 1.0 / a, tail_shape + 1.0, z))
        if not (np.isfinite(h_survival) and np.isfinite(h_moment)):
            raise ArithmeticError("GB2 incomplete-beta tail ratio did not converge")
        if h_survival <= 0.0 or h_moment <= 0.0:
            raise ArithmeticError("GB2 incomplete-beta tail ratio is invalid")
        log_conditional_ratio = -np.log1p(-1.0 / (a * q)) - np.log1p(-z) / a + np.log(h_moment) - np.log(h_survival)
    else:
        log_survival = float(_gb2_log_regularized_beta(q, p, np.asarray(log_z)))
        log_moment = float(_gb2_log_regularized_beta(q - 1.0 / a, p + 1.0 / a, np.asarray(log_z)))
        log_conditional_ratio = (
            np.log(b)
            + special.betaln(p + 1.0 / a, q - 1.0 / a)
            - special.betaln(p, q)
            + log_moment
            - log_survival
            - np.log(attachment)
        )
    # Computing E[X|X>d]/d - 1 using expm1 avoids cancellation when
    # the mean residual life is small compared with attachment.
    return float(attachment * np.expm1(log_conditional_ratio))


def _gb2_excess(ds: np.ndarray, a: float, b: float, p: float, q: float) -> np.ndarray:
    """Evaluate GB2 residual life, retaining the log-beta fallback."""
    out = _beta_prime_excess(ds, a, b, p, q)
    pars = {"a": a, "b": b, "p": p, "q": q}
    return _repair_tail(ds, out, lambda d: _gb2_tail_excess(pars, d))


def _inverse_burr_excess(ds: np.ndarray, power: float, shape: float, scale: float, loc: float) -> np.ndarray:
    """Use GB2 with second beta shape equal to one."""
    return _beta_prime_excess(ds - loc, power, scale, shape, 1.0)


def _inverse_paralogistic_excess(ds: np.ndarray, shape: float, scale: float, loc: float) -> np.ndarray:
    """Use inverse Burr with equal shape parameters."""
    return _inverse_burr_excess(ds, shape, shape, scale, loc)


def _inverse_generalized_gamma_excess(x: np.ndarray, a: float, b: float, power: float) -> np.ndarray:
    """Evaluate the shared inverse-gamma and inverse-Weibull moment ratio."""
    k = 1 / power
    z = np.power(b / x, power)
    log_mean = np.log(b) + special.gammaln(a - k) - special.gammaln(a)
    lr = log_mean + np.log(special.gammainc(a - k, z)) - np.log(special.gammainc(a, z)) - np.log(x)
    # P(a,z) = z**a / Gamma(a+1) * 1F1(a;a+1;-z).
    # Cancel powers analytically for small z (even if z underflows).
    far = (x > 0) & (z < 1e-4)
    lr[far] = (
        -np.log1p(-k / a)
        + np.log(special.hyp1f1(a - k, a - k + 1, -z[far]))
        - np.log(special.hyp1f1(a, a + 1, -z[far]))
    )
    out = np.where(lr > 1e-8, _positive_log_excess(np.log(x), lr), np.nan)
    out = np.where(x <= 0, np.exp(log_mean) - x, out)
    return out


def _inverse_gamma_excess(ds: np.ndarray, alpha: float, theta: float, loc: float) -> np.ndarray:
    """Use inverse generalized gamma with power one."""
    return _inverse_generalized_gamma_excess(ds - loc, alpha, theta, 1.0)


def _inverse_weibull_excess(ds: np.ndarray, shape: float, scale: float, loc: float) -> np.ndarray:
    """Use inverse generalized gamma with gamma shape one."""
    return _inverse_generalized_gamma_excess(ds - loc, 1.0, scale, shape)


def _inverse_gaussian_excess(ds: np.ndarray, mu: float, lambda_: float) -> np.ndarray:
    """Evaluate inverse Gaussian residual life via normal tail moments."""
    lam = lambda_
    out = np.full(ds.shape, np.nan, dtype=float)
    x = ds[ds > 0]
    root = np.sqrt(lam / x)
    normal_a = root * (x / mu - 1)
    normal_b = root * (x / mu + 1)
    # The exponential tilt cancels the normal density exponent.
    log_a = special.log_ndtr(-normal_a)
    log_b = np.log(0.5 * special.erfcx(normal_b / np.sqrt(2))) - 0.5 * normal_a**2
    # A+B and A-B are the first moment / mu and survival.
    log_sf = log_a + np.log(-np.expm1(log_b - log_a))
    log_moment = np.log(mu) + np.logaddexp(log_a, log_b)
    lr = log_moment - log_sf - np.log(x)
    out[ds > 0] = np.where((lr > 1e-6) & (log_a - log_b > 1e-6), x * np.expm1(lr), np.nan)
    out[ds <= 0] = mu - ds[ds <= 0]
    return out


def _gev_excess(ds: np.ndarray, shape: float, loc: float, scale: float) -> np.ndarray:
    """Evaluate the GEV truncated gamma moment away from the Gumbel limit."""
    xi = shape
    out = np.full(ds.shape, np.nan, dtype=float)
    if abs(xi) >= 1e-5:
        z = (ds - loc) / scale
        base = 1 + xi * z
        if xi < 0:
            base = (-xi / scale) * ((loc - scale / xi) - ds)
        inside = base > 0
        log_y = -np.log(base[inside]) / xi
        y = np.exp(log_y)
        log_ratio = np.empty(y.shape)
        small = y < 1e-4
        # Cancel powers of y before evaluating the lower incomplete gamma.
        ys = y[small]
        correction = special.hyp1f1(1 - xi, 2 - xi, -ys)
        survival_factor = np.ones_like(ys)
        nonzero = ys > 0
        survival_factor[nonzero] = -np.expm1(-ys[nonzero]) / ys[nonzero]
        log_ratio[small] = -np.log1p(-xi) + np.log(correction) - np.log(survival_factor)
        yc = y[~small]
        log_ratio[~small] = (
            special.gammaln(1 - xi)
            + np.log(special.gammainc(1 - xi, yc))
            - np.log(-np.expm1(-yc))
            - np.log(base[inside][~small])
        )
        excess = scale * base[inside] / xi * np.expm1(log_ratio)
        out[inside] = np.where(abs(log_ratio) > 1e-7, excess, np.nan)
        if xi > 0:
            out[~inside] = loc + scale * np.expm1(special.gammaln(1 - xi)) / xi - ds[~inside]
    return out


def _loggamma_tail_excess(alpha: float, beta: float, d: float) -> float:
    """Integrate a gamma excess moment without subtracting the attachment."""
    z = np.log(d) / beta
    # Integrate in units of the tilted exponential tail. The numerator
    # includes 1-exp(-beta*u), so it stays bounded even for beta near one.
    rate = 1 - beta
    slope = (alpha - 1) / z
    if slope < rate / 2:
        width = 1 / (rate - slope)
        nodes, weights = special.roots_laguerre(32)
        u = nodes * width
        correction = np.exp((alpha - 1) * (np.log1p(u / z) - u / z))
        numerator = width * np.sum(weights * correction * (-np.expm1(-beta * u)))
        width_den = 1 / (1 - slope)
        v = nodes * width_den
        denominator = width_den * np.sum(weights * np.exp((alpha - 1) * (np.log1p(v / z) - v / z)))
        return float(d * (numerator / denominator))
    log_num = -alpha * np.log1p(-beta) + float(_gamma_logsf(alpha, z * rate))
    log_den = float(_gamma_logsf(alpha, z))
    log_ratio = log_num - log_den - np.log(d)
    if log_ratio <= 0:
        raise ArithmeticError("LogGamma conditional moment lost numerical precision")
    return float(np.exp(np.log(d) + log_ratio + np.log(-np.expm1(-log_ratio))))


def _loggamma_excess(ds: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Evaluate the log-gamma moment ratio with stable endpoint and tail handling."""
    x = np.maximum(ds, 1)
    if alpha == 1:
        return x * (beta / (1 - beta)) + np.maximum(1 - ds, 0)
    log_x = np.log(x)
    z = log_x / beta
    log_ratio = -alpha * np.log1p(-beta) + np.log(special.gammaincc(alpha, z * (1 - beta)))
    log_ratio -= np.log(special.gammaincc(alpha, z)) + log_x
    out = np.where(log_ratio > 1e-5, _positive_log_excess(log_x, log_ratio), np.nan)
    out += np.maximum(1 - ds, 0)
    out = np.where(ds <= 1, np.expm1(-alpha * np.log1p(-beta)) + 1 - ds, out)
    return _repair_tail(ds, out, lambda d: _loggamma_tail_excess(alpha, beta, d))


def _closed_form(family: type[DistributionBase], pars: dict[str, float], ds: np.ndarray) -> np.ndarray:
    """Select the vectorised formula by distribution family."""
    formulas: dict[type[DistributionBase], t.Callable[..., np.ndarray]] = {
        dist.Pareto: _pareto_excess,
        dist.GPD: _gpd_excess,
        dist.Exponential: _exponential_excess,
        dist.Gamma: _gamma_excess,
        dist.LogNormal: _lognormal_excess,
        dist.Weibull: _weibull_excess,
        dist.Burr: _burr_excess,
        dist.LogLogistic: _loglogistic_excess,
        dist.Paralogistic: _paralogistic_excess,
        dist.GeneralizedBetaPrime: _gb2_excess,
        dist.InverseBurr: _inverse_burr_excess,
        dist.InverseParalogistic: _inverse_paralogistic_excess,
        dist.InverseGamma: _inverse_gamma_excess,
        dist.InverseWeibull: _inverse_weibull_excess,
        dist.InverseGaussian: _inverse_gaussian_excess,
        dist.GEV: _gev_excess,
        dist.LogGamma: _loggamma_excess,
    }
    formula = formulas.get(family)
    if formula is None:
        return np.full(ds.shape, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        out = formula(ds, **pars)
    return np.where(out > 0, out, np.nan)


def severity_mean_excess(distribution: DistributionBase, threshold: float | np.ndarray) -> float | np.ndarray:
    """Calculate E[X - d | X > d] at the supplied thresholds.

    Args:
        distribution: A PAL severity distribution with fixed, scalar parameters.
        threshold: One or more attachment levels d.

    Returns:
        Mean residual loss for each threshold. Infinite means produce infinity,
        and thresholds beyond a finite upper endpoint return NaN.

    Notes:
        Vectorised analytical expressions are used where available. Other
        severity families use a conditional survival integral with reuse
        across nearby decreasing thresholds. This function uses the CPU, without sampling.
    """
    family = type(distribution)
    if family not in (*_supported_families(), *_other_families()):
        raise ValueError(f"No mean excess implementation for {family.__name__}")
    params = distribution._params
    pars: dict[str, float] = {}
    for key, value in params.items():
        if isinstance(value, StochasticScalar):
            raise ValueError("mean excess requires scalar distribution parameters")
        pars[key] = float(value)
    frozen = None if family in (dist.LogGamma, dist.GeneralizedBetaPrime) else _frozen_distribution(family, pars)
    inputs = np.asarray(threshold, dtype=float)
    result = np.full(inputs.shape, np.nan, dtype=float)
    inf_mean = (
        (family is dist.GEV and pars["shape"] >= 1.0)
        or (family is dist.GeneralizedBetaPrime and pars["a"] * pars["q"] <= 1.0)
        or (family is dist.LogGamma and pars["beta"] >= 1.0)
        or (family is dist.Pareto and pars["shape"] <= 1.0)
        or (family is dist.GPD and pars["shape"] >= 1.0)
        or (family is dist.Burr and pars["power"] * pars["shape"] <= 1.0)
        or (family is dist.LogLogistic and pars["shape"] <= 1.0)
        or (family is dist.Paralogistic and pars["shape"] <= 1.0)
        or (family is dist.InverseBurr and pars["power"] <= 1.0)
        or (family is dist.InverseGamma and pars["alpha"] <= 1.0)
        or (family is dist.InverseWeibull and pars["shape"] <= 1.0)
        or family is dist.InverseExponential
        or (family is dist.InverseParalogistic and pars["shape"] <= 1.0)
    )
    if family is dist.GeneralizedBetaPrime:
        lower, upper = 0.0, np.inf
    else:
        lower, upper = (1.0, np.inf) if frozen is None else frozen.support()
    valid = np.isfinite(inputs) & (inputs < upper)
    if inf_mean:
        result[valid] = np.inf
        return float(result) if result.ndim == 0 else result
    # Work only on distinct, valid thresholds, preserving input order on return.
    levels, inverse = np.unique(inputs[valid], return_inverse=True)
    values = _closed_form(family, pars, levels)
    _integrate_unresolved(family, frozen, lower, upper, levels, values)
    result[valid] = values[inverse]
    return float(result) if result.ndim == 0 else result


def _integrate_unresolved(
    family: type[DistributionBase],
    frozen: t.Any,
    lower: float,
    upper: float,
    levels: np.ndarray,
    values: np.ndarray,
) -> None:
    """Integrate unresolved thresholds backwards, reusing conditional tail areas."""
    previous_start: float | None = None
    previous_excess = 0.0
    previous_log_sf = 0.0
    for index in np.flatnonzero(np.isnan(values))[::-1]:
        d = float(levels[index])
        if frozen is None:
            raise ArithmeticError(f"No resolved mean excess formula for {family.__name__} at {d}")
        # Scale by the local inverse hazard. Integrating in units of tail
        # width prevents quadrature missing a narrow tail at a high threshold.
        start = max(float(d), float(lower))
        log_start_sf = float(frozen.logsf(start))
        if not np.isfinite(log_start_sf):
            raise ArithmeticError(f"{family.__name__} tail survival is numerically unresolved at {d}")
        log_start_pdf = float(frozen.logpdf(start))
        tail_probability = float(frozen.sf(start))
        # The conditional median gives a representative tail width even if
        # the density at the attachment is vanishingly small.
        local_scale = np.nan
        if tail_probability > 0:
            median = float(frozen.isf(tail_probability * 0.5))
            local_scale = median - start
        if not np.isfinite(local_scale) or local_scale <= 0:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                local_scale = float(np.exp(log_start_sf - log_start_pdf))
        if not np.isfinite(local_scale) or local_scale <= 0:
            local_scale = max(abs(start) * 0.1, 1.0)
        remaining = upper - start
        limit = remaining / local_scale if np.isfinite(remaining) else np.inf

        def conditional_survival(
            u: float,
            start: float = start,
            local_scale: float = local_scale,
            log_start_sf: float = log_start_sf,
        ) -> float:
            x = start + local_scale * u
            return float(np.exp(float(frozen.logsf(x)) - log_start_sf))

        # Reuse the integrated tail at the preceding (higher) attachment.
        # Work with conditional survival ratios, not tiny unconditional areas.
        reuse = previous_start is not None and 0 <= (previous_start - start) / local_scale <= 100
        if family is dist.GeneralizedInverseGaussian:
            # SciPy's GIG survival may suffer cancellation in the far tail.
            # Integrate the conditional density instead, avoiding noisy S(x).
            def scaled_excess_density(
                u: float,
                start: float = start,
                local_scale: float = local_scale,
                log_start_sf: float = log_start_sf,
            ) -> float:
                x = start + local_scale * u
                return float(u * np.exp(float(frozen.logpdf(x)) - log_start_sf))

            interval = (previous_start - start) / local_scale if reuse and previous_start is not None else limit
            area, error = integrate.quad(scaled_excess_density, 0.0, interval, epsrel=1e-9, limit=300)
            excess = local_scale**2 * area
            if reuse and previous_start is not None:
                excess += np.exp(previous_log_sf - log_start_sf) * (previous_excess + previous_start - start)
        elif reuse and previous_start is not None:
            interval = (previous_start - start) / local_scale
            area, error = integrate.quad(conditional_survival, 0.0, interval, epsrel=1e-9, limit=300)
            excess = local_scale * area + np.exp(previous_log_sf - log_start_sf) * previous_excess
        else:
            area, error = integrate.quad(conditional_survival, 0.0, limit, epsrel=1e-9, limit=300)
            excess = local_scale * area

        if not np.isfinite(area) or area < 0 or not np.isfinite(error) or error > 1e-7 * max(1.0, abs(area)):
            raise ArithmeticError(f"Mean excess integration did not converge for {family.__name__} at {d}")
        values[index] = (start - d) + float(np.exp(log_start_sf - float(frozen.logsf(d)))) * excess
        previous_start, previous_excess, previous_log_sf = start, float(excess), log_start_sf


def _repair_tail(ds: np.ndarray, out: np.ndarray, fallback: t.Callable[[float], float]) -> np.ndarray:
    """Evaluate only unresolved entries with a more robust scalar expression."""
    for i in np.flatnonzero(np.isnan(out) | (out <= 0)):
        out[i] = fallback(float(ds[i]))
    return out

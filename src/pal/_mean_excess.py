"""Deterministic severity mean excess formulas and conditional tail integration."""

from __future__ import annotations

import typing as t
from decimal import Decimal, localcontext
from functools import lru_cache

import numpy as np
from scipy import integrate, special

from . import distributions as dist
from .fitting import _frozen_distribution, _gamma_logsf, _gb2_log_regularized_beta, _other_families, _supported_families
from .stochastic_scalar import StochasticScalar

if t.TYPE_CHECKING:
    from .distributions import DistributionBase


@lru_cache(maxsize=1)
def _laguerre_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Cache the fixed quadrature rules used in conditional tail calculations."""
    return special.roots_laguerre(order)


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


def _tail_fraction_pair(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a tail continued fraction and its divided difference.

    The denominator of each fraction is linear in its argument.
    Backward recurrence propagates the divided difference analytically,
    including when both arguments round to the same value. Doubling the
    depth provides a convergence check; unresolved entries return NaN.
    """
    # Normal Mills: 1/R(x) = x + C(x).
    # Upper gamma: x**alpha * exp(-x) / Gamma(alpha,x) = x+1-alpha+C(x).
    if not x.size:
        return np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    previous = None
    converged = np.zeros(x.shape, dtype=bool)
    for depth in (8, 16, 32, 64, 128, 256, 512, 1024):
        cx = np.zeros_like(x)
        cy = np.zeros_like(y)
        difference = np.zeros_like(x)
        for n in range(depth, 0, -1):
            numerator = n if alpha is None else n * (alpha - n)
            offset = 0 if alpha is None else 2 * n + 1 - alpha
            dx = x + offset + cx
            dy = y + offset + cy
            cx = numerator / dx
            cy = numerator / dy
            difference = -(cx / dy) * (1 + difference)
        current = (cx, cy, difference)
        if previous is not None:
            converged = np.ones(x.shape, dtype=bool)
            for value, old in zip(current, previous, strict=True):
                converged &= np.abs(value - old) <= 4e-15 * np.maximum(np.abs(value), 1e-300)
            if np.all(converged):
                return current
        previous = current
    return (np.where(converged, cx, np.nan), np.where(converged, cy, np.nan), np.where(converged, difference, np.nan))


def _gamma_far_excess(alpha: float, z: np.ndarray) -> np.ndarray:
    """Cancel the attachment analytically using the upper-gamma fraction."""
    correction, _, _ = _tail_fraction_pair(z, z, alpha)
    return 1 + correction


def _gamma_tail_excess(alpha: float, theta: float, d: float) -> float:
    """Integrate conditional gamma survival when the moment ratio is ill-conditioned."""
    z = d / theta
    if z > alpha + 1:
        candidate = float(_gamma_far_excess(alpha, np.array([z]))[0])
        if np.isfinite(candidate) and candidate > 0:
            return theta * candidate
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
    if np.any(far):
        out[far] = theta * _gamma_far_excess(alpha, z[far])
    out = np.where(x <= 0, alpha * theta - x, out)
    return _repair_tail(x, out, lambda d: _gamma_tail_excess(alpha, theta, d))


def _positive_log_excess(log_x: np.ndarray, log_ratio: np.ndarray) -> np.ndarray:
    """Compute x*expm1(log_ratio) without intermediate overflow."""
    return np.exp(log_x + log_ratio + np.log(-np.expm1(-log_ratio)))


@lru_cache(maxsize=64)
def _lognormal_center(mu: float) -> tuple[float, float]:
    """Return a rounded median and the correction to its logarithm."""
    center = float(np.exp(mu))
    # One compensated constant per log-location, not per threshold.
    with localcontext() as context:
        context.prec = 50
        correction = float(Decimal.from_float(center).ln() - Decimal.from_float(mu))
    return center, correction


def _lognormal_coordinate(ds: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Resolve log(d)-mu accurately when the distribution is very concentrated."""
    z = (np.log(ds) - mu) / sigma
    if sigma < 1e-4 and -700 < mu < 700:
        center, correction = _lognormal_center(mu)
        near = (ds >= 0.5 * center) & (ds <= 1.5 * center)
        z[near] = (np.log1p((ds[near] - center) / center) + correction) / sigma
    return z


def _lognormal_fraction_excess(z: np.ndarray, sigma: float) -> np.ndarray:
    """Evaluate the Mills-ratio difference without subtracting close arguments."""
    y = z - sigma
    _, correction, difference = _tail_fraction_pair(z, y)
    return sigma * (1 + difference) / (y + correction)


def _lognormal_series_excess(z: np.ndarray, sigma: float) -> np.ndarray:
    """Expand exp(sigma*u)-1 in conditional normal moments.

    Used only for small sigma and moderate z, where the moment recurrence
    is stable and the series converges in a few terms.
    """
    mills = np.sqrt(np.pi / 2) * special.erfcx(z / np.sqrt(2))
    previous = np.ones_like(z)
    moment = 1 / mills - z
    coefficient = sigma
    total = coefficient * moment
    converged = np.zeros(z.shape, dtype=bool)
    for n in range(2, 33):
        following = (n - 1) * previous - z * moment
        coefficient *= sigma / n
        term = coefficient * following
        total += term
        converged = np.abs(term) <= 2e-15 * total
        if np.all(converged):
            break
        previous, moment = moment, following
    return np.where(converged, total, np.nan)


def _lognormal_narrow_excess(ds: np.ndarray, z: np.ndarray, sigma: float) -> np.ndarray:
    """Use a moment series near the mean and a continued fraction in the tail."""
    result = np.empty_like(z)
    tail = z - sigma >= 4
    result[tail] = _lognormal_fraction_excess(z[tail], sigma)
    result[~tail] = _lognormal_series_excess(z[~tail], sigma)
    return ds * result


def _lognormal_tail_excess(sigma: float, d: float, z: float) -> float:
    """Integrate a conditional normal density when two moments nearly cancel."""
    width = 1 / max(1.0, z)

    def density(t: float) -> float:
        u = width * t
        return float(np.exp(-z * u - 0.5 * u**2))

    def excess_density(t: float) -> float:
        # expm1 resolves arbitrarily small conditional excesses.
        u = width * t
        tilt = sigma * u
        return float(np.exp(-z * u - 0.5 * u**2 + tilt) * (-np.expm1(-tilt)))

    denominator, den_error = integrate.quad(density, 0, np.inf, epsabs=0, epsrel=1e-10)
    numerator, num_error = integrate.quad(excess_density, 0, np.inf, epsabs=0, epsrel=1e-10)
    if denominator <= 0 or numerator <= 0 or den_error > 1e-7 * denominator or num_error > 1e-7 * numerator:
        raise ArithmeticError("LogNormal conditional moment integration did not converge")
    return float(d * (numerator / denominator))


def _lognormal_excess(ds: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Evaluate the normal-tail moment ratio with a cancellation-safe fallback."""
    log_x = np.log(ds)
    z = _lognormal_coordinate(ds, mu, sigma)
    log_ratio = sigma * (0.5 * sigma - z) + special.log_ndtr(sigma - z) - special.log_ndtr(-z)
    hi = z >= sigma
    den = special.erfcx(z[hi] / np.sqrt(2.0))
    relative = (special.erfcx((z[hi] - sigma) / np.sqrt(2.0)) - den) / den
    log_ratio[hi] = np.log1p(relative)
    out = _positive_log_excess(log_x, log_ratio)
    out = np.where((log_ratio > 1e-5) | (z < -8), out, np.nan)
    if sigma < 1e-4:
        narrow = (ds > 0) & (z >= -12) & np.isfinite(z)
        out[narrow] = _lognormal_narrow_excess(ds[narrow], z[narrow], sigma)
    out = np.where(ds <= 0, np.exp(mu + 0.5 * sigma**2) - ds, out)
    unresolved = (np.isnan(out) | (out <= 0)) & (z - sigma >= 4)
    if np.any(unresolved):
        out[unresolved] = ds[unresolved] * _lognormal_fraction_excess(z[unresolved], sigma)
    for i in np.flatnonzero(np.isnan(out) | (out <= 0)):
        try:
            out[i] = _lognormal_tail_excess(sigma, float(ds[i]), float(z[i]))
        except ArithmeticError:
            out[i] = np.nan
    return out


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
        for i in np.flatnonzero(unresolved):
            try:
                tail_log_w = log_w[i]
                log_beta = special.betaln(a, b) + _gb2_log_regularized_beta(a, b, np.asarray(tail_log_w))
                out[i] = np.exp(np.log(scale / power) + log_beta - shape * tail_log_w)
            except ArithmeticError:
                out[i] = np.nan
    return out


def _loglogistic_excess(ds: np.ndarray, shape: float, scale: float, loc: float) -> np.ndarray:
    """Use Burr XII with second shape equal to one."""
    return _burr_excess(ds, shape, 1.0, scale, loc)


def _paralogistic_excess(ds: np.ndarray, shape: float, scale: float, loc: float) -> np.ndarray:
    """Use Burr XII with equal shape parameters."""
    return _burr_excess(ds, shape, shape, scale, loc)


def _beta_fraction_coefficients(
    j: int,
    q: float,
    p: float,
    shift: float,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return paired incomplete-beta fraction coefficients and divided difference."""
    total = q + p
    shifted_q = q - shift
    if j == 0:
        return -total * z / (q + 1), -total * z / (shifted_q + 1), total * z / (q + 1) / (shifted_q + 1)
    m = (j + 1) // 2
    if j % 2:
        h, i = q + 2 * m - 1, q + 2 * m
        numerator = m * (p - m) * z
        shifted = m * (p + shift - m) * z
        delta = -m * z
    else:
        h, i = q + 2 * m, q + 2 * m + 1
        numerator = -(q + m) * (total + m) * z
        shifted = -(shifted_q + m) * (total + m) * z
        delta = -(total + m) * z
    c = numerator / h / i
    d = shifted / (h - shift) / (i - shift)
    difference = delta / h / i - d * (1 / h + (h - shift) / h / i)
    return c, d, difference


def _beta_prime_tail_excess(x: np.ndarray, log_power: np.ndarray, a: float, p: float, q: float) -> np.ndarray:
    """Cancel beta normalisers, endpoint powers and attachment analytically.

    The unregularized incomplete beta is the endpoint powers divided
    by the shape parameter and the continued-fraction denominator.
    The first moment shifts the two shape parameters in opposite
    directions; the paired recurrence avoids subtracting close fractions.
    """
    z = special.expit(-log_power)
    shift = 1 / a
    previous = None
    converged = np.zeros(x.shape, dtype=bool)
    for depth in (8, 16, 32, 64, 128, 256, 512, 1024):
        original = np.ones_like(x)
        shifted = np.ones_like(x)
        difference = np.zeros_like(x)
        for j in range(depth, -1, -1):
            c, d, delta = _beta_fraction_coefficients(j, q, p, shift, z)
            difference = (delta - (d / shifted) * difference) / original
            original = 1 + c / original
            shifted = 1 + d / shifted
        current = (original, shifted, difference)
        if previous is not None:
            converged = np.ones(x.shape, dtype=bool)
            for value, old in zip(current, previous, strict=True):
                converged &= np.abs(value - old) <= 4e-15 * np.maximum(np.abs(value), 1e-300)
            if np.all(converged):
                break
        previous = current
    relative = shift * (difference + original / (q - shift)) / shifted
    return np.where(converged & (relative > 0), x * relative, np.nan)


def _beta_prime_fraction_mask(log_power: np.ndarray, a: float, p: float, q: float) -> np.ndarray:
    """Select upper tails with a well-conditioned incomplete-beta fraction."""
    z = special.expit(-log_power)
    w = special.expit(log_power)
    rate = a * (q * w - p * z)
    curvature = a**2 * (p + q) * w * z
    return (rate > 2) & (curvature < 0.1 * (rate - 1) ** 2)


def _beta_prime_excess(x: np.ndarray, a: float, b: float, p: float, q: float) -> np.ndarray:
    """Evaluate the shared GB2 and inverse-Burr truncated first moment."""
    out = np.full(x.shape, np.nan, dtype=float)
    log_mean = np.log(b) + special.betaln(p + 1 / a, q - 1 / a) - special.betaln(p, q)
    positive = x > 0
    xp = x[positive]
    log_power = a * (np.log(xp) - np.log(b))
    eligible = _beta_prime_fraction_mask(log_power, a, p, q)
    use_fraction = eligible & (max(p, q) >= 100 or a * q >= 100)
    values = np.full_like(xp, np.nan)
    if np.any(use_fraction):
        values[use_fraction] = _beta_prime_tail_excess(xp[use_fraction], log_power[use_fraction], a, p, q)
    regular = ~use_fraction | ~np.isfinite(values)
    x_regular = xp[regular]
    log_power = log_power[regular]
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
    log_ratio[central] = log_mean + np.log(numerator) - np.log(denominator) - np.log(x_regular[central])
    values[regular] = np.where(log_ratio > 1e-8, _positive_log_excess(np.log(x_regular), log_ratio), np.nan)
    # A nonfinite ratio may be intermediate underflow rather than infinite mean.
    repair = ~np.isfinite(values) & eligible & regular
    if np.any(repair):
        lp = a * (np.log(xp[repair]) - np.log(b))
        values[repair] = _beta_prime_tail_excess(xp[repair], lp, a, p, q)
    values[~np.isfinite(values)] = np.nan
    out[positive] = values
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


def _inverse_gaussian_fraction(
    mu: float,
    x: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    gap: np.ndarray,
) -> np.ndarray:
    """Cancel the attachment in the ratio of normal Mills functions."""
    ca = np.empty_like(a)
    cb = np.empty_like(b)
    difference = np.empty_like(a)
    tail = a >= 4
    cb[tail], ca[tail], difference[tail] = _tail_fraction_pair(b[tail], a[tail])
    central = ~tail
    # Here a*b>20, so b is well into the positive tail and the gap
    # is large enough for direct subtraction of the Mills corrections.
    ac = a[central]
    bc = b[central]
    ca[central] = np.sqrt(2 / np.pi) / special.erfcx(ac / np.sqrt(2)) - ac
    cb[central] = _tail_fraction_pair(bc, bc)[0]
    difference[central] = (cb[central] - ca[central]) / gap[central]
    return (mu * ((ca + cb) / gap) - x * difference) / (1 + difference)


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
    values = np.where((lr > 1e-5) & (log_a - log_b > 1e-5), _positive_log_excess(np.log(x), lr), np.nan)
    # Express both normal tails as inverse Mills ratios. Cancelling
    # the attachment before evaluation leaves only small positive terms.
    rate = lam / (2 * mu**2) * ((x - mu) / x) * (1 + mu / x)
    far = (x > mu) & (rate * x > 10)
    if np.any(far):
        values[far] = _inverse_gaussian_fraction(mu, x[far], normal_a[far], normal_b[far], 2 * root[far])
    out[ds > 0] = values
    out[ds <= 0] = mu - ds[ds <= 0]
    return out


def _gev_small_tail(shape: float, y: np.ndarray) -> np.ndarray:
    """Evaluate the normalized survival integral as a convergent small-y series."""
    # Integral from 0 to y of (1-exp(-t))*t**(-shape-1), divided by
    # y**(1-shape). Cancel the factor y before forming the survival ratio.
    series = 1 / (1 - shape) + y * (
        -1 / (2 * (2 - shape)) + y * (1 / (6 * (3 - shape)) + y * (-1 / (24 * (4 - shape))))
    )
    sf_factor = np.ones_like(y)
    nonzero = y > 0
    sf_factor[nonzero] = -np.expm1(-y[nonzero]) / y[nonzero]
    return series / sf_factor


def _gev_excess(ds: np.ndarray, shape: float, loc: float, scale: float) -> np.ndarray:
    """Evaluate GEV residual life, including the Gumbel and endpoint limits."""
    xi = shape
    out = np.full(ds.shape, np.nan, dtype=float)
    z = (ds - loc) / scale
    if xi == 0:
        y = np.exp(-z)
        small = y < 1e-4
        out[small] = scale * _gev_small_tail(0, y[small])
        central = ~small & np.isfinite(y)
        yc = y[central]
        out[central] = scale * (np.euler_gamma + np.log(yc) + special.exp1(yc)) / (-np.expm1(-yc))
        out[np.isposinf(y)] = scale * np.euler_gamma + loc - ds[np.isposinf(y)]
        return out

    base = 1 + xi * z
    log_base = np.log1p(xi * z)
    if xi < 0:
        near_endpoint = base < 0.5
        base[near_endpoint] = (-xi / scale) * ((loc - scale / xi) - ds[near_endpoint])
        log_base[near_endpoint] = np.log(base[near_endpoint])
    inside = base > 0
    y = np.exp(-log_base / xi)
    small = inside & (y < 1e-4)
    out[small] = scale * base[small] * _gev_small_tail(xi, y[small])
    # The central formula subtracts close quantities for xi near zero.
    # Retain conditional integration for this narrow parameter region.
    if abs(xi) >= 1e-5:
        central = inside & ~small
        yc = y[central]
        log_ratio = (
            special.gammaln(1 - xi) + np.log(special.gammainc(1 - xi, yc)) - np.log(-np.expm1(-yc)) - log_base[central]
        )
        excess = scale * base[central] / xi * np.expm1(log_ratio)
        out[central] = np.where(abs(log_ratio) > 1e-7, excess, np.nan)
    if xi > 0:
        out[~inside] = loc + scale * np.expm1(special.gammaln(1 - xi)) / xi - ds[~inside]
    return out


def _gig_excess(ds: np.ndarray, p: float, chi: float, psi: float, loc: float) -> np.ndarray:
    """Evaluate the deep GIG tail using a scaled conditional density integral."""
    x = ds - loc
    if p == -0.5:
        return _inverse_gaussian_excess(x, np.sqrt(chi / psi), chi)
    out = np.full(ds.shape, np.nan)
    width = 2 / psi
    far = x > max(2 * np.sqrt(chi / psi), 10 * width * max(1, abs(p - 1)))
    if np.any(far):
        nodes, weights = _laguerre_rule(32)
        u = width * nodes
        ratio = u / x[far, None]
        correction = (p - 1) * np.log1p(ratio) + chi / (2 * x[far, None]) * ratio / (1 + ratio)
        density = weights * np.exp(correction)
        out[far] = np.sum(density * u, axis=1) / np.sum(density, axis=1)
    return out


def _loggamma_fraction_excess(alpha: float, beta: float, d: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Evaluate the divided difference of the upper-gamma fraction in a batch."""
    y = (1 - beta) * z
    _, correction, difference = _tail_fraction_pair(z, y, alpha)
    return d * (np.log(d) * (1 + difference) / (y + 1 - alpha + correction))


def _loggamma_tail_excess(alpha: float, beta: float, d: float) -> float:
    """Evaluate the remaining log-gamma moment ratios in log space."""
    z = np.log(d) / beta
    rate = 1 - beta
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
    out = np.where(ds <= 1, np.expm1(-alpha * np.log1p(-beta)) + (1 - ds), out)
    # The conditional exponential moment is D(z)/D((1-beta)*z),
    # where D is the upper-gamma fraction denominator. Its divided
    # difference avoids rounding away beta*z for small beta.
    tail = np.isnan(out) & ((1 - beta) * z > max(1.0, alpha + 1))
    if np.any(tail):
        out[tail] = _loggamma_fraction_excess(alpha, beta, ds[tail], z[tail])
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
        dist.GeneralizedInverseGaussian: _gig_excess,
        dist.LogGamma: _loggamma_excess,
    }
    formula = formulas.get(family)
    if formula is None:
        return np.full(ds.shape, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        out = formula(ds, **pars)
    return np.where(np.isfinite(out) & (out > 0), out, np.nan)


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
    _integrate_unresolved(family, pars, frozen, lower, upper, levels, values)
    result[valid] = values[inverse]
    return float(result) if result.ndim == 0 else result


def _log_lower_gamma(alpha: float, log_z: float) -> float:
    """Evaluate the inverse-gamma survival even below probability underflow."""
    if log_z > 709:
        return 0.0
    z = np.exp(log_z)
    probability = special.gammainc(alpha, z)
    if probability > 0:
        return float(np.log(probability))
    correction = special.hyp1f1(alpha, alpha + 1, -z)
    return float(alpha * log_z - special.gammaln(alpha + 1) + np.log(correction))


def _positive_log_survival(family: type[DistributionBase], pars: dict[str, float], log_x: float) -> float | None:
    """Tail probabilities in log-loss coordinates, relative to location."""
    p = pars
    if family is dist.Pareto:
        return min(0.0, -p["shape"] * (log_x - np.log(p["scale"])))
    if family in (dist.Burr, dist.LogLogistic, dist.Paralogistic):
        power = p["power"] if family is dist.Burr else p["shape"]
        shape = 1.0 if family is dist.LogLogistic else p["shape"]
        return float(-shape * np.logaddexp(0, power * (log_x - np.log(p["scale"]))))
    if family in (dist.InverseBurr, dist.InverseParalogistic):
        power = p["power"] if family is dist.InverseBurr else p["shape"]
        log_u = power * (np.log(p["scale"]) - log_x)
        if log_u < -36:
            log_hazard = np.log(p["shape"]) + log_u
            if log_hazard < -36:
                return float(log_hazard)
            return float(np.log(-np.expm1(-np.exp(log_hazard)))) if log_hazard < 709 else 0.0
        return float(np.log(-np.expm1(-p["shape"] * np.logaddexp(0, log_u))))
    if family is dist.InverseGamma:
        return _log_lower_gamma(p["alpha"], np.log(p["theta"]) - log_x)
    if family is dist.InverseWeibull:
        log_z = p["shape"] * (np.log(p["scale"]) - log_x)
        if log_z < -36:
            return float(log_z)
        if log_z > 709:
            return 0.0
        return float(np.log(-np.expm1(-np.exp(log_z))))
    if family is dist.GeneralizedBetaPrime:
        log_z = -np.logaddexp(0, p["a"] * (log_x - np.log(p["b"])))
        return float(_gb2_log_regularized_beta(p["q"], p["p"], np.asarray(log_z)))
    if family is dist.LogGamma:
        return float(_gamma_logsf(p["alpha"], max(0.0, log_x) / p["beta"]))
    if family is dist.LogNormal:
        return float(special.log_ndtr((p["mu"] - log_x) / p["sigma"]))
    if family is dist.Gamma:
        log_z = log_x - np.log(p["theta"])
        return float(_gamma_logsf(p["alpha"], np.exp(log_z))) if log_z < 709 else -np.inf
    if family is dist.Weibull:
        log_z = p["shape"] * (log_x - np.log(p["scale"]))
        return float(-np.exp(log_z)) if log_z < 709 else -np.inf
    if family is dist.Exponential:
        log_z = log_x - np.log(p["scale"])
        return float(-np.exp(log_z)) if log_z < 709 else -np.inf
    if family is dist.GPD and p["shape"] > 0:
        return float(-np.logaddexp(0, log_x + np.log(p["shape"]) - np.log(p["scale"])) / p["shape"])
    return None


def _log_survival(family: type[DistributionBase], pars: dict[str, float], frozen: t.Any, x: float) -> float:
    """Prefer cancellation-free family formulas to generic survival calculations."""
    shifted = x - pars.get("loc", 0.0)
    if shifted > 0:
        value = _positive_log_survival(family, pars, float(np.log(shifted)))
        if value is not None:
            return value
    if family in (dist.GeneralizedBetaPrime, dist.LogGamma):
        return 0.0
    return float(frozen.logsf(x))


def _integrate_log_interval(
    family: type[DistributionBase],
    pars: dict[str, float],
    frozen: t.Any,
    start: float,
    stop: float,
    width: float,
    log_start_sf: float,
) -> float:
    """Integrate conditional survival in log-increment coordinates.

    Scaling the increment handles concentrated distributions. Its logarithm
    converts polynomial tails to exponentially decaying integrands.
    """
    if stop == start:
        return 0.0
    location = pars.get("loc", 0.0)
    shifted = start - location
    log_width = float(np.log(width))
    log_start = float(np.log(shifted)) if shifted > 0 else -np.inf
    limit = float(np.logaddexp(0, np.log(stop - start) - log_width)) if np.isfinite(stop) else np.inf

    def integrand(u: float) -> float:
        if u == 0:
            return 1.0
        log_increment = log_width + u + np.log(-np.expm1(-u))
        if shifted >= 0:
            log_x = float(np.logaddexp(log_start, log_increment))
            log_sf = _positive_log_survival(family, pars, log_x)
        else:
            log_sf = None
        if log_sf is None:
            if log_increment > 709:
                return 0.0
            x = start + float(np.exp(log_increment))
            log_sf = _log_survival(family, pars, frozen, x)
        return float(np.exp(u + log_sf - log_start_sf))

    if limit == 0:
        return 0.0
    # Integrate the local interval first, then longer logarithmic gaps.
    # An infinite-interval transform is appropriate for heavy tails;
    # a huge finite interval can otherwise miss a narrow boundary layer.
    split = min(1.0, limit)
    area, error = integrate.quad(integrand, 0.0, split, epsabs=0, epsrel=1e-10, limit=300)
    if split < limit:
        rest, rest_error = integrate.quad(integrand, split, limit, epsabs=0, epsrel=1e-10, limit=300)
        area += rest
        error += rest_error
    if not np.isfinite(area) or area < 0 or not np.isfinite(error) or error > 1e-7 * area:
        raise ArithmeticError(f"Mean excess integration did not converge for {family.__name__} at {start}")
    return width * area


def _integrate_unresolved(
    family: type[DistributionBase],
    pars: dict[str, float],
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
    unresolved = np.flatnonzero(~np.isfinite(values))
    if unresolved.size == 0:
        return
    for index in range(min(levels.size - 1, int(unresolved[-1]) + 1), int(unresolved[0]) - 1, -1):
        d = float(levels[index])
        if np.isfinite(values[index]):
            # A stable analytical value is also an exact recursion anchor.
            if d >= lower:
                log_sf = _log_survival(family, pars, frozen, d)
                if np.isfinite(log_sf):
                    previous_start, previous_excess, previous_log_sf = d, float(values[index]), log_sf
            continue
        # Scale by the local inverse hazard. Integrating in units of tail
        # width prevents quadrature missing a narrow tail at a high threshold.
        start = max(float(d), float(lower))
        if previous_start == start:
            values[index] = start - d + previous_excess
            continue
        log_start_sf = _log_survival(family, pars, frozen, start)
        if not np.isfinite(log_start_sf):
            # Do not let one unresolved probability invalidate other attachments.
            values[index] = np.nan
            continue
        log_start_pdf = float(frozen.logpdf(start)) if frozen is not None else np.nan
        tail_probability = float(np.exp(log_start_sf))
        # The conditional median gives a representative tail width even if
        # the density at the attachment is vanishingly small.
        local_scale = np.nan
        if frozen is not None and tail_probability > 0:
            median = float(frozen.isf(tail_probability * 0.5))
            local_scale = median - start
        if not np.isfinite(local_scale) or local_scale <= 0:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                local_scale = float(np.exp(log_start_sf - log_start_pdf))
        if not np.isfinite(local_scale) or local_scale <= 0:
            local_scale = max(abs(start - pars.get("loc", 0.0)) * 0.1, 1.0)
        remaining = upper - start
        limit = remaining / local_scale if np.isfinite(remaining) else np.inf

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
            if not np.isfinite(area) or area < 0 or not np.isfinite(error) or error > 1e-7 * area:
                raise ArithmeticError(f"Mean excess integration did not converge for {family.__name__} at {d}")
        else:
            stop = previous_start if reuse and previous_start is not None else upper
            excess = _integrate_log_interval(family, pars, frozen, start, stop, local_scale, log_start_sf)
            if reuse and previous_start is not None:
                excess += np.exp(previous_log_sf - log_start_sf) * previous_excess

        values[index] = (start - d) + excess
        previous_start, previous_excess, previous_log_sf = start, float(excess), log_start_sf


def _repair_tail(ds: np.ndarray, out: np.ndarray, fallback: t.Callable[[float], float]) -> np.ndarray:
    """Evaluate only unresolved entries with a more robust scalar expression."""
    for i in np.flatnonzero(np.isnan(out) | (out <= 0)):
        try:
            out[i] = fallback(float(ds[i]))
        except ArithmeticError:
            # The generic conditional-survival fallback may still succeed.
            out[i] = np.nan
    return out

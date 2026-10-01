"""Numerically stable kernels for the FRA1 copula.

The implementation follows the elementary transforms in Hua (2026) and works
primarily on logarithmic scales so that the full parameter range remains usable
close to the asymptotic-dependence boundaries.
"""

from __future__ import annotations

import typing as t

from ._maths import xp as np

_LOG_TWO = np.log(2.0)
_LOG_SMALL = np.log(1e-5)


def _log1pexp(x: t.Any) -> t.Any:
    """Return log(1 + exp(x)) without overflow."""
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def _logexpm1(x: t.Any) -> t.Any:
    """Return log(exp(x) - 1) accurately for nonnegative x."""
    zero = x == 0.0
    safe_x = np.where(zero, 1.0, x)
    value = safe_x + np.log(-np.expm1(-safe_x))
    return np.where(zero, -np.inf, value)


def _log_cosh_minus_one(x: t.Any) -> t.Any:
    """Return log(cosh(x) - 1) accurately for nonnegative x."""
    zero = x == 0.0
    safe_x = np.where(zero, 1.0, x)
    regular = safe_x - _LOG_TWO + 2.0 * np.log(-np.expm1(-safe_x))
    small = safe_x < 1e-5
    series = 2.0 * np.log(safe_x) - _LOG_TWO + safe_x**2 / 12.0
    value = np.where(small, series, regular)
    return np.where(zero, -np.inf, value)


def _acosh_one_plus_exp(log_x: t.Any) -> t.Any:
    """Return acosh(1 + exp(log_x)) without overflowing exp(log_x)."""
    regular = log_x < 20.0
    safe_log_x = np.where(regular, log_x, 20.0)
    x = np.exp(safe_log_x)
    exact = np.log1p(x + np.sqrt(x) * np.sqrt(x + 2.0))
    return np.where(regular, exact, log_x + _LOG_TWO)


def _logsumexp_pair(x: t.Any, y: t.Any) -> t.Any:
    """Return log(exp(x) + exp(y)) elementwise."""
    maximum = np.maximum(x, y)
    return maximum + np.log(np.exp(x - maximum) + np.exp(y - maximum))


def _logdiffexp(x: t.Any, y: t.Any) -> t.Any:
    """Return log(exp(x) - exp(y)) for x >= y."""
    gap = -np.expm1(y - x)
    zero = gap == 0.0
    safe_gap = np.maximum(gap, np.finfo(np.float64).tiny)
    value = x + np.log(safe_gap)
    return np.where(zero, -np.inf, value)


def _log_f(log_s: t.Any, p: float) -> tuple[t.Any, t.Any, t.Any]:
    """Return log(F_p), log(F_p') and log(-F_p'')."""
    if p == 1.0:
        return log_s, np.zeros_like(log_s), np.full_like(log_s, -np.inf)

    a = _acosh_one_plus_exp(log_s)
    log_delta = 0.5 * (log_s + _LOG_TWO + _log1pexp(log_s - _LOG_TWO))

    if p < 1e-7:
        log_value = 2.0 * np.log(a) - _LOG_TWO
        log_first = np.log(a) - log_delta
        ratio = 1.0 + 2.0 / np.expm1(2.0 * a) - 1.0 / a
    else:
        pa = p * a
        log_value = _log_cosh_minus_one(pa) - 2.0 * np.log(p)
        log_first = pa - _LOG_TWO + np.log(-np.expm1(-2.0 * pa)) - np.log(p) - log_delta
        ratio = (1.0 - p) + 2.0 / np.expm1(2.0 * a) - 2.0 * p / np.expm1(2.0 * pa)

    log_second_magnitude = log_first + np.log(np.maximum(ratio, np.finfo(np.float64).tiny)) - log_delta

    small = log_s < _LOG_SMALL
    series_s = np.exp(np.minimum(log_s, _LOG_SMALL))
    one_minus_p2 = 1.0 - p**2
    value_series = log_s + np.log1p(-one_minus_p2 * series_s / 6.0 + one_minus_p2 * (4.0 - p**2) * series_s**2 / 90.0)
    first_series = np.log1p(-one_minus_p2 * series_s / 3.0 + one_minus_p2 * (4.0 - p**2) * series_s**2 / 30.0)
    second_series = np.log(one_minus_p2 / 3.0) + np.log1p(-(4.0 - p**2) * series_s / 5.0)

    return (
        np.where(small, value_series, log_value),
        np.where(small, first_series, log_first),
        np.where(small, second_series, log_second_magnitude),
    )


def _log_f_inv(log_x: t.Any, p: float) -> t.Any:
    """Return log(F_p^{-1}(x)) from log(x)."""
    if p < 1e-7:
        log_a = 0.5 * (log_x + _LOG_TWO)
        capped = np.minimum(log_a, np.log(np.finfo(np.float64).max))
        a = np.exp(capped)
        a = np.where(log_a > np.log(np.finfo(np.float64).max), np.inf, a)
    else:
        a = _acosh_one_plus_exp(log_x + 2.0 * np.log(p)) / p

    result = _log_cosh_minus_one(a)
    small = log_x < -20.0
    approximation = log_x + np.log1p((1.0 - p**2) * np.exp(log_x) / 6.0)
    return np.where(small, approximation, result)


def _log_j(log_t: t.Any, p: float) -> tuple[t.Any, t.Any]:
    """Return log(J_p(t)) and log(J_p'(t))."""
    if p == 1.0:
        return log_t, np.zeros_like(log_t)

    log_x = -log_t
    log_d = _log_f_inv(np.zeros_like(log_t), p)
    log_fx, log_fpx, _ = _log_f(log_x, p)
    log_transformed = _log_f_inv(_log1pexp(log_fx), p)
    _, log_fp_transformed, _ = _log_f(log_transformed, p)

    gap = -np.expm1(log_d - log_transformed)
    log_s = log_transformed + np.log(np.maximum(gap, np.finfo(np.float64).tiny))
    log_sp = log_fpx - log_fp_transformed

    small = log_x < _LOG_SMALL
    _, log_fp_d, log_neg_fpp_d = _log_f(log_d, p)
    first_at_zero = np.exp(-log_fp_d)
    fpp_at_zero = -(1.0 - p**2) / 3.0
    fp_d = np.exp(log_fp_d)
    fpp_d = -np.exp(log_neg_fpp_d)
    second_at_zero = fpp_at_zero / fp_d - fpp_d / fp_d**3

    x_for_series = np.where(small, np.exp(log_x), 0.0)
    s_ratio = second_at_zero / first_at_zero
    log_s_series = log_x + np.log(first_at_zero) + np.log1p(0.5 * s_ratio * x_for_series)
    log_sp_series = np.log(first_at_zero) + np.log1p(s_ratio * x_for_series)
    log_s = np.where(small, log_s_series, log_s)
    log_sp = np.where(small, log_sp_series, log_sp)

    log_value = -log_s
    log_first = log_sp - 2.0 * log_t - 2.0 * log_s
    return log_value, log_first


def _log_j_inv(log_y: t.Any, p: float) -> t.Any:
    """Return log(J_p^{-1}(y)) from log(y)."""
    if p == 1.0:
        return log_y

    log_d = _log_f_inv(np.zeros_like(log_y), p)
    log_z = -log_y
    log_argument = _logsumexp_pair(log_d, log_z)
    log_f_value, _, _ = _log_f(log_argument, p)
    log_delta = _logexpm1(np.maximum(log_f_value, 0.0))

    small = log_z < _LOG_SMALL
    _, log_fp_d, log_neg_fpp_d = _log_f(log_d, p)
    correction = 0.5 * np.exp(log_neg_fpp_d - log_fp_d + log_z)
    correction = np.minimum(correction, 1.0 - np.finfo(np.float64).eps)
    approximation = log_fp_d + log_z + np.log1p(-correction)
    log_delta = np.where(small, approximation, log_delta)
    return -_log_f_inv(log_delta, p)


def _log_k(log_t: t.Any, a: float) -> tuple[t.Any, t.Any]:
    """Return log(K_a(t)) and log(K_a'(t))."""
    h = _log1pexp(a * log_t)
    log_value = _logexpm1(h / a)
    tiny = a * log_t < -700.0
    log_value = np.where(tiny, a * log_t - np.log(a), log_value)
    log_first = (a - 1.0) * log_t + (1.0 / a - 1.0) * h
    return log_value, log_first


def _log_k_inv(log_y: t.Any, a: float) -> t.Any:
    """Return log(K_a^{-1}(y)) from log(y)."""
    h = _log1pexp(log_y)
    result = _logexpm1(a * h) / a
    tiny = log_y < -700.0
    approximation = (np.log(a) + log_y) / a
    return np.where(tiny, approximation, result)


def _log_w(log_t: t.Any, theta: float) -> tuple[t.Any, t.Any]:
    """Return log(W_theta(t)) and log(W_theta'(t))."""
    if theta <= 0.0:
        return _log_j(log_t, -theta)

    log_k, log_kp = _log_k(log_t, 1.0 - theta)
    log_j, log_jp = _log_j(log_k, 0.0)
    return log_j, log_jp + log_kp


def _log_w_inv(log_y: t.Any, theta: float) -> t.Any:
    """Return log(W_theta^{-1}(y)) from log(y)."""
    if theta <= 0.0:
        return _log_j_inv(log_y, -theta)

    log_k = _log_j_inv(log_y, 0.0)
    return _log_k_inv(log_k, 1.0 - theta)


def _b(eta: float) -> float:
    """Return b_eta for positive eta."""
    return (1.0 - eta) / eta**2


def log_inverse_generator(u: t.Any, eta: float, theta: float) -> t.Any:
    """Return log(phi(u)), where phi=psi^{-1}, for interior u."""
    log_target = np.log(-np.log(u))

    if eta > 0.0:
        b = _b(eta)
        log_target = np.log(b) + _logexpm1(-np.log(u) / b)

    log_g_inv = _log_f_inv(log_target, abs(eta))
    return _log_w_inv(log_g_inv, theta)


def _log_generator_and_negative_prime(log_t: t.Any, eta: float, theta: float) -> tuple[t.Any, t.Any]:
    """Return log(psi(t)) and log(-psi'(t)) from log(t)."""
    log_w, log_wp = _log_w(log_t, theta)
    log_f_value, log_fp, _ = _log_f(log_w, abs(eta))

    if eta <= 0.0:
        log_generator = -np.exp(log_f_value)
        log_negative_g_prime = log_generator + log_fp
    else:
        b = _b(eta)
        log_q = _log1pexp(log_f_value - np.log(b))
        log_generator = -b * log_q
        log_negative_g_prime = log_fp - (b + 1.0) * log_q

    return log_generator, log_negative_g_prime + log_wp


def _log_negative_generator_prime(log_t: t.Any, eta: float, theta: float) -> t.Any:
    """Return log(-psi'(t)) from log(t)."""
    return _log_generator_and_negative_prime(log_t, eta, theta)[1]


_STEHFEST_COEFFICIENTS = np.asarray(
    [
        4.96031746031746e-05,
        -0.6095734126984128,
        274.5940476190476,
        -26306.956746031745,
        957257.2013888889,
        -17358694.84583333,
        182421222.64722222,
        -1218533288.3091269,
        5491680025.283035,
        -17362131115.206844,
        39455096903.52738,
        -65266516985.175,
        78730068328.22083,
        -68556444196.120834,
        41984343475.05357,
        -17160934711.839287,
        4204550039.102679,
        -467172226.56696427,
    ],
    dtype=np.float64,
)
_STEHFEST_INDICES = np.arange(1, 19, dtype=np.float64)
_STEHFEST_LOG_ARGUMENTS = np.log(_STEHFEST_INDICES * _LOG_TWO)


def generator_from_log_argument(log_t: t.Any, eta: float, theta: float) -> t.Any:
    """Evaluate the FRA1 generator from log(t)."""
    log_t = np.asarray(log_t)
    at_zero = np.isneginf(log_t)
    safe_log_t = np.where(at_zero, 0.0, log_t)
    log_generator = _log_generator_and_negative_prime(safe_log_t, eta, theta)[0]
    return np.where(at_zero, 1.0, np.exp(log_generator))


def frailty_cdf_from_log(log_v: t.Any, eta: float, theta: float) -> t.Any:
    """Numerically invert the frailty Laplace transform at log(v).

    The FRA1 generator is the Laplace transform of a positive frailty variable.
    This evaluates its CDF using an 18-term Gaver-Stehfest inversion of
    psi(s) / s. Working in log coordinates keeps the inversion stable over
    the very wide frailty ranges that occur near the parameter boundaries.
    """
    log_v = np.atleast_1d(np.asarray(log_v, dtype=np.float64))
    log_s = _STEHFEST_LOG_ARGUMENTS[:, np.newaxis] - log_v[np.newaxis, :]
    psi = generator_from_log_argument(log_s, eta, theta)
    weights = _STEHFEST_COEFFICIENTS / _STEHFEST_INDICES
    return np.sum(weights[:, np.newaxis] * psi, axis=0)


def frailty_log_quantile_table(
    eta: float,
    theta: float,
    grid_size: int = 8193,
) -> tuple[t.Any, t.Any]:
    """Build a numerical quantile table for the FRA1 frailty distribution."""
    if grid_size < 257:
        raise ValueError("grid_size must be at least 257")

    # A sinh grid provides fine resolution around the main body while retaining
    # an enormous dynamic range for the heavy frailty tails as theta approaches 1.
    z = np.linspace(-1.0, 1.0, grid_size)
    stretch = 4.5
    log_v = 650.0 * np.sinh(stretch * z) / np.sinh(stretch)
    cdf = frailty_cdf_from_log(log_v, eta, theta)
    cdf = np.sort(np.clip(cdf, 0.0, 1.0))
    cdf[0] = 0.0
    cdf[-1] = 1.0

    # Numerical Laplace inversion can introduce tiny local violations of
    # monotonicity. Sorting performs a monotone rearrangement and is supported
    # efficiently by both NumPy and CuPy. Remove flat sections at machine
    # precision so interpolation always sees a strictly increasing CDF.
    keep = np.concatenate((np.asarray([True]), np.diff(cdf) > 1e-10))
    indices = np.flatnonzero(keep)
    return np.take(cdf, indices), np.take(log_v, indices)


def sample_log_frailty(
    n_sims: int,
    rng: t.Any,
    eta: float,
    theta: float,
) -> t.Any:
    """Sample log frailties for the multivariate Marshall-Olkin construction."""
    probabilities = rng.uniform(size=n_sims)
    cdf, log_v = frailty_log_quantile_table(eta, theta)
    return np.interp(probabilities, cdf, log_v)


def generate_multivariate(
    dimension: int,
    n_sims: int,
    rng: t.Any,
    eta: float,
    theta: float,
) -> t.Any:
    """Generate a d-variate FRA1 sample using its frailty representation."""
    if dimension < 2:
        raise ValueError("dimension must be at least 2")

    if eta == -1.0 and theta == -1.0:
        return rng.uniform(size=(dimension, n_sims))

    log_frailty = sample_log_frailty(n_sims, rng, eta, theta)
    exponentials = np.maximum(
        rng.exponential(size=(dimension, n_sims)),
        np.finfo(np.float64).tiny,
    )
    log_arguments = np.log(exponentials) - log_frailty[np.newaxis, :]
    return generator_from_log_argument(log_arguments, eta, theta)


def conditional_ppf(q: t.Any, v: t.Any, eta: float, theta: float) -> t.Any:
    """Invert the FRA1 conditional CDF for bivariate sampling."""
    log_y = log_inverse_generator(v, eta, theta)
    target = np.log(q) + _log_negative_generator_prime(log_y, eta, theta)

    lower = log_y
    upper = log_y + 1.0

    # The inversion is monotone. Fixed-count vector operations avoid repeated
    # host/device synchronisation when PAL is using CuPy.
    for _ in range(64):
        needs_expansion = _log_negative_generator_prime(upper, eta, theta) > target
        upper = np.where(needs_expansion, upper + 4.0, upper)

    for _ in range(64):
        midpoint = 0.5 * (lower + upper)
        needs_larger = _log_negative_generator_prime(midpoint, eta, theta) > target
        lower = np.where(needs_larger, midpoint, lower)
        upper = np.where(needs_larger, upper, midpoint)

    log_total = 0.5 * (lower + upper)
    log_argument = _logdiffexp(log_total, log_y)
    at_zero = ~np.isfinite(log_argument)
    safe_argument = np.where(at_zero, 0.0, log_argument)
    log_generator = _log_generator_and_negative_prime(safe_argument, eta, theta)[0]
    result = np.exp(log_generator)
    return np.where(at_zero, 1.0, result)

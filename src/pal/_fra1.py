"""Numerical kernels for the FRA1 copula.

The formulas implement the full-range Archimedean type I construction from
Hua (2026).  They are kept separate from :mod:`pal.copulas` so the public
copula module stays focused on modelling classes.
"""

from __future__ import annotations

import typing as t

from ._maths import xp as np


def _acosh1p(x: t.Any) -> t.Any:
    """Evaluate acosh(1 + x) accurately for nonnegative x."""
    return np.log1p(x + np.sqrt(x) * np.sqrt(x + 2.0))


def _coshm1(x: t.Any) -> t.Any:
    """Evaluate cosh(x) - 1 accurately near zero."""
    return 2.0 * np.sinh(x / 2.0) ** 2


def _f(s: t.Any, p: float) -> t.Any:
    """Evaluate Hua's F_p transform."""
    a = _acosh1p(s)
    if p == 0.0:
        return 0.5 * a**2
    return _coshm1(p * a) / p**2


def _f_inv(y: t.Any, p: float) -> t.Any:
    """Evaluate the closed-form inverse of F_p."""
    if p == 0.0:
        return _coshm1(np.sqrt(2.0 * y))
    return _coshm1(_acosh1p(p**2 * y) / p)


def _f_prime(s: t.Any, p: float) -> t.Any:
    """Evaluate the first derivative of F_p with a series near zero."""
    small = s < 1e-6
    safe_s = np.where(small, 1.0, s)
    a = _acosh1p(safe_s)
    delta = np.sqrt(safe_s) * np.sqrt(safe_s + 2.0)

    if p == 0.0:
        regular = a / delta
    else:
        regular = np.sinh(p * a) / (p * delta)

    series = (
        1.0
        - (1.0 - p**2) * s / 3.0
        + (1.0 - p**2) * (4.0 - p**2) * s**2 / 30.0
    )
    return np.where(small, series, regular)


def _f_second(s: t.Any, p: float) -> t.Any:
    """Evaluate the second derivative of F_p with a series near zero."""
    small = s < 1e-5
    safe_s = np.where(small, 1.0, s)
    a = _acosh1p(safe_s)
    delta = np.sqrt(safe_s) * np.sqrt(safe_s + 2.0)

    if p == 0.0:
        regular = (delta - (1.0 + safe_s) * a) / delta**3
    else:
        regular = (
            p * np.cosh(p * a) * delta - (1.0 + safe_s) * np.sinh(p * a)
        ) / (p * delta**3)

    series = (
        -(1.0 - p**2) / 3.0
        + (1.0 - p**2) * (4.0 - p**2) * s / 15.0
    )
    return np.where(small, series, regular)


def _s_bundle(x: t.Any, p: float) -> tuple[t.Any, t.Any, t.Any]:
    """Return S_p, S_p' and S_p''."""
    d = _f_inv(np.asarray(1.0), p)
    transformed = _f_inv(_f(x, p) + 1.0, p)
    s = transformed - d

    fp_d = _f_prime(d, p)
    fpp_d = _f_second(d, p)
    first_at_zero = 1.0 / fp_d
    second_at_zero = _f_second(np.asarray(0.0), p) / fp_d - fpp_d / fp_d**3

    small = x < 1e-5 * (1.0 + d)
    s_series = first_at_zero * x + 0.5 * second_at_zero * x**2
    s = np.where(small, s_series, s)
    transformed = np.where(small, d + s, transformed)

    fp_x = _f_prime(x, p)
    fp_t = _f_prime(transformed, p)
    fpp_x = _f_second(x, p)
    fpp_t = _f_second(transformed, p)

    first = fp_x / fp_t
    second = fpp_x / fp_t - fp_x**2 * fpp_t / fp_t**3
    return s, first, second


def _j_bundle(t: t.Any, p: float) -> tuple[t.Any, t.Any, t.Any]:
    """Return J_p, J_p' and J_p'' for positive t."""
    if p == 1.0:
        return t, np.ones_like(t), np.zeros_like(t)

    x = 1.0 / t
    s, sp, spp = _s_bundle(x, p)
    value = 1.0 / s
    first = sp / (t**2 * s**2)
    second = (
        -2.0 * sp / (t**3 * s**2)
        - spp / (t**4 * s**2)
        + 2.0 * sp**2 / (t**4 * s**3)
    )
    return value, first, second


def _j_inv(y: t.Any, p: float) -> t.Any:
    """Evaluate the closed-form inverse of J_p."""
    if p == 1.0:
        return y
    d = _f_inv(np.asarray(1.0), p)
    return 1.0 / _f_inv(_f(d + 1.0 / y, p) - 1.0, p)


def _k_bundle(t: t.Any, a: float) -> tuple[t.Any, t.Any, t.Any]:
    """Return K_a, K_a' and K_a''."""
    ta = t**a
    value = np.expm1(np.log1p(ta) / a)
    first = t ** (a - 1.0) * (1.0 + ta) ** (1.0 / a - 1.0)
    second = (a - 1.0) * t ** (a - 2.0) * (1.0 + ta) ** (1.0 / a - 2.0)
    return value, first, second


def _k_inv(y: t.Any, a: float) -> t.Any:
    """Evaluate the closed-form inverse of K_a."""
    return np.exp(np.log(np.expm1(a * np.log1p(y))) / a)


def _w_bundle(t: t.Any, theta: float) -> tuple[t.Any, t.Any, t.Any]:
    """Return W_theta, W_theta' and W_theta''."""
    if theta <= 0.0:
        return _j_bundle(t, -theta)

    a = 1.0 - theta
    k, kp, kpp = _k_bundle(t, a)
    j, jp, jpp = _j_bundle(k, 0.0)
    return j, jp * kp, jpp * kp**2 + jp * kpp


def _w_inv(y: t.Any, theta: float) -> t.Any:
    """Evaluate the inverse of W_theta."""
    if theta <= 0.0:
        return _j_inv(y, -theta)
    return _k_inv(_j_inv(y, 0.0), 1.0 - theta)


def _b(eta: float) -> float:
    """Return b_eta for positive eta."""
    return (1.0 - eta) / eta**2


def inverse_generator(u: t.Any, eta: float, theta: float) -> t.Any:
    """Evaluate phi(u) = psi^{-1}(u) for interior probabilities."""
    if eta <= 0.0:
        g_inv = _f_inv(-np.log(u), -eta)
    else:
        b = _b(eta)
        g_inv = _f_inv(b * np.expm1(-np.log(u) / b), eta)
    return _w_inv(g_inv, theta)


def generator(t: t.Any, eta: float, theta: float) -> t.Any:
    """Evaluate the FRA1 Archimedean generator psi(t)."""
    t = np.asarray(t)
    at_zero = t == 0.0
    safe_t = np.where(at_zero, 1.0, t)
    w, _, _ = _w_bundle(safe_t, theta)
    f = _f(w, abs(eta))

    if eta <= 0.0:
        value = np.exp(-f)
    else:
        b = _b(eta)
        value = np.exp(-b * np.log1p(f / b))

    return np.where(at_zero, 1.0, value)


def log_negative_generator_prime(t: t.Any, eta: float, theta: float) -> t.Any:
    """Evaluate log(-psi'(t)) for positive t."""
    w, wp, _ = _w_bundle(t, theta)
    f = _f(w, abs(eta))
    fp = _f_prime(w, abs(eta))

    if eta <= 0.0:
        log_negative_g_prime = np.log(fp) - f
    else:
        b = _b(eta)
        log_q = np.log1p(f / b)
        log_negative_g_prime = np.log(fp) - (b + 1.0) * log_q

    return log_negative_g_prime + np.log(wp)


def conditional_ppf(q: t.Any, v: t.Any, eta: float, theta: float) -> t.Any:
    """Invert P(U <= u | V=v) using monotonicity of -psi'."""
    y = inverse_generator(v, eta, theta)
    target = np.log(q) + log_negative_generator_prime(y, eta, theta)

    lower = y
    upper = np.maximum(1.0, 2.0 * y + 1.0)

    # Fixed-count vectorised expansion keeps the routine compatible with both
    # NumPy and CuPy without a host synchronisation in each iteration.
    for _ in range(64):
        needs_expansion = log_negative_generator_prime(upper, eta, theta) > target
        upper = np.where(needs_expansion, 2.0 * upper + 1.0, upper)

    for _ in range(64):
        midpoint = 0.5 * (lower + upper)
        needs_larger = log_negative_generator_prime(midpoint, eta, theta) > target
        lower = np.where(needs_larger, midpoint, lower)
        upper = np.where(needs_larger, upper, midpoint)

    total_argument = 0.5 * (lower + upper)
    return generator(total_argument - y, eta, theta)

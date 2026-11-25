"""Batched adaptive integration of non-negative interval integrands."""

from __future__ import annotations

import typing as t
from functools import lru_cache

import numpy as np
from scipy import integrate, special


@lru_cache(maxsize=2)
def _rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Cache Legendre quadrature nodes and logarithmic weights.

    Args:
        order: Number of quadrature nodes.
    """
    nodes, weights = special.roots_legendre(order)
    return nodes, np.log(weights)


def _estimate(
    log_integrand: t.Callable[[np.ndarray], np.ndarray], a: np.ndarray, b: np.ndarray, order: int
) -> np.ndarray:
    """Estimate interval integrals using a specified quadrature order.

    Args:
        log_integrand: Vectorised logarithm of the nonnegative integrand.
        a: Lower endpoints of log-coordinate intervals.
        b: Upper endpoints of log-coordinate intervals.
        order: Number of Legendre quadrature nodes.
    """
    nodes, log_weights = _rule(order)
    half = (b - a) / 2
    points = a[:, None] + half[:, None] * (1 + nodes)
    logs = log_integrand(points)
    return np.exp(np.log(half) + special.logsumexp(logs + log_weights, axis=1))


def _adaptive_intervals(
    log_integrand: t.Callable[[np.ndarray], np.ndarray],
    starts: np.ndarray,
    stops: np.ndarray,
) -> np.ndarray:
    """Integrate intervals in bounded batches, refining only unconverged pieces.

    Args:
        log_integrand: Vectorised function returning the log of a nonnegative integrand.
        starts: Lower endpoints of the log-coordinate intervals.
        stops: Upper endpoints of the intervals.
    """
    output = np.zeros_like(starts)
    # Long gaps are split before quadrature so that an interior transition
    # cannot be overlooked by both low- and high-order rules.
    counts = np.maximum(1, np.ceil((stops - starts) / 2).astype(int))
    ends = np.cumsum(counts)
    owner = np.repeat(np.arange(starts.size), counts)
    step = (stops - starts) / counts
    offset = np.arange(owner.size) - np.repeat(ends - counts, counts)
    left = starts[owner] + offset * step[owner]
    right = np.where(offset == counts[owner] - 1, stops[owner], left + step[owner])
    for start in range(0, owner.size, 1024):
        selection = slice(start, start + 1024)
        a, b, indices = left[selection], right[selection], owner[selection]
        active = b > a
        a, b, indices = a[active], b[active], indices[active]
        for _ in range(20):
            if not a.size:
                break
            coarse = _estimate(log_integrand, a, b, 8)
            fine = _estimate(log_integrand, a, b, 16)
            good = np.isfinite(fine) & (np.abs(coarse - fine) <= 2e-11 * fine + 2e-323)
            np.add.at(output, indices[good], fine[good])
            a, b, indices = a[~good], b[~good], indices[~good]
            mid = a + (b - a) / 2
            # If floating-point endpoints cannot be separated, the better
            # quadrature estimate is the last available resolution.
            stuck = (mid == a) | (mid == b)
            if np.any(stuck):
                value = _estimate(log_integrand, a[stuck], b[stuck], 16)
                np.add.at(output, indices[stuck], value)
            a, b, mid, indices = a[~stuck], b[~stuck], mid[~stuck], indices[~stuck]
            a, b, indices = np.r_[a, mid], np.r_[mid, b], np.tile(indices, 2)
            if a.size > 8192:
                break
        # Adaptive scalar integration is only a last resort on remaining
        # small intervals, not a separate tail integral for every limit.
        for lo, hi, index in zip(a, b, indices, strict=True):
            value, error = integrate.quad(
                lambda u: float(np.exp(log_integrand(np.asarray(u)))),
                lo,
                hi,
                epsabs=0,
                epsrel=1e-10,
                limit=300,
            )
            if not np.isfinite(value) or not np.isfinite(error) or error > 1e-7 * value:
                raise ArithmeticError("Limited expected value interval integration did not converge")
            output[index] += value
    return output


def _positive_survival_integrals(
    log_survival: t.Callable[[np.ndarray], np.ndarray],
    log_limits: np.ndarray,
    log_scale: float,
) -> np.ndarray:
    """Integrate survival from zero to sorted positive limits in one sweep.

    Args:
        log_survival: Vectorised log-survival function evaluated in log-loss coordinates.
        log_limits: Sorted logarithms of positive policy limits.
        log_scale: Logarithm of the characteristic loss scale.
    """
    if not log_limits.size:
        return np.empty_like(log_limits)

    def log_integrand(t_: np.ndarray) -> np.ndarray:
        """Evaluate the log integrand including the coordinate Jacobian.

        Args:
            t_: Log-loss or transformed integration coordinate.
        """
        return t_ + log_survival(t_)

    lower = max(-800.0, min(float(log_limits[0]), log_scale) - 40)
    # Split exactly at the characteristic scale if it falls inside the grid.
    grid = np.unique(np.r_[log_limits, log_scale]) if lower < log_scale < log_limits[-1] else log_limits
    starts = np.r_[lower, grid[:-1]]
    values = _adaptive_intervals(log_integrand, starts, grid)
    # The omitted interval has mass at most exp(lower), independently of
    # the severity family. Extend only when this is material to the anchor.
    for _ in range(40):
        if lower <= np.log(max(float(values[0]), np.nextafter(0.0, 1.0))) + np.log(1e-12):
            break
        new_lower = lower - 40
        values[0] += _adaptive_intervals(log_integrand, np.array([new_lower]), np.array([lower]))[0]
        lower = new_lower
    else:
        raise ArithmeticError("Unable to bound the lower limited expected value integral")
    return np.cumsum(values)[np.searchsorted(grid, log_limits)]

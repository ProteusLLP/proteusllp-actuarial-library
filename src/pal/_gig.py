"""Piecewise probability integration for a fixed generalized inverse Gaussian."""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate, special


def _weighted_remainder(coefficient: float, delta: float) -> float:
    """Evaluate coefficient times exp(delta) minus its linear Taylor polynomial.

    Args:
        coefficient: Nonnegative coefficient multiplying the exponential remainder.
        delta: Displacement from the expansion point.
    """
    if coefficient == 0:
        return 0.0
    if abs(delta) < 1e-3:
        return coefficient * delta**2 * (0.5 + delta * (1 / 6 + delta * (1 / 24 + delta * (1 / 120 + delta / 720))))
    if delta > 50:
        exponent = math.log(coefficient) + delta
        return math.exp(exponent) if exponent < 709 else math.inf
    return coefficient * (math.expm1(delta) - delta)


class _LogDensity:
    """Unnormalised log-coordinate density, scaled to one at its mode."""

    def __init__(self, p: float, b: float) -> None:
        """Initialize the numerical evaluator.

        Args:
            p: GIG index controlling the power of loss in the density.
            b: Positive standardized GIG shape.
        """
        ratio = p / b
        self.mode = (
            math.asinh(ratio)
            if math.isfinite(ratio)
            else math.copysign(math.log(2) + math.log(abs(p)) - math.log(b), p)
        )
        self.log_half = math.log(b) - math.log(2)
        self.a0 = math.exp(self.log_half + self.mode)
        self.b0 = math.exp(self.log_half - self.mode)

    def log_mass(self, near: float, far: float, direction: float) -> float:
        """Integrate one interval in the direction away from the log-density mode.

        Args:
            near: Interval endpoint nearer the mode of the log-density.
            far: Interval endpoint farther from the mode of the log-density.
            direction: Direction of integration away from the mode, either -1 or +1.
        """
        if near == far:
            return -math.inf
        log_a = self.log_half + near
        log_b = self.log_half - near
        if max(log_a, log_b) > 709:
            return -math.inf
        a = math.exp(log_a)
        b = math.exp(log_b)
        v = near - self.mode
        height = -_weighted_remainder(self.a0, v) - _weighted_remainder(self.b0, -v)
        if abs(v) < 0.5:
            slope = abs(-self.a0 * math.expm1(v) + self.b0 * math.expm1(-v))
        else:
            slope = abs((self.a0 - self.b0) - (a - b))
        # A local scale based on the slope and curvature prevents adaptive
        # integration from overlooking a narrow tail or concentrated mode.
        width = 1 / max(1.0, slope + math.sqrt(a + b))
        length = abs(far - near) / width

        def log_density(u: float) -> float:
            """Evaluate the local log density at the integration coordinate.

            Args:
                u: Probability for an inverse CDF or dimensionless integration coordinate.
            """
            delta = direction * width * u
            return -slope * width * u - _weighted_remainder(a, delta) - _weighted_remainder(b, -delta)

        # Log density is concave and decreasing in this direction. After
        # a drop of 45 the omitted tail is bounded by the secant slope.
        end = min(length, 1.0)
        while end < length and log_density(end) > -45:
            end = min(length, 2 * end)
        area, error = integrate.quad(lambda u: math.exp(log_density(u)), 0, end, epsabs=0, epsrel=2e-12)
        if area <= 0 or not math.isfinite(area) or error > 1e-9 * area:
            raise ArithmeticError("GIG interval integration did not converge")
        return height + math.log(width) + math.log(area)


def _gig_log_probabilities(x: np.ndarray, p: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """Return log CDF and log survival for standardised losses.

    Sort and deduplicate positive thresholds, and accumulate interval masses
    towards the mode from either tail. Normalise using the sum of both half
    integrals; this avoids extreme Bessel normalisers. Small probabilities
    are always accumulated directly rather than by subtracting from one.

    Args:
        x: Standardized positive loss thresholds.
        p: GIG index controlling the power of loss in the density.
        b: Positive standardized GIG shape.
    """
    values = np.asarray(x, dtype=float)
    logcdf = np.full(values.shape, np.nan)
    logsf = np.full(values.shape, np.nan)
    logcdf[values <= 0] = -np.inf
    logsf[values <= 0] = 0
    logcdf[np.isposinf(values)] = 0
    logsf[np.isposinf(values)] = -np.inf
    valid = np.isfinite(values) & (values > 0)
    if not np.any(valid):
        return logcdf, logsf
    if not math.isfinite(p) or not math.isfinite(b) or b <= 0:
        raise ValueError("GIG shape parameters must be finite and b must be positive")
    ts, inverse = np.unique(np.log(values[valid]), return_inverse=True)
    density = _LogDensity(p, b)
    mode = density.mode
    split = int(np.searchsorted(ts, mode, side="right"))
    lower = np.r_[ts[:split], mode]
    upper = np.r_[mode, ts[split:]]
    left = np.empty_like(lower)
    right = np.empty_like(upper)
    previous, area = -math.inf, -math.inf
    for i, t in enumerate(lower):
        area = float(np.logaddexp(area, density.log_mass(float(t), previous, -1.0)))
        left[i] = area
        previous = float(t)
    previous, area = math.inf, -math.inf
    for i in range(upper.size - 1, -1, -1):
        t = float(upper[i])
        area = float(np.logaddexp(area, density.log_mass(t, previous, 1.0)))
        right[i] = area
        previous = t
    total = np.logaddexp(left[-1], right[0])
    c = np.empty_like(ts)
    s = np.empty_like(ts)
    c[:split] = left[:-1] - total
    s[split:] = right[1:] - total
    # The complementary side is evaluated with expm1, not 1-exp(log_p).
    with np.errstate(divide="ignore", invalid="ignore"):
        s[:split] = np.log(-np.expm1(np.minimum(c[:split], 0)))
        c[split:] = np.log(-np.expm1(np.minimum(s[split:], 0)))
    logcdf[valid] = c[inverse]
    logsf[valid] = s[inverse]
    return logcdf, logsf


def _gig_log_moments(
    x: np.ndarray, p: float, chi: float, psi: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return lower/upper probability and lower/upper first moment in log units.

    Args:
        x: Positive losses in original units.
        p: GIG index controlling the power of loss in the density.
        chi: Positive inverse-loss coefficient of the GIG density.
        psi: Positive loss coefficient of the GIG density.
    """
    b = np.sqrt(chi) * np.sqrt(psi)
    scale = np.sqrt(chi) / np.sqrt(psi)
    lower, upper = _gig_log_probabilities(x / scale, p, b)
    moment_lower, moment_upper = _gig_log_probabilities(x / scale, p + 1, b)
    log_mean = np.log(scale) + np.log(special.kve(p + 1, b)) - np.log(special.kve(p, b))
    return lower, upper, log_mean + moment_lower, log_mean + moment_upper

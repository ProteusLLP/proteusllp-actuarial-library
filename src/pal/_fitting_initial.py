"""Statistical starting estimates for CPU maximum-likelihood fitting."""

from __future__ import annotations

import numpy as np


def _conditional_quantiles(
    x: np.ndarray, censored: np.ndarray, weights: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    """Estimate conditional quantiles using a weighted product-limit estimator.

    Args:
        x: Recorded observations, including censoring limits.
        censored: Indicators of right-censoring.
        weights: Positive weights aligned with the observations.
        probabilities: Conditional cumulative probabilities to estimate.

    Returns:
        Interpolated quantiles; NaN where censoring prevents identification.

    Notes:
        The reporting threshold is already conditioned on by the input sample.
        Interpolation at the midpoints of the jumps reduces discretisation error.
    """
    values, inverse = np.unique(x, return_inverse=True)
    counts = np.bincount(inverse, weights=weights, minlength=values.size)
    events = np.bincount(inverse, weights=weights * ~censored, minlength=values.size)
    risk = np.cumsum(counts[::-1])[::-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_survival = np.cumsum(np.log1p(-np.minimum(events / risk, 1.0)))
    cdf = -np.expm1(log_survival)
    observed = events > 0
    midpoints = (np.r_[0.0, cdf[:-1]] + cdf) / 2
    result = np.interp(probabilities, midpoints[observed], values[observed])
    return np.where(probabilities <= cdf[-1], result, np.nan)


def _log_location_scale(x: np.ndarray, censored: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Estimate log-loss location and dispersion from the conditional quartiles.

    Args:
        x: Positive recorded losses.
        censored: Indicators of right-censoring.
        weights: Positive observation weights.

    Returns:
        Median log loss and its interquartile range, with a moment fallback.
    """
    logs = np.log(x)
    lo, med, hi = _conditional_quantiles(logs, censored, weights, np.array([0.25, 0.5, 0.75]))
    mean = float(np.average(logs, weights=weights))
    std = float(np.sqrt(np.average((logs - mean) ** 2, weights=weights)))
    location = float(med) if np.isfinite(med) else mean
    spread = float(hi - lo) if np.isfinite(hi - lo) and hi > lo else 1.349 * std
    return location, max(spread, 1e-4)


def _gpd_initial(x: np.ndarray, censored: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Estimate GPD shape and scale from two conditional excess quantiles.

    Args:
        x: Nonnegative excesses above the reporting threshold.
        censored: Indicators of right-censoring.
        weights: Positive observation weights.

    Returns:
        A feasible shape and positive scale in natural coordinates.
    """
    exposure = float(np.dot(weights, x))
    events = float(np.sum(weights[~censored]))
    scale = exposure / events
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("GPD scale is not identifiable for these observations")
    quantiles = _conditional_quantiles(x, censored, weights, np.array([0.25, 0.4375, 0.5, 0.75]))
    for lo, hi, hazard in ((quantiles[2], quantiles[3], np.log(2.0)), (quantiles[0], quantiles[1], -np.log(0.75))):
        if np.isfinite(lo) and np.isfinite(hi) and lo > 0 and hi > lo:
            shape = float(np.clip(np.log(hi / lo - 1) / hazard, -0.9, 2.0))
            scale = float(lo / hazard if abs(shape) < 1e-7 else lo * shape / np.expm1(shape * hazard))
            scale = max(scale, -shape * float(np.max(x)) * 1.01, np.finfo(float).tiny)
            return shape, scale
    return 0.0, scale


def _support_shape(raw: float, scale: float, left: float, right: float) -> float:
    """Map an unconstrained shape to one satisfying finite observed-data support.

    Args:
        raw: Unconstrained shape coordinate.
        scale: Positive natural scale.
        left: Distance from the location to the smallest observation.
        right: Distance from the location to the largest observation.

    Returns:
        Shape satisfying 1 + shape * (x - loc) / scale > 0 on the sample.
    """
    distance = max(0.0, left if raw >= 0 else right)
    if distance == 0:
        return raw
    return raw / (1 + abs(raw) * (distance / scale))


def _gpd_shape_coordinate(shape: float, scale: float, largest: float) -> float:
    """Invert the negative-shape support transformation for a feasible GPD start.

    Args:
        shape: Feasible natural GPD shape.
        scale: Positive natural scale.
        largest: Largest observed excess.
    """
    return shape if shape >= 0 else shape / (1 + shape * largest / scale)

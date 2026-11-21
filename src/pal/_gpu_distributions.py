"""GPU-native vectorised probability kernels built from existing CuPy operations.

These algorithms avoid host SciPy calls and require no custom CUDA kernels.
The noncentral chi-square Poisson mixture uses a finite window around the
Poisson mode; the window is chosen for a negligible omitted tail.
"""

from __future__ import annotations

import math
import typing as t

from ._maths import special, xp


def noncentral_chisquare_cdf(x: t.Any, df: t.Any, nonc: t.Any) -> t.Any:
    """Evaluate the noncentral chi-square Poisson mixture on the active backend.

    Uses a Poisson-mode-centred window, evaluated in memory-bounded chunks.
    The only host synchronisation determines the shared truncation window;
    all probability arithmetic and special-function evaluations remain on
    the device. Noncentralities needing more than 4096 offsets are rejected,
    rather than silently approximated.
    """
    x, df, nonc = xp.broadcast_arrays(
        xp.asarray(x, dtype=xp.float64),
        xp.asarray(df, dtype=xp.float64),
        xp.asarray(nonc, dtype=xp.float64),
    )
    shape = x.shape
    xx, degrees, mean = x.ravel(), df.ravel(), nonc.ravel() / 2.0
    max_mean = float(xp.max(mean)) if mean.size else 0.0
    if max_mean == 0:
        return xp.where(x <= 0, 0.0, special.gammainc(df / 2.0, xp.maximum(x, 0) / 2.0))

    # A conservative Chernoff-tail window: extra 40 indices protect small
    # means where Gaussian tail bounds are less accurate.
    offset = math.ceil(12.0 * math.sqrt(max_mean + 1.0) + 40.0)
    if offset > 4096:
        raise NotImplementedError("GPU noncentral chi-square CDF: noncentrality exceeds tested Poisson-series range")

    mode = xp.floor(mean).astype(xp.int64)
    log_mean = xp.log(xp.where(mean > 0, mean, 1.0))
    z = xp.maximum(xx, 0.0) / 2.0
    result = xp.zeros_like(xx)
    for first in range(-offset, offset + 1, 32):
        offsets = xp.arange(first, min(first + 32, offset + 1), dtype=xp.int64)
        indexes = mode[:, None] + offsets[None, :]
        count = xp.maximum(indexes, 0)
        log_weights = -mean[:, None] + count * log_mean[:, None] - special.gammaln(count + 1.0)
        weights = xp.where(
            (indexes >= 0) & (mean[:, None] > 0),
            xp.exp(log_weights),
            xp.where(indexes == 0, xp.exp(-mean[:, None]), 0.0),
        )
        result += xp.sum(weights * special.gammainc(degrees[:, None] / 2.0 + count, z[:, None]), axis=1)

    return xp.where(x <= 0, 0.0, xp.clip(result.reshape(shape), 0.0, 1.0))


def noncentral_chisquare_ppf(u: t.Any, df: t.Any, nonc: t.Any) -> t.Any:
    """Invert the GPU-native CDF with a vectorised bracketed bisection.

    This prioritises numerical robustness; benchmarking may justify a
    safeguarded Newton iteration in a later optimisation.
    """
    u, df, nonc = xp.broadcast_arrays(
        xp.asarray(u, dtype=xp.float64),
        xp.asarray(df, dtype=xp.float64),
        xp.asarray(nonc, dtype=xp.float64),
    )
    if not u.size:
        return xp.empty_like(u)
    invalid = (u < 0) | (u > 1) | xp.isnan(u)
    target = xp.clip(u, 1e-300, 1 - 1e-15)
    mean = df + nonc
    deviation = xp.sqrt(2.0 * (df + 2.0 * nonc))
    high = mean + 32.0 * deviation + 128.0
    low = xp.full_like(high, xp.log(xp.finfo(xp.float64).tiny))
    upper_log = xp.log(high)
    # A single GPU-wide bound check, rather than a per-path host transfer.
    if bool(xp.any(noncentral_chisquare_cdf(high, df, nonc) < target)):
        raise RuntimeError("GPU noncentral chi-square inverse CDF upper bracket was insufficient")
    for _ in range(56):
        middle = (low + upper_log) / 2.0
        smaller = noncentral_chisquare_cdf(xp.exp(middle), df, nonc) < target
        low = xp.where(smaller, middle, low)
        upper_log = xp.where(smaller, upper_log, middle)
    return xp.where(
        invalid, xp.nan, xp.where(u == 0, 0.0, xp.where(u == 1, xp.inf, xp.exp((low + upper_log) / 2.0)))
    )


def generalized_inverse_gaussian_mc(rng: t.Any, n_sims: int, p: t.Any, shape: t.Any, scale: t.Any, loc: t.Any) -> t.Any:
    """Sample a GIG distribution using log-density rejection on the GPU.

    With Z=log(Y), the GIG log density is p*Z-shape*cosh(Z), which is
    strictly concave. A uniform centre and two exponential tails constitute
    a valid envelope at all positive shape parameters. Rejection is
    vectorised over simulations; host work is limited to loop completion.
    """
    index = xp.broadcast_to(xp.asarray(p, dtype=xp.float64), (n_sims,))
    b = xp.broadcast_to(xp.asarray(shape, dtype=xp.float64), (n_sims,))
    factor = xp.broadcast_to(xp.asarray(scale, dtype=xp.float64), (n_sims,))
    location = xp.broadcast_to(xp.asarray(loc, dtype=xp.float64), (n_sims,))
    curvature = xp.hypot(index, b)
    a = (curvature + index) / 2.0
    c = (curvature - index) / 2.0
    # A single-width choice covers both the concentrated and diffuse cases.
    half_width = xp.log1p((1.0 + xp.sqrt(1.0 + 2.0 * curvature)) / curvature)
    exp_width = xp.exp(half_width)
    left_slope = index - a / exp_width + c * exp_width
    right_slope = index - a * exp_width + c / exp_width
    left_mass = 1.0 / left_slope
    right_rate = -right_slope
    total_mass = left_mass + 2.0 * half_width + 1.0 / right_rate

    accepted = xp.zeros(n_sims, dtype=xp.bool_)
    deviations = xp.zeros(n_sims, dtype=xp.float64)
    tiny = xp.finfo(xp.float64).tiny
    for _ in range(128):
        mass_position = rng.uniform(size=n_sims) * total_mass
        left = -half_width + xp.log(xp.maximum(mass_position * left_slope, tiny)) / left_slope
        centre = mass_position - left_mass - half_width
        right = half_width - xp.log(xp.maximum((total_mass - mass_position) * right_rate, tiny)) / right_rate
        trial = xp.where(
            mass_position < left_mass,
            left,
            xp.where(mass_position < left_mass + 2 * half_width, centre, right),
        )
        # Express the log-density relative to its mode for stability.
        log_density = index * trial - a * xp.expm1(trial) - c * xp.expm1(-trial)
        log_envelope = xp.where(
            trial < -half_width,
            left_slope * (trial + half_width),
            xp.where(trial > half_width, right_slope * (trial - half_width), 0.0),
        )
        keep = ~accepted & (xp.log(rng.uniform(size=n_sims)) <= log_density - log_envelope)
        deviations = xp.where(keep, trial, deviations)
        accepted |= keep
        if bool(xp.all(accepted)):
            break
    else:
        raise RuntimeError("GPU GIG rejection sampler did not accept all paths in 128 rounds")
    return location + factor * xp.exp(xp.arcsinh(index / b) + deviations)

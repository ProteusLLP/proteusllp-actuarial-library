"""Maximum-likelihood fitting for PAL severity distributions.

This module works with observed CPU arrays. Fitted distributions can subsequently
be used for PAL probability calculations and simulation on supported backends.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass

import numpy as np
from scipy import integrate, optimize, stats

if t.TYPE_CHECKING:
    from .distributions import DistributionBase

_T = t.TypeVar("_T", bound="DistributionBase")
_SUPPORTED = ("Pareto", "LogNormal", "Gamma", "GPD")


@dataclass(frozen=True)
class FitResult(t.Generic[_T]):
    """Parameter estimates and likelihood diagnostics for a fitted distribution.

    The likelihood is conditional on exceeding `threshold`. A censored
    observation contributes its conditional survival probability. AICc uses
    the total frequency weight when weights are non-negative integers;
    otherwise it is NaN because a frequency-sample interpretation is absent.
    """

    distribution: _T
    parameters: dict[str, float]
    log_likelihood: float
    aic: float
    aicc: float
    converged: bool
    n_observations: int
    n_censored: int
    threshold: float
    method: str = "mle"


def _prepare(
    observations: t.Sequence[float] | np.ndarray,
    threshold: float,
    censored: t.Sequence[bool] | np.ndarray | None,
    weights: t.Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate sample alignment, truncation and frequency weights."""
    x = np.asarray(observations, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("observations must be a nonempty one-dimensional array")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    if not np.all(np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("observations must be finite and strictly positive")
    if censored is None:
        c = np.zeros(x.size, dtype=bool)
    else:
        c = np.asarray(censored, dtype=bool)
        if c.shape != x.shape:
            raise ValueError("censored must match observations")
    if weights is None:
        w = np.ones(x.size, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != x.shape or not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("weights must be finite, non-negative and match observations")
    keep = w > 0
    if not np.any(keep):
        raise ValueError("at least one observation must have positive weight")
    x, c, w = x[keep], c[keep], w[keep]
    if np.any(x < threshold) or np.any((x == threshold) & ~c):
        raise ValueError("observations must exceed threshold unless censored at threshold")
    if not np.any(~c):
        raise ValueError("at least one uncensored observation is required for fitting")
    return x, c, w


def _log_likelihood(
    family: str,
    parameters: dict[str, float],
    x: np.ndarray,
    c: np.ndarray,
    w: np.ndarray,
    threshold: float,
) -> float:
    """Return a conditional right-censored, left-truncated log-likelihood."""
    if family == "Pareto":
        shape, scale = parameters["shape"], parameters["scale"]
        if shape <= 0 or scale <= 0 or np.any(x < scale):
            return -np.inf
        log_pdf = np.log(shape) + shape * np.log(scale) - (shape + 1) * np.log(x)
        log_sf = shape * (np.log(scale) - np.log(x))
        log_sf_threshold = min(0.0, shape * (np.log(scale) - np.log(threshold))) if threshold > 0 else 0.0
    elif family == "LogNormal":
        mu, sigma = parameters["mu"], parameters["sigma"]
        if sigma <= 0:
            return -np.inf
        z = (np.log(x) - mu) / sigma
        log_pdf = stats.norm.logpdf(z) - np.log(x) - np.log(sigma)
        log_sf = stats.norm.logsf(z)
        log_sf_threshold = float(stats.norm.logsf((np.log(threshold) - mu) / sigma)) if threshold > 0 else 0.0
    elif family == "Gamma":
        alpha, theta = parameters["alpha"], parameters["theta"]
        if alpha <= 0 or theta <= 0:
            return -np.inf
        log_pdf = stats.gamma.logpdf(x, a=alpha, scale=theta)
        log_sf = stats.gamma.logsf(x, a=alpha, scale=theta)
        log_sf_threshold = float(stats.gamma.logsf(threshold, a=alpha, scale=theta)) if threshold > 0 else 0.0
    elif family == "GPD":
        shape, scale, loc = parameters["shape"], parameters["scale"], parameters["loc"]
        if scale <= 0 or np.any(x < loc):
            return -np.inf
        log_pdf = stats.genpareto.logpdf(x, c=shape, loc=loc, scale=scale)
        log_sf = stats.genpareto.logsf(x, c=shape, loc=loc, scale=scale)
        log_sf_threshold = float(stats.genpareto.logsf(threshold, c=shape, loc=loc, scale=scale))
    else:
        raise ValueError(f"Unsupported fitting family: {family}")
    terms = np.where(c, log_sf, log_pdf) - log_sf_threshold
    ll = float(np.sum(w * terms))
    return ll if np.isfinite(ll) else -np.inf


def fit_distribution(
    distribution: type[_T],
    observations: t.Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.0,
    censored: t.Sequence[bool] | np.ndarray | None = None,
    weights: t.Sequence[float] | np.ndarray | None = None,
    free_location: bool = False,
) -> FitResult[_T]:
    """Fit a supported PAL distribution by maximum likelihood.

    Args:
        distribution: PAL Pareto, LogNormal, Gamma or GPD class.
        observations: Positive, observed values. Right-censored values are
            the censoring levels, not imputed ultimate values.
        threshold: Left-truncation level, conditional on X exceeding it.
            For GPD this also fixes the location parameter.
        censored: Indicators for right-censored observations.
        weights: Non-negative observation weights. Integer weights represent
            repeated observations for information criteria.
        free_location: Fit the three-parameter GPD when True. Only supported
            for GPD; the location defaults to the reporting threshold.

    Returns:
        Fitted distribution with maximum-likelihood diagnostics.

    Raises:
        ValueError: For invalid input, an unsupported family or an
            unidentifiable sample.
        RuntimeError: If numerical optimisation cannot find a finite fit.

    Notes:
        Fitting uses SciPy on the CPU; it does not draw simulations or modify
        PAL's random-number state. Pareto has a closed-form censored MLE.
        The other families use deterministic numerical optimisation.
    """
    family = distribution.__name__
    if family not in (*_SUPPORTED, *_OTHER_FAMILIES):
        raise ValueError(f"Unsupported fitting family {family!r}; expected one of {_SUPPORTED + _OTHER_FAMILIES}")
    threshold = float(threshold)
    x, c, w = _prepare(observations, threshold, censored, weights)
    if free_location:
        if family != "GPD":
            raise ValueError("free_location is supported only for GPD")
        if threshold > 0:
            raise ValueError("The GPD location is not identifiable from a left-truncated sample")
        return _fit_gpd_free_location(distribution, x, c, w, threshold)
    if family in _OTHER_FAMILIES:
        return _fit_other(distribution, x, c, w, threshold)
    total = float(np.sum(w))
    mean = float(np.average(x, weights=w))
    variance = float(np.average((x - mean) ** 2, weights=w))
    if family == "Pareto":
        scale = threshold if threshold > 0 else float(x.min())
        denominator = float(np.sum(w * np.log(x / scale)))
        if denominator <= 0 or not np.isfinite(denominator):
            raise ValueError("Pareto shape is not identifiable for these observations")
        parameters = {"shape": float(np.sum(w[~c])) / denominator, "scale": scale}
        converged = True
    else:
        if family == "LogNormal":
            logs = np.log(x)
            mu0 = float(np.average(logs, weights=w))
            sigma0 = float(np.sqrt(np.average((logs - mu0) ** 2, weights=w)))
            starts = [[mu0, np.log(max(sigma0, 0.01))]]

            def transform(p: np.ndarray) -> dict[str, float]:
                return {"mu": float(p[0]), "sigma": float(np.exp(p[1]))}
        elif family == "Gamma":
            alpha0 = max(min(mean * mean / max(variance, np.finfo(float).tiny), 1e4), 0.05)
            starts = [
                [np.log(alpha0), np.log(mean / alpha0)],
                [0.0, np.log(mean)],
                [np.log(2.0), np.log(mean / 2.0)],
            ]

            def transform(p: np.ndarray) -> dict[str, float]:
                return {"alpha": float(np.exp(p[0])), "theta": float(np.exp(p[1]))}
        else:
            exceedances = x - threshold
            mean_exc = float(np.average(exceedances, weights=w))
            if mean_exc <= 0:
                raise ValueError("GPD scale is not identifiable for these observations")
            var_exc = float(np.average((exceedances - mean_exc) ** 2, weights=w))
            ratio = mean_exc * mean_exc / var_exc if var_exc > 0 else 1.0
            xi0 = float(np.clip(0.5 * (1 - ratio), -0.45, 2.0))
            starts = [
                [xi0, np.log(mean_exc)],
                [0.0, np.log(mean_exc)],
                [0.5, np.log(mean_exc)],
                [-0.2, np.log(mean_exc * 1.5)],
            ]

            def transform(p: np.ndarray) -> dict[str, float]:
                return {"shape": float(p[0]), "scale": float(np.exp(p[1])), "loc": threshold}

        def objective(p: np.ndarray) -> float:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
                ll = _log_likelihood(family, transform(p), x, c, w, threshold)
            return -ll if np.isfinite(ll) else 1e100

        results = [
            optimize.minimize(
                objective,
                np.asarray(start, dtype=np.float64),
                method="Nelder-Mead",
                options={"maxiter": 5000, "xatol": 1e-7, "fatol": 1e-7},
            )
            for start in starts
        ]
        best = min(results, key=lambda result: float(result.fun))
        if not np.isfinite(best.fun) or best.fun >= 1e100:
            raise RuntimeError(f"Unable to obtain a finite {family} fit")
        parameters = transform(best.x)
        converged = bool(best.success)
    ll = _log_likelihood(family, parameters, x, c, w, threshold)
    if not np.isfinite(ll):
        raise RuntimeError(f"Fitted {family} has an invalid log-likelihood")
    model = distribution(**parameters)
    if family == "Pareto":
        k = 1 if threshold > 0 else 2
    else:
        k = len(parameters) - int(family == "GPD")
    aic = 2.0 * k - 2.0 * ll
    # AICc is defined for integer frequency weights, including repeated rows.
    is_frequency_weight = bool(np.all(w == np.floor(w)))
    aicc = aic + 2.0 * k * (k + 1) / (total - k - 1) if is_frequency_weight and total > k + 1 else np.nan
    return FitResult(
        distribution=model,
        parameters=parameters,
        log_likelihood=ll,
        aic=aic,
        aicc=aicc,
        converged=converged,
        n_observations=len(x),
        n_censored=int(np.sum(c)),
        threshold=threshold,
    )


# The additional severity families use PAL's established zero-location
# parameterisations. Location is fixed at zero unless GPD has its location
# fixed at the truncation threshold.
_OTHER_FAMILIES = (
    "Weibull",
    "Burr",
    "LogLogistic",
    "InverseGaussian",
    "Paralogistic",
    "InverseBurr",
    "InverseGamma",
    "InverseWeibull",
    "LogGamma",
)


def _frozen_distribution(family: str, params: dict[str, float]) -> t.Any:
    """Build an independent SciPy reference distribution for inference."""
    p = params
    if family == "Pareto":
        return stats.pareto(b=p["shape"], scale=p["scale"])
    if family == "GPD":
        return stats.genpareto(c=p["shape"], loc=p["loc"], scale=p["scale"])
    if family == "LogNormal":
        return stats.lognorm(s=p["sigma"], scale=np.exp(p["mu"]))
    if family == "Gamma":
        return stats.gamma(a=p["alpha"], scale=p["theta"])
    if family == "Weibull":
        return stats.weibull_min(c=p["shape"], scale=p["scale"], loc=p["loc"])
    if family == "Burr":
        return stats.burr12(c=p["power"], d=p["shape"], scale=p["scale"], loc=p["loc"])
    if family == "LogLogistic":
        return stats.fisk(c=p["shape"], scale=p["scale"], loc=p["loc"])
    if family == "InverseGaussian":
        return stats.invgauss(mu=p["mu"] / p["lambda_"], scale=p["lambda_"])
    if family == "Paralogistic":
        return stats.burr12(c=p["shape"], d=p["shape"], scale=p["scale"], loc=p["loc"])
    if family == "InverseBurr":
        return stats.burr(c=p["power"], d=p["shape"], scale=p["scale"], loc=p["loc"])
    if family == "InverseGamma":
        return stats.invgamma(a=p["alpha"], scale=p["theta"], loc=p["loc"])
    if family == "InverseWeibull":
        return stats.invweibull(c=p["shape"], scale=p["scale"], loc=p["loc"])
    raise ValueError(f"Unsupported severity family: {family}")


def _fit_other(
    distribution: type[_T],
    x: np.ndarray,
    censored: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> FitResult[_T]:
    """Fit additional two- and three-parameter severity distributions."""
    family = distribution.__name__
    mean = float(np.average(x, weights=weights))
    median = float(np.median(x))
    variance = float(np.average((x - mean) ** 2, weights=weights))
    scale = max(median, np.finfo(float).tiny)
    if family == "LogGamma":
        names = ("alpha", "beta")
    elif family == "Weibull":
        names = ("shape", "scale")
    elif family == "Burr":
        names = ("power", "shape", "scale")
    elif family == "LogLogistic":
        names = ("shape", "scale")
    elif family == "InverseGaussian":
        names = ("mu", "lambda_")
    elif family == "Paralogistic":
        names = ("shape", "scale")
    elif family == "InverseBurr":
        names = ("power", "shape", "scale")
    elif family == "InverseGamma":
        names = ("alpha", "theta")
    else:
        names = ("shape", "scale")

    starts: list[list[float]]
    if family == "LogGamma":
        if np.any(x < 1.0):
            raise ValueError("log-gamma observations must be at least one")
        logs = np.log(x)
        log_mean = float(np.average(logs, weights=weights))
        log_var = float(np.average((logs - log_mean) ** 2, weights=weights))
        alpha = max(log_mean**2 / max(log_var, np.finfo(float).tiny), 0.1)
        beta = max(log_var / max(log_mean, np.finfo(float).tiny), 0.01)
        starts = [[np.log(alpha), np.log(beta)], [0.0, 0.0], [np.log(2.0), np.log(beta)]]
    elif family == "InverseGaussian":
        lam = max(mean**3 / max(variance, np.finfo(float).tiny), np.finfo(float).tiny)
        starts = [[np.log(mean), np.log(lam)], [np.log(mean), np.log(mean)]]
    elif family in {"Burr", "InverseBurr"}:
        starts = [
            [np.log(1.5), np.log(1.5), np.log(scale)],
            [0.0, 0.0, np.log(scale)],
            [np.log(2.5), 0.0, np.log(scale)],
            [0.0, np.log(2.5), np.log(scale)],
        ]
    elif family == "InverseGamma":
        alpha = max(mean * mean / max(variance, np.finfo(float).tiny) + 2.0, 1.5)
        starts = [[np.log(alpha), np.log(max(mean * (alpha - 1.0), 1e-12))], [0.0, np.log(scale)]]
    else:
        starts = [[np.log(1.0), np.log(scale)], [np.log(2.0), np.log(scale)], [np.log(0.75), np.log(scale)]]

    def make_params(raw: np.ndarray) -> dict[str, float]:
        pars: dict[str, float] = dict(zip(names, (float(np.exp(value)) for value in raw)))
        if family not in {"InverseGaussian", "LogGamma"}:
            pars["loc"] = 0.0
        return pars

    def loglik(pars: dict[str, float]) -> float:
        if family == "LogGamma":
            log_x = np.log(x)
            alpha, beta = pars["alpha"], pars["beta"]
            base = float(stats.gamma.logsf(np.log(threshold), a=alpha, scale=beta)) if threshold > 1.0 else 0.0
            terms = (
                np.where(
                    censored,
                    stats.gamma.logsf(log_x, a=alpha, scale=beta),
                    stats.gamma.logpdf(log_x, a=alpha, scale=beta) - log_x,
                )
                - base
            )
        else:
            frozen = _frozen_distribution(family, pars)
            base = float(frozen.logsf(threshold)) if threshold > 0 else 0.0
            terms = np.where(censored, frozen.logsf(x), frozen.logpdf(x)) - base
        if not np.isfinite(base):
            return -np.inf
        value = float(np.dot(weights, terms))
        return value if np.isfinite(value) else -np.inf

    def objective(raw: np.ndarray) -> float:
        with np.errstate(all="ignore"):
            val = loglik(make_params(raw))
        return -val if np.isfinite(val) else 1e100

    results = [
        optimize.minimize(
            objective,
            np.asarray(start, dtype=np.float64),
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-7, "fatol": 1e-7},
        )
        for start in starts
    ]
    best = min(results, key=lambda item: float(item.fun))
    if best.fun >= 1e100 or not np.isfinite(best.fun):
        raise RuntimeError(f"Unable to obtain finite {family} likelihood")
    pars = make_params(best.x)
    ll = loglik(pars)
    model = distribution(**pars)
    n_params = len(names)
    total_weight = float(np.sum(weights))
    aic = 2 * n_params - 2 * ll
    is_frequency = bool(np.all(weights == np.floor(weights)))
    aicc = (
        aic + 2 * n_params * (n_params + 1) / (total_weight - n_params - 1)
        if is_frequency and total_weight > n_params + 1
        else np.nan
    )
    return FitResult(
        distribution=model,
        parameters=pars,
        log_likelihood=ll,
        aic=aic,
        aicc=aicc,
        converged=bool(best.success),
        n_observations=len(x),
        n_censored=int(np.sum(censored)),
        threshold=threshold,
    )


def mean_excess(distribution: DistributionBase, threshold: float | np.ndarray) -> float | np.ndarray:
    """Calculate E[X - d | X > d] at the supplied thresholds.

    Args:
        distribution: A PAL severity distribution with fixed, scalar parameters.
        threshold: One or more attachment levels d.

    Returns:
        Mean residual loss for each threshold. Infinite means produce infinity,
        and thresholds beyond a finite upper endpoint return NaN.

    Notes:
        Closed forms are used for Pareto and GPD. For other severity families,
        stable conditional survival integration avoids subtracting close tail
        expectations. This function uses the CPU, without sampling.
    """
    family = type(distribution).__name__
    if family not in (*_SUPPORTED, *_OTHER_FAMILIES):
        raise ValueError(f"No mean excess implementation for {family}")
    params = distribution._params
    from .stochastic_scalar import StochasticScalar

    pars: dict[str, float] = {}
    for key, value in params.items():
        if isinstance(value, StochasticScalar):
            raise ValueError("mean excess requires scalar distribution parameters")
        pars[key] = float(value)
    frozen = None if family == "LogGamma" else _frozen_distribution(family, pars)
    inputs = np.asarray(threshold, dtype=float)
    result = np.full(inputs.shape, np.nan, dtype=float)
    inf_mean = (
        (family == "LogGamma" and pars["beta"] >= 1.0)
        or (family == "Pareto" and pars["shape"] <= 1.0)
        or (family == "GPD" and pars["shape"] >= 1.0)
        or (family == "Burr" and pars["power"] * pars["shape"] <= 1.0)
        or (family == "LogLogistic" and pars["shape"] <= 1.0)
        or (family == "Paralogistic" and pars["shape"] <= 1.0)
        or (family == "InverseBurr" and pars["power"] <= 1.0)
        or (family == "InverseGamma" and pars["alpha"] <= 1.0)
        or (family == "InverseWeibull" and pars["shape"] <= 1.0)
    )
    lower, upper = (1.0, np.inf) if frozen is None else frozen.support()
    for index, d in np.ndenumerate(inputs):
        if not np.isfinite(d):
            continue
        if d >= upper or (frozen is not None and frozen.logsf(float(d)) == -np.inf):
            continue
        if inf_mean:
            result[index] = np.inf
            continue
        if family == "LogGamma":
            alpha, beta = pars["alpha"], pars["beta"]
            log_cutoff = max(np.log(max(float(d), 1.0)), 0.0)
            log_num = -alpha * np.log1p(-beta) + stats.gamma.logsf(log_cutoff, a=alpha, scale=beta / (1.0 - beta))
            log_den = stats.gamma.logsf(log_cutoff, a=alpha, scale=beta)
            result[index] = float(np.exp(log_num - log_den)) - d
            continue
        if family == "Pareto":
            alpha, xm = pars["shape"], pars["scale"]
            result[index] = max(xm - d, 0.0) + max(d, xm) / (alpha - 1.0)
            continue
        if family == "GPD":
            xi, sigma, loc = pars["shape"], pars["scale"], pars["loc"]
            base = max(float(d), loc)
            result[index] = (base - d) + (sigma + xi * (base - loc)) / (1.0 - xi)
            continue

        assert frozen is not None
        # Scale by the local inverse hazard. Integrating in units of tail
        # width prevents quadrature missing a narrow tail at a high threshold.
        start = max(float(d), float(lower))
        log_start_sf = float(frozen.logsf(start))
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

        area, error = integrate.quad(
            conditional_survival,
            0.0,
            limit,
            epsrel=1e-9,
            limit=300,
        )
        if not np.isfinite(area) or area < 0 or not np.isfinite(error) or error > 1e-7 * max(1.0, abs(area)):
            raise ArithmeticError(f"Mean excess integration did not converge for {family} at {d}")
        result[index] = (start - d) + float(np.exp(log_start_sf - float(frozen.logsf(d)))) * local_scale * area
    return float(result) if result.ndim == 0 else result


def _fit_gpd_free_location(
    distribution: type[_T],
    x: np.ndarray,
    c: np.ndarray,
    w: np.ndarray,
    threshold: float,
) -> FitResult[_T]:
    """Estimate GPD shape, scale and location subject to support constraints."""
    smallest = float(np.min(x))
    delta = max(smallest * 1e-8, 1e-9)
    max_loc = smallest - delta
    average_excess = float(np.average(x - threshold, weights=w))
    start_scale = max(average_excess, delta)
    base_loc = min(threshold, max_loc - delta)

    def transform(raw: np.ndarray) -> dict[str, float]:
        # Keeping the location below the smallest observation avoids invalid
        # density evaluations. The truncation correction is retained.
        return {
            "shape": float(raw[0]),
            "scale": float(np.exp(raw[1])),
            "loc": float(max_loc - np.exp(raw[2])),
        }

    def objective(raw: np.ndarray) -> float:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
            ll = _log_likelihood("GPD", transform(raw), x, c, w, threshold)
        return -ll if np.isfinite(ll) else 1e100

    distance = max(max_loc - base_loc, delta)
    starts = [
        [0.0, np.log(start_scale), np.log(distance)],
        [0.3, np.log(start_scale), np.log(distance)],
        [-0.2, np.log(start_scale), np.log(distance)],
    ]
    results = [
        optimize.minimize(
            objective,
            np.asarray(start, dtype=float),
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-7, "fatol": 1e-7},
        )
        for start in starts
    ]
    best = min(results, key=lambda item: float(item.fun))
    if best.fun >= 1e100 or not np.isfinite(best.fun):
        raise RuntimeError("Unable to obtain a finite three-parameter GPD fit")
    params = transform(best.x)
    ll = _log_likelihood("GPD", params, x, c, w, threshold)
    n = float(np.sum(w))
    aic = 6.0 - 2.0 * ll
    aicc = aic + 24.0 / (n - 4) if np.all(w == np.floor(w)) and n > 4 else np.nan
    return FitResult(
        distribution=distribution(**params),
        parameters=params,
        log_likelihood=ll,
        aic=aic,
        aicc=aicc,
        converged=bool(best.success),
        n_observations=len(x),
        n_censored=int(np.sum(c)),
        threshold=threshold,
    )

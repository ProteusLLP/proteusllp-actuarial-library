"""Maximum-likelihood fitting for PAL severity distributions.

This module works with observed CPU arrays. Fitted distributions can subsequently
be used for PAL probability calculations and simulation on supported backends.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import integrate, optimize, special, stats

if t.TYPE_CHECKING:
    from .distributions import DistributionBase
    from .fitting_uncertainty import AsymptoticParameterUncertainty

_T = t.TypeVar("_T", bound="DistributionBase", covariant=True)


def _supported_families() -> tuple[type[DistributionBase], ...]:
    """Resolve PAL class identities lazily to avoid circular imports."""
    from . import distributions as _distributions

    return (
        _distributions.Pareto,
        _distributions.LogNormal,
        _distributions.Gamma,
        _distributions.GPD,
    )


class _FitMethod(str, Enum):
    """Stable estimator labels; FitResult exposes their string values."""

    MLE = "mle"
    ANALYTIC = "analytic_mle"
    PROFILE = "profile_mle"
    INTEGER = "integer_mle"
    ENDPOINT = "endpoint_mle"
    MULTINOMIAL = "multinomial_mle"


@dataclass(frozen=True)
class FitResult(t.Generic[_T]):
    """Parameter estimates and likelihood diagnostics for a fitted distribution.

    The likelihood is conditional on exceeding `threshold`. A censored
    observation contributes its conditional survival probability. AICc uses
    the total frequency weight when weights are non-negative integers;
    otherwise it is NaN because a frequency-sample interpretation is absent.
    """

    distribution: _T
    parameters: dict[str, t.Any]
    log_likelihood: float
    aic: float
    aicc: float
    converged: bool
    n_observations: int
    n_censored: int
    threshold: float
    method: str = _FitMethod.MLE.value

    def asymptotic_uncertainty(
        self,
        observations: t.Sequence[float] | np.ndarray,
        *,
        censored: t.Sequence[bool] | np.ndarray | None = None,
        weights: t.Sequence[float] | np.ndarray | None = None,
    ) -> AsymptoticParameterUncertainty[_T]:
        """Estimate joint parameter covariance from the fitted likelihood.

        Supply the same observations, censoring and frequency weights used
        for fitting. The default point-estimate distribution is unchanged.
        The resulting object can create a PAL distribution whose parameters
        vary together by simulation path.
        """
        from .fitting_uncertainty import asymptotic_parameter_uncertainty

        return asymptotic_parameter_uncertainty(
            self, observations, censored=censored, weights=weights
        )


def _gamma_log_survival_underflow(shape: float, z: float) -> float:
    """Evaluate log Q(shape, z) after the ordinary gamma survival underflows.

    The upper incomplete-gamma continued fraction converges in this regime.
    The computation stays on the CPU because fitting is deterministic and
    does not depend on PAL's simulation backend.
    """
    if not np.isfinite(z) or z <= shape + 1.0:
        raise ArithmeticError("Gamma survival continuation requires z > shape + 1")
    tiny = 1e-300
    b = z + 1.0 - shape
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 10001):
        an = -float(i) * (float(i) - shape)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-13:
            break
    else:
        raise ArithmeticError("Gamma survival continued fraction did not converge")
    if not np.isfinite(h) or h <= 0.0:
        raise ArithmeticError("Invalid Gamma survival continued fraction")
    return float(shape * np.log(z) - z - special.gammaln(shape) + np.log(h))


def _gamma_logsf(shape: float, z: float | np.ndarray) -> float | np.ndarray:
    """Stable log survival of a unit-scale Gamma, including underflowed tails."""
    values = np.asarray(z, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        result = np.asarray(np.log(special.gammaincc(shape, values)))
    underflow = np.isneginf(result) & np.isfinite(values) & (values > shape + 1.0)
    if np.any(underflow):
        result = result.copy()
        for index in np.ndindex(values.shape):
            if underflow[index]:
                result[index] = _gamma_log_survival_underflow(shape, float(values[index]))
    return float(result) if result.ndim == 0 else result


def _gamma_mean_excess(alpha: float, theta: float, d: float) -> float:
    """Gamma mean residual life, avoiding loss of tail survival precision."""
    if d <= 0.0:
        return alpha * theta - d
    if alpha == 1.0:
        return theta
    z = d / theta
    if not np.isfinite(z):
        raise ArithmeticError("Gamma tail attachment exceeds numeric range")
    if z > max(1e6, 100.0 * alpha):
        # Leading terms of the upper incomplete Gamma expansion. At this
        # threshold the omitted terms are negligible for practical shapes.
        return float(theta * (1.0 + (alpha - 1.0) / z))
    log_survival = float(_gamma_logsf(alpha, z))
    log_next_survival = float(_gamma_logsf(alpha + 1.0, z))
    excess = alpha * np.exp(log_next_survival - log_survival) - z
    if np.isfinite(excess) and excess > 1e-7 * max(1.0, z):
        return float(theta * excess)
    # When subtracting the attachment loses precision, integrate conditional
    # survival in dimensionless tail-width units instead.
    integral, error = integrate.quad(
        lambda u: np.exp(float(_gamma_logsf(alpha, z + u)) - log_survival),
        0.0,
        np.inf,
        epsrel=1e-10,
        limit=300,
    )
    if not np.isfinite(integral) or not np.isfinite(error) or error > 1e-7:
        raise ArithmeticError("Gamma conditional survival integration did not converge")
    return float(theta * integral)


def _lognormal_mean_excess(mu: float, sigma: float, d: float) -> float:
    """Conditional first-moment formula with erfcx for upper-tail accuracy."""
    if d <= 0.0:
        return float(np.exp(mu + 0.5 * sigma**2) - d)
    z = (np.log(d) - mu) / sigma
    if z >= sigma:
        numerator = special.erfcx((z - sigma) / np.sqrt(2.0))
        denominator = special.erfcx(z / np.sqrt(2.0))
        if denominator > 0.0 and np.isfinite(numerator):
            relative_excess = (numerator - denominator) / denominator
            if relative_excess > 1e-8:
                return float(d * relative_excess)
    log_ratio = mu + 0.5 * sigma**2 - np.log(d)
    log_ratio += special.log_ndtr(sigma - z) - special.log_ndtr(-z)
    if np.isfinite(log_ratio) and log_ratio > 1e-10:
        return float(d * np.expm1(log_ratio))
    if z > 1e7:
        # Extreme normal-tail asymptotic, avoiding subtraction of close values.
        return float(d * sigma / z)
    raise ArithmeticError("LogNormal mean excess lost numerical precision")


def _weibull_mean_excess(shape: float, scale: float, loc: float, d: float) -> float:
    """Integrate a Weibull survival using the upper incomplete gamma ratio."""
    a = 1.0 / shape
    if d <= loc:
        return float(loc + scale * np.exp(special.gammaln(1.0 + a)) - d)
    if shape == 1.0:
        return scale
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        z = float(np.power((d - loc) / scale, shape))
    if not np.isfinite(z):
        raise ArithmeticError("Weibull attachment exceeds numeric range")
    if z > 1e7:
        log_value = np.log(scale / shape) + (a - 1.0) * np.log(z)
        log_value += np.log1p((a - 1.0) / z)
    else:
        log_value = np.log(scale) + special.gammaln(1.0 + a)
        log_value += float(_gamma_logsf(a, z)) + z
    return float(np.exp(log_value))


def _burr_mean_excess(power: float, shape: float, scale: float, loc: float, d: float) -> float:
    """Burr XII mean residual life via the complementary incomplete beta."""
    if d <= loc:
        mean = scale * shape * special.beta(1.0 + 1.0 / power, shape - 1.0 / power)
        return float(loc + mean - d)
    x = d - loc
    if power == 1.0:
        return float((scale + x) / (shape - 1.0))
    a = shape - 1.0 / power
    b = 1.0 / power
    log_w = -np.logaddexp(0.0, power * (np.log(x) - np.log(scale)))
    w = float(np.exp(log_w))
    if w < 1e-12 and abs(b - 1.0) * w < 1e-12:
        # B_w(a,b) = w**a/a * (1 + O(w)) in the far tail.
        log_value = np.log(scale) - np.log(power) - np.log(a) - b * log_w
    else:
        incomplete = float(special.betainc(a, b, w))
        if incomplete <= 0.0 or not np.isfinite(incomplete):
            raise ArithmeticError("Burr mean excess incomplete beta underflowed")
        log_value = np.log(scale) - np.log(power) + special.betaln(a, b) + np.log(incomplete) - shape * log_w
    return float(np.exp(log_value))


def _prepare(
    observations: t.Sequence[float] | np.ndarray,
    threshold: float,
    censored: t.Sequence[bool] | np.ndarray | None,
    weights: t.Sequence[float] | np.ndarray | None,
    *,
    allow_any_real: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate sample alignment, truncation and frequency weights."""
    x = np.asarray(observations, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("observations must be a nonempty one-dimensional array")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    if not np.all(np.isfinite(x)) or (not allow_any_real and np.any(x <= 0)):
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
    if threshold > 0 and (np.any(x < threshold) or np.any((x == threshold) & ~c)):
        raise ValueError("observations must exceed threshold unless censored at threshold")
    if not np.any(~c):
        raise ValueError("at least one uncensored observation is required for fitting")
    return x, c, w


def _log_likelihood(
    family: type[DistributionBase],
    parameters: dict[str, float],
    x: np.ndarray,
    c: np.ndarray,
    w: np.ndarray,
    threshold: float,
) -> float:
    """Return a conditional right-censored, left-truncated log-likelihood."""
    from . import distributions as _distributions

    if family is _distributions.Pareto:
        shape, scale = parameters["shape"], parameters["scale"]
        if shape <= 0 or scale <= 0 or np.any(x < scale):
            return -np.inf
        log_pdf = np.log(shape) + shape * np.log(scale) - (shape + 1) * np.log(x)
        log_sf = shape * (np.log(scale) - np.log(x))
        log_sf_threshold = min(0.0, shape * (np.log(scale) - np.log(threshold))) if threshold > 0 else 0.0
    elif family is _distributions.LogNormal:
        mu, sigma = parameters["mu"], parameters["sigma"]
        if sigma <= 0:
            return -np.inf
        z = (np.log(x) - mu) / sigma
        log_pdf = stats.norm.logpdf(z) - np.log(x) - np.log(sigma)
        log_sf = stats.norm.logsf(z)
        log_sf_threshold = float(stats.norm.logsf((np.log(threshold) - mu) / sigma)) if threshold > 0 else 0.0
    elif family is _distributions.Gamma:
        alpha, theta = parameters["alpha"], parameters["theta"]
        if alpha <= 0 or theta <= 0:
            return -np.inf
        log_pdf = stats.gamma.logpdf(x, a=alpha, scale=theta)
        log_sf = _gamma_logsf(alpha, x / theta)
        log_sf_threshold = float(_gamma_logsf(alpha, threshold / theta)) if threshold > 0 else 0.0
    elif family is _distributions.GPD:
        shape, scale, loc = parameters["shape"], parameters["scale"], parameters["loc"]
        if scale <= 0 or np.any(x < loc):
            return -np.inf
        log_pdf = stats.genpareto.logpdf(x, c=shape, loc=loc, scale=scale)
        log_sf = stats.genpareto.logsf(x, c=shape, loc=loc, scale=scale)
        log_sf_threshold = float(stats.genpareto.logsf(threshold, c=shape, loc=loc, scale=scale))
    else:
        raise ValueError(f"Unsupported fitting family: {family.__name__}")
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
    from . import distributions as _distributions

    family = distribution
    supported = (*_supported_families(), *_other_families())
    if family not in supported:
        expected = tuple(cls.__name__ for cls in supported)
        raise ValueError(f"Unsupported fitting family {family.__name__!r}; expected one of {expected}")
    threshold = float(threshold)
    if free_location and family is not _distributions.GPD:
        raise ValueError("free_location is supported only for GPD")
    if free_location and threshold > 0:
        raise ValueError("The GPD location is not identifiable from a left-truncated sample")
    x, c, w = _prepare(observations, threshold, censored, weights, allow_any_real=family is _distributions.GEV)
    if free_location:
        return _fit_gpd_free_location(distribution, x, c, w, threshold)
    if family is _distributions.Exponential:
        return _fit_exponential(distribution, x, c, w, threshold)
    if family is _distributions.GeneralizedBetaPrime:
        return _fit_gb2(distribution, x, c, w, threshold)
    if family in _other_families():
        return _fit_other(distribution, x, c, w, threshold)
    total = float(np.sum(w))
    mean = float(np.average(x, weights=w))
    variance = float(np.average((x - mean) ** 2, weights=w))
    if family is _distributions.Pareto:
        scale = threshold if threshold > 0 else float(x.min())
        denominator = float(np.sum(w * np.log(x / scale)))
        if denominator <= 0 or not np.isfinite(denominator):
            raise ValueError("Pareto shape is not identifiable for these observations")
        parameters = {"shape": float(np.sum(w[~c])) / denominator, "scale": scale}
        converged = True
    else:
        if family is _distributions.LogNormal:
            logs = np.log(x)
            mu0 = float(np.average(logs, weights=w))
            sigma0 = float(np.sqrt(np.average((logs - mu0) ** 2, weights=w)))
            starts = [[mu0, np.log(max(sigma0, 0.01))]]

            def transform(p: np.ndarray) -> dict[str, float]:
                return {"mu": float(p[0]), "sigma": float(np.exp(p[1]))}
        elif family is _distributions.Gamma:
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
            raise RuntimeError(f"Unable to obtain a finite {family.__name__} fit")
        parameters = transform(best.x)
        converged = bool(best.success)
    ll = _log_likelihood(family, parameters, x, c, w, threshold)
    if not np.isfinite(ll):
        raise RuntimeError(f"Fitted {family.__name__} has an invalid log-likelihood")
    model = distribution(**parameters)
    if family is _distributions.Pareto:
        k = 1 if threshold > 0 else 2
    else:
        k = len(parameters) - int(family is _distributions.GPD)
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
        method=_FitMethod.ANALYTIC.value if family is _distributions.Pareto else _FitMethod.MLE.value,
    )


# The additional severity families use PAL's established zero-location
# parameterisations. Location is fixed at zero unless GPD has its location
# fixed at the truncation threshold.
def _other_families() -> tuple[type[DistributionBase], ...]:
    """Return the other supported severity distribution classes."""
    from . import distributions as _distributions

    return (
        _distributions.Weibull,
        _distributions.Burr,
        _distributions.LogLogistic,
        _distributions.InverseGaussian,
        _distributions.Paralogistic,
        _distributions.InverseBurr,
        _distributions.InverseGamma,
        _distributions.InverseWeibull,
        _distributions.LogGamma,
        _distributions.Exponential,
        _distributions.InverseExponential,
        _distributions.InverseParalogistic,
        _distributions.GeneralizedBetaPrime,
        _distributions.GeneralizedInverseGaussian,
        _distributions.GEV,
    )


def _frozen_distribution(family: type[DistributionBase], params: dict[str, float]) -> t.Any:
    """Build an independent SciPy reference distribution for inference."""
    from . import distributions as _distributions

    p = params
    if family is _distributions.Pareto:
        return stats.pareto(b=p["shape"], scale=p["scale"])
    if family is _distributions.GPD:
        return stats.genpareto(c=p["shape"], loc=p["loc"], scale=p["scale"])
    if family is _distributions.LogNormal:
        return stats.lognorm(s=p["sigma"], scale=np.exp(p["mu"]))
    if family is _distributions.Gamma:
        return stats.gamma(a=p["alpha"], scale=p["theta"])
    if family is _distributions.Weibull:
        return stats.weibull_min(c=p["shape"], scale=p["scale"], loc=p["loc"])
    if family is _distributions.Burr:
        return stats.burr12(c=p["power"], d=p["shape"], scale=p["scale"], loc=p["loc"])
    if family is _distributions.LogLogistic:
        return stats.fisk(c=p["shape"], scale=p["scale"], loc=p["loc"])
    if family is _distributions.InverseGaussian:
        return stats.invgauss(mu=p["mu"] / p["lambda_"], scale=p["lambda_"])
    if family is _distributions.Paralogistic:
        return stats.burr12(c=p["shape"], d=p["shape"], scale=p["scale"], loc=p["loc"])
    if family is _distributions.InverseBurr:
        return stats.burr(c=p["power"], d=p["shape"], scale=p["scale"], loc=p["loc"])
    if family is _distributions.InverseGamma:
        return stats.invgamma(a=p["alpha"], scale=p["theta"], loc=p["loc"])
    if family is _distributions.InverseWeibull:
        return stats.invweibull(c=p["shape"], scale=p["scale"], loc=p["loc"])
    if family is _distributions.Exponential:
        return stats.expon(scale=p["scale"], loc=p["loc"])
    if family is _distributions.InverseExponential:
        return stats.invweibull(c=1.0, scale=p["scale"], loc=p["loc"])
    if family is _distributions.InverseParalogistic:
        return stats.burr(c=p["shape"], d=p["shape"], scale=p["scale"], loc=p["loc"])
    if family is _distributions.GeneralizedInverseGaussian:
        return stats.geninvgauss(
            p=p["p"], b=np.sqrt(p["chi"] * p["psi"]), scale=np.sqrt(p["chi"] / p["psi"]), loc=p["loc"]
        )
    if family is _distributions.GEV:
        return stats.genextreme(c=-p["shape"], loc=p["loc"], scale=p["scale"])
    raise ValueError(f"Unsupported severity family: {family.__name__}")


def _fit_other(
    distribution: type[_T],
    x: np.ndarray,
    censored: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> FitResult[_T]:
    """Fit additional two- and three-parameter severity distributions."""
    from . import distributions as _distributions

    family = distribution
    mean = float(np.average(x, weights=weights))
    median = float(np.median(x))
    variance = float(np.average((x - mean) ** 2, weights=weights))
    scale = max(median, np.finfo(float).tiny)
    if family is _distributions.GeneralizedInverseGaussian:
        names = ("p", "chi", "psi")
    elif family is _distributions.GEV:
        names = ("shape", "scale", "loc")
    elif family in (_distributions.Exponential, _distributions.InverseExponential):
        names = ("scale",)
    elif family is _distributions.LogGamma:
        names = ("alpha", "beta")
    elif family is _distributions.Weibull:
        names = ("shape", "scale")
    elif family is _distributions.Burr:
        names = ("power", "shape", "scale")
    elif family is _distributions.LogLogistic:
        names = ("shape", "scale")
    elif family is _distributions.InverseGaussian:
        names = ("mu", "lambda_")
    elif family is _distributions.Paralogistic:
        names = ("shape", "scale")
    elif family is _distributions.InverseBurr:
        names = ("power", "shape", "scale")
    elif family is _distributions.InverseGamma:
        names = ("alpha", "theta")
    else:
        names = ("shape", "scale")

    starts: list[list[float]]
    if family is _distributions.GeneralizedInverseGaussian:
        starts = [[-0.5, np.log(scale), np.log(1.0 / scale)], [0.5, np.log(scale), np.log(1.0 / scale)]]
    elif family is _distributions.GEV:
        spread = max(float(np.std(x)), 0.01)
        starts = [[0.0, np.log(spread), median], [0.2, np.log(spread), median], [-0.2, np.log(spread), median]]
    elif family in (_distributions.Exponential, _distributions.InverseExponential):
        starts = [[np.log(max(mean, 1e-10))]]
    elif family is _distributions.LogGamma:
        if np.any(x < 1.0):
            raise ValueError("log-gamma observations must be at least one")
        logs = np.log(x)
        log_mean = float(np.average(logs, weights=weights))
        log_var = float(np.average((logs - log_mean) ** 2, weights=weights))
        alpha = max(log_mean**2 / max(log_var, np.finfo(float).tiny), 0.1)
        beta = max(log_var / max(log_mean, np.finfo(float).tiny), 0.01)
        starts = [[np.log(alpha), np.log(beta)], [0.0, 0.0], [np.log(2.0), np.log(beta)]]
    elif family is _distributions.InverseGaussian:
        lam = max(mean**3 / max(variance, np.finfo(float).tiny), np.finfo(float).tiny)
        starts = [[np.log(mean), np.log(lam)], [np.log(mean), np.log(mean)]]
    elif family in (_distributions.Burr, _distributions.InverseBurr):
        starts = [
            [np.log(1.5), np.log(1.5), np.log(scale)],
            [0.0, 0.0, np.log(scale)],
            [np.log(2.5), 0.0, np.log(scale)],
            [0.0, np.log(2.5), np.log(scale)],
        ]
    elif family is _distributions.InverseGamma:
        alpha = max(mean * mean / max(variance, np.finfo(float).tiny) + 2.0, 1.5)
        starts = [[np.log(alpha), np.log(max(mean * (alpha - 1.0), 1e-12))], [0.0, np.log(scale)]]
    else:
        starts = [[np.log(1.0), np.log(scale)], [np.log(2.0), np.log(scale)], [np.log(0.75), np.log(scale)]]

    def make_params(raw: np.ndarray) -> dict[str, float]:
        if family is _distributions.GeneralizedInverseGaussian:
            return {"p": float(raw[0]), "chi": float(np.exp(raw[1])), "psi": float(np.exp(raw[2])), "loc": 0.0}
        if family is _distributions.GEV:
            return {"shape": float(raw[0]), "scale": float(np.exp(raw[1])), "loc": float(raw[2])}
        pars: dict[str, float] = dict(zip(names, (float(np.exp(value)) for value in raw)))
        if family not in (_distributions.InverseGaussian, _distributions.LogGamma):
            pars["loc"] = 0.0
        return pars

    def loglik(pars: dict[str, float]) -> float:
        if family is _distributions.LogGamma:
            log_x = np.log(x)
            alpha, beta = pars["alpha"], pars["beta"]
            base = float(_gamma_logsf(alpha, np.log(threshold) / beta)) if threshold > 1.0 else 0.0
            terms = (
                np.where(
                    censored,
                    _gamma_logsf(alpha, log_x / beta),
                    stats.gamma.logpdf(log_x, a=alpha, scale=beta) - log_x,
                )
                - base
            )
        else:
            frozen = _frozen_distribution(family, pars)
            base = float(frozen.logsf(threshold)) if threshold > 0 else 0.0
            terms = np.empty_like(x)
            if np.any(~censored):
                terms[~censored] = frozen.logpdf(x[~censored])
            if np.any(censored):
                terms[censored] = frozen.logsf(x[censored])
            terms -= base
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
        raise RuntimeError(f"Unable to obtain finite {family.__name__} likelihood")
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
        Analytical expressions are used for Pareto, GPD, Gamma, LogNormal,
        Weibull and Burr. Other severity families use a conditional survival
        integral. This function uses the CPU, without sampling.
    """
    from . import distributions as _distributions

    family = type(distribution)
    if family not in (*_supported_families(), *_other_families()):
        raise ValueError(f"No mean excess implementation for {family.__name__}")
    params = distribution._params
    from .stochastic_scalar import StochasticScalar

    pars: dict[str, float] = {}
    for key, value in params.items():
        if isinstance(value, StochasticScalar):
            raise ValueError("mean excess requires scalar distribution parameters")
        pars[key] = float(value)
    frozen = (
        None
        if family in (_distributions.LogGamma, _distributions.GeneralizedBetaPrime)
        else _frozen_distribution(family, pars)
    )
    inputs = np.asarray(threshold, dtype=float)
    result = np.full(inputs.shape, np.nan, dtype=float)
    inf_mean = (
        (family is _distributions.GEV and pars["shape"] >= 1.0)
        or (family is _distributions.GeneralizedBetaPrime and pars["a"] * pars["q"] <= 1.0)
        or (family is _distributions.LogGamma and pars["beta"] >= 1.0)
        or (family is _distributions.Pareto and pars["shape"] <= 1.0)
        or (family is _distributions.GPD and pars["shape"] >= 1.0)
        or (family is _distributions.Burr and pars["power"] * pars["shape"] <= 1.0)
        or (family is _distributions.LogLogistic and pars["shape"] <= 1.0)
        or (family is _distributions.Paralogistic and pars["shape"] <= 1.0)
        or (family is _distributions.InverseBurr and pars["power"] <= 1.0)
        or (family is _distributions.InverseGamma and pars["alpha"] <= 1.0)
        or (family is _distributions.InverseWeibull and pars["shape"] <= 1.0)
        or family is _distributions.InverseExponential
        or (family is _distributions.InverseParalogistic and pars["shape"] <= 1.0)
    )
    if family is _distributions.GeneralizedBetaPrime:
        lower, upper = 0.0, np.inf
    else:
        lower, upper = (1.0, np.inf) if frozen is None else frozen.support()
    for index, d in np.ndenumerate(inputs):
        if not np.isfinite(d):
            continue
        if d >= upper:
            continue
        if inf_mean:
            result[index] = np.inf
            continue
        if family is _distributions.GeneralizedBetaPrime:
            result[index] = _gb2_mean_excess(pars, float(d))
            continue
        if family is _distributions.LogGamma:
            alpha, beta = pars["alpha"], pars["beta"]
            log_cutoff = max(np.log(max(float(d), 1.0)), 0.0)
            log_num = -alpha * np.log1p(-beta) + float(_gamma_logsf(alpha, log_cutoff * (1.0 - beta) / beta))
            log_den = float(_gamma_logsf(alpha, log_cutoff / beta))
            result[index] = float(np.exp(log_num - log_den)) - d
            continue
        if family is _distributions.Pareto:
            alpha, xm = pars["shape"], pars["scale"]
            result[index] = max(xm - d, 0.0) + max(d, xm) / (alpha - 1.0)
            continue
        if family is _distributions.GPD:
            xi, sigma, loc = pars["shape"], pars["scale"], pars["loc"]
            base = max(float(d), loc)
            result[index] = (base - d) + (sigma + xi * (base - loc)) / (1.0 - xi)
            continue

        if family is _distributions.Gamma:
            result[index] = _gamma_mean_excess(pars["alpha"], pars["theta"], float(d))
            continue
        if family is _distributions.LogNormal:
            result[index] = _lognormal_mean_excess(pars["mu"], pars["sigma"], float(d))
            continue
        if family is _distributions.Weibull:
            result[index] = _weibull_mean_excess(pars["shape"], pars["scale"], pars["loc"], float(d))
            continue
        if family is _distributions.Burr:
            result[index] = _burr_mean_excess(pars["power"], pars["shape"], pars["scale"], pars["loc"], float(d))
            continue

        if frozen is None:
            raise ValueError(f"No continuous survival function for {family.__name__}")
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

        if family is _distributions.GeneralizedInverseGaussian:
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

            area, error = integrate.quad(scaled_excess_density, 0.0, limit, epsrel=1e-9, limit=300)
            excess = local_scale**2 * area
        else:
            area, error = integrate.quad(conditional_survival, 0.0, limit, epsrel=1e-9, limit=300)
            excess = local_scale * area

        if not np.isfinite(area) or area < 0 or not np.isfinite(error) or error > 1e-7 * max(1.0, abs(area)):
            raise ArithmeticError(f"Mean excess integration did not converge for {family.__name__} at {d}")
        result[index] = (start - d) + float(np.exp(log_start_sf - float(frozen.logsf(d)))) * excess
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
            ll = _log_likelihood(distribution, transform(raw), x, c, w, threshold)
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
    ll = _log_likelihood(distribution, params, x, c, w, threshold)
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


def _fit_exponential(
    distribution: type[_T],
    x: np.ndarray,
    censored: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> FitResult[_T]:
    """Use the exact censored MLE for an exponential above the threshold."""
    events = float(np.sum(weights[~censored]))
    total = float(np.sum(weights * (x - threshold)))
    if events <= 0 or total <= 0:
        raise ValueError("Exponential scale is not identifiable")
    scale = total / events
    params = {"scale": scale, "loc": 0.0}
    ll = float(-events * np.log(scale) - total / scale)
    aic = 2 - 2 * ll
    sample_size = float(np.sum(weights))
    freq = bool(np.all(weights == np.floor(weights)))
    aicc = aic + 4 / (sample_size - 2) if freq and sample_size > 2 else np.nan
    return FitResult(
        distribution=distribution(**params),
        parameters=params,
        log_likelihood=ll,
        aic=aic,
        aicc=aicc,
        converged=True,
        n_observations=len(x),
        n_censored=int(np.sum(censored)),
        threshold=threshold,
        method=_FitMethod.ANALYTIC.value,
    )


def _gb2_log_regularized_beta(alpha: float, beta: float, log_z: np.ndarray) -> np.ndarray:
    """Stable log I_z(alpha, beta), including probabilities below float underflow."""
    log_z = np.asarray(log_z, dtype=float)
    z = np.exp(log_z)
    with np.errstate(divide="ignore", under="ignore", invalid="ignore"):
        result = np.log(special.betainc(alpha, beta, z))
    # Large beta shape parameters can underflow even when z exceeds 1e-8.
    small = (log_z < np.log(1e-8)) | (np.isneginf(result) & np.isfinite(log_z))
    if np.any(small):
        # B_z(alpha,beta) = z**alpha / alpha *
        # 2F1(alpha, 1-beta; alpha+1; z). Evaluate the correction only
        # where z is small, avoiding unnecessary special-function calls.
        correction = special.hyp2f1(alpha, 1.0 - beta, alpha + 1.0, z[small])
        if np.any(~np.isfinite(correction)) or np.any(correction <= 0.0):
            raise ArithmeticError("Incomplete-beta tail correction did not converge")
        result = np.asarray(result).copy()
        result[small] = alpha * log_z[small] - np.log(alpha) - special.betaln(alpha, beta) + np.log(correction)
    return result


def _gb2_terms(x: np.ndarray, pars: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Log-density and log-survival for the generalised beta prime family."""
    a, b, p, q = (pars[name] for name in ("a", "b", "p", "q"))
    log_ratio = np.log(x) - np.log(b)
    log_power = a * log_ratio
    log_pdf = np.log(a) - np.log(b) - special.betaln(p, q) + (a * p - 1) * log_ratio
    log_pdf -= (p + q) * np.logaddexp(0.0, log_power)
    log_survival_argument = -np.logaddexp(0.0, log_power)
    log_sf = _gb2_log_regularized_beta(q, p, log_survival_argument)
    return log_pdf, log_sf


def _fit_gb2(
    distribution: type[_T],
    x: np.ndarray,
    censored: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> FitResult[_T]:
    """Fit the four-parameter GB2 distribution in log-parameter coordinates."""
    base_scale = float(np.median(x))
    starts = (
        (1.0, base_scale, 1.0, 2.0),
        (2.0, base_scale, 1.0, 1.0),
        (1.0, base_scale, 2.0, 3.0),
        (0.7, base_scale, 2.0, 2.0),
    )

    def transform(params: np.ndarray) -> dict[str, float]:
        return dict(zip(("a", "b", "p", "q"), (float(np.exp(v)) for v in params)))

    def likelihood(params: dict[str, float]) -> float:
        pdf, sf = _gb2_terms(x, params)
        if threshold > 0:
            _, log_sf_thresh = _gb2_terms(np.array([threshold]), params)
            truncation = float(log_sf_thresh[0])
        else:
            truncation = 0.0
        terms = np.where(censored, sf, pdf) - truncation
        ll = float(np.sum(weights * terms))
        return ll if np.isfinite(ll) else -np.inf

    def objective(v: np.ndarray) -> float:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            ll = likelihood(transform(v))
        return -ll if np.isfinite(ll) else 1e100

    candidates = [
        optimize.minimize(
            objective,
            np.log(start),
            method="Nelder-Mead",
            options={"maxiter": 6000, "xatol": 1e-7, "fatol": 1e-7},
        )
        for start in starts
    ]
    best = min(candidates, key=lambda fit: float(fit.fun))
    if not np.isfinite(best.fun) or best.fun >= 1e100:
        raise RuntimeError("No finite generalised beta prime fit")
    pars = transform(best.x)
    ll = likelihood(pars)
    total_weight = float(np.sum(weights))
    aic = 8 - 2 * ll
    frequency = bool(np.all(weights == np.floor(weights)))
    aicc = aic + 40 / (total_weight - 5) if frequency and total_weight > 5 else np.nan
    return FitResult(
        distribution=distribution(**pars),
        parameters=pars,
        log_likelihood=ll,
        aic=aic,
        aicc=aicc,
        converged=bool(best.success),
        n_observations=len(x),
        n_censored=int(np.sum(censored)),
        threshold=threshold,
    )


def _gb2_mean_excess(pars: dict[str, float], attachment: float) -> float:
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

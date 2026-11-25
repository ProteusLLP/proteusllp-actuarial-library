"""Maximum-likelihood fitting for PAL severity distributions.

This module works with observed CPU arrays. Fitted distributions can subsequently
be used for PAL probability calculations and simulation on supported backends.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np
from scipy import optimize, special, stats

from ._fitting_initial import (
    _gpd_initial,
    _gpd_shape_coordinate,
    _initial_coordinates,
    _log_location_scale,
    _override_initial,
    _shape_coordinate,
    _support_shape,
)
from ._gig import _gig_log_probabilities
from ._severity_probability import _burr_terms, _inverse_weibull_terms, _StableSeverityReference

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


class UncertaintyMethod(str, Enum):
    """Optional frequentist parameter uncertainty attached to a fit result.

    Attributes:
        NONE: Retain fitted point estimates only.
        ASYMPTOTIC: Estimate joint parameter covariance using the fitted likelihood.
    """

    NONE = "none"
    ASYMPTOTIC = "asymptotic"


class OptimizerMethod(str, Enum):
    """Numerical optimizer for severity maximum-likelihood estimation.

    Attributes:
        NELDER_MEAD: Derivative-free simplex optimisation.
        L_BFGS_B: Limited-memory quasi-Newton optimisation with bound support.
    """

    NELDER_MEAD = "Nelder-Mead"
    L_BFGS_B = "L-BFGS-B"


class _FitMethod(str, Enum):
    """Stable estimator labels; FitResult exposes their string values.

    Attributes:
        MLE: Numerical maximum likelihood.
        ANALYTIC: Closed-form maximum likelihood.
        PROFILE: Profile maximum likelihood over a reduced parameter space.
        INTEGER: Maximum likelihood over discrete parameter candidates.
        ENDPOINT: Maximum likelihood from sample support endpoints.
        MULTINOMIAL: Maximum likelihood from observed category frequencies.
    """

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

    Attributes:
        distribution: Distribution instantiated at the fitted parameters.
        parameters: Estimated natural parameters, including any fixed support values.
        log_likelihood: Maximised weighted conditional log-likelihood.
        aic: Akaike information criterion.
        aicc: Small-sample corrected AIC, or NaN when unavailable.
        converged: Whether the selected estimator reported convergence.
        n_observations: Number of positive-weight observation rows used.
        n_censored: Number of right-censored observation rows used.
        threshold: Left-truncation threshold used in the likelihood.
        method: Identifier for the estimation method.
        uncertainty: Optional joint asymptotic parameter uncertainty.
        n_parameters: Number of independently estimated parameters.
        warnings: Diagnostics for potentially limiting or extreme estimates.
        optimizer: Numerical optimiser that produced the result, if applicable.
        optimizer_fallback: Whether L-BFGS-B required a simplex fallback.
        threshold_specified: Whether a left-truncation threshold was supplied, including zero.
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
    uncertainty: AsymptoticParameterUncertainty[_T] | None = field(default=None, repr=False, compare=False)
    n_parameters: int = field(default=-1, kw_only=True)
    warnings: tuple[str, ...] = field(default=(), kw_only=True)
    optimizer: str | None = field(default=None, kw_only=True)
    optimizer_fallback: bool = field(default=False, kw_only=True)
    threshold_specified: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        """Preserve compatibility with callers constructing older FitResult objects."""
        if self.n_parameters == -1:
            object.__setattr__(self, "n_parameters", round((self.aic + 2 * self.log_likelihood) / 2))

    @property
    def at_boundary(self) -> bool:
        """Whether numerical diagnostics indicate a possible limiting fit."""
        return bool(self.warnings)

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

        Args:
            observations: Observed values used in the likelihood or empirical estimate.
            censored: Indicators that observations are right-censored at their recorded values.
            weights: Nonnegative observation weights; zero-weight observations are excluded.
        """
        from .fitting_uncertainty import asymptotic_parameter_uncertainty

        return asymptotic_parameter_uncertainty(self, observations, censored=censored, weights=weights)


def _minimize(
    objective: t.Callable[[np.ndarray], float],
    initial: np.ndarray,
    optimizer: OptimizerMethod | str,
    options: t.Mapping[str, t.Any] | None,
    maxiter: int,
    *,
    alternative: np.ndarray | None = None,
) -> optimize.OptimizeResult:
    """Optimise once from a statistical estimate, with failure-driven recovery.

    Args:
        objective: Function returning the negative log-likelihood.
        initial: Initial parameter vector in unconstrained coordinates.
        optimizer: Numerical optimiser for the primary fit.
        options: Options passed to the primary optimiser.
        maxiter: Default iteration budget for each attempt.
        alternative: Optional recovery start, used only when the primary fit fails.

    Returns:
        Best finite optimisation result and diagnostics describing any recovery.
    """
    method = OptimizerMethod(optimizer)
    simplex_options = {"maxiter": maxiter, "xatol": 1e-7, "fatol": 1e-7}
    defaults: dict[str, t.Any] = (
        simplex_options
        if method is OptimizerMethod.NELDER_MEAD
        else {"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-6, "maxls": 50}
    )
    settings = {**defaults, **(options or {})}

    def safe_objective(raw: np.ndarray) -> float:
        """Return a finite penalty for invalid parameter proposals.

        Args:
            raw: Trial vector in unconstrained coordinates.
        """
        try:
            value = objective(raw)
        except (ArithmeticError, ValueError):
            return 1e100
        return float(value) if np.isfinite(value) else 1e100

    def run(start: np.ndarray, selected: OptimizerMethod) -> optimize.OptimizeResult:
        """Execute one numerical optimisation.

        Args:
            start: Feasible starting parameter vector.
            selected: Method to execute.
        """
        chosen_options = settings if selected is method else simplex_options
        return optimize.minimize(
            safe_objective,
            start,
            method=selected.value,
            jac="3-point" if selected is OptimizerMethod.L_BFGS_B else None,
            options=t.cast(t.Any, chosen_options),
        )

    def acceptable(result: optimize.OptimizeResult, selected: OptimizerMethod) -> bool:
        """Check objective validity, convergence and gradient stationarity.

        Args:
            result: Result of one numerical optimisation.
            selected: Method that produced the result.
        """
        if not result.success or not np.isfinite(result.fun) or result.fun >= 1e100:
            return False
        if selected is OptimizerMethod.NELDER_MEAD:
            return True
        gradient = np.asarray(result.jac)
        return bool(np.all(np.isfinite(gradient)) and np.max(np.abs(gradient)) <= max(1e-4, 1e-7 * abs(result.fun)))

    fallback = False
    if safe_objective(initial) >= 1e100 and alternative is not None:
        initial, alternative = alternative, None
        fallback = True
    best = run(initial, method)
    used = method
    if not acceptable(best, used):
        fallback = True
        if method is OptimizerMethod.L_BFGS_B and best.fun < 1e100:
            polished = run(best.x, OptimizerMethod.NELDER_MEAD)
            if polished.fun <= best.fun:
                best, used = polished, OptimizerMethod.NELDER_MEAD
        if not acceptable(best, used) and alternative is not None and safe_objective(alternative) < 1e100:
            retry = run(alternative, method)
            if retry.fun <= best.fun:
                best, used = retry, method
            if method is OptimizerMethod.L_BFGS_B and not acceptable(best, used) and best.fun < 1e100:
                polished = run(best.x, OptimizerMethod.NELDER_MEAD)
                if polished.fun <= best.fun:
                    best, used = polished, OptimizerMethod.NELDER_MEAD
    # OptimizeResult is a dict at runtime, but its stubs omit custom entries.
    diagnostics = t.cast(dict[str, t.Any], best)
    diagnostics["optimizer_used"] = used.value
    diagnostics["optimizer_fallback"] = fallback
    return best


def _boundary_warnings(parameters: dict[str, float], observations: np.ndarray, threshold: float) -> tuple[str, ...]:
    """Flag extreme transformed estimates relative to the observed loss scale.

    These are diagnostics, not proof that a finite maximum fails to exist.
    An optimizer can report convergence even on a nearly flat limiting ridge.

    Args:
        parameters: Distribution parameters in their natural coordinates.
        observations: Observed values used in the likelihood or empirical estimate.
        threshold: Reporting threshold for fitting, or attachment for mean excess.
    """
    scale = max(float(np.median(np.abs(observations))), abs(threshold), np.finfo(float).tiny)
    messages = []
    positive_shapes = {"alpha", "beta", "shape", "power", "p", "q", "a", "nu"}
    scales = {"scale", "theta", "b", "lambda_"}
    for name, value in parameters.items():
        if not np.isfinite(value):
            messages.append(f"Non-finite parameter: {name}")
        elif name in scales and value > 0 and abs(np.log(value) - np.log(scale)) > 30:
            messages.append(f"{name} is extreme relative to the observation scale; likelihood may be on a ridge")
        elif name in positive_shapes and value > 0 and abs(np.log(value)) > 30:
            messages.append(f"{name} is extreme; likelihood may be on a limiting boundary")
    return tuple(messages)


def _with_uncertainty(
    fitted: FitResult[_T],
    observations: t.Sequence[t.Any] | np.ndarray,
    *,
    uncertainty_method: UncertaintyMethod,
    censored: t.Sequence[bool] | np.ndarray | None = None,
    weights: t.Sequence[float] | np.ndarray | None = None,
    threshold_specified: bool = False,
) -> FitResult[_T]:
    """Attach optional uncertainty without changing the MLE fitting result.

    Args:
        fitted: Previously fitted distribution and its likelihood diagnostics.
        observations: Observed values used in the likelihood or empirical estimate.
        uncertainty_method: Method for estimating joint parameter uncertainty; none retains point estimates.
        censored: Indicators that observations are right-censored at their recorded values.
        weights: Nonnegative observation weights; zero-weight observations are excluded.
        threshold_specified: Whether a reporting threshold was supplied to the fit.
    """
    method = UncertaintyMethod(uncertainty_method)
    if method is UncertaintyMethod.NONE:
        return fitted
    from .fitting_uncertainty import asymptotic_parameter_uncertainty

    result = asymptotic_parameter_uncertainty(
        fitted,
        observations,
        censored=censored,
        weights=weights,
        threshold_specified=threshold_specified,
    )
    return replace(fitted, uncertainty=result)


def _gamma_log_survival_underflow(shape: float, z: float) -> float:
    """Evaluate log Q(shape, z) after the ordinary gamma survival underflows.

    The upper incomplete-gamma continued fraction converges in this regime.
    The computation stays on the CPU because fitting is deterministic and
    does not depend on PAL's simulation backend.

    Args:
        shape: Gamma shape parameter.
        z: Unit-scale threshold beyond the ordinary survival underflow point.
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
    """Stable log survival of a unit-scale Gamma, including underflowed tails.

    Args:
        shape: Gamma shape parameter.
        z: Unit-scale gamma evaluation points.
    """
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


def _prepare(
    observations: t.Sequence[float] | np.ndarray,
    threshold: float,
    censored: t.Sequence[bool] | np.ndarray | None,
    weights: t.Sequence[float] | np.ndarray | None,
    *,
    allow_any_real: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and align loss observations, censoring flags and weights.

    Args:
        observations: Observed losses before weight filtering.
        threshold: Left-truncation threshold.
        censored: Indicators of right-censoring.
        weights: Nonnegative observation weights.
        allow_any_real: Allow signed observations for families with real-valued support.

    Returns:
        Validated observations, censoring flags and weights after zero-weight filtering.
    """
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
    """Return a conditional right-censored, left-truncated log-likelihood.

    Args:
        family: PAL distribution class selecting the mathematical parameterisation.
        parameters: Distribution parameters in their natural coordinates.
        x: Loss values at which to evaluate the calculation.
        c: Right-censoring indicators aligned with the observations.
        w: Observation weights aligned with the observations.
        threshold: Reporting threshold for fitting, or attachment for mean excess.
    """
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
    optimizer: OptimizerMethod | str = OptimizerMethod.L_BFGS_B,
    optimizer_options: t.Mapping[str, t.Any] | None = None,
    initial_parameters: t.Mapping[str, float] | None = None,
    uncertainty_method: UncertaintyMethod = UncertaintyMethod.NONE,
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
        optimizer: Numerical method for non-analytical fits.
        optimizer_options: Options passed to the selected SciPy optimizer.
        initial_parameters: Optional natural starting estimates for a numerical fit. Unspecified parameters
            use automatic estimates; fixed parameters cannot be changed.
        uncertainty_method: Optional joint parameter uncertainty method; defaults to None.

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
    if UncertaintyMethod(uncertainty_method) is not UncertaintyMethod.NONE:
        fitted = fit_distribution(
            distribution,
            observations,
            threshold=threshold,
            censored=censored,
            weights=weights,
            free_location=free_location,
            optimizer=optimizer,
            optimizer_options=optimizer_options,
            initial_parameters=initial_parameters,
        )
        return _with_uncertainty(
            fitted,
            observations,
            uncertainty_method=uncertainty_method,
            censored=censored,
            weights=weights,
            threshold_specified=bool(threshold > 0),
        )

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
    if initial_parameters is not None and family in (_distributions.Pareto, _distributions.Exponential):
        raise ValueError(f"{family.__name__} uses an analytical fit and does not require initial_parameters")
    if free_location:
        return _fit_gpd_free_location(
            distribution, x, c, w, threshold, optimizer, optimizer_options, initial_parameters
        )
    if family is _distributions.Exponential:
        return _fit_exponential(distribution, x, c, w, threshold)
    if family is _distributions.GeneralizedBetaPrime:
        return _fit_gb2(distribution, x, c, w, threshold, optimizer, optimizer_options, initial_parameters)
    if family in _other_families():
        return _fit_other(distribution, x, c, w, threshold, optimizer, optimizer_options, initial_parameters)
    total = float(np.sum(w))
    mean = float(np.average(x, weights=w))
    variance = float(np.average((x - mean) ** 2, weights=w))
    optimizer_used: str | None = None
    optimizer_fallback = False
    if family is _distributions.Pareto:
        scale = threshold if threshold > 0 else float(x.min())
        denominator = float(np.sum(w * np.log(x / scale)))
        if denominator <= 0 or not np.isfinite(denominator):
            raise ValueError("Pareto shape is not identifiable for these observations")
        parameters = {"shape": float(np.sum(w[~c])) / denominator, "scale": scale}
        converged = True
    else:
        largest = float(np.max(x)) - threshold
        if family is _distributions.LogNormal:
            logs = np.log(x)
            mu0 = float(np.average(logs, weights=w))
            sigma0 = float(np.sqrt(np.average((logs - mu0) ** 2, weights=w)))
            initial = [mu0, np.log(max(sigma0, 0.01))]
            alternative = None

            def transform(p: np.ndarray) -> dict[str, float]:
                """Convert trial optimisation coordinates to distribution parameters.

                Args:
                    p: Distribution parameter values in their natural coordinates.
                """
                return {"mu": float(p[0]), "sigma": float(np.exp(p[1]))}
        elif family is _distributions.Gamma:
            alpha0 = max(min(mean * mean / max(variance, np.finfo(float).tiny), 1e4), 0.05)
            initial = [np.log(alpha0), np.log(mean / alpha0)]
            alternative = np.array([0.0, np.log(mean)])

            def transform(p: np.ndarray) -> dict[str, float]:
                """Convert trial optimisation coordinates to distribution parameters.

                Args:
                    p: Distribution parameter values in their natural coordinates.
                """
                return {"alpha": float(np.exp(p[0])), "theta": float(np.exp(p[1]))}
        else:
            exceedances = x - threshold
            xi0, sigma0 = _gpd_initial(exceedances, c, w)
            initial = [_gpd_shape_coordinate(xi0, sigma0, largest), np.log(sigma0)]
            alternative = np.array([0.0, np.log(float(np.dot(w, exceedances)) / np.sum(w[~c]))])

            def transform(p: np.ndarray) -> dict[str, float]:
                """Convert trial optimisation coordinates to distribution parameters.

                Args:
                    p: Distribution parameter values in their natural coordinates.
                """
                scale = float(np.exp(p[1]))
                shape = _support_shape(float(p[0]), scale, 0.0, largest) if scale > 0 else np.nan
                return {"shape": shape, "scale": scale, "loc": threshold}

        def objective(p: np.ndarray) -> float:
            """Evaluate the negative log-likelihood in trial coordinates.

            Args:
                p: Distribution parameter values in their natural coordinates.
            """
            with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
                ll = _log_likelihood(family, transform(p), x, c, w, threshold)
            return -ll if np.isfinite(ll) else 1e100

        if family is _distributions.GPD:
            free_names = ("shape", "scale")

            def encode(params: dict[str, float]) -> np.ndarray:
                """Encode natural GPD estimates.

                Args:
                    params: Natural GPD parameters.
                """
                shape = _shape_coordinate(params["shape"], params["scale"], 0.0, largest)
                return np.array([shape, np.log(params["scale"])])

        else:
            free_names = ("mu", "sigma") if family is _distributions.LogNormal else ("alpha", "theta")

            def encode(params: dict[str, float]) -> np.ndarray:
                """Encode ordinary severity estimates.

                Args:
                    params: Natural severity parameters.
                """
                return _initial_coordinates(params, free_names, linear=("mu",))

        start, alternative = _override_initial(
            np.asarray(initial), alternative, initial_parameters, transform, encode, free_names, objective
        )
        best = _minimize(objective, start, optimizer, optimizer_options, 5000, alternative=alternative)
        if not np.isfinite(best.fun) or best.fun >= 1e100:
            raise RuntimeError(f"Unable to obtain a finite {family.__name__} fit")
        parameters = transform(best.x)
        converged = bool(best.success)
        optimizer_used = str(best.optimizer_used)
        optimizer_fallback = bool(best.optimizer_fallback)
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
        n_parameters=k,
        optimizer=optimizer_used,
        optimizer_fallback=optimizer_fallback,
        warnings=() if family is _distributions.Pareto else _boundary_warnings(parameters, x, threshold),
        distribution=model,
        parameters=parameters,
        log_likelihood=ll,
        aic=aic,
        aicc=aicc,
        converged=converged,
        n_observations=len(x),
        n_censored=int(np.sum(c)),
        threshold=threshold,
        threshold_specified=threshold > 0,
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
    """Build an independent SciPy reference distribution for inference.

    Args:
        family: PAL distribution class selecting the mathematical parameterisation.
        params: Distribution parameter values in the family's natural coordinates.
    """
    from . import distributions as _distributions

    p = params
    if family is _distributions.Pareto:
        return stats.pareto(b=p["shape"], scale=p["scale"])
    if family is _distributions.GPD:
        return stats.genpareto(c=p["shape"], loc=p["loc"], scale=p["scale"])
    if family is _distributions.LogNormal:
        return stats.lognorm(s=p["sigma"], scale=np.exp(p["mu"]))
    if family is _distributions.Gamma:
        return stats.gamma(a=p["alpha"], scale=p["theta"], loc=p.get("loc", 0.0))
    if family is _distributions.Weibull:
        return stats.weibull_min(c=p["shape"], scale=p["scale"], loc=p["loc"])
    if family is _distributions.Burr:
        return _StableSeverityReference(
            stats.burr12(c=p["power"], d=p["shape"], scale=p["scale"], loc=p["loc"]),
            lambda x: _burr_terms(x, p["power"], p["shape"], p["scale"], p["loc"]),
        )
    if family is _distributions.LogLogistic:
        return _StableSeverityReference(
            stats.fisk(c=p["shape"], scale=p["scale"], loc=p["loc"]),
            lambda x: _burr_terms(x, p["shape"], 1.0, p["scale"], p["loc"]),
        )
    if family is _distributions.InverseGaussian:
        return stats.invgauss(mu=p["mu"] / p["lambda_"], scale=p["lambda_"])
    if family is _distributions.Paralogistic:
        return _StableSeverityReference(
            stats.burr12(c=p["shape"], d=p["shape"], scale=p["scale"], loc=p["loc"]),
            lambda x: _burr_terms(x, p["shape"], p["shape"], p["scale"], p["loc"]),
        )
    if family is _distributions.InverseBurr:
        return _StableSeverityReference(
            stats.burr(c=p["power"], d=p["shape"], scale=p["scale"], loc=p["loc"]),
            lambda x: _burr_terms(x, p["power"], p["shape"], p["scale"], p["loc"], inverse=True),
        )
    if family is _distributions.InverseGamma:
        return stats.invgamma(a=p["alpha"], scale=p["theta"], loc=p["loc"])
    if family is _distributions.InverseWeibull:
        return _StableSeverityReference(
            stats.invweibull(c=p["shape"], scale=p["scale"], loc=p["loc"]),
            lambda x: _inverse_weibull_terms(x, p["shape"], p["scale"], p["loc"]),
        )
    if family is _distributions.Exponential:
        return stats.expon(scale=p["scale"], loc=p["loc"])
    if family is _distributions.InverseExponential:
        return _StableSeverityReference(
            stats.invweibull(c=1.0, scale=p["scale"], loc=p["loc"]),
            lambda x: _inverse_weibull_terms(x, 1.0, p["scale"], p["loc"]),
        )
    if family is _distributions.InverseParalogistic:
        return _StableSeverityReference(
            stats.burr(c=p["shape"], d=p["shape"], scale=p["scale"], loc=p["loc"]),
            lambda x: _burr_terms(x, p["shape"], p["shape"], p["scale"], p["loc"], inverse=True),
        )
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
    optimizer: OptimizerMethod | str,
    optimizer_options: t.Mapping[str, t.Any] | None,
    initial_parameters: t.Mapping[str, float] | None,
) -> FitResult[_T]:
    """Fit additional two- and three-parameter severity distributions.

    Args:
        distribution: PAL distribution class or instance used for the calculation.
        x: Loss values at which to evaluate the calculation.
        censored: Indicators that observations are right-censored at their recorded values.
        weights: Nonnegative observation weights; zero-weight observations are excluded.
        threshold: Reporting threshold for fitting, or attachment for mean excess.
        optimizer: Numerical optimiser used for non-analytical fitting.
        optimizer_options: Options passed to the selected SciPy optimiser.
        initial_parameters: Optional natural starting estimates.
    """
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

    alternative: np.ndarray | None = None
    # Standardise the GEV location coordinate to the scale of the data.
    spread = max(float(np.sqrt(variance)), np.finfo(float).eps * max(abs(median), 1.0))
    smallest, largest = float(np.min(x)), float(np.max(x))
    if family is _distributions.GeneralizedInverseGaussian:
        lam = max(mean**3 / max(variance, np.finfo(float).tiny), np.finfo(float).tiny)
        initial = [-0.5, np.log(lam), np.log(lam) - 2 * np.log(mean)]
        alternative = np.array([0.5, np.log(scale), -np.log(scale)])
    elif family is _distributions.GEV:
        gumbel_scale = spread * np.sqrt(6) / np.pi
        gumbel_loc = mean - np.euler_gamma * gumbel_scale
        initial = [0.0, np.log(gumbel_scale), (gumbel_loc - median) / spread]
        alternative = np.array([0.0, np.log(spread), 0.0])
    elif family in (_distributions.Exponential, _distributions.InverseExponential):
        initial = [np.log(max(mean, 1e-10))]
    elif family is _distributions.LogGamma:
        if np.any(x < 1.0):
            raise ValueError("log-gamma observations must be at least one")
        logs = np.log(x)
        log_mean = float(np.average(logs, weights=weights))
        log_var = float(np.average((logs - log_mean) ** 2, weights=weights))
        alpha = max(log_mean**2 / max(log_var, np.finfo(float).tiny), 0.1)
        beta = max(log_var / max(log_mean, np.finfo(float).tiny), 0.01)
        initial = [np.log(alpha), np.log(beta)]
        alternative = np.zeros(2)
    elif family is _distributions.InverseGaussian:
        lam = max(mean**3 / max(variance, np.finfo(float).tiny), np.finfo(float).tiny)
        initial = [np.log(mean), np.log(lam)]
        alternative = np.array([np.log(mean), np.log(mean)])
    elif family is _distributions.InverseGamma:
        alpha = max(mean * mean / max(variance, np.finfo(float).tiny) + 2.0, 1.5)
        initial = [np.log(alpha), np.log(max(mean * (alpha - 1.0), 1e-12))]
        alternative = np.array([0.0, np.log(scale)])
    else:
        center, iqr = _log_location_scale(x, censored, weights)
        logistic_shape = float(np.clip(2 * np.log(3) / iqr, 0.05, 100.0))
        if family in (_distributions.Burr, _distributions.InverseBurr):
            initial = [np.log(logistic_shape), 0.0, center]
            alternative = np.array([np.log(1.5), np.log(1.5), np.log(scale)])
        elif family in (_distributions.Weibull, _distributions.InverseWeibull):
            logs = np.log(x)
            log_mean = float(np.average(logs, weights=weights))
            log_std = float(np.sqrt(np.average((logs - log_mean) ** 2, weights=weights)))
            shape = float(np.clip(np.pi / (np.sqrt(6) * max(log_std, 1e-4)), 0.05, 100.0))
            sign = 1 if family is _distributions.Weibull else -1
            initial = [np.log(shape), log_mean + sign * np.euler_gamma / shape]
            alternative = np.array([0.0, np.log(scale)])
        else:
            initial = [np.log(logistic_shape), center]
            alternative = np.array([0.0, np.log(scale)])

    def make_params(raw: np.ndarray) -> dict[str, float]:
        """Convert trial optimisation coordinates to distribution parameters.

        Args:
            raw: Trial parameters in unconstrained optimisation coordinates.
        """
        if family is _distributions.GeneralizedInverseGaussian:
            return {"p": float(raw[0]), "chi": float(np.exp(raw[1])), "psi": float(np.exp(raw[2])), "loc": 0.0}
        if family is _distributions.GEV:
            gev_scale = float(np.exp(raw[1]))
            loc = float(median + spread * raw[2])
            shape = _support_shape(float(raw[0]), gev_scale, loc - smallest, largest - loc) if gev_scale > 0 else np.nan
            return {"shape": shape, "scale": gev_scale, "loc": loc}
        pars: dict[str, float] = dict(zip(names, (float(np.exp(value)) for value in raw), strict=False))
        if family not in (_distributions.InverseGaussian, _distributions.LogGamma):
            pars["loc"] = 0.0
        return pars

    def loglik(pars: dict[str, float]) -> float:
        """Evaluate the weighted conditional log-likelihood.

        Args:
            pars: Distribution parameter values in the family's natural coordinates.
        """
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
        elif family is _distributions.GeneralizedInverseGaussian:
            frozen = _frozen_distribution(family, pars)
            terms = np.empty_like(x)
            terms[~censored] = frozen.logpdf(x[~censored])
            # Evaluate the reporting threshold and distinct censoring limits
            # together. SciPy's generic survival integrates each point anew.
            points = np.concatenate((np.array([threshold]), x[censored]))
            root_chi, root_psi = np.sqrt(pars["chi"]), np.sqrt(pars["psi"])
            _, survival = _gig_log_probabilities(points / (root_chi / root_psi), pars["p"], root_chi * root_psi)
            base = float(survival[0])
            terms[censored] = survival[1:]
            terms -= base
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
        """Evaluate the negative log-likelihood in trial coordinates.

        Args:
            raw: Trial parameters in unconstrained optimisation coordinates.
        """
        with np.errstate(all="ignore"):
            val = loglik(make_params(raw))
        return -val if np.isfinite(val) else 1e100

    def encode(params: dict[str, float]) -> np.ndarray:
        """Encode natural estimates, including support-safe GEV coordinates.

        Args:
            params: Natural distribution parameters.
        """
        if family is _distributions.GEV:
            loc, scale = params["loc"], params["scale"]
            shape = _shape_coordinate(params["shape"], scale, loc - smallest, largest - loc)
            return np.array([shape, np.log(scale), (loc - median) / spread])
        linear = ("p",) if family is _distributions.GeneralizedInverseGaussian else ()
        return _initial_coordinates(params, names, linear=linear)

    start, alternative = _override_initial(
        np.asarray(initial), alternative, initial_parameters, make_params, encode, names, objective
    )
    best = _minimize(objective, start, optimizer, optimizer_options, 5000, alternative=alternative)
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
        n_parameters=n_params,
        optimizer=str(best.optimizer_used),
        optimizer_fallback=bool(best.optimizer_fallback),
        warnings=_boundary_warnings(pars, x, threshold),
        distribution=model,
        parameters=pars,
        log_likelihood=ll,
        aic=aic,
        aicc=aicc,
        converged=bool(best.success),
        n_observations=len(x),
        n_censored=int(np.sum(censored)),
        threshold=threshold,
        threshold_specified=threshold > 0,
    )


def empirical_mean_excess(
    observations: t.Sequence[float] | np.ndarray,
    threshold: float | np.ndarray,
) -> float | np.ndarray:
    """Empirical mean excess for uncensored observations, with strict exceedances.

    Sort once and accumulate tail excesses using positive adjacent spacings.
    Evaluation takes O(n log n + m log n) time and O(n + m) storage for n
    observations and m thresholds. No exceedances (or nonfinite thresholds)
    produce NaN. Threshold order and shape are preserved.

    Args:
        observations: Observed values used in the likelihood or empirical estimate.
        threshold: Reporting threshold for fitting, or attachment for mean excess.
    """
    x = np.asarray(observations, dtype=float)
    if x.ndim != 1 or not x.size or not np.all(np.isfinite(x)):
        raise ValueError("observations must be a nonempty finite one-dimensional array")
    x = np.sort(x)
    ds = np.asarray(threshold, dtype=float)
    result = np.full(ds.shape, np.nan, dtype=float)
    indices = np.searchsorted(x, ds, side="right")
    valid = np.isfinite(ds) & (indices < x.size)
    i = indices[valid]
    # A sum of positive spacings avoids cancellation of sum(x) - n*d.
    counts = np.arange(x.size - 1, 0, -1, dtype=float)
    # Recurrence in mean units would avoid overflow but requires a Python
    # loop; divide each spacing by n first to protect the cumulative sum.
    areas = np.empty(x.size, dtype=float)
    areas[-1] = 0.0
    fractions = counts / x.size
    with np.errstate(over="ignore"):
        spacing = np.diff(x)
        weighted = spacing * fractions
        overflow = np.isinf(spacing)
        weighted[overflow] = x[1:][overflow] * fractions[overflow] - x[:-1][overflow] * fractions[overflow]
        areas[:-1] = np.cumsum(weighted[::-1])[::-1]
    result[valid] = x[i] - ds[valid] + areas[i] / ((x.size - i) / x.size)
    return float(result) if result.ndim == 0 else result


def mean_excess(distribution: DistributionBase, threshold: float | np.ndarray) -> float | np.ndarray:
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
    from ._mean_excess import severity_mean_excess

    return severity_mean_excess(distribution, threshold)


def _fit_gpd_free_location(
    distribution: type[_T],
    x: np.ndarray,
    c: np.ndarray,
    w: np.ndarray,
    threshold: float,
    optimizer: OptimizerMethod | str,
    optimizer_options: t.Mapping[str, t.Any] | None,
    initial_parameters: t.Mapping[str, float] | None,
) -> FitResult[_T]:
    """Estimate GPD shape, scale and location subject to support constraints.

    Args:
        distribution: PAL distribution class or instance used for the calculation.
        x: Loss values at which to evaluate the calculation.
        c: Right-censoring indicators aligned with the observations.
        w: Observation weights aligned with the observations.
        threshold: Reporting threshold for fitting, or attachment for mean excess.
        optimizer: Numerical optimiser used for non-analytical fitting.
        optimizer_options: Options passed to the selected SciPy optimiser.
        initial_parameters: Optional natural starting estimates.
    """
    smallest = float(np.min(x))
    delta = max(smallest * 1e-8, 1e-9)
    max_loc = smallest - delta
    average_excess = float(np.average(x - threshold, weights=w))
    start_scale = max(average_excess, delta)
    base_loc = min(threshold, max_loc - delta)

    def transform(raw: np.ndarray) -> dict[str, float]:
        """Convert trial optimisation coordinates to distribution parameters.

        Args:
            raw: Trial parameters in unconstrained optimisation coordinates.
        """
        # Keeping the location below the smallest observation avoids invalid
        # density evaluations. The truncation correction is retained.
        scale = float(np.exp(raw[1]))
        loc = float(max_loc - np.exp(raw[2]))
        shape = _support_shape(float(raw[0]), scale, 0.0, float(np.max(x)) - loc) if scale > 0 else np.nan
        return {"shape": shape, "scale": scale, "loc": loc}

    def objective(raw: np.ndarray) -> float:
        """Evaluate the negative log-likelihood in trial coordinates.

        Args:
            raw: Trial parameters in unconstrained optimisation coordinates.
        """
        with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
            ll = _log_likelihood(distribution, transform(raw), x, c, w, threshold)
        return -ll if np.isfinite(ll) else 1e100

    distance = max(max_loc - base_loc, delta)
    xi0, sigma0 = _gpd_initial(x - base_loc, c, w)
    largest = float(np.max(x)) - base_loc
    initial = np.array([_gpd_shape_coordinate(xi0, sigma0, largest), np.log(sigma0), np.log(distance)])
    alternative = np.array([0.0, np.log(start_scale), np.log(distance)])

    def encode(params: dict[str, float]) -> np.ndarray:
        """Encode free-location GPD estimates.

        Args:
            params: Natural GPD shape, scale and location.
        """
        loc, scale = params["loc"], params["scale"]
        if loc >= max_loc:
            raise ValueError("initial GPD location must be below the smallest observation")
        shape = _shape_coordinate(params["shape"], scale, 0.0, float(np.max(x)) - loc)
        return np.array([shape, np.log(scale), np.log(max_loc - loc)])

    start, alternative = _override_initial(
        initial, alternative, initial_parameters, transform, encode, ("shape", "scale", "loc"), objective
    )
    best = _minimize(objective, start, optimizer, optimizer_options, 5000, alternative=alternative)
    if best.fun >= 1e100 or not np.isfinite(best.fun):
        raise RuntimeError("Unable to obtain a finite three-parameter GPD fit")
    params = transform(best.x)
    ll = _log_likelihood(distribution, params, x, c, w, threshold)
    n = float(np.sum(w))
    aic = 6.0 - 2.0 * ll
    aicc = aic + 24.0 / (n - 4) if np.all(w == np.floor(w)) and n > 4 else np.nan
    return FitResult(
        n_parameters=3,
        optimizer=str(best.optimizer_used),
        optimizer_fallback=bool(best.optimizer_fallback),
        distribution=distribution(**params),
        parameters=params,
        log_likelihood=ll,
        aic=aic,
        aicc=aicc,
        converged=bool(best.success),
        n_observations=len(x),
        n_censored=int(np.sum(c)),
        threshold=threshold,
        threshold_specified=threshold > 0,
    )


def _fit_exponential(
    distribution: type[_T],
    x: np.ndarray,
    censored: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> FitResult[_T]:
    """Use the exact censored MLE for an exponential above the threshold.

    Args:
        distribution: PAL distribution class or instance used for the calculation.
        x: Loss values at which to evaluate the calculation.
        censored: Indicators that observations are right-censored at their recorded values.
        weights: Nonnegative observation weights; zero-weight observations are excluded.
        threshold: Reporting threshold for fitting, or attachment for mean excess.
    """
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
        n_parameters=1,
        distribution=distribution(**params),
        parameters=params,
        log_likelihood=ll,
        aic=aic,
        aicc=aicc,
        converged=True,
        n_observations=len(x),
        n_censored=int(np.sum(censored)),
        threshold=threshold,
        threshold_specified=threshold > 0,
        method=_FitMethod.ANALYTIC.value,
    )


def _gb2_log_regularized_beta(alpha: float, beta: float, log_z: np.ndarray) -> np.ndarray:
    """Stable log I_z(alpha, beta), including probabilities below float underflow.

    Args:
        alpha: First shape parameter of the gamma or beta family.
        beta: Second shape or gamma scale parameter, according to the family.
        log_z: Logarithm of the dimensionless transformed loss coordinate.
    """
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


def _gb2_logpdf(log_x: np.ndarray, pars: dict[str, float]) -> np.ndarray:
    """Evaluate the GB2 log density without calculating survival.

    Args:
        log_x: Natural logarithms of positive losses.
        pars: Natural GB2 parameters a, b, p and q.

    Returns:
        Log densities in the original loss units.
    """
    a, b, p, q = (pars[name] for name in ("a", "b", "p", "q"))
    log_ratio = log_x - np.log(b)
    log_power = a * log_ratio
    log_pdf = np.log(a) - np.log(b) - special.betaln(p, q) + (a * p - 1) * log_ratio
    return log_pdf - (p + q) * np.logaddexp(0.0, log_power)


def _gb2_logsf(log_x: np.ndarray, pars: dict[str, float]) -> np.ndarray:
    """Evaluate the GB2 log survival without calculating density.

    Args:
        log_x: Natural logarithms of positive losses.
        pars: Natural GB2 parameters a, b, p and q.

    Returns:
        Log survival probabilities, including representable logs of underflowed probabilities.
    """
    a, b, p, q = (pars[name] for name in ("a", "b", "p", "q"))
    log_survival_argument = -np.logaddexp(0.0, a * (log_x - np.log(b)))
    return _gb2_log_regularized_beta(q, p, log_survival_argument)


def _gb2_terms(x: np.ndarray, pars: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Log-density and log-survival for the generalised beta prime family.

    Args:
        x: Positive losses in original loss units.
        pars: Natural GB2 parameters a, b, p and q.
    """
    log_x = np.log(x)
    return _gb2_logpdf(log_x, pars), _gb2_logsf(log_x, pars)


def _fit_gb2(
    distribution: type[_T],
    x: np.ndarray,
    censored: np.ndarray,
    weights: np.ndarray,
    threshold: float,
    optimizer: OptimizerMethod | str,
    optimizer_options: t.Mapping[str, t.Any] | None,
    initial_parameters: t.Mapping[str, float] | None,
) -> FitResult[_T]:
    """Fit the four-parameter GB2 distribution in log-parameter coordinates.

    Args:
        distribution: PAL distribution class or instance used for the calculation.
        x: Loss values at which to evaluate the calculation.
        censored: Indicators that observations are right-censored at their recorded values.
        weights: Nonnegative observation weights; zero-weight observations are excluded.
        threshold: Reporting threshold for fitting, or attachment for mean excess.
        optimizer: Numerical optimiser used for non-analytical fitting.
        optimizer_options: Options passed to the selected SciPy optimiser.
        initial_parameters: Optional natural starting estimates.
    """
    center, iqr = _log_location_scale(x, censored, weights)
    shape = float(np.clip(2 * np.log(3) / iqr, 0.05, 100.0))
    initial = np.array([np.log(shape), center, 0.0, 0.0])
    alternative = np.array([0.0, np.log(float(np.median(x))), 0.0, np.log(2.0)])

    def transform(params: np.ndarray) -> dict[str, float]:
        """Convert trial optimisation coordinates to distribution parameters.

        Args:
            params: Distribution parameter values in the family's natural coordinates.
        """
        return dict(zip(("a", "b", "p", "q"), (float(np.exp(v)) for v in params), strict=False))

    log_x = np.log(x)
    log_observed = log_x[~censored]
    observed_weights = weights[~censored]
    # Policy-limit censoring usually repeats the same loss value. Compress
    # those terms once before optimisation, retaining their total weights.
    log_capped, inverse = np.unique(log_x[censored], return_inverse=True)
    capped_weights = np.bincount(inverse, weights=weights[censored], minlength=log_capped.size)
    log_threshold = np.array([np.log(threshold)]) if threshold > 0 else np.empty(0)
    survival_points = np.concatenate((log_threshold, log_capped))

    def likelihood(params: dict[str, float]) -> float:
        """Evaluate the likelihood at the supplied natural parameters.

        Args:
            params: Distribution parameter values in the family's natural coordinates.
        """
        density = _gb2_logpdf(log_observed, params)
        survival = _gb2_logsf(survival_points, params) if survival_points.size else np.empty(0)
        truncation = float(survival[0]) if log_threshold.size else 0.0
        ll = float(np.dot(observed_weights, density - truncation))
        ll += float(np.dot(capped_weights, survival[log_threshold.size :] - truncation))
        return ll if np.isfinite(ll) else -np.inf

    def objective(v: np.ndarray) -> float:
        """Evaluate the negative log-likelihood in trial coordinates.

        Args:
            v: Trial parameter vector in unconstrained optimisation coordinates.
        """
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            ll = likelihood(transform(v))
        return -ll if np.isfinite(ll) else 1e100

    names = ("a", "b", "p", "q")
    start, alternative = _override_initial(
        initial, alternative, initial_parameters, transform, lambda p: _initial_coordinates(p, names), names, objective
    )
    best = _minimize(objective, start, optimizer, optimizer_options, 6000, alternative=alternative)
    if not np.isfinite(best.fun) or best.fun >= 1e100:
        raise RuntimeError("No finite generalised beta prime fit")
    pars = transform(best.x)
    ll = likelihood(pars)
    total_weight = float(np.sum(weights))
    aic = 8 - 2 * ll
    frequency = bool(np.all(weights == np.floor(weights)))
    aicc = aic + 40 / (total_weight - 5) if frequency and total_weight > 5 else np.nan
    return FitResult(
        n_parameters=4,
        optimizer=str(best.optimizer_used),
        optimizer_fallback=bool(best.optimizer_fallback),
        warnings=_boundary_warnings(pars, x, threshold),
        distribution=distribution(**pars),
        parameters=pars,
        log_likelihood=ll,
        aic=aic,
        aicc=aicc,
        converged=bool(best.success),
        n_observations=len(x),
        n_censored=int(np.sum(censored)),
        threshold=threshold,
        threshold_specified=threshold > 0,
    )

"""Joint asymptotic parameter uncertainty for PAL maximum-likelihood fits.

The inverse observed-information approximation is local and large-sample.
It is not a calibrated predictive distribution or a Bayesian posterior.
"""

from __future__ import annotations

import typing as t
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from ._maths import to_backend, xp
from .config import config
from .couplings import couple_stochastic_results
from .fitting import FitResult, _prepare
from .qmc import _sample_rng
from .stochastic_scalar import StochasticScalar

if t.TYPE_CHECKING:
    from .distributions import DistributionBase

_T = t.TypeVar("_T", bound="DistributionBase")


def _coordinate_value(raw: t.Any, kind: str, origin: float, scale: float, module: t.Any) -> t.Any:
    """Transform one unconstrained coordinate back to its model parameter."""
    if kind == "log":
        return module.exp(raw)
    if kind == "logit":
        return 1.0 / (1.0 + module.exp(-raw))
    if kind == "shiftlog":
        return 1.0 + module.exp(raw)
    if kind == "lowerlog":
        return origin + module.exp(raw)
    return origin + scale * raw


@dataclass(frozen=True)
class AsymptoticParameterUncertainty(t.Generic[_T]):
    """Joint observed-information uncertainty in identifiable parameters.

    Covariance and standard errors use the original fitted coordinates.
    Parameter draws use continuous unconstrained coordinates, including
    log-positive and logit-probability transformations where appropriate.
    """

    fit: FitResult[_T]
    parameter_names: tuple[str, ...]
    covariance: np.ndarray
    transformed_covariance: np.ndarray
    _transformed_estimates: np.ndarray = field(repr=False)
    _transform_kinds: tuple[str, ...] = field(repr=False)
    _origins: np.ndarray = field(repr=False)
    _scales: np.ndarray = field(repr=False)
    _observed_min: float | None = field(default=None, repr=False)
    _observed_max: float | None = field(default=None, repr=False)

    @property
    def standard_errors(self) -> dict[str, float]:
        """Approximate marginal standard errors in original parameter units."""
        return dict(zip(self.parameter_names, np.sqrt(np.diag(self.covariance)).tolist(), strict=False))

    def confidence_intervals(self, level: float = 0.95) -> dict[str, tuple[float, float]]:
        """Marginal transformed Wald intervals (not predictive intervals)."""
        if not np.isfinite(level) or not 0 < level < 1:
            raise ValueError("level must lie strictly between zero and one")
        z = float(stats.norm.ppf((1.0 + level) / 2.0))
        deviations = z * np.sqrt(np.diag(self.transformed_covariance))
        result: dict[str, tuple[float, float]] = {}
        for i, name in enumerate(self.parameter_names):
            lower = _coordinate_value(
                self._transformed_estimates[i] - deviations[i],
                self._transform_kinds[i],
                float(self._origins[i]),
                float(self._scales[i]),
                np,
            )
            upper = _coordinate_value(
                self._transformed_estimates[i] + deviations[i],
                self._transform_kinds[i],
                float(self._origins[i]),
                float(self._scales[i]),
                np,
            )
            result[name] = float(lower), float(upper)
        return result

    def _valid_parameter_rows(self, raw_draws: t.Any) -> t.Any:
        """Validate *joint* transformed draws on the active simulation backend.

        Marginal bounds are enforced by bijections; additionally check finite
        machine-precision results and the parameter-dependent likelihood
        support for GPD/GEV. Invalid rows must be replaced as whole vectors.
        """
        from . import distributions as d

        valid = xp.ones(raw_draws.shape[0], dtype=bool)
        values_by_name: dict[str, t.Any] = {}
        for i, (name, kind) in enumerate(zip(self.parameter_names, self._transform_kinds, strict=False)):
            values = _coordinate_value(raw_draws[:, i], kind, float(self._origins[i]), float(self._scales[i]), xp)
            values_by_name[name] = values
            valid &= xp.isfinite(values)
            if kind == "log":
                valid &= values > 0.0
            elif kind == "logit":
                valid &= (values > 0.0) & (values < 1.0)
            elif kind == "shiftlog":
                valid &= values > 1.0
            elif kind == "lowerlog":
                valid &= values > float(self._origins[i])

        family = type(self.fit.distribution)
        if family is d.GPD and self._observed_max is not None:
            shape = values_by_name["shape"]
            scale = values_by_name["scale"]
            loc = float(self.fit.parameters["loc"])
            valid &= 1.0 + shape * (self._observed_max - loc) / scale > 0.0
        elif family is d.GEV and self._observed_min is not None and self._observed_max is not None:
            shape = values_by_name["shape"]
            scale = values_by_name["scale"]
            loc = values_by_name["loc"]
            valid &= 1.0 + shape * (self._observed_min - loc) / scale > 0.0
            valid &= 1.0 + shape * (self._observed_max - loc) / scale > 0.0
        return valid

    def parameter_samples(self, n_sims: int | None = None, rng: t.Any = None) -> dict[str, t.Any]:
        """Draw joint parameters, preserving bounds, correlations and coupling.

        Infeasible rows are resampled as *entire joint vectors*. Independent
        clipping or per-parameter rejection would destroy estimated dependence.
        """
        if n_sims is None:
            n_sims = config.n_sims
        if not isinstance(n_sims, int) or isinstance(n_sims, bool) or n_sims < 1:
            raise ValueError("n_sims must be a positive integer")
        if rng is None:
            rng = config.rng
        if rng is None:
            raise RuntimeError("No PAL random generator is configured")
        dimension = len(self.parameter_names)
        factor = to_backend(np.linalg.cholesky(self.transformed_covariance))
        center = to_backend(self._transformed_estimates)

        def draw_joint(count: int) -> t.Any:
            normals = xp.stack(
                [to_backend(_sample_rng(rng, "normal", count, 0.0, 1.0)) for _ in range(dimension)],
                axis=1,
            )
            return center + normals @ factor.T

        raw_draws = draw_joint(n_sims)
        rejected_total = 0
        for attempt in range(33):
            invalid = ~self._valid_parameter_rows(raw_draws)
            n_invalid = int(xp.count_nonzero(invalid))
            if n_invalid == 0:
                break
            if attempt == 32:
                raise ValueError("Unable to simulate admissible parameters from the asymptotic approximation")
            rejected_total += n_invalid
            raw_draws[invalid] = draw_joint(n_invalid)

        if rejected_total > 0.05 * n_sims:
            warnings.warn(
                "More than 5% of joint asymptotic parameter draws required rejection; "
                "the Gaussian approximation may be unreliable near parameter/support boundaries",
                RuntimeWarning,
                stacklevel=2,
            )

        parameters = dict(self.fit.parameters)
        stochastic = []
        for index, name in enumerate(self.parameter_names):
            values = _coordinate_value(
                raw_draws[:, index],
                self._transform_kinds[index],
                float(self._origins[index]),
                float(self._scales[index]),
                xp,
            )
            value = StochasticScalar(values)
            parameters[name] = value
            stochastic.append(value)
        # This *one* group contains every uncertain parameter. Distribution
        # simulation subsequently couples its output to all of these sources.
        couple_stochastic_results(stochastic, n_sims=n_sims)
        return parameters

    def sample_distribution(self, n_sims: int | None = None, rng: t.Any = None) -> _T:
        """Create the fitted PAL family with correlated stochastic parameters."""
        return type(self.fit.distribution)(**self.parameter_samples(n_sims, rng))


def _observed_hessian(objective: t.Callable[[np.ndarray], float], center: np.ndarray) -> np.ndarray:
    """Centered finite-difference Hessian in transformed coordinates."""
    size = len(center)
    steps = 5e-4 * (1.0 + np.abs(center))
    value = float(objective(center))
    hessian = np.empty((size, size), dtype=float)
    if not np.isfinite(value):
        raise ValueError("Fitted likelihood is not finite")
    for i in range(size):
        e_i = np.zeros(size)
        e_i[i] = steps[i]
        plus = float(objective(center + e_i))
        minus = float(objective(center - e_i))
        hessian[i, i] = (plus - 2.0 * value + minus) / steps[i] ** 2
        for j in range(i):
            e_j = np.zeros(size)
            e_j[j] = steps[j]
            cross = (
                objective(center + e_i + e_j)
                - objective(center + e_i - e_j)
                - objective(center - e_i + e_j)
                + objective(center - e_i - e_j)
            ) / (4.0 * steps[i] * steps[j])
            hessian[i, j] = hessian[j, i] = cross
    if not np.all(np.isfinite(hessian)):
        raise ValueError("Observed information is not finite near the fitted parameters")
    return hessian


def asymptotic_parameter_uncertainty(
    fit: FitResult[_T],
    observations: t.Sequence[float] | np.ndarray,
    *,
    censored: t.Sequence[bool] | np.ndarray | None = None,
    weights: t.Sequence[float] | np.ndarray | None = None,
    threshold_specified: bool = False,
) -> AsymptoticParameterUncertainty[_T]:
    """Estimate regular fitted-family uncertainty from observed information.

    Uses the original likelihood, censoring, truncation and frequency weights.
    Returns an error for unsupported nonregular estimators and for a singular
    or indefinite information matrix. Numerical fitting is always on CPU;
    subsequent PAL parameter draws use the configured simulation backend.
    """
    from . import distributions as d
    from .fitting import _other_families, _supported_families
    from .fitting_remaining import _input
    from .fitting_uncertainty_likelihood import _loglik, _parameter_names, _transform_types

    if not fit.converged:
        raise ValueError("Parameter uncertainty requires a converged maximum-likelihood fit")
    family = type(fit.distribution)
    params = fit.parameters
    names = _parameter_names(family, params, fit.threshold)
    kinds = _transform_types(family, names)
    is_severity = family in (*_supported_families(), *_other_families())

    if is_severity:
        x, c, w = _prepare(observations, fit.threshold, censored, weights, allow_any_real=family is d.GEV)
    else:
        x, w = _input(observations, weights)
        x = np.asarray(x, dtype=float)
        if censored is None:
            c = np.zeros(x.size, dtype=bool)
        else:
            source_weights = np.ones(len(observations)) if weights is None else np.asarray(weights, dtype=float)
            indicators = np.asarray(censored, dtype=bool)
            if indicators.shape != source_weights.shape:
                raise ValueError("censored must match observations")
            c = indicators[source_weights > 0]

    if not np.all(np.isfinite(x)):
        raise ValueError("observations must be finite")
    if not np.all(w == np.floor(w)):
        raise ValueError("Parameter covariance requires integer frequency weights")
    if float(np.sum(w)) <= len(names) + 2:
        raise ValueError("Too few observations to estimate parameter covariance")
    if any(
        (kind == "log" and float(params[name]) <= 0)
        or (kind == "logit" and not 0 < float(params[name]) < 1)
        or (kind == "shiftlog" and float(params[name]) <= 1)
        or (kind == "lowerlog" and float(params[name]) <= -0.5)
        for name, kind in zip(names, kinds, strict=False)
    ):
        raise ValueError("MLE is at a parameter boundary; Gaussian parameter uncertainty is invalid")

    # Convert unconstrained large-magnitude location parameters to
    # locally dimensionless coordinates, improving Hessian accuracy.
    origin_list: list[float] = []
    scale_list: list[float] = []
    transformed_list: list[float] = []
    jacobian_list: list[float] = []
    for name, kind in zip(names, kinds, strict=False):
        val = float(params[name])
        if kind == "log":
            origin_list.append(0.0)
            scale_list.append(1.0)
            transformed_list.append(float(np.log(val)))
            jacobian_list.append(val)
        elif kind == "logit":
            origin_list.append(0.0)
            scale_list.append(1.0)
            transformed_list.append(float(np.log(val / (1.0 - val))))
            jacobian_list.append(val * (1.0 - val))
        elif kind == "shiftlog":
            origin_list.append(0.0)
            scale_list.append(1.0)
            transformed_list.append(float(np.log(val - 1.0)))
            jacobian_list.append(val - 1.0)
        elif kind == "lowerlog":
            origin_list.append(-0.5)
            scale_list.append(1.0)
            transformed_list.append(float(np.log(val + 0.5)))
            jacobian_list.append(val + 0.5)
        else:
            reference = float(params.get("scale", params.get("sigma", 1.0)))
            if family is d.LogNormal and name == "mu":
                reference = float(params["sigma"])
            reference = max(abs(reference), 1e-9)
            origin_list.append(val)
            scale_list.append(reference)
            transformed_list.append(0.0)
            jacobian_list.append(reference)

    center = np.asarray(transformed_list)
    origins = np.asarray(origin_list)
    scales = np.asarray(scale_list)
    original_ll = _loglik(family, params, x, c, w, fit.threshold, threshold_specified)
    if not np.isclose(original_ll, fit.log_likelihood, rtol=1e-8, atol=1e-5):
        raise ValueError("Observations, censoring and weights must match the original fit")

    def objective(raw: np.ndarray) -> float:
        trial = dict(params)
        for i, name in enumerate(names):
            trial[name] = float(_coordinate_value(raw[i], kinds[i], origins[i], scales[i], np))
        with np.errstate(all="ignore"):
            ll = _loglik(family, trial, x, c, w, fit.threshold, threshold_specified)
        return -ll if np.isfinite(ll) else np.inf

    information = _observed_hessian(objective, center)
    eigenvalues = np.linalg.eigvalsh(information)
    if np.min(eigenvalues) <= max(np.max(eigenvalues), 1.0) * 1e-10:
        raise ValueError("Observed information is singular or not positive definite")
    transformed_covariance = np.linalg.inv(information)
    jacobian = np.diag(jacobian_list)
    covariance = jacobian @ transformed_covariance @ jacobian
    return AsymptoticParameterUncertainty(
        fit=fit,
        parameter_names=names,
        covariance=covariance,
        transformed_covariance=transformed_covariance,
        _transformed_estimates=center,
        _transform_kinds=kinds,
        _origins=origins,
        _scales=scales,
        _observed_min=float(np.min(x)) if family is d.GEV else None,
        _observed_max=float(np.max(x)) if family in (d.GPD, d.GEV) else None,
    )

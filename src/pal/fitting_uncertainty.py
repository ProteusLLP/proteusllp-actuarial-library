"""Joint asymptotic parameter uncertainty for PAL maximum-likelihood fits.

The inverse observed-information approximation is local and large-sample.
It is not a calibrated predictive distribution or a Bayesian posterior.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from ._maths import to_backend, xp
from .config import config
from .couplings import couple_stochastic_results
from .fitting import FitResult, _log_likelihood, _prepare
from .qmc import _sample_rng
from .stochastic_scalar import StochasticScalar

if t.TYPE_CHECKING:
    from .distributions import DistributionBase

_T = t.TypeVar("_T", bound="DistributionBase")


@dataclass(frozen=True)
class AsymptoticParameterUncertainty(t.Generic[_T]):
    """Approximate joint uncertainty, including parameter correlations.

    Covariance and standard errors use the original parameterisation. Joint
    parameter draws use unconstrained coordinates with log-positive variables.
    Fixed parameters remain deterministic.
    """

    fit: FitResult[_T]
    parameter_names: tuple[str, ...]
    covariance: np.ndarray
    transformed_covariance: np.ndarray
    _transformed_estimates: np.ndarray = field(repr=False)
    _positive_parameters: frozenset[str] = field(repr=False)

    @property
    def standard_errors(self) -> dict[str, float]:
        """Approximate marginal standard errors of estimated parameters."""
        return dict(zip(self.parameter_names, np.sqrt(np.diag(self.covariance)).tolist()))

    def confidence_intervals(self, level: float = 0.95) -> dict[str, tuple[float, float]]:
        """Marginal Wald intervals, transformed for positive parameters."""
        if not np.isfinite(level) or not 0 < level < 1:
            raise ValueError("level must lie strictly between zero and one")
        z = float(stats.norm.ppf((1.0 + level) / 2.0))
        deviations = z * np.sqrt(np.diag(self.transformed_covariance))
        result: dict[str, tuple[float, float]] = {}
        for i, name in enumerate(self.parameter_names):
            lower = self._transformed_estimates[i] - deviations[i]
            upper = self._transformed_estimates[i] + deviations[i]
            if name in self._positive_parameters:
                lower, upper = np.exp(lower), np.exp(upper)
            result[name] = float(lower), float(upper)
        return result

    def parameter_samples(self, n_sims: int | None = None, rng: t.Any = None) -> dict[str, t.Any]:
        """Joint parameter draws using PAL's CPU/GPU and MC/RQMC generator.

        The scenario-indexed parameters are coupled to preserve dependency
        through severity and frequency-severity simulation.
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
        normals = xp.stack(
            [_sample_rng(rng, "normal", n_sims, 0.0, 1.0) for _ in range(dimension)],
            axis=1,
        )
        factor = to_backend(np.linalg.cholesky(self.transformed_covariance))
        raw_draws = to_backend(self._transformed_estimates) + normals @ factor.T

        parameters = dict(self.fit.parameters)
        stochastic = []
        for index, name in enumerate(self.parameter_names):
            values = raw_draws[:, index]
            if name in self._positive_parameters:
                values = xp.exp(values)
            value = StochasticScalar(values)
            parameters[name] = value
            stochastic.append(value)
        couple_stochastic_results(stochastic, n_sims=n_sims)
        return parameters

    def sample_distribution(self, n_sims: int | None = None, rng: t.Any = None) -> _T:
        """Create the same PAL distribution with scenario-varying parameters."""
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
) -> AsymptoticParameterUncertainty[_T]:
    """Estimate joint covariance from the original fit likelihood.

    Initially supports Gamma, LogNormal, fixed-location GPD and Pareto with
    known positive scale (the truncation threshold). Right censoring, left
    truncation and integer frequency weights retain the original likelihood.

    The observations, censoring and weights must match the original fit.
    Free locations, estimated endpoints and nonregular GPD shapes are excluded.
    """
    from . import distributions

    if not fit.converged:
        raise ValueError("Parameter uncertainty requires a converged maximum-likelihood fit")
    family = type(fit.distribution)
    if family is distributions.Gamma:
        names = ("alpha", "theta")
        positive = frozenset(names)
    elif family is distributions.LogNormal:
        names = ("mu", "sigma")
        positive = frozenset(("sigma",))
    elif family is distributions.GPD:
        if not np.isclose(fit.parameters["loc"], fit.threshold, rtol=0, atol=1e-12):
            raise ValueError("Asymptotic GPD uncertainty requires a fixed location")
        if fit.parameters["shape"] <= -0.5:
            raise ValueError("The usual asymptotic GPD covariance requires shape > -0.5")
        names = ("shape", "scale")
        positive = frozenset(("scale",))
    elif family is distributions.Pareto:
        if fit.threshold <= 0 or not np.isclose(fit.parameters["scale"], fit.threshold, rtol=0, atol=1e-12):
            raise ValueError("Pareto asymptotic uncertainty requires a known scale (positive threshold)")
        names = ("shape",)
        positive = frozenset(names)
    else:
        raise NotImplementedError(f"Asymptotic parameter uncertainty is not yet implemented for {family.__name__}")

    x, c, w = _prepare(observations, fit.threshold, censored, weights)
    if not np.all(w == np.floor(w)):
        raise ValueError("Parameter covariance requires integer frequency weights")
    if float(np.sum(w)) <= len(names) + 2:
        raise ValueError("Too few observations to estimate parameter covariance")
    params = fit.parameters
    original_ll = _log_likelihood(family, params, x, c, w, fit.threshold)
    if not np.isclose(original_ll, fit.log_likelihood, rtol=1e-8, atol=1e-5):
        raise ValueError("Observations, censoring and weights must match the original fit")

    center = np.array(
        [np.log(float(params[name])) if name in positive else float(params[name]) for name in names],
        dtype=float,
    )

    def objective(raw: np.ndarray) -> float:
        trial = dict(params)
        for name, value in zip(names, raw):
            trial[name] = float(np.exp(value)) if name in positive else float(value)
        with np.errstate(all="ignore"):
            ll = _log_likelihood(family, trial, x, c, w, fit.threshold)
        return -ll if np.isfinite(ll) else np.inf

    information = _observed_hessian(objective, center)
    eigenvalues = np.linalg.eigvalsh(information)
    if np.min(eigenvalues) <= max(np.max(eigenvalues), 1.0) * 1e-10:
        raise ValueError("Observed information is singular or not positive definite")
    transformed_covariance = np.linalg.inv(information)
    jacobian = np.diag([float(params[name]) if name in positive else 1.0 for name in names])
    covariance = jacobian @ transformed_covariance @ jacobian
    return AsymptoticParameterUncertainty(
        fit=fit,
        parameter_names=names,
        covariance=covariance,
        transformed_covariance=transformed_covariance,
        _transformed_estimates=center,
        _positive_parameters=positive,
    )

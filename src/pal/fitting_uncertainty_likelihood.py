"""Likelihood evaluation for regular PAL fitted-distribution uncertainty.

This module follows the fitted families' original likelihood conventions and
handles frequency weights, censoring, and truncation. It is not a separate
fitting engine: the point estimates always come from the existing MLE fitters.
"""

from __future__ import annotations

import typing as t

import numpy as np
from scipy import stats

from .fitting import (
    _frozen_distribution,
    _gamma_logsf,
    _gb2_terms,
    _log_likelihood,
    _other_families,
    _supported_families,
)
from .fitting_remaining import _continuous_families, _discrete_families


def _parameter_names(family: type[t.Any], parameters: dict[str, t.Any], threshold: float) -> tuple[str, ...]:
    """Choose identifiable continuous parameters, excluding known constants."""
    from . import distributions as d

    if family is d.Uniform:
        raise ValueError("Uniform endpoint estimators are nonregular; no Gaussian inverse-Hessian approximation exists")
    if family is d.HyperGeometric:
        raise ValueError("HyperGeometric fitting estimates an integer count; Gaussian Hessian uncertainty is invalid")
    if family is d.Categorical:
        raise ValueError("Categorical probabilities require simplex-constrained uncertainty")
    if family is d.Pareto:
        if threshold <= 0 or not np.isclose(parameters["scale"], threshold, rtol=0, atol=1e-12):
            raise ValueError("Pareto uncertainty requires a known scale/lower endpoint (positive threshold)")
        return ("shape",)
    if family is d.GPD:
        if not np.isclose(parameters["loc"], threshold, rtol=0, atol=1e-12):
            raise ValueError("Three-parameter GPD with free location is nonregular")
        if parameters["shape"] <= -0.5:
            raise ValueError("Usual asymptotic GPD inference requires shape > -0.5")
        return ("shape", "scale")
    if family is d.Beta:
        return ("alpha", "beta")
    if family is d.Binomial or family is d.Bernoulli:
        return ("p",)
    if family is d.MBBEFD and parameters["g"] <= 1.0:
        raise ValueError("MBBEFD fitted g=1 is a nonregular boundary")
    if family is d.NonCentralChiSquared and parameters["nonc"] <= 0.0:
        raise ValueError("Noncentral chi-squared fitted noncentrality is at the boundary")
    if family is d.GEV:
        return ("shape", "scale", "loc")
    if family in (*_supported_families(), *_other_families(), *_continuous_families(), *_discrete_families(), d.MBBEFD):
        return tuple(name for name in parameters if name not in ("loc", "n", "n_draws", "ngood", "nbad"))
    raise NotImplementedError(f"Asymptotic uncertainty unavailable for {family.__name__}")


def _transform_types(family: type[t.Any], names: tuple[str, ...]) -> tuple[str, ...]:
    from . import distributions as d

    real = {"mu"} if family in (d.Normal, d.Logistic, d.LogNormal, d.StudentsT) else set()
    if family in (d.GPD, d.GEV):
        real.add("shape")
    if family is d.GeneralizedInverseGaussian:
        real.add("p")
    if family is d.GEV:
        real.add("loc")
    probability = {"p"} if family in (d.Binomial, d.Bernoulli, d.NegBinomial) else set()
    return tuple(
        "logit"
        if name in probability
        else "shiftlog"
        if family is d.MBBEFD and name == "g"
        else "linear"
        if name in real
        else "log"
        for name in names
    )


def _mbbefd_log_likelihood(pars: dict[str, float], x: np.ndarray, w: np.ndarray) -> float:
    """Exact mixed density/atom likelihood, including limiting cases."""
    g, b = pars["g"], pars["b"]
    if g <= 1 or b <= 0 or np.any((x <= 0) | (x > 1)):
        return -np.inf
    interior = x[x < 1]
    inner_weights = w[x < 1]
    atom_weight = float(np.sum(w[x == 1]))
    logb = float(np.log(b))
    if abs(logb) < 1e-7:
        density = (g - 1) / (1 + (g - 1) * interior) ** 2
    elif abs(g * b - 1) < 1e-7:
        density = -logb * np.exp(logb * interior)
    else:
        bx = np.exp(logb * interior)
        denominator = (g - 1) * b + (1 - g * b) * bx
        density = -(1 - b) * (g - 1) * logb * b * bx / denominator**2
    if np.any(density <= 0) or not np.all(np.isfinite(density)):
        return -np.inf
    return float(np.dot(inner_weights, np.log(density)) - atom_weight * np.log(g))


def _loglik(
    family: type[t.Any],
    params: dict[str, float],
    x: np.ndarray,
    c: np.ndarray,
    w: np.ndarray,
    threshold: float,
    threshold_specified: bool,
) -> float:
    """Re-evaluate the same conditional likelihood used for the fitted MLE."""
    from . import distributions as d

    if family in _supported_families():
        return _log_likelihood(family, params, x, c, w, threshold)
    if family in _other_families():
        if family is d.GeneralizedBetaPrime:
            density, survival = _gb2_terms(x, params)
            truncation = float(_gb2_terms(np.array([threshold]), params)[1][0]) if threshold > 0 else 0.0
            terms = np.where(c, survival, density) - truncation
            value = float(np.dot(w, terms))
            return value if np.isfinite(value) else -np.inf
        if family is d.LogGamma:
            alpha, beta = params["alpha"], params["beta"]
            lx = np.log(x)
            base = float(_gamma_logsf(alpha, np.log(threshold) / beta)) if threshold > 1 else 0.0
            terms = (
                np.where(
                    c,
                    _gamma_logsf(alpha, lx / beta),
                    stats.gamma.logpdf(lx, a=alpha, scale=beta) - lx,
                )
                - base
            )
            value = float(np.dot(w, terms))
            return value if np.isfinite(value) else -np.inf
        rv = _frozen_distribution(family, params)
        base = float(rv.logsf(threshold)) if threshold > 0 else 0.0
        terms = np.where(c, rv.logsf(x), rv.logpdf(x)) - base
    elif family in _continuous_families():
        if family is d.Beta:
            rv = stats.beta(params["alpha"], params["beta"], loc=params["loc"], scale=params["scale"])
        elif family is d.Normal:
            rv = stats.norm(loc=params["mu"], scale=params["sigma"])
        elif family is d.Logistic:
            rv = stats.logistic(loc=params["mu"], scale=params["sigma"])
        elif family is d.StudentsT:
            rv = stats.t(df=params["nu"], loc=params["mu"], scale=params["sigma"])
        else:
            rv = stats.ncx2(df=params["df"], nc=params["nonc"])
        base = float(rv.logsf(threshold)) if threshold_specified else 0.0
        terms = np.where(c, rv.logsf(x), rv.logpdf(x)) - base
    elif family in _discrete_families():
        if family is d.Poisson:
            rv = stats.poisson(params["mean"])
        elif family in (d.Binomial, d.Bernoulli):
            rv = stats.binom(1 if family is d.Bernoulli else params["n"], params["p"])
        elif family is d.NegBinomial:
            rv = stats.nbinom(params["n"], params["p"])
        else:
            raise ValueError("Integer HyperGeometric information requires a discrete method")
        terms = rv.logpmf(x)
    elif family is d.MBBEFD:
        return _mbbefd_log_likelihood(params, x, w)
    else:
        raise ValueError(f"No likelihood implementation for {family.__name__}")

    value = float(np.dot(w, terms))
    return value if np.isfinite(value) else -np.inf

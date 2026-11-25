"""Independent likelihood references for less frequently fitted families."""

import numpy as np
import pytest
from scipy import stats

from pal import distributions as d
from pal.fitting_uncertainty_likelihood import _loglik, _mbbefd_log_likelihood, _parameter_names
from pal.maths import asnumpy


@pytest.mark.parametrize("family", [d.LogGamma, d.GeneralizedBetaPrime, d.StudentsT, d.NonCentralChiSquared])
@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize("censored", [False, True])
@pytest.mark.parametrize("offset", [0.0, 0.12])
def test_weighted_conditional_likelihood_against_scipy(family, truncated, censored, offset):
    """Check transformed densities, survival and normalisation away from the MLE."""
    x = np.array([1.4, 2.1, 3.5, 6.0, 10.0])
    weights = np.array([1, 3, 2, 4, 1])
    flags = np.array([False, False, False, censored, censored])
    threshold = 1.2 if truncated else 0.0
    if family is d.LogGamma:
        params = {"alpha": 2.3 + offset, "beta": 0.7 + offset}
        reference = stats.gamma(a=params["alpha"], scale=params["beta"])
        logpdf = reference.logpdf(np.log(x)) - np.log(x)
        logsf = reference.logsf(np.log(x))
        normaliser = reference.logsf(np.log(threshold)) if truncated else 0.0
    elif family is d.GeneralizedBetaPrime:
        params = {"a": 1.7 + offset, "b": 2.4, "p": 2.1, "q": 3.2 + offset}
        reference = stats.betaprime(a=params["p"], b=params["q"])
        transformed = (x / params["b"]) ** params["a"]
        jacobian = params["a"] / params["b"] * (x / params["b"]) ** (params["a"] - 1)
        logpdf = reference.logpdf(transformed) + np.log(jacobian)
        logsf = reference.logsf(transformed)
        normaliser = reference.logsf((threshold / params["b"]) ** params["a"]) if truncated else 0.0
    else:
        if family is d.StudentsT:
            params = {"mu": -0.3 + offset, "sigma": 1.8, "nu": 6.5 + offset}
            reference = stats.t(df=params["nu"], loc=params["mu"], scale=params["sigma"])
        else:
            params = {"df": 4.2 + offset, "nonc": 2.5 + offset}
            reference = stats.ncx2(df=params["df"], nc=params["nonc"])
        logpdf, logsf = reference.logpdf(x), reference.logsf(x)
        normaliser = reference.logsf(threshold) if truncated else 0.0
    expected = np.sum(weights * (np.where(flags, logsf, logpdf) - normaliser))
    actual = _loglik(family, params, x, flags, weights, threshold, truncated)
    assert actual == pytest.approx(expected, rel=2e-12, abs=2e-12)


@pytest.mark.parametrize(("g", "b"), [(3.0, 1.0), (4.0, 0.25), (2.8, 0.6), (2.8, 1.7)])
def test_mbbefd_weighted_density_and_total_loss_atom(g, b):
    """Numerically differentiate the CDF and add the independent atom 1/g."""
    x = np.array([0.15, 0.35, 0.7, 1.0, 1.0])
    weights = np.array([2, 3, 1, 4, 2])
    interior = x[x < 1]

    def cdf(t):
        if b == 1:
            return (g - 1) * t / (1 + (g - 1) * t)
        if g * b == 1:
            return 1 - b**t
        return 1 - (1 - b) / (1 - g * b + (g - 1) * b ** (1 - t))

    step = 1e-5
    derivative = (
        cdf(interior - 2 * step) - 8 * cdf(interior - step) + 8 * cdf(interior + step) - cdf(interior + 2 * step)
    ) / (12 * step)
    expected = np.dot(weights[x < 1], np.log(derivative)) + weights[x == 1].sum() * np.log(1 / g)
    params = {"g": g, "b": b}
    actual = _loglik(d.MBBEFD, params, x, np.zeros(x.size, dtype=bool), weights, 0.0, False)
    assert actual == pytest.approx(expected, abs=2e-9)
    assert cdf(1.0) == pytest.approx(1 - 1 / g)


@pytest.mark.parametrize(("g", "b", "x"), [(1, 0.5, 0.4), (2, 0, 0.4), (2, 0.5, 0), (2, 0.5, 1.1)])
def test_mbbefd_invalid_parameters_or_support_are_rejected(g, b, x):
    assert _mbbefd_log_likelihood({"g": g, "b": b}, np.array([x]), np.ones(1)) == -np.inf


@pytest.mark.parametrize(
    ("family", "params", "message"),
    [
        (d.MBBEFD, {"g": 1.0, "b": 0.5}, "nonregular"),
        (d.NonCentralChiSquared, {"df": 3.0, "nonc": 0.0}, "boundary"),
        (d.Uniform, {"low": 0.0, "high": 1.0}, "nonregular"),
        (d.Categorical, {}, "simplex"),
        (d.HyperGeometric, {}, "integer"),
    ],
)
def test_nonregular_information_is_rejected(family, params, message):
    with pytest.raises(ValueError, match=message):
        _parameter_names(family, params, 0.0)


@pytest.mark.parametrize("family", [d.LogGamma, d.StudentsT, d.NonCentralChiSquared])
def test_nonfinite_likelihood_is_rejected(family):
    params = {
        d.LogGamma: {"alpha": 2.0, "beta": 1.0},
        d.StudentsT: {"mu": 0.0, "sigma": -1.0, "nu": 4.0},
        d.NonCentralChiSquared: {"df": -1.0, "nonc": 2.0},
    }[family]
    value = _loglik(family, params, np.array([0.5]), np.array([False]), np.ones(1), 0.0, False)
    assert value == -np.inf


@pytest.mark.parametrize("family", [d.LogGamma, d.GeneralizedBetaPrime, d.StudentsT, d.NonCentralChiSquared])
def test_fitted_likelihood_covariance_and_parameter_coupling(family):
    """Deferred information uses the fitted conditional likelihood and joint draws."""
    quantiles = np.linspace(0.001, 0.999, 700)
    if family is d.LogGamma:
        sample = np.exp(stats.gamma(a=2.5, scale=0.45).ppf(quantiles))
        threshold = 1.2
    elif family is d.GeneralizedBetaPrime:
        sample = 2.4 * stats.betaprime(a=2.1, b=3.2).ppf(quantiles) ** (1 / 1.7)
        threshold = 0.4
    elif family is d.StudentsT:
        sample = stats.t(df=7, loc=2, scale=1.4).ppf(quantiles)
        threshold = -1.0
    else:
        sample = stats.ncx2(df=4, nc=3).ppf(quantiles)
        threshold = 0.5
    sample = sample[sample > threshold]
    limit = np.quantile(sample, 0.85)
    flags = sample > limit
    observed = np.minimum(sample, limit)
    weights = np.ones(observed.size, dtype=int)
    fitted = family.fit(observed, threshold=threshold, censored=flags, weights=weights)
    assert fitted.log_likelihood == pytest.approx(
        _loglik(family, fitted.parameters, observed, flags, weights, threshold, True), abs=1e-8
    )
    uncertainty = fitted.asymptotic_uncertainty(observed, censored=flags, weights=weights)
    assert np.linalg.eigvalsh(uncertainty.covariance).min() > 0
    draws = uncertainty.parameter_samples(128, rng=np.random.default_rng(944))
    group = draws[uncertainty.parameter_names[0]].coupled_variable_group
    for name in uncertainty.parameter_names:
        assert draws[name].coupled_variable_group is group
        assert np.all(np.isfinite(asnumpy(draws[name].values)))


@pytest.mark.parametrize("threshold", [None, -1.0, 0.0, 1.0])
@pytest.mark.parametrize("family", [d.Normal, d.Logistic, d.StudentsT])
def test_continuous_fit_preserves_supplied_threshold(threshold, family):
    """Zero truncation and no truncation remain distinct in deferred uncertainty."""
    sample = stats.t(df=7, loc=2, scale=1.4).ppf(np.linspace(0.001, 0.999, 700))
    if threshold is not None:
        sample = sample[sample > threshold]
    weights = 1 + np.arange(sample.size) % 3
    limit = np.quantile(sample, 0.9)
    flags = sample > limit
    observed = np.minimum(sample, limit)
    fitted = family.fit(observed, threshold=threshold, censored=flags, weights=weights)
    assert fitted.threshold_specified is (threshold is not None)
    deferred = fitted.asymptotic_uncertainty(observed, censored=flags, weights=weights)
    direct = family.fit(
        observed, threshold=threshold, censored=flags, weights=weights, uncertainty_method="asymptotic"
    ).uncertainty
    assert direct is not None
    np.testing.assert_allclose(deferred.covariance, direct.covariance, rtol=1e-10)


@pytest.mark.parametrize("threshold", [None, 0.0, 0.2])
def test_beta_deferred_uncertainty_retains_truncation(threshold):
    sample = stats.beta(a=3.5, b=4.2).ppf(np.linspace(0.001, 0.999, 700))
    if threshold is not None:
        sample = sample[sample > threshold]
    fit = d.Beta.fit(sample, threshold=threshold)
    uncertainty = fit.asymptotic_uncertainty(sample)
    assert fit.threshold_specified is (threshold is not None)
    assert np.linalg.eigvalsh(uncertainty.covariance).min() > 0


def test_mbbefd_fitted_likelihood_and_joint_uncertainty():
    """The fitted atom and interior density agree with deferred information."""
    g, b = 3.0, 0.6
    probability = np.linspace(0.001, 0.999, 900)
    interior = probability < 1 - 1 / g
    sample = np.ones(probability.size)
    power = ((1 - b) / (1 - probability[interior]) - (1 - g * b)) / (g - 1)
    sample[interior] = 1 - np.log(power) / np.log(b)
    fit = d.MBBEFD.fit(sample)
    flags, weights = np.zeros(sample.size, dtype=bool), np.ones(sample.size)
    assert _loglik(d.MBBEFD, fit.parameters, sample, flags, weights, 0.0, False) == pytest.approx(fit.log_likelihood)
    uncertainty = fit.asymptotic_uncertainty(sample)
    assert np.linalg.eigvalsh(uncertainty.covariance).min() > 0
    draws = uncertainty.parameter_samples(128, rng=np.random.default_rng(41))
    assert draws["g"].coupled_variable_group is draws["b"].coupled_variable_group
    assert np.all(asnumpy(draws["g"].values) > 1)
    assert np.all(asnumpy(draws["b"].values) > 0)

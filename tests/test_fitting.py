"""Regression tests for PAL's CPU severity maximum-likelihood fitters."""

import numpy as np
import pytest
from scipy import stats

from pal.distributions import GPD, Gamma, LogNormal, Pareto
from pal.fitting import fit_distribution


def test_pareto_censored_frequency_weight_mle_is_analytic() -> None:
    """Censored observations contribute survival, not density, terms."""
    x = np.array([12.0, 20.0, 40.0, 80.0])
    censored = np.array([False, False, True, False])
    weights = np.array([1.0, 2.0, 3.0, 1.0])
    fitted = Pareto.fit(x, threshold=10.0, censored=censored, weights=weights)
    expected_shape = 4.0 / np.sum(weights * np.log(x / 10.0))
    assert fitted.distribution._params["shape"] == pytest.approx(expected_shape, rel=1e-12)
    assert fitted.distribution._params["scale"] == 10.0
    assert fitted.parameters["shape"] == pytest.approx(expected_shape)
    expected_loglik = np.sum(
        weights
        * np.where(
            censored,
            expected_shape * np.log(10.0 / x),
            np.log(expected_shape) - np.log(x) + expected_shape * np.log(10.0 / x),
        )
    )
    assert fitted.log_likelihood == pytest.approx(expected_loglik)
    assert fitted.aic == pytest.approx(2.0 - 2.0 * expected_loglik)
    assert fitted.aicc == pytest.approx(fitted.aic + 4.0 / (weights.sum() - 2.0))
    assert fitted.converged


def test_lognormal_uncensored_parameters_match_closed_form_mle() -> None:
    x = np.array([1.2, 2.4, 4.2, 8.1, 9.5, 12.0])
    fitted = LogNormal.fit(x)
    log_x = np.log(x)
    assert fitted.parameters["mu"] == pytest.approx(np.mean(log_x), rel=1e-5)
    assert fitted.parameters["sigma"] == pytest.approx(np.std(log_x, ddof=0), rel=1e-5)
    assert fitted.distribution.cdf(4.0) == pytest.approx(
        stats.lognorm.cdf(4.0, s=fitted.parameters["sigma"], scale=np.exp(fitted.parameters["mu"]))
    )


def test_lognormal_truncated_censored_likelihood_matches_scipy() -> None:
    rng = np.random.default_rng(789)
    raw = np.exp(rng.normal(2.0, 0.5, 2000))
    raw = raw[raw > 6.0]
    x = np.minimum(raw, 15.0)
    c = raw > 15.0
    fitted = fit_distribution(LogNormal, x, threshold=6.0, censored=c)
    mu, sigma = fitted.parameters["mu"], fitted.parameters["sigma"]
    expected = (
        np.sum(stats.lognorm.logpdf(x[~c], sigma, scale=np.exp(mu)))
        + np.sum(stats.lognorm.logsf(x[c], sigma, scale=np.exp(mu)))
        - len(x) * stats.lognorm.logsf(6.0, sigma, scale=np.exp(mu))
    )
    assert fitted.log_likelihood == pytest.approx(expected, rel=1e-10)
    assert mu == pytest.approx(2.0, abs=0.12)
    assert sigma == pytest.approx(0.5, abs=0.12)


def test_gamma_truncated_censored_parameter_recovery() -> None:
    rng = np.random.default_rng(714)
    raw = rng.gamma(shape=2.0, scale=3.0, size=3500)
    raw = raw[raw > 2.0]
    c = raw > 15.0
    fitted = Gamma.fit(np.minimum(raw, 15.0), threshold=2.0, censored=c)
    assert fitted.converged
    assert fitted.parameters["alpha"] == pytest.approx(2.0, abs=0.4)
    assert fitted.parameters["theta"] == pytest.approx(3.0, abs=0.6)
    assert fitted.distribution.invcdf(0.9) == pytest.approx(
        stats.gamma.ppf(0.9, fitted.parameters["alpha"], scale=fitted.parameters["theta"])
    )


def test_gpd_censored_mle_uses_threshold_as_location() -> None:
    rng = np.random.default_rng(221)
    raw = stats.genpareto.rvs(c=0.3, loc=10.0, scale=4.0, size=4000, random_state=rng)
    c = raw > 30.0
    fitted = GPD.fit(np.minimum(raw, 30.0), threshold=10.0, censored=c)
    assert fitted.converged
    assert fitted.parameters["loc"] == 10.0
    assert fitted.parameters["shape"] == pytest.approx(0.3, abs=0.18)
    assert fitted.parameters["scale"] == pytest.approx(4.0, abs=0.8)
    assert fitted.distribution.invcdf(0.9) == pytest.approx(
        stats.genpareto.ppf(0.9, fitted.parameters["shape"], loc=10.0, scale=fitted.parameters["scale"])
    )


def test_zero_and_fractional_weights_and_input_validation() -> None:
    x = np.array([3.0, 6.0, 12.0, 99.0])
    fitted = Pareto.fit(x, threshold=2.0, weights=[1.0, 0.5, 1.5, 0.0])
    assert np.isnan(fitted.aicc)
    assert fitted.n_observations == 3
    assert fitted.parameters["shape"] == pytest.approx(3.0 / np.sum([np.log(1.5), 0.5 * np.log(3), 1.5 * np.log(6)]))
    with pytest.raises(ValueError, match="observations"):
        Gamma.fit([])
    with pytest.raises(ValueError, match="finite"):
        Gamma.fit([1.0, np.nan])
    with pytest.raises(ValueError, match="censored"):
        Gamma.fit([1.0, 2.0], censored=[True])
    with pytest.raises(ValueError, match="weights"):
        Gamma.fit([1.0, 2.0], weights=[1.0, -1.0])
    with pytest.raises(ValueError, match="threshold"):
        Gamma.fit([1.0, 2.0], threshold=-1.0)
    with pytest.raises(ValueError, match="threshold"):
        Gamma.fit([1.0, 2.0], threshold=1.5)
    with pytest.raises(ValueError, match="uncensored"):
        Pareto.fit([2.0, 3.0], censored=[True, True])
    with pytest.raises(ValueError, match="Unsupported"):
        fit_distribution(type("Unsupported", (), {}), [2.0, 3.0])

def test_fitting_dispatch_requires_the_actual_pal_distribution_class() -> None:
    """An unrelated class with a matching name must not enter a PAL fitter."""
    fake_gamma = type("Gamma", (), {})
    with pytest.raises(ValueError, match="Unsupported fitting family"):
        fit_distribution(fake_gamma, [2.0, 3.0, 6.0])


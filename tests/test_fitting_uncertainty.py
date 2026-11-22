"""Tests for joint likelihood-based uncertainty in fitted severity parameters."""

import numpy as np
import pytest
from scipy import stats

from pal.config import set_random_seed, set_sampling_method
from pal.distributions import GPD, Gamma, LogNormal, Pareto
from pal.stochastic_scalar import StochasticScalar
from pal.types import SamplingMethod


def test_lognormal_information_and_coupled_predictive_simulation() -> None:
    rng = np.random.default_rng(1023)
    observations = rng.lognormal(mean=2.0, sigma=0.65, size=800)
    fitted = LogNormal.fit(observations)
    uncertainty = fitted.asymptotic_uncertainty(observations)

    assert uncertainty.parameter_names == ("mu", "sigma")
    assert uncertainty.covariance.shape == (2, 2)
    assert np.linalg.eigvalsh(uncertainty.covariance).min() > 0
    assert uncertainty.covariance[0, 0] == pytest.approx(
        fitted.parameters["sigma"] ** 2 / len(observations), rel=0.02
    )
    assert uncertainty.transformed_covariance[1, 1] == pytest.approx(
        1.0 / (2 * len(observations)), rel=0.02
    )
    intervals = uncertainty.confidence_intervals()
    assert intervals["sigma"][0] > 0
    assert intervals["sigma"][0] < fitted.parameters["sigma"] < intervals["sigma"][1]
    assert uncertainty.standard_errors["mu"] > 0

    parameters = uncertainty.parameter_samples(2048, rng=np.random.default_rng(91))
    assert isinstance(parameters["mu"], StochasticScalar)
    assert isinstance(parameters["sigma"], StochasticScalar)
    assert parameters["mu"].coupled_variable_group is parameters["sigma"].coupled_variable_group
    assert np.all(parameters["sigma"].values > 0)
    assert np.mean(parameters["mu"].values) == pytest.approx(fitted.parameters["mu"], abs=0.04)

    severity = uncertainty.sample_distribution(2048, rng=np.random.default_rng(42))
    simulations = severity.generate(2048, rng=np.random.default_rng(43))
    assert simulations.n_sims == 2048
    assert simulations.coupled_variable_group is severity._params["mu"].coupled_variable_group
    assert severity._params["mu"].coupled_variable_group is severity._params["sigma"].coupled_variable_group


def test_gamma_joint_uncertainty_preserves_parameter_correlation() -> None:
    rng = np.random.default_rng(284)
    observations = rng.gamma(shape=1.8, scale=3000, size=900)
    fit = Gamma.fit(observations)
    uncertainty = fit.asymptotic_uncertainty(observations)
    assert uncertainty.parameter_names == ("alpha", "theta")
    assert uncertainty.covariance[0, 1] < 0
    parameter_draws = uncertainty.parameter_samples(5000, rng=np.random.default_rng(123))
    draws = np.column_stack((parameter_draws["alpha"].values, parameter_draws["theta"].values))
    assert np.min(draws) > 0
    observed_corr = np.corrcoef(draws.T)[0, 1]
    target_corr = uncertainty.covariance[0, 1] / np.sqrt(
        uncertainty.covariance[0, 0] * uncertainty.covariance[1, 1]
    )
    assert observed_corr == pytest.approx(target_corr, abs=0.05)


def test_censored_and_truncated_gpd_uses_full_likelihood_and_fixed_location() -> None:
    rng = np.random.default_rng(22)
    ultimate = stats.genpareto.rvs(
        c=0.25, loc=10.0, scale=6.0, size=1200, random_state=rng
    )
    censored = ultimate >= 35
    observations = np.minimum(ultimate, 35)
    fit = GPD.fit(observations, threshold=10, censored=censored)
    uncertainty = fit.asymptotic_uncertainty(observations, censored=censored)
    assert uncertainty.parameter_names == ("shape", "scale")
    assert np.linalg.eigvalsh(uncertainty.transformed_covariance).min() > 0
    distribution = uncertainty.sample_distribution(500, rng=np.random.default_rng(313))
    assert distribution._params["loc"] == 10
    assert isinstance(distribution._params["shape"], StochasticScalar)
    assert np.min(distribution._params["scale"].values) > 0
    with pytest.raises(ValueError, match="match the original fit"):
        fit.asymptotic_uncertainty(observations)


def test_known_scale_pareto_information_matches_analytic_value() -> None:
    observations = np.array([12.0, 15.0, 20.0, 30.0, 45.0, 55.0, 90.0, 120.0])
    censored = np.array([False, False, True, False, False, True, False, False])
    weights = np.array([2, 1, 3, 1, 1, 1, 2, 1])
    fit = Pareto.fit(observations, threshold=10, censored=censored, weights=weights)
    uncertainty = fit.asymptotic_uncertainty(observations, censored=censored, weights=weights)
    n_events = weights[~censored].sum()
    assert uncertainty.transformed_covariance[0, 0] == pytest.approx(1 / n_events, rel=0.002)
    assert uncertainty.covariance[0, 0] == pytest.approx(
        fit.parameters["shape"] ** 2 / n_events, rel=0.002
    )
    values = uncertainty.parameter_samples(256, rng=np.random.default_rng(98))
    assert values["scale"] == 10
    assert np.all(values["shape"].values > 0)


def test_unsupported_or_nonregular_fits_raise_instead_of_clipping_information() -> None:
    x = np.array([12.0, 15.0, 19.0, 22.0, 35.0, 51.0, 62.0])
    with pytest.raises(ValueError, match="known scale"):
        Pareto.fit(x).asymptotic_uncertainty(x)
    weighted = Pareto.fit(x, threshold=10, weights=[1, 1, 0.5, 1, 1, 1, 1])
    with pytest.raises(ValueError, match="integer frequency weights"):
        weighted.asymptotic_uncertainty(x, weights=[1, 1, 0.5, 1, 1, 1, 1])
    short = Pareto.fit([12, 25, 40], threshold=10)
    with pytest.raises(ValueError, match="Too few"):
        short.asymptotic_uncertainty([12, 25, 40])
    with pytest.raises(ValueError, match="level"):
        Pareto.fit(x, threshold=10).asymptotic_uncertainty(x).confidence_intervals(1.0)


@pytest.mark.parametrize("method", [SamplingMethod.MC, SamplingMethod.RQMC])
def test_uncertainty_respects_pal_sampling_method(method: SamplingMethod) -> None:
    rng = np.random.default_rng(172)
    x = rng.lognormal(mean=2.0, sigma=0.5, size=200)
    fit = LogNormal.fit(x)
    uncertainty = fit.asymptotic_uncertainty(x)
    try:
        set_sampling_method(method, dimensions=4)
        set_random_seed(9281)
        parameters = uncertainty.parameter_samples(256)
        assert np.all(np.isfinite(parameters["sigma"].values))
        distribution = uncertainty.sample_distribution(256)
        simulation = distribution.generate(256)
        assert np.all(np.isfinite(simulation.values))
    finally:
        set_sampling_method(SamplingMethod.MC)

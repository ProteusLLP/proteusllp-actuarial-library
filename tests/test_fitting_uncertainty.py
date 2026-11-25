"""Tests for joint likelihood-based uncertainty in fitted severity parameters."""

import inspect
import warnings
from dataclasses import replace

import numpy as np
import pytest
from scipy import stats

from pal.config import set_random_seed, set_sampling_method
from pal.distributions import GPD, Gamma, LogNormal, Pareto
from pal.fitting import UncertaintyMethod
from pal.maths import asnumpy
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
    assert uncertainty.covariance[0, 0] == pytest.approx(fitted.parameters["sigma"] ** 2 / len(observations), rel=0.02)
    assert uncertainty.transformed_covariance[1, 1] == pytest.approx(1.0 / (2 * len(observations)), rel=0.02)
    intervals = uncertainty.confidence_intervals()
    assert intervals["sigma"][0] > 0
    assert intervals["sigma"][0] < fitted.parameters["sigma"] < intervals["sigma"][1]
    assert uncertainty.standard_errors["mu"] > 0

    parameters = uncertainty.parameter_samples(2048, rng=np.random.default_rng(91))
    assert isinstance(parameters["mu"], StochasticScalar)
    assert isinstance(parameters["sigma"], StochasticScalar)
    assert parameters["mu"].coupled_variable_group is parameters["sigma"].coupled_variable_group
    assert np.all(asnumpy(parameters["sigma"].values) > 0)
    assert np.mean(asnumpy(parameters["mu"].values)) == pytest.approx(fitted.parameters["mu"], abs=0.04)

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
    draws = np.column_stack((asnumpy(parameter_draws["alpha"].values), asnumpy(parameter_draws["theta"].values)))
    assert np.min(draws) > 0
    observed_corr = np.corrcoef(draws.T)[0, 1]
    target_corr = uncertainty.covariance[0, 1] / np.sqrt(uncertainty.covariance[0, 0] * uncertainty.covariance[1, 1])
    assert observed_corr == pytest.approx(target_corr, abs=0.05)


def test_censored_and_truncated_gpd_uses_full_likelihood_and_fixed_location() -> None:
    rng = np.random.default_rng(22)
    ultimate = stats.genpareto.rvs(c=0.25, loc=10.0, scale=6.0, size=1200, random_state=rng)
    censored = ultimate >= 35
    observations = np.minimum(ultimate, 35)
    fit = GPD.fit(observations, threshold=10, censored=censored)
    uncertainty = fit.asymptotic_uncertainty(observations, censored=censored)
    assert uncertainty.parameter_names == ("shape", "scale")
    assert np.linalg.eigvalsh(uncertainty.transformed_covariance).min() > 0
    distribution = uncertainty.sample_distribution(500, rng=np.random.default_rng(313))
    assert distribution._params["loc"] == 10
    assert isinstance(distribution._params["shape"], StochasticScalar)
    assert np.min(asnumpy(distribution._params["scale"].values)) > 0
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
    assert uncertainty.covariance[0, 0] == pytest.approx(fit.parameters["shape"] ** 2 / n_events, rel=0.002)
    values = uncertainty.parameter_samples(256, rng=np.random.default_rng(98))
    assert values["scale"] == 10
    assert np.all(asnumpy(values["shape"].values) > 0)


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
        assert np.all(np.isfinite(asnumpy(parameters["sigma"].values)))
        distribution = uncertainty.sample_distribution(256)
        simulation = distribution.generate(256)
        assert np.all(np.isfinite(asnumpy(simulation.values)))
    finally:
        set_sampling_method(SamplingMethod.MC)


def test_uncertainty_method_is_an_option_in_every_distribution_fit() -> None:
    """All PAL distribution fits expose one consistently named method option."""
    from pal import distributions as d

    families = [
        cls
        for cls in vars(d).values()
        if inspect.isclass(cls)
        and issubclass(cls, d.DistributionBase)
        and cls.__module__ == d.__name__
        and "fit" in cls.__dict__
    ]
    assert len(families) >= 30
    for family in families:
        signature = inspect.signature(family.fit)
        assert "uncertainty_method" in signature.parameters
        assert signature.parameters["uncertainty_method"].default is UncertaintyMethod.NONE


def test_fit_populates_uncertainty_when_requested() -> None:
    rng = np.random.default_rng(142)
    observations = rng.gamma(2.0, 3.0, 400)
    standard = Gamma.fit(observations)
    assert standard.uncertainty is None
    fitted = Gamma.fit(observations, uncertainty_method=UncertaintyMethod.ASYMPTOTIC)
    assert fitted.uncertainty is not None
    assert fitted.parameters == standard.parameters
    assert fitted.log_likelihood == standard.log_likelihood
    assert fitted.uncertainty.standard_errors["alpha"] > 0
    assert fitted.uncertainty.standard_errors["theta"] > 0
    random_severity = fitted.uncertainty.sample_distribution(256, rng=np.random.default_rng(51))
    assert random_severity.generate(256).n_sims == 256


@pytest.mark.parametrize(
    ("family_name", "kind"),
    [
        ("Normal", "normal"),
        ("Logistic", "logistic"),
        ("Beta", "beta"),
        ("Poisson", "poisson"),
        ("Binomial", "binomial"),
        ("Bernoulli", "bernoulli"),
        ("NegBinomial", "nbinom"),
        ("Weibull", "weibull"),
        ("Exponential", "exponential"),
        ("InverseGamma", "invgamma"),
        ("LogLogistic", "fisk"),
    ],
)
def test_regular_families_support_uncertainty(family_name: str, kind: str) -> None:
    """Smoke-test observed information, covariance and parameter simulation."""
    from pal import distributions as d

    rng = np.random.default_rng(622)
    sample = {
        "normal": lambda: rng.normal(2.0, 2.0, 550),
        "logistic": lambda: stats.logistic.rvs(loc=1.0, scale=3.0, size=550, random_state=rng),
        "beta": lambda: rng.beta(3.5, 4.2, 550),
        "poisson": lambda: rng.poisson(3.5, 550),
        "binomial": lambda: rng.binomial(12, 0.4, 550),
        "bernoulli": lambda: rng.binomial(1, 0.45, 550),
        "nbinom": lambda: rng.negative_binomial(3, 0.4, 550),
        "weibull": lambda: rng.weibull(1.6, 550) * 5,
        "exponential": lambda: rng.exponential(4, 550),
        "invgamma": lambda: stats.invgamma.rvs(a=3.5, scale=5.0, size=550, random_state=rng),
        "fisk": lambda: stats.fisk.rvs(c=2.5, scale=3, size=550, random_state=rng),
    }[kind]()
    options = {"n": 12} if kind == "binomial" else {}
    fit = getattr(d, family_name).fit(sample, uncertainty_method=UncertaintyMethod.ASYMPTOTIC, **options)
    assert fit.uncertainty is not None
    assert np.linalg.eigvalsh(fit.uncertainty.covariance).min() > 0
    sampled = fit.uncertainty.parameter_samples(128, rng=np.random.default_rng(47))
    for name in fit.uncertainty.parameter_names:
        assert isinstance(sampled[name], StochasticScalar)
        assert np.all(np.isfinite(asnumpy(sampled[name].values)))


def test_negative_binomial_size_and_probability_have_joint_uncertainty_and_coupling() -> None:
    """The size n is estimated, unlike fixed binomial trial count n."""
    from pal.distributions import NegBinomial

    rng = np.random.default_rng(182)
    observed = rng.negative_binomial(4.2, 0.46, size=700)
    fit = NegBinomial.fit(observed, uncertainty_method=UncertaintyMethod.ASYMPTOTIC)
    assert fit.uncertainty is not None
    assert fit.uncertainty.parameter_names == ("n", "p")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        draws = fit.uncertainty.parameter_samples(2048, rng=np.random.default_rng(183))
    size, probability = draws["n"], draws["p"]
    assert isinstance(size, StochasticScalar)
    assert isinstance(probability, StochasticScalar)
    assert size.coupled_variable_group is probability.coupled_variable_group
    assert np.all(asnumpy(size.values) > 0)
    assert np.all((asnumpy(probability.values) > 0) & (asnumpy(probability.values) < 1))
    assert np.std(asnumpy(size.values)) > 0.0
    normalised_n = np.log(asnumpy(size.values))
    normalised_p = np.log(asnumpy(probability.values) / (1.0 - asnumpy(probability.values)))
    drawn_correlation = np.corrcoef(normalised_n, normalised_p)[0, 1]
    covariance = fit.uncertainty.transformed_covariance
    expected_correlation = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
    assert drawn_correlation == pytest.approx(expected_correlation, abs=0.065)
    stochastic_frequency = fit.uncertainty.sample_distribution(512, rng=np.random.default_rng(184))
    frequency = stochastic_frequency.generate(512, rng=np.random.default_rng(185))
    assert frequency.coupled_variable_group is stochastic_frequency._params["n"].coupled_variable_group
    assert frequency.coupled_variable_group is stochastic_frequency._params["p"].coupled_variable_group


def test_gpd_joint_draws_remain_within_support_without_independent_clipping() -> None:
    """Reject full joint parameter vectors when the fitted data violate support."""
    rng = np.random.default_rng(192)
    observations = stats.genpareto.rvs(c=-0.14, scale=8.0, size=900, random_state=rng)
    fitted = GPD.fit(observations, uncertainty_method=UncertaintyMethod.ASYMPTOTIC)
    assert fitted.uncertainty is not None
    # Stress the approximation so its joint shape/scale support matters.
    stressed = replace(
        fitted.uncertainty,
        transformed_covariance=fitted.uncertainty.transformed_covariance * 20,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        samples = stressed.parameter_samples(1500, rng=np.random.default_rng(193))
    shape = asnumpy(samples["shape"].values)
    scale = asnumpy(samples["scale"].values)
    assert np.all(shape > -0.5)
    assert np.all(scale > 0.0)
    assert np.all(1.0 + shape * np.max(observations) / scale > 0)
    assert samples["shape"].coupled_variable_group is samples["scale"].coupled_variable_group


def test_gev_joint_shape_scale_location_and_support_remain_feasible() -> None:
    """Preserve full shape/scale/location dependence and observed-data support."""
    from pal.distributions import GEV

    rng = np.random.default_rng(802)
    observed = stats.genextreme.rvs(c=0.12, loc=100, scale=15, size=800, random_state=rng)
    fitted = GEV.fit(observed, uncertainty_method=UncertaintyMethod.ASYMPTOTIC)
    assert fitted.uncertainty is not None
    assert fitted.uncertainty.parameter_names == ("shape", "scale", "loc")
    samples = fitted.uncertainty.parameter_samples(1200, rng=np.random.default_rng(803))
    shape = asnumpy(samples["shape"].values)
    scale = asnumpy(samples["scale"].values)
    location = asnumpy(samples["loc"].values)
    assert np.all(shape > -0.5)
    assert np.all(scale > 0.0)
    assert np.all(1 + shape * (np.min(observed) - location) / scale > 0)
    assert np.all(1 + shape * (np.max(observed) - location) / scale > 0)
    assert samples["shape"].coupled_variable_group is samples["scale"].coupled_variable_group
    assert samples["shape"].coupled_variable_group is samples["loc"].coupled_variable_group
    random_gev = fitted.uncertainty.sample_distribution(256, rng=np.random.default_rng(804))
    observations_sim = random_gev.generate(256, rng=np.random.default_rng(805))
    for parameter in random_gev._params.values():
        assert isinstance(parameter, StochasticScalar)
        assert parameter.coupled_variable_group is observations_sim.coupled_variable_group


@pytest.mark.parametrize("method", [SamplingMethod.MC, SamplingMethod.RQMC])
def test_fitted_parameter_coupling_propagates_through_frequency_severity(
    method: SamplingMethod,
) -> None:
    """Keep one severity parameter value per simulation, not per claim."""
    from pal import FrequencySeverityModel
    from pal.distributions import Poisson

    observations = np.random.default_rng(511).gamma(2.0, 1000.0, 500)
    fitted = Gamma.fit(observations, uncertainty_method=UncertaintyMethod.ASYMPTOTIC)
    assert fitted.uncertainty is not None
    try:
        set_sampling_method(method, dimensions=8)
        set_random_seed(925)
        severity = fitted.uncertainty.sample_distribution(n_sims=256)
        model = FrequencySeverityModel(Poisson(2.0), severity)
        simulations = model.generate(256)
        alpha, theta = severity._params["alpha"], severity._params["theta"]
        assert isinstance(alpha, StochasticScalar)
        assert isinstance(theta, StochasticScalar)
        assert alpha.coupled_variable_group is theta.coupled_variable_group
        assert simulations.coupled_variable_group is alpha.coupled_variable_group
        assert simulations.aggregate().coupled_variable_group is alpha.coupled_variable_group
        assert simulations.n_sims == alpha.n_sims == theta.n_sims == 256
        assert np.all(np.isfinite(asnumpy(simulations.values)))
    finally:
        set_sampling_method(SamplingMethod.MC)

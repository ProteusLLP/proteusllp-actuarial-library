"""Regression tests for weighted model uncertainty and AICc fitting."""

import numpy as np
import pytest

from pal.config import set_random_seed, set_sampling_method
from pal.distributions import (
    Gamma,
    LogNormal,
    Normal,
    Poisson,
    WeightedDistribution,
)
from pal.fitting import UncertaintyMethod
from pal.maths import asnumpy
from pal.stochastic_scalar import StochasticScalar
from pal.types import SamplingMethod


def test_manual_weights_cdf_and_quantile() -> None:
    gamma = Gamma(2.0, 1.0)
    lognormal = LogNormal(0.0, 0.8)
    mixture = WeightedDistribution([gamma, lognormal], [3.0, 1.0])
    assert np.allclose(mixture.weights, [0.75, 0.25])
    x = np.array([0.25, 1.0, 4.0])
    expected = 0.75 * gamma.cdf(x) + 0.25 * lognormal.cdf(x)
    assert np.allclose(mixture.cdf(x), expected)
    assert np.allclose(mixture.cdf(mixture.invcdf(expected)), expected, atol=2e-7)
    assert mixture.cdf(mixture.invcdf(0.98)) == pytest.approx(0.98, abs=2e-7)


def test_aicc_fitting_matches_information_criterion_weights() -> None:
    rng = np.random.default_rng(116)
    sample = rng.gamma(2.6, 150.0, 450)
    result = WeightedDistribution.fit(sample, [Gamma, LogNormal])
    score = np.array([Gamma.fit(sample).aicc, LogNormal.fit(sample).aicc])
    expected = np.exp(-0.5 * (score - np.min(score)))
    expected /= expected.sum()
    assert np.allclose(result.weights, expected)
    assert np.allclose(result.distribution.weights, expected)
    assert set(result.model_weights) == {Gamma, LogNormal}
    assert result.aicc[Gamma] == pytest.approx(score[0])
    assert np.isfinite(result.distribution.invcdf(0.995))
    draws = result.distribution.generate(1024, rng=np.random.default_rng(10))
    assert draws.n_sims == 1024
    assert np.all(np.isfinite(asnumpy(draws.values)))


def test_fit_accepts_user_specified_model_weights() -> None:
    """Manually selected weights override AICc averaging, not MLE diagnostics."""
    sample = np.random.default_rng(66).gamma(3.0, 200.0, 500)
    manual = WeightedDistribution.fit(
        sample, [Gamma, LogNormal], model_weights={Gamma: 0.9, LogNormal: 0.1}
    )
    assert manual.model_weights[Gamma] == pytest.approx(0.9)
    assert manual.model_weights[LogNormal] == pytest.approx(0.1)
    assert manual.aicc[Gamma] == pytest.approx(Gamma.fit(sample).aicc)
    assert manual.aicc[LogNormal] == pytest.approx(LogNormal.fit(sample).aicc)
    sequence = WeightedDistribution.fit(sample, [Gamma, LogNormal], model_weights=[3.0, 1.0])
    assert np.allclose(sequence.weights, [0.75, 0.25])
    with pytest.raises(ValueError, match="model_weights mapping"):
        WeightedDistribution.fit(sample, [Gamma, LogNormal], model_weights={Gamma: 1.0})
    with pytest.raises(ValueError, match="weights"):
        WeightedDistribution.fit(sample, [Gamma, LogNormal], model_weights=[0, 0])


def test_joint_parameter_uncertainty_and_persistent_model_choice() -> None:
    rng = np.random.default_rng(222)
    sample = rng.lognormal(2.0, 0.6, 600)
    fit = WeightedDistribution.fit(
        sample, [Gamma, LogNormal], uncertainty_method=UncertaintyMethod.ASYMPTOTIC
    )
    assert all(candidate.uncertainty is not None for candidate in fit.fits)
    ensemble = fit.sample_distribution(2048, rng=np.random.default_rng(22))
    assert ensemble.model_indices is not None
    assert isinstance(ensemble.model_indices, StochasticScalar)
    selection_before = asnumpy(ensemble.model_indices.values).copy()
    for component in ensemble.distributions:
        assert all(
            isinstance(value, StochasticScalar)
            for name, value in component._params.items()
            if name in ("alpha", "theta", "mu", "sigma")
        )
    first = ensemble.generate(2048, rng=np.random.default_rng(23))
    second = ensemble.generate(2048, rng=np.random.default_rng(24))
    assert np.array_equal(asnumpy(ensemble.model_indices.values), selection_before)
    assert not np.array_equal(asnumpy(first.values), asnumpy(second.values))
    assert first.coupled_variable_group is ensemble.model_indices.coupled_variable_group
    assert second.coupled_variable_group is first.coupled_variable_group
    for component in ensemble.distributions:
        for parameter in component._params.values():
            if isinstance(parameter, StochasticScalar):
                assert parameter.coupled_variable_group is first.coupled_variable_group
    quantile = ensemble.invcdf(0.8)
    assert isinstance(quantile, StochasticScalar)
    assert np.allclose(asnumpy(ensemble.cdf(quantile).values), 0.8, atol=1e-6)


@pytest.mark.parametrize("method", [SamplingMethod.MC, SamplingMethod.RQMC])
def test_weighted_generation_supports_both_sampling_methods(method: SamplingMethod) -> None:
    try:
        set_sampling_method(method, dimensions=4)
        set_random_seed(244)
        model = WeightedDistribution([Gamma(2.0, 5.0), Gamma(7.0, 2.0)], [0.2, 0.8])
        ensemble = model.sample_models(512)
        first = ensemble.generate(512)
        assert first.n_sims == 512
        assert np.all(np.isfinite(asnumpy(first.values)))
        assert np.all(np.isin(asnumpy(ensemble.model_indices.values), [0, 1]))
    finally:
        set_sampling_method(SamplingMethod.MC)


def test_discrete_model_averaging_and_constraints() -> None:
    rng = np.random.default_rng(22)
    counts = rng.poisson(2.0, 300)
    # Same-family manual construction is valid for counts.
    mixture = WeightedDistribution([Poisson(2), Poisson(5)], [0.6, 0.4])
    assert float(mixture.cdf(3)) == pytest.approx(0.6 * Poisson(2).cdf(3) + 0.4 * Poisson(5).cdf(3))
    assert mixture.generate(256, rng=np.random.default_rng(1)).n_sims == 256
    with pytest.raises(ValueError, match="discrete and continuous"):
        WeightedDistribution([Poisson(2.0), Normal(0, 1)], [0.5, 0.5])
    with pytest.raises(ValueError, match="weights"):
        WeightedDistribution([Gamma(2, 3)], [0])
    with pytest.raises(ValueError, match="finite AICc"):
        WeightedDistribution.fit(counts[:2], [Poisson])


def test_aicc_fit_accepts_right_censoring_and_preserves_likelihood_diagnostics() -> None:
    rng = np.random.default_rng(131)
    ultimate = rng.lognormal(np.log(5000), 0.8, 650)
    threshold = 1000.0
    reported = ultimate[ultimate > threshold]
    censor = reported > 12000
    observations = np.minimum(reported, 12000)
    fitted = WeightedDistribution.fit(
        observations, [Gamma, LogNormal], threshold=threshold, censored=censor
    )
    assert len(fitted.fits) == 2
    assert all(f.threshold == threshold for f in fitted.fits)
    assert all(f.n_censored == np.sum(censor) for f in fitted.fits)
    assert fitted.weights.sum() == pytest.approx(1.0)
    assert np.all(fitted.weights >= 0)


def test_invalid_incomparable_candidates_and_premature_parameter_draws() -> None:
    sample = np.random.default_rng(2).gamma(2.0, 5.0, 300)
    with pytest.raises(ValueError, match="duplicates"):
        WeightedDistribution.fit(sample, [Gamma, Gamma])
    with pytest.raises(ValueError, match="nonregular"):
        from pal.distributions import Uniform

        WeightedDistribution.fit(sample, [Gamma, Uniform])
    with pytest.raises(ValueError, match="probabilities"):
        WeightedDistribution([Gamma(2, 3)], [1]).invcdf(-0.1)
    with pytest.raises(ValueError, match="matching n_sims"):
        # Scenario-level model selection cannot be used with a different axis.
        fitted = WeightedDistribution.fit(sample, [Gamma, LogNormal])
        fitted.sample_distribution(128).generate(64)


@pytest.mark.parametrize("method", [SamplingMethod.MC, SamplingMethod.RQMC])
def test_compound_model_keeps_same_family_for_each_claim_in_a_scenario(method: SamplingMethod) -> None:
    from pal.distributions import Uniform
    from pal.frequency_severity import FrequencySeverityModel

    try:
        set_sampling_method(method, dimensions=6)
        set_random_seed(829)
        base = WeightedDistribution([Uniform(0.0, 1.0), Uniform(10.0, 11.0)], [0.4, 0.6])
        scenario = base.sample_models(n_sims=256)
        results = FrequencySeverityModel(Poisson(3.0), scenario).generate(n_sims=256)
        choices = asnumpy(scenario.model_indices.values)[asnumpy(results.sim_index)]
        severities = asnumpy(results.values)
        assert len(severities) > 0
        assert np.all(severities[choices == 0] <= 1.0)
        assert np.all(severities[choices == 1] >= 10.0)
        assert results.coupled_variable_group is scenario.model_indices.coupled_variable_group
    finally:
        set_sampling_method(SamplingMethod.MC)


@pytest.mark.parametrize("method", [SamplingMethod.MC, SamplingMethod.RQMC])
def test_compound_model_preserves_uncertain_parameter_scenario_axis(method: SamplingMethod) -> None:
    from pal.frequency_severity import FrequencySeverityModel

    rng = np.random.default_rng(723)
    observations = rng.lognormal(mean=3.0, sigma=0.6, size=450)
    try:
        set_sampling_method(method, dimensions=6)
        set_random_seed(924)
        fitted = WeightedDistribution.fit(
            observations, [Gamma, LogNormal], uncertainty_method=UncertaintyMethod.ASYMPTOTIC
        )
        scenario = fitted.sample_distribution(n_sims=256)
        result = FrequencySeverityModel(Poisson(1.7), scenario).generate(n_sims=256)
        assert result.n_sims == 256
        assert np.all(np.isfinite(asnumpy(result.values)))
        for family in scenario.distributions:
            for parameter in family._params.values():
                if isinstance(parameter, StochasticScalar):
                    assert parameter.coupled_variable_group is result.coupled_variable_group
    finally:
        set_sampling_method(SamplingMethod.MC)


def test_compound_generation_uses_standard_parameter_expansion() -> None:
    """No special-case compound model logic is needed for weighted distributions."""
    import copy

    rng = np.random.default_rng(473)
    observations = rng.lognormal(2.0, 0.7, 400)
    fit = WeightedDistribution.fit(
        observations, [Gamma, LogNormal], uncertainty_method=UncertaintyMethod.ASYMPTOTIC
    )
    scenario = fit.sample_distribution(256, rng=np.random.default_rng(474))
    event_indices = np.repeat(np.arange(256), 2)
    expanded = copy.copy(scenario)
    expanded._params = {
        key: (
            StochasticScalar(value.values[event_indices])
            if isinstance(value, StochasticScalar)
            else value
        )
        for key, value in scenario._params.items()
    }
    result = expanded.generate(512, rng=np.random.default_rng(475))
    selected = asnumpy(scenario.model_indices.values[event_indices])
    assert np.array_equal(asnumpy(expanded.model_indices.values), selected)
    assert result.n_sims == 512
    assert np.all(asnumpy(result.values) >= 0)
    assert not np.array_equal(
        asnumpy(expanded._params["component_0_alpha"].values),
        asnumpy(scenario._params["component_0_alpha"].values),
    )



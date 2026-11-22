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

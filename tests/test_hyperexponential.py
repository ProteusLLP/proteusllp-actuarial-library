"""Tests for the hyperexponential distribution."""

import math

import numpy as np
import pytest

from pal import HyperExponential, distributions
from pal.config import set_random_seed
from pal.stochastic_scalar import StochasticScalar
from tests._assertions import allclose


def test_hyperexponential_cdf_and_inverse_cdf() -> None:
    """CDF and numerical inverse CDF agree across the body and tail."""
    weights = [0.25, 0.75]
    rates = [1.0, 0.25]
    loc = 2.0
    dist = HyperExponential(weights=weights, rates=rates, loc=loc)

    assert dist.cdf(loc) == 0.0
    expected = sum(weight * (1 - math.exp(-rate)) for weight, rate in zip(weights, rates))
    assert dist.cdf(loc + 1) == pytest.approx(expected)
    assert dist.invcdf(0) == loc
    assert math.isinf(float(dist.invcdf(1)))

    probabilities = StochasticScalar([0.001, 0.1, 0.5, 0.9, 0.999])
    quantiles = dist.invcdf(probabilities)
    assert allclose(dist.cdf(quantiles), probabilities, rtol=1e-11, atol=1e-12)


def test_hyperexponential_reduces_to_exponential() -> None:
    """A one-component mixture is the existing exponential distribution."""
    rate = 0.4
    loc = 3.0
    hyperexponential = HyperExponential(weights=[1.0], rates=[rate], loc=loc)
    exponential = distributions.Exponential(scale=1 / rate, loc=loc)
    probabilities = StochasticScalar([0.01, 0.2, 0.5, 0.9, 0.999])

    assert allclose(hyperexponential.invcdf(probabilities), exponential.invcdf(probabilities), rtol=1e-11)
    assert allclose(
        hyperexponential.cdf(hyperexponential.invcdf(probabilities)),
        probabilities,
        rtol=1e-11,
        atol=1e-12,
    )


def test_hyperexponential_from_scales() -> None:
    """The scale convenience constructor matches reciprocal rates."""
    weights = [0.3, 0.7]
    scales = [0.5, 4.0]
    from_scales = HyperExponential.from_scales(weights=weights, scales=scales)
    from_rates = HyperExponential(weights=weights, rates=[2.0, 0.25])
    probabilities = StochasticScalar([0.05, 0.25, 0.75, 0.95])

    assert allclose(from_scales.invcdf(probabilities), from_rates.invcdf(probabilities), rtol=1e-11)


def test_hyperexponential_simulated_moments() -> None:
    """Generated samples reproduce the analytical mean and variance."""
    set_random_seed(12345678910)
    weights = [0.2, 0.3, 0.5]
    rates = [2.0, 0.5, 1 / 7]
    loc = 10.0
    dist = HyperExponential(weights=weights, rates=rates, loc=loc)

    sims = dist.generate(500_000)
    centred_mean = sum(weight / rate for weight, rate in zip(weights, rates))
    expected_mean = loc + centred_mean
    expected_variance = 2 * sum(weight / rate**2 for weight, rate in zip(weights, rates)) - centred_mean**2

    assert sims.mean() == pytest.approx(expected_mean, rel=5e-3)
    assert sims.std() == pytest.approx(math.sqrt(expected_variance), rel=1e-2)


def test_hyperexponential_stochastic_component_parameters_preserve_coupling() -> None:
    """Scenario-varying mixture weights and rates are evaluated elementwise."""
    probability = StochasticScalar([0.2, 0.4, 0.6])
    rate = StochasticScalar([0.5, 1.0, 2.0])
    dist = HyperExponential(weights=[probability, 1 - probability], rates=[rate, 3.0])

    result = dist.cdf(1.0)
    expected = probability * (1 - np.exp(-rate)) + (1 - probability) * (1 - np.exp(-3.0))

    assert allclose(result, expected)
    assert result.coupled_variable_group == probability.coupled_variable_group
    assert result.coupled_variable_group == rate.coupled_variable_group


def test_hyperexponential_stochastic_sampling_preserves_coupling() -> None:
    """Sampling supports scenario-varying component parameters."""
    set_random_seed(12345678910)
    n_sims = 10_000
    probability = distributions.Beta(2.0, 3.0).generate(n_sims)
    rate = distributions.Gamma(2.0, 0.5).generate(n_sims)
    dist = HyperExponential(weights=[probability, 1 - probability], rates=[rate, 3.0])

    sims = dist.generate(n_sims)

    assert sims.coupled_variable_group == probability.coupled_variable_group
    assert sims.coupled_variable_group == rate.coupled_variable_group


@pytest.mark.parametrize(
    ("weights", "rates"),
    [
        ([], []),
        ([0.5, 0.5], [1.0]),
        ([-0.1, 1.1], [1.0, 2.0]),
        ([0.4, 0.4], [1.0, 2.0]),
        ([0.5, 0.5], [0.0, 2.0]),
        ([0.5, 0.5], [1.0, math.inf]),
    ],
)
def test_hyperexponential_rejects_invalid_parameters(weights: list[float], rates: list[float]) -> None:
    """Invalid mixture definitions fail at construction."""
    with pytest.raises(ValueError):
        HyperExponential(weights=weights, rates=rates)


def test_hyperexponential_is_exposed_in_distributions_namespace() -> None:
    """The public distributions namespace and name registry expose the class."""
    assert distributions.__dict__["HyperExponential"] is HyperExponential
    assert distributions.AVAILABLE_CONTINUOUS_DISTRIBUTIONS["hyperexponential"] is HyperExponential

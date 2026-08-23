"""Tests for the empirical distribution."""

import math

import pytest

from pal import Empirical, distributions
from pal.config import set_random_seed
from pal.stochastic_scalar import StochasticScalar
from tests._assertions import allclose


def test_empirical_unweighted_cdf_and_inverse_cdf() -> None:
    """The unweighted empirical CDF and generalized inverse are exact."""
    dist = Empirical([3.0, 1.0, 2.0, 2.0])

    assert dist.cdf(0.5) == 0.0
    assert dist.cdf(1.0) == pytest.approx(0.25)
    assert dist.cdf(2.0) == pytest.approx(0.75)
    assert dist.cdf(10.0) == 1.0

    probabilities = StochasticScalar([0.0, 0.1, 0.25, 0.250001, 0.75, 0.750001, 1.0])
    expected = StochasticScalar([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    assert allclose(dist.invcdf(probabilities), expected)


def test_empirical_weighted_cdf_and_inverse_cdf() -> None:
    """Arbitrary observation weights are normalized internally."""
    dist = Empirical(samples=[1.0, 2.0, 4.0], weights=[2.0, 6.0, 2.0])

    assert dist.cdf(0.0) == 0.0
    assert dist.cdf(1.0) == pytest.approx(0.2)
    assert dist.cdf(2.0) == pytest.approx(0.8)
    assert dist.cdf(4.0) == 1.0

    probabilities = StochasticScalar([0.0, 0.2, 0.200001, 0.8, 0.800001, 1.0])
    expected = StochasticScalar([1.0, 1.0, 2.0, 2.0, 4.0, 4.0])
    assert allclose(dist.invcdf(probabilities), expected)


def test_empirical_duplicate_samples_accumulate_probability() -> None:
    """Repeated observations contribute their combined mass at one support value."""
    dist = Empirical(samples=[1.0, 2.0, 2.0, 4.0], weights=[1.0, 2.0, 3.0, 4.0])

    assert dist.cdf(1.5) == pytest.approx(0.1)
    assert dist.cdf(2.0) == pytest.approx(0.6)
    assert dist.invcdf(0.6) == 2.0
    assert dist.invcdf(0.600001) == 4.0


def test_empirical_zero_weight_observations_are_ignored() -> None:
    """Zero-weight observations do not affect support or quantiles."""
    dist = Empirical(samples=[-100.0, 1.0, 3.0, 100.0], weights=[0.0, 2.0, 2.0, 0.0])

    assert dist.cdf(0.0) == 0.0
    assert dist.invcdf(0.0) == 1.0
    assert dist.invcdf(1.0) == 3.0


def test_empirical_generation_reproduces_weighted_probabilities() -> None:
    """Resampling reproduces the weighted empirical probabilities."""
    set_random_seed(12345678910)
    dist = Empirical(samples=[1.0, 4.0, 10.0], weights=[1.0, 3.0, 6.0])

    sims = dist.generate(300_000)
    expected_mean = 0.1 * 1.0 + 0.3 * 4.0 + 0.6 * 10.0
    expected_second_moment = 0.1 * 1.0**2 + 0.3 * 4.0**2 + 0.6 * 10.0**2
    expected_std = math.sqrt(expected_second_moment - expected_mean**2)

    assert sims.mean() == pytest.approx(expected_mean, rel=5e-3)
    assert sims.std() == pytest.approx(expected_std, rel=5e-3)


def test_empirical_accepts_stochastic_scalar_as_observed_sample() -> None:
    """A StochasticScalar can be reused as fixed empirical support."""
    source = StochasticScalar([5.0, 1.0, 3.0, 7.0])
    dist = Empirical(source)

    assert dist.cdf(3.0) == pytest.approx(0.5)
    assert dist.invcdf(0.75) == 5.0


def test_empirical_cdf_and_inverse_preserve_argument_coupling() -> None:
    """Vector CDF and inverse-CDF results remain coupled to their arguments."""
    dist = Empirical(samples=[1.0, 2.0, 5.0], weights=[1.0, 2.0, 1.0])
    values = StochasticScalar([0.0, 2.0, 10.0])
    probabilities = StochasticScalar([0.1, 0.6, 0.9])

    cdf_result = dist.cdf(values)
    inverse_result = dist.invcdf(probabilities)

    assert cdf_result.coupled_variable_group == values.coupled_variable_group
    assert inverse_result.coupled_variable_group == probabilities.coupled_variable_group


def test_empirical_invalid_probabilities_return_nan() -> None:
    """Inverse-CDF probabilities outside the unit interval return NaN."""
    dist = Empirical([1.0, 2.0, 3.0])

    result = dist.invcdf(StochasticScalar([-0.1, 0.5, 1.1]))
    assert math.isnan(float(result[0]))
    assert result[1] == 2.0
    assert math.isnan(float(result[2]))


@pytest.mark.parametrize(
    ("samples", "weights", "error_type"),
    [
        ([], None, ValueError),
        ([[1.0, 2.0]], None, ValueError),
        ([1.0, math.nan], None, ValueError),
        ([1.0, math.inf], None, ValueError),
        ([1.0, 2.0], [1.0], ValueError),
        ([1.0, 2.0], [[1.0, 1.0]], ValueError),
        ([1.0, 2.0], [-1.0, 2.0], ValueError),
        ([1.0, 2.0], [0.0, 0.0], ValueError),
        ([1.0, 2.0], [1.0, math.nan], ValueError),
        (["a", "b"], None, TypeError),
    ],
)
def test_empirical_rejects_invalid_inputs(samples: object, weights: object, error_type: type[Exception]) -> None:
    """Invalid empirical samples and weights fail at construction."""
    with pytest.raises(error_type):
        Empirical(samples=samples, weights=weights)


def test_empirical_is_exposed_in_distributions_namespace() -> None:
    """The public distributions namespace and discrete registry expose the class."""
    assert distributions.Empirical is Empirical
    assert distributions.AVAILABLE_DISCRETE_DISTRIBUTIONS["empirical"] is Empirical

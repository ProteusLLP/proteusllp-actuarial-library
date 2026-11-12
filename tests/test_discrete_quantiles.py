"""Quantile regression tests for PAL's discrete count distributions."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binom as scipy_binom
from scipy.stats import nbinom as scipy_nbinom
from scipy.stats import poisson as scipy_poisson

from pal import StochasticScalar
from pal._maths import xp
from pal.copulas import apply_copula
from pal.distributions import Binomial, NegBinomial, Poisson
from tests._assertions import assert_array_equal, host_values


@pytest.mark.parametrize(
    ("distribution", "reference", "parameters"),
    [
        (Binomial, scipy_binom, (10, 0.3)),
        (Binomial, scipy_binom, (200, 0.015)),
        (Binomial, scipy_binom, (1000, 0.98)),
        (NegBinomial, scipy_nbinom, (5, 0.25)),
        (NegBinomial, scipy_nbinom, (2.5, 0.6)),
        (NegBinomial, scipy_nbinom, (100, 0.04)),
    ],
)
def test_count_quantiles_match_scipy_and_cdf_brackets(distribution, reference, parameters) -> None:
    """Check trusted count quantiles and the mathematical generalized-inverse condition."""
    probabilities = np.array([1e-12, 1e-8, 0.005, 0.1, 0.3, 0.5, 0.7, 0.9, 0.995, 1 - 1e-10])
    dist = distribution(*parameters)
    output = dist.invcdf(StochasticScalar(probabilities))
    assert isinstance(output, StochasticScalar)
    quantiles = host_values(output)
    expected = reference.ppf(probabilities, *parameters)
    np.testing.assert_array_equal(quantiles, expected)
    assert np.all(quantiles >= 0)
    assert np.all(np.floor(quantiles) == quantiles)
    assert np.all(reference.cdf(quantiles - 1, *parameters) < probabilities)
    assert np.all(reference.cdf(quantiles, *parameters) >= probabilities)


@pytest.mark.parametrize(
    ("distribution", "reference", "parameters"),
    [
        (Binomial, scipy_binom, (12, 0.4)),
        (NegBinomial, scipy_nbinom, (5, 0.3)),
    ],
)
def test_count_quantile_scalar_and_endpoints(distribution, reference, parameters) -> None:
    dist = distribution(*parameters)
    assert dist.invcdf(0.0) == 0
    assert dist.invcdf(0.5) == reference.ppf(0.5, *parameters)
    assert dist.invcdf(1.0) == reference.ppf(1.0, *parameters)
    assert np.isnan(float(dist.invcdf(-0.1)))
    assert np.isnan(float(dist.invcdf(1.1)))


@pytest.mark.parametrize(
    ("distribution", "reference", "parameters"),
    [
        (Binomial, scipy_binom, (20, 0.0)),
        (Binomial, scipy_binom, (20, 1.0)),
        (Binomial, scipy_binom, (0, 0.25)),
        (NegBinomial, scipy_nbinom, (5, 1.0)),
    ],
)
def test_count_quantiles_for_degenerate_parameters(distribution, reference, parameters) -> None:
    probabilities = StochasticScalar([0.01, 0.2, 0.5, 0.99])
    output = distribution(*parameters).invcdf(probabilities)
    assert isinstance(output, StochasticScalar)
    assert_array_equal(output, reference.ppf([0.01, 0.2, 0.5, 0.99], *parameters))


@pytest.mark.parametrize(
    ("distribution", "reference", "params"),
    [
        (Binomial, scipy_binom, ([10, 20, 30, 40], [0.1, 0.2, 0.4, 0.7])),
        (NegBinomial, scipy_nbinom, ([2.0, 3.5, 5.0, 8.0], [0.15, 0.3, 0.6, 0.9])),
    ],
)
def test_count_quantiles_preserve_stochastic_parameter_ancestry(distribution, reference, params) -> None:
    n = StochasticScalar(params[0])
    p = StochasticScalar(params[1])
    u = StochasticScalar([0.15, 0.4, 0.75, 0.95])
    assert n.coupled_variable_group is not p.coupled_variable_group
    assert n.coupled_variable_group is not u.coupled_variable_group

    output = distribution(n, p).invcdf(u)
    expected = reference.ppf(host_values(u), host_values(n), host_values(p))
    assert isinstance(output, StochasticScalar)
    assert_array_equal(output, expected)
    assert output.coupled_variable_group is n.coupled_variable_group
    assert output.coupled_variable_group is p.coupled_variable_group
    assert output.coupled_variable_group is u.coupled_variable_group


@pytest.mark.parametrize("distribution", [Binomial, NegBinomial])
def test_count_quantiles_with_scalar_probability_and_stochastic_parameters(distribution) -> None:
    n = StochasticScalar([4, 7, 10])
    p = StochasticScalar([0.25, 0.5, 0.75])
    result = distribution(n, p).invcdf(0.7)
    assert isinstance(result, StochasticScalar)
    assert result.coupled_variable_group is n.coupled_variable_group
    assert result.coupled_variable_group is p.coupled_variable_group


@pytest.mark.parametrize("distribution", [Binomial, NegBinomial])
def test_count_quantile_rejects_incompatible_simulation_lengths_without_coupling(distribution) -> None:
    n = StochasticScalar([5, 10, 15])
    p = StochasticScalar([0.2, 0.4, 0.6])
    u = StochasticScalar([0.1, 0.3, 0.5, 0.7])
    with pytest.raises(ValueError):
        distribution(n, p).invcdf(u)
    assert n.coupled_variable_group is not p.coupled_variable_group
    assert n.coupled_variable_group is not u.coupled_variable_group
    assert p.coupled_variable_group is not u.coupled_variable_group


@pytest.mark.parametrize(
    ("distribution", "params", "probabilities"),
    [
        (Binomial, ([15, 20, 25, 30], [0.1, 0.3, 0.5, 0.7]), [0.1, 0.4, 0.7, 0.9]),
        (NegBinomial, ([3.0, 5.0, 7.0, 10.0], [0.8, 0.6, 0.4, 0.2]), [0.1, 0.4, 0.7, 0.9]),
    ],
)
def test_count_quantile_copula_reordering_preserves_parameter_alignment(distribution, params, probabilities) -> None:
    n = StochasticScalar(params[0])
    p = StochasticScalar(params[1])
    u = StochasticScalar(probabilities)
    quantiles = distribution(n, p).invcdf(u)
    assert isinstance(quantiles, StochasticScalar)
    original_n = host_values(n).copy()
    original_p = host_values(p).copy()
    original_u = host_values(u).copy()
    original_q = host_values(quantiles).copy()
    scores = [3, 0, 2, 1]
    selection = np.argsort(original_q, kind="stable")[np.argsort(np.argsort(scores, kind="stable"), kind="stable")]
    anchor = StochasticScalar(np.arange(4))
    apply_copula(
        [anchor, quantiles],
        [StochasticScalar(np.arange(4)), StochasticScalar(scores)],
    )
    assert_array_equal(n, original_n[selection])
    assert_array_equal(p, original_p[selection])
    assert_array_equal(u, original_u[selection])
    assert_array_equal(quantiles, original_q[selection])
    assert all(item.coupled_variable_group is anchor.coupled_variable_group for item in (n, p, u, quantiles))


def test_count_quantiles_return_backend_values_for_stochastic_inputs() -> None:
    """CPU and GPU runners exercise the same host fallback and backend result."""
    result = Binomial(20, 0.2).invcdf(StochasticScalar([0.2, 0.5, 0.9]))
    assert isinstance(result, StochasticScalar)
    assert isinstance(result.values, xp.ndarray)
    assert_array_equal(result, scipy_binom.ppf([0.2, 0.5, 0.9], 20, 0.2))


@pytest.mark.parametrize("mean", [0.5, 3.5, 100.0, 5000.0])
def test_poisson_count_quantiles_match_scipy_and_cdf_brackets(mean: float) -> None:
    probabilities = np.array([1e-12, 1e-8, 0.01, 0.1, 0.5, 0.9, 0.99, 1 - 1e-10])
    result = Poisson(mean).invcdf(StochasticScalar(probabilities))
    assert isinstance(result, StochasticScalar)
    quantiles = host_values(result)
    np.testing.assert_array_equal(quantiles, scipy_poisson.ppf(probabilities, mean))
    assert np.all(quantiles >= 0)
    assert np.all(quantiles == np.floor(quantiles))
    assert np.all(scipy_poisson.cdf(quantiles - 1, mean) < probabilities)
    assert np.all(scipy_poisson.cdf(quantiles, mean) >= probabilities)


def test_poisson_quantile_support_and_parameter_coupling() -> None:
    assert Poisson(3.5).invcdf(0) == 0
    assert np.isinf(float(Poisson(3.5).invcdf(1)))
    assert Poisson(0).invcdf(0.99) == 0

    mean = StochasticScalar([0.5, 3.5, 10, 50])
    u = StochasticScalar([0.1, 0.5, 0.75, 0.99])
    result = Poisson(mean).invcdf(u)
    assert isinstance(result, StochasticScalar)
    assert_array_equal(result, scipy_poisson.ppf(host_values(u), host_values(mean)))
    assert result.coupled_variable_group is mean.coupled_variable_group
    assert result.coupled_variable_group is u.coupled_variable_group


def test_poisson_quantile_mismatched_lengths_do_not_merge_groups() -> None:
    mean = StochasticScalar([1.0, 2.0, 3.0])
    u = StochasticScalar([0.1, 0.3, 0.5, 0.7])
    with pytest.raises(ValueError):
        Poisson(mean).invcdf(u)
    assert mean.coupled_variable_group is not u.coupled_variable_group

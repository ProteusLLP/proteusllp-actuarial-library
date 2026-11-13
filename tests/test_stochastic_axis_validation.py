"""Regression tests for simulation-axis validation and atomic stochastic coupling."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from pal import StochasticScalar
from pal._maths import asnumpy
from pal.copulas import apply_copula
from pal.distributions import Binomial, Gamma, LogNormal, NegBinomial, Normal, Poisson


def _assert_separate_and_unchanged(*variables: StochasticScalar) -> None:
    groups = [item.coupled_variable_group for item in variables]
    assert all(groups[i] is not groups[j] for i in range(len(groups)) for j in range(i + 1, len(groups)))


@pytest.mark.parametrize("ufunc", [np.add, np.multiply, np.maximum])
def test_ufunc_rejects_broadcasting_between_different_scenario_counts(ufunc) -> None:
    long = StochasticScalar([1.0, 2.0, 3.0])
    derived = long * 2
    singleton = StochasticScalar([10.0])
    prior_long = asnumpy(long.values).copy()
    prior_derived = asnumpy(derived.values).copy()
    prior_singleton = asnumpy(singleton.values).copy()
    long_group = long.coupled_variable_group
    singleton_group = singleton.coupled_variable_group

    with pytest.raises(ValueError, match="matching n_sims"):
        ufunc(derived, singleton)

    assert long.coupled_variable_group is long_group
    assert derived.coupled_variable_group is long_group
    assert singleton.coupled_variable_group is singleton_group
    _assert_separate_and_unchanged(long, singleton)
    np.testing.assert_array_equal(asnumpy(long.values), prior_long)
    np.testing.assert_array_equal(asnumpy(derived.values), prior_derived)
    np.testing.assert_array_equal(asnumpy(singleton.values), prior_singleton)


def test_numpy_where_rejects_broadcasting_stochastic_scenarios_atomically() -> None:
    condition = StochasticScalar([True])
    values = StochasticScalar([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="matching n_sims"):
        np.where(condition, values, 0.0)
    _assert_separate_and_unchanged(condition, values)


def test_elementwise_ufunc_cannot_expand_simulation_axis_through_raw_array() -> None:
    variable = StochasticScalar([1.0])
    original_group = variable.coupled_variable_group
    original_values = asnumpy(variable.values).copy()
    with pytest.raises(ValueError, match="incompatible n_sims"):
        np.add(variable, np.array([10.0, 20.0, 30.0]))
    assert variable.coupled_variable_group is original_group
    np.testing.assert_array_equal(asnumpy(variable.values), original_values)


def test_numpy_array_functions_changing_axis_length_do_not_couple() -> None:
    variable = StochasticScalar([1.0, 3.0, 6.0, 10.0])
    original_group = variable.coupled_variable_group
    difference = np.diff(variable)
    assert not isinstance(difference, StochasticScalar)
    np.testing.assert_array_equal(asnumpy(difference), [2.0, 3.0, 4.0])

    singleton = StochasticScalar([1.0])
    singleton_group = singleton.coupled_variable_group
    expanded = np.where(singleton > 0, np.array([10.0, 20.0, 30.0]), 0.0)
    assert not isinstance(expanded, StochasticScalar)
    np.testing.assert_array_equal(asnumpy(expanded), [10.0, 20.0, 30.0])
    assert variable.coupled_variable_group is original_group
    assert singleton.coupled_variable_group is singleton_group


def test_length_changing_reduceat_returns_array_without_coupling() -> None:
    source = StochasticScalar([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    group = source.coupled_variable_group
    reduced = np.add.reduceat(source, [0, 3, 5])
    assert not isinstance(reduced, StochasticScalar)
    np.testing.assert_allclose(asnumpy(reduced), [6.0, 9.0, 21.0])
    assert source.coupled_variable_group is group
    assert len(group) == 1


@pytest.mark.parametrize("method", ["cdf", "invcdf"])
def test_gamma_probability_methods_reject_mismatched_parameter_axes(method: str) -> None:
    shape = StochasticScalar([1.0, 2.0, 3.0])
    scale = StochasticScalar([1.0])
    dist = Gamma(alpha=shape, theta=scale)
    with pytest.raises(ValueError, match="matching n_sims"):
        getattr(dist, method)(0.5 if method == "invcdf" else 2.0)
    _assert_separate_and_unchanged(shape, scale)


@pytest.mark.parametrize(
    ("family", "first_values", "second_values"),
    [
        (Gamma, [1.0, 2.0, 3.0], [0.7]),
        (Normal, [1.0], [0.5, 1.0, 1.5]),
        (LogNormal, [1.0], [0.5, 1.0, 1.5]),
    ],
)
@pytest.mark.parametrize("method", ["cdf", "invcdf"])
def test_multi_step_probability_failure_does_not_merge_partial_ancestry(
    family, first_values, second_values, method: str
) -> None:
    first = StochasticScalar(first_values)
    second = StochasticScalar(second_values)
    argument = StochasticScalar([0.2, 0.5, 0.8] if method == "invcdf" else [1.5, 2.0, 2.5])
    with pytest.raises(ValueError, match="matching n_sims"):
        getattr(family(first, second), method)(argument)
    _assert_separate_and_unchanged(first, second, argument)


@pytest.mark.parametrize("distribution", [Poisson, Binomial, NegBinomial])
def test_discrete_quantiles_reject_mismatched_axes_atomically(distribution) -> None:
    probability = StochasticScalar([0.2, 0.5, 0.8])
    param = StochasticScalar([3.0])
    dist = distribution(param) if distribution is Poisson else distribution(param, 0.4)
    with pytest.raises(ValueError, match="matching n_sims"):
        dist.invcdf(probability)
    _assert_separate_and_unchanged(probability, param)


def test_distribution_generate_checks_parameter_length_before_rng_use() -> None:
    param = StochasticScalar([2.0])
    rng = np.random.default_rng(317)
    with pytest.raises(ValueError, match="matching n_sims"):
        Gamma(alpha=param, theta=1.0).generate(n_sims=3, rng=rng)
    assert rng.random() == np.random.default_rng(317).random()
    assert len(param.coupled_variable_group) == 1


def test_three_level_distribution_chain_retains_values_and_ancestry_after_reordering() -> None:
    shape = StochasticScalar([1.0, 2.0, 3.0, 4.0])
    scale = StochasticScalar([0.5, 0.8, 1.1, 1.4])
    probability = StochasticScalar([0.2, 0.4, 0.6, 0.8])
    level_one = Gamma(alpha=shape, theta=scale).invcdf(probability)
    level_two = Normal(mu=level_one, sigma=0.8).cdf(1.5)
    result = LogNormal(mu=level_two, sigma=0.4).invcdf(0.7)

    expected_one = stats.gamma.ppf([0.2, 0.4, 0.6, 0.8], a=[1, 2, 3, 4], scale=[0.5, 0.8, 1.1, 1.4])
    expected_two = stats.norm.cdf(1.5, loc=expected_one, scale=0.8)
    expected_result = stats.lognorm.ppf(0.7, s=0.4, scale=np.exp(expected_two))
    np.testing.assert_allclose(asnumpy(result.values), expected_result, rtol=1e-12)

    common_group = shape.coupled_variable_group
    assert all(
        variable.coupled_variable_group is common_group
        for variable in (scale, probability, level_one, level_two, result)
    )
    original = {
        id(variable): asnumpy(variable.values).copy()
        for variable in (shape, scale, probability, level_one, level_two, result)
    }
    scores = [3, 0, 2, 1]
    ranking = np.argsort(original[id(result)], kind="stable")
    selection = ranking[np.argsort(np.argsort(scores, kind="stable"), kind="stable")]

    anchor = StochasticScalar(np.arange(4))
    apply_copula(
        [anchor, result],
        [StochasticScalar(np.arange(4)), StochasticScalar(scores)],
    )

    for variable in (shape, scale, probability, level_one, level_two, result):
        assert variable.coupled_variable_group is anchor.coupled_variable_group
        np.testing.assert_allclose(asnumpy(variable.values), original[id(variable)][selection], rtol=1e-12)

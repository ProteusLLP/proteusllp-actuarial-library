"""Behavioural contract for linking scenario-indexed stochastic results."""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from pal import FreqSevSims, StochasticScalar
from pal._maths import asnumpy
from pal.couplings import couple_stochastic_results
from pal.distributions import Gamma, HyperGeometric, Normal
from pal.hyperexponential import HyperExponential
from pal.multivariate_distributions import MultivariateNormal, Wishart


def test_shared_helper_links_multiple_outputs_and_upstream_ancestry() -> None:
    first = StochasticScalar([1.0, 2.0, 3.0])
    upstream = first * 2
    second = StochasticScalar([4.0, 5.0, 6.0])
    left = StochasticScalar([7.0, 8.0, 9.0])
    right = StochasticScalar([10.0, 11.0, 12.0])
    original_group = first.coupled_variable_group

    couple_stochastic_results((left, right), (upstream, second), n_sims=3)

    assert all(item.coupled_variable_group is original_group for item in (first, upstream, second, left, right))


def test_shared_helper_rejects_late_result_axis_failure_atomically() -> None:
    source = StochasticScalar([1.0, 2.0, 3.0])
    other = StochasticScalar([3.0, 4.0, 5.0])
    first_leaf = StochasticScalar([6.0, 7.0, 8.0])
    bad_leaf = StochasticScalar([9.0])
    variables = (source, other, first_leaf, bad_leaf)
    groups = tuple(variable.coupled_variable_group for variable in variables)
    values = tuple(asnumpy(variable.values).copy() for variable in variables)

    with pytest.raises(ValueError, match="matching n_sims"):
        couple_stochastic_results((first_leaf, bad_leaf), (source, other))

    assert all(variable.coupled_variable_group is group for variable, group in zip(variables, groups))
    assert all(len(group) == 1 for group in groups)
    for variable, before in zip(variables, values):
        np.testing.assert_array_equal(asnumpy(variable.values), before)


def test_multi_output_scalar_ufunc_preserves_one_group() -> None:
    source = StochasticScalar([1.2, 2.5, -3.7])
    fraction, integer = np.modf(source)
    assert fraction.coupled_variable_group is source.coupled_variable_group
    assert integer.coupled_variable_group is source.coupled_variable_group
    np.testing.assert_allclose(asnumpy(fraction.values), np.modf([1.2, 2.5, -3.7])[0])
    np.testing.assert_allclose(asnumpy(integer.values), np.modf([1.2, 2.5, -3.7])[1])


def test_distribution_chains_preserve_all_ancestors() -> None:
    shape = StochasticScalar([1.0, 2.0, 3.0])
    scale = StochasticScalar([1.0, 1.5, 2.0])
    first = Gamma(alpha=shape, theta=scale).invcdf(0.6)
    second = Normal(mu=first, sigma=0.5).cdf(2.0)
    third = Gamma(alpha=second + 1, theta=scale).generate(3, np.random.default_rng(42))

    assert all(
        variable.coupled_variable_group is shape.coupled_variable_group for variable in (scale, first, second, third)
    )


def test_multivariate_and_matrix_leaves_share_parameter_ancestry() -> None:
    mean = StochasticScalar([0.0, 1.0, 2.0, 3.0])
    multivariate = MultivariateNormal([mean, 1.0], np.eye(2))
    generated = multivariate.generate(4, np.random.default_rng(51))
    density = multivariate.logpdf([0.0, 1.0])
    assert all(entry.coupled_variable_group is mean.coupled_variable_group for entry in generated)
    assert density.coupled_variable_group is mean.coupled_variable_group

    df = StochasticScalar([4.0, 5.0, 6.0, 7.0])
    matrices = Wishart(df=df, scale=np.eye(2)).generate(4, np.random.default_rng(52))
    leaves = [entry for row in matrices for entry in row]
    assert len(leaves) == 4
    assert all(entry.coupled_variable_group is df.coupled_variable_group for entry in leaves)


def test_frequency_severity_ufunc_checks_simulation_axis_before_merging() -> None:
    events = FreqSevSims([0, 2, 2], [3.0, 4.0, 5.0], 3)
    mismatched = StochasticScalar([10.0])
    event_group = events.coupled_variable_group
    other_group = mismatched.coupled_variable_group
    original = asnumpy(events.values).copy()

    with pytest.raises(ValueError, match="matching n_sims"):
        np.add(events, mismatched)

    assert events.coupled_variable_group is event_group
    assert mismatched.coupled_variable_group is other_group
    np.testing.assert_array_equal(asnumpy(events.values), original)


def test_frequency_severity_reductions_link_to_event_group() -> None:
    events = FreqSevSims([0, 2, 2], [3.0, 4.0, 5.0], 3)
    derived = events + 1.0
    total = derived.aggregate()
    count = derived.count()
    assert all(item.coupled_variable_group is events.coupled_variable_group for item in (derived, total, count))
    np.testing.assert_allclose(asnumpy(total.values), [4.0, 0.0, 11.0])
    np.testing.assert_allclose(asnumpy(count.values), [1.0, 0.0, 2.0])


def test_coupling_helper_does_not_retain_temporary_variables() -> None:
    source = StochasticScalar([1.0, 2.0, 3.0])
    output = StochasticScalar([4.0, 5.0, 6.0])
    temporary = StochasticScalar([7.0, 8.0, 9.0])
    weak = weakref.ref(temporary)
    couple_stochastic_results((output,), (source, temporary))
    del temporary
    gc.collect()
    assert weak() is None
    assert len(source.coupled_variable_group) == 2


def test_hypergeometric_probability_methods_preserve_parameter_ancestry() -> None:
    from scipy.stats import hypergeom

    good = StochasticScalar([5.0, 6.0, 7.0])
    bad = StochasticScalar([4.0, 5.0, 6.0])
    draw = HyperGeometric(ngood=good, nbad=bad, n_draws=3)
    cdf = draw.cdf(1)
    quantile = draw.invcdf(0.6)
    assert all(item.coupled_variable_group is good.coupled_variable_group for item in (bad, cdf, quantile))
    np.testing.assert_allclose(asnumpy(cdf.values), hypergeom.cdf(1, [9, 11, 13], [5, 6, 7], 3))
    np.testing.assert_allclose(asnumpy(quantile.values), hypergeom.ppf(0.6, [9, 11, 13], [5, 6, 7], 3))


def test_python_divmod_links_two_outputs_without_losing_ancestry() -> None:
    numerator = StochasticScalar([8.0, 9.0, 10.0])
    divisor = StochasticScalar([2.0, 4.0, 3.0])
    quotient, remainder = divmod(numerator, divisor)
    assert quotient.coupled_variable_group is numerator.coupled_variable_group
    assert remainder.coupled_variable_group is numerator.coupled_variable_group
    assert divisor.coupled_variable_group is numerator.coupled_variable_group
    np.testing.assert_allclose(asnumpy(quotient.values), [4.0, 2.0, 3.0])
    np.testing.assert_allclose(asnumpy(remainder.values), [0.0, 1.0, 1.0])


def test_divmod_mismatched_axes_do_not_merge() -> None:
    numerator = StochasticScalar([1.0, 2.0, 3.0])
    divisor = StochasticScalar([2.0])
    first_group = numerator.coupled_variable_group
    second_group = divisor.coupled_variable_group
    with pytest.raises(ValueError, match="matching n_sims"):
        divmod(numerator, divisor)
    assert numerator.coupled_variable_group is first_group
    assert divisor.coupled_variable_group is second_group


def test_mixture_prevalidates_all_stochastic_parameter_axes_before_rng_use() -> None:
    rate = StochasticScalar([1.0, 2.0, 3.0])
    location = StochasticScalar([0.0])
    mixture = HyperExponential(weights=[0.3, 0.7], rates=[rate, 2.0], loc=location)
    rng = np.random.default_rng(270)

    with pytest.raises(ValueError, match="matching n_sims"):
        mixture.generate(3, rng)

    assert rng.random() == np.random.default_rng(270).random()
    assert rate.coupled_variable_group is not location.coupled_variable_group

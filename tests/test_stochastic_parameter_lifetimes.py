"""Regression tests for stochastic parameters and coupling-group object lifetimes."""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest
import scipy.special

from pal import FreqSevSims, FrequencySeverityModel, StochasticScalar
from pal.copulas import apply_copula
from pal.distributions import Gamma, Normal, Poisson
from tests._assertions import assert_allclose, assert_array_equal, host_values


def _expected_copula_selection(marginal: np.ndarray, sample_scores: list[int]) -> np.ndarray:
    """Use Python rank matching with stable ties, independently of PAL."""
    sorted_positions = sorted(range(len(marginal)), key=lambda i: (marginal[i], i))
    sorted_scores = sorted(sample_scores)
    return np.asarray([sorted_positions[sorted_scores.index(score)] for score in sample_scores])


def test_stochastic_frequency_parameter_and_compound_losses_remain_aligned() -> None:
    """Independent RNG draws and copula reordering must respect scenario rates."""
    n_sims = 6
    exposure = StochasticScalar([1.0, 3.0, 0.0, 4.0, 7.0, 2.0])
    rate = exposure * 0.75
    seed = 287
    rng = np.random.default_rng(seed)
    losses = FrequencySeverityModel(Poisson(mean=rate), Gamma(alpha=2.0, theta=3.0)).generate(n_sims, rng)

    reference_rng = np.random.default_rng(seed)
    initial_exposure = host_values(exposure).copy()
    initial_rate = host_values(rate).copy()
    counts = reference_rng.poisson(initial_rate, size=n_sims)
    severities = reference_rng.gamma(shape=2.0, scale=3.0, size=int(counts.sum()))
    assert_array_equal(losses.sim_index, np.repeat(np.arange(n_sims), counts))
    assert_allclose(losses.values, severities, rtol=0, atol=1e-12)
    assert_array_equal(losses.count(), counts)

    aggregate = losses.aggregate()
    adjusted = losses * exposure
    adjusted_aggregate = adjusted.aggregate()
    before_aggregate = host_values(aggregate).copy()
    before_values = host_values(losses).copy()
    before_counts = counts.copy()
    first = StochasticScalar(np.arange(n_sims, dtype=float))
    scores = [4, 0, 3, 1, 5, 2]
    expected_selection = _expected_copula_selection(before_aggregate, scores)

    apply_copula(
        [first, aggregate],
        [StochasticScalar(list(range(n_sims))), StochasticScalar(scores)],
    )

    assert_array_equal(exposure, initial_exposure[expected_selection])
    assert_array_equal(rate, initial_rate[expected_selection])
    assert_array_equal(losses.count(), before_counts[expected_selection])
    assert_allclose(aggregate, before_aggregate[expected_selection])
    assert_allclose(losses.values, before_values)
    assert_allclose(adjusted.aggregate(), adjusted_aggregate)
    assert_allclose(adjusted_aggregate, host_values(aggregate) * host_values(exposure))
    assert_allclose(adjusted.values, host_values(losses) * host_values(exposure)[host_values(losses.sim_index)])
    assert all(
        item.coupled_variable_group is first.coupled_variable_group
        for item in (exposure, rate, losses, aggregate, adjusted, adjusted_aggregate)
    )


class _FixedClaimCounts:
    """Generate deterministic claim counts, including zero-claim simulations."""

    def __init__(self, counts: list[int]) -> None:
        self.counts = counts

    def generate(self, n_sims: int, rng: object) -> StochasticScalar:
        assert n_sims == len(self.counts)
        return StochasticScalar(self.counts)


def test_stochastic_severity_parameters_are_per_simulation_not_per_claim() -> None:
    """All claims in one simulation must use the same stochastic parameters."""
    counts = [2, 0, 3, 1, 0]
    n_sims = len(counts)
    indices = np.repeat(np.arange(n_sims), counts)
    location = StochasticScalar([10.0, 30.0, 50.0, 70.0, 90.0])
    scale = StochasticScalar([1.0, 2.0, 3.0, 4.0, 5.0])
    original_location = host_values(location).copy()
    original_scale = host_values(scale).copy()
    seed = 731
    distribution = Normal(mu=location, sigma=scale)
    model = FrequencySeverityModel(_FixedClaimCounts(counts), distribution)
    losses = model.generate(n_sims, np.random.default_rng(seed))

    z = scipy.special.ndtri(np.random.default_rng(seed).uniform(size=len(indices)))
    expected_severities = original_location[indices] + original_scale[indices] * z
    assert_array_equal(losses.sim_index, indices)
    assert_allclose(losses.values, expected_severities, rtol=1e-10, atol=1e-10)
    assert_array_equal(losses.count(), counts)
    assert_allclose(losses.aggregate(), np.bincount(indices, weights=expected_severities, minlength=n_sims))
    # The model must not replace the distribution's scenario-level parameters.
    assert distribution._params["mu"] is location
    assert distribution._params["sigma"] is scale

    aggregate = losses.aggregate()
    driver = StochasticScalar([1, 2, 3, 4, 5])
    scores = [4, 1, 3, 0, 2]
    original_aggregates = host_values(aggregate).copy()
    selection = _expected_copula_selection(original_aggregates, scores)
    apply_copula(
        [driver, aggregate],
        [StochasticScalar([0, 1, 2, 3, 4]), StochasticScalar(scores)],
    )

    assert_array_equal(location, original_location[selection])
    assert_array_equal(scale, original_scale[selection])
    assert_array_equal(losses.sim_index, np.argsort(selection)[indices])
    assert_allclose(losses.values, expected_severities, rtol=1e-10, atol=1e-10)
    assert_allclose(aggregate, original_aggregates[selection])
    assert_allclose(
        (host_values(losses) - host_values(location)[host_values(losses.sim_index)])
        / host_values(scale)[host_values(losses.sim_index)],
        z,
        rtol=1e-10,
        atol=1e-10,
    )
    assert all(
        variable.coupled_variable_group is driver.coupled_variable_group
        for variable in (location, scale, losses, aggregate)
    )


def test_gamma_severity_shape_and_scale_follow_simulation_claim_counts() -> None:
    """Direct gamma sampling must expand both stochastic parameters by claim count."""
    counts = [0, 3, 1, 2, 0, 4]
    indices = np.repeat(np.arange(len(counts)), counts)
    shape = StochasticScalar([0.5, 1.0, 2.0, 3.0, 5.0, 7.0])
    scale = StochasticScalar([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    original_shape = host_values(shape).copy()
    original_scale = host_values(scale).copy()
    distribution = Gamma(alpha=shape, theta=scale)
    seed = 129
    actual = FrequencySeverityModel(_FixedClaimCounts(counts), distribution).generate(
        len(counts), np.random.default_rng(seed)
    )
    expected_values = np.random.default_rng(seed).gamma(
        original_shape[indices], original_scale[indices], size=len(indices)
    )
    assert_array_equal(actual.sim_index, indices)
    assert_allclose(actual.values, expected_values, rtol=1e-10, atol=1e-10)
    assert_array_equal(actual.count(), counts)
    assert distribution._params["alpha"] is shape
    assert distribution._params["theta"] is scale
    assert actual.coupled_variable_group is shape.coupled_variable_group
    assert actual.coupled_variable_group is scale.coupled_variable_group


def test_no_claims_with_stochastic_severity_preserves_coupling() -> None:
    """Zero events should not index past parameters or detach the model result."""
    counts = [0, 0, 0, 0]
    shape = StochasticScalar([1.0, 2.0, 3.0, 4.0])
    scale = StochasticScalar([2.0, 4.0, 6.0, 8.0])
    actual = FrequencySeverityModel(_FixedClaimCounts(counts), Gamma(alpha=shape, theta=scale)).generate(
        len(counts), np.random.default_rng(75)
    )
    assert_array_equal(actual.sim_index, [])
    assert_array_equal(actual.values, [])
    assert_array_equal(actual.count(), counts)
    assert actual.coupled_variable_group is shape.coupled_variable_group
    assert actual.coupled_variable_group is scale.coupled_variable_group


def test_stochastic_severity_parameter_rejects_event_length_input() -> None:
    """A per-claim parameter vector must not be mistaken for scenario parameters."""
    counts = [2, 0, 3]
    shape = StochasticScalar([1.0, 1.5, 2.0, 2.5, 3.0])
    model = FrequencySeverityModel(_FixedClaimCounts(counts), Gamma(alpha=shape, theta=2.0))
    with pytest.raises(ValueError, match="one value per simulation"):
        model.generate(len(counts), np.random.default_rng(13))


def test_event_copy_transform_and_detached_clone_under_copula_reordering() -> None:
    """Copies keep their own buffers; only coupled copies are permuted."""
    losses = FreqSevSims([0, 0, 2, 3], [10.0, 20.0, 30.0, 40.0], 4)
    copied = losses.copy()
    transformed = losses.apply(lambda values: values * 2)
    clipped = np.minimum(losses, 25.0)
    detached = losses.deep_copy()
    old_indices = host_values(losses.sim_index).copy()
    old_values = host_values(losses).copy()
    aggregate = losses.aggregate()

    anchor = StochasticScalar([1, 2, 3, 4])
    apply_copula(
        [anchor, aggregate],
        [StochasticScalar([0, 1, 2, 3]), StochasticScalar([2, 0, 3, 1])],
    )

    assert not np.array_equal(host_values(losses.sim_index), old_indices)
    for dependent, expected in (
        (copied, old_values),
        (transformed, 2 * old_values),
        (clipped, np.minimum(old_values, 25.0)),
    ):
        assert_array_equal(dependent.sim_index, losses.sim_index)
        assert_allclose(dependent.values, expected)
        assert dependent.coupled_variable_group is losses.coupled_variable_group

    assert_array_equal(detached.sim_index, old_indices)
    assert_allclose(detached.values, old_values)
    assert detached.coupled_variable_group is not losses.coupled_variable_group
    assert_allclose(copied.aggregate(), losses.aggregate())
    assert_allclose(transformed.aggregate(), 2 * host_values(losses.aggregate()))

    losses.values[0] = 99.0
    assert_allclose(copied.values, old_values)
    assert_allclose(detached.values, old_values)


def test_stochastic_scalar_wrapper_remains_in_group_across_copula_reordering() -> None:
    """Constructing a scalar from another scalar must preserve scenario identity."""
    source = StochasticScalar([3, 1, 4, 2])
    wrapper = StochasticScalar(source)
    derived = wrapper * 10
    driver = StochasticScalar([1, 2, 3, 4])
    apply_copula(
        [driver, source],
        [StochasticScalar([0, 1, 2, 3]), StochasticScalar([2, 0, 3, 1])],
    )
    assert_array_equal(wrapper, source)
    assert_allclose(derived, 10 * host_values(source))
    assert wrapper.coupled_variable_group is source.coupled_variable_group
    assert derived.coupled_variable_group is driver.coupled_variable_group


def test_discarded_intermediate_variables_are_weakly_held() -> None:
    """Dead derived variables must not survive solely through a coupling group."""
    losses = FreqSevSims([0, 2, 2], [10.0, 30.0, 40.0], 4)
    aggregate = losses.aggregate()
    transformed = (aggregate + 5) * 2
    group = losses.coupled_variable_group
    assert len(group) == 3
    aggregate_ref = weakref.ref(aggregate)
    transformed_ref = weakref.ref(transformed)
    del aggregate
    del transformed
    gc.collect()

    assert aggregate_ref() is None
    assert transformed_ref() is None
    assert len(group) == 1
    assert list(group)[0] is losses

    independent = StochasticScalar([1, 2, 3, 4])
    remaining_aggregate = losses.aggregate()
    apply_copula(
        [independent, remaining_aggregate],
        [StochasticScalar([0, 1, 2, 3]), StochasticScalar([2, 0, 3, 1])],
    )
    assert_allclose(losses.aggregate(), remaining_aggregate)
    assert losses.coupled_variable_group is independent.coupled_variable_group


def test_compound_distribution_does_not_keep_dead_parameters_alive() -> None:
    """The dependency graph must not retain a parameter after model disposal."""
    rate = StochasticScalar([0.5, 1.0, 2.0, 0.0])
    model = FrequencySeverityModel(Poisson(mean=rate), Gamma(alpha=2.0, theta=3.0))
    losses = model.generate(4, np.random.default_rng(118))
    group = losses.coupled_variable_group
    assert rate in group

    rate_ref = weakref.ref(rate)
    del rate
    del model
    gc.collect()

    assert rate_ref() is None
    assert len(group) == 1
    assert list(group)[0] is losses
    assert_array_equal(losses.count(), np.bincount(host_values(losses.sim_index), minlength=losses.n_sims))

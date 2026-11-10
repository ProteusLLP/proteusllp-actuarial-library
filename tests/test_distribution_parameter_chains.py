"""End-to-end tests for distribution outputs used as stochastic parameters.

These cover entire dependency chains rather than checking only one generator.
Numerical references use independent NumPy and SciPy operations.
"""

from __future__ import annotations

import numpy as np
import scipy.special

from pal import FrequencySeverityModel, StochasticScalar
from pal.copulas import apply_copula
from pal.distributions import Gamma, LogNormal, Normal, Poisson
from tests._assertions import assert_allclose, assert_array_equal, host_values


def _copula_permutation(marginal: np.ndarray, copula_scores: list[int]) -> np.ndarray:
    """Stable rank matching when the first copula margin is already sorted."""
    sorted_positions = np.argsort(marginal, kind="stable")
    sample_ranks = np.argsort(np.argsort(copula_scores, kind="stable"), kind="stable")
    return sorted_positions[sample_ranks]


def test_three_stage_distribution_parameter_chain_reorders_every_ancestor() -> None:
    """Gamma -> Poisson -> Normal preserves all parameters and seed-level draws."""
    n_sims = 7
    seed = 419
    generator = np.random.default_rng(seed)

    root = Gamma(alpha=2.5, theta=3.0).generate(n_sims, generator)
    rate = 0.25 + root / 2.0
    counts = Poisson(mean=rate).generate(n_sims, generator)
    location = 1.0 + counts
    output = Normal(mu=location, sigma=0.5).generate(n_sims, generator)

    reference = np.random.default_rng(seed)
    expected_root = reference.gamma(2.5, 3.0, size=n_sims)
    expected_rate = 0.25 + expected_root / 2.0
    expected_counts = reference.poisson(expected_rate, size=n_sims)
    expected_location = 1.0 + expected_counts
    expected_output = expected_location + 0.5 * scipy.special.ndtri(reference.uniform(size=n_sims))

    assert_allclose(root, expected_root)
    assert_allclose(rate, expected_rate)
    assert_array_equal(counts, expected_counts)
    assert_allclose(location, expected_location)
    assert_allclose(output, expected_output)
    assert all(
        value.coupled_variable_group is root.coupled_variable_group
        for value in (rate, counts, location, output)
    )

    scores = [5, 2, 6, 0, 4, 1, 3]
    selection = _copula_permutation(expected_output, scores)
    anchor = StochasticScalar(np.arange(n_sims))
    apply_copula(
        [anchor, output],
        [StochasticScalar(np.arange(n_sims)), StochasticScalar(scores)],
    )

    assert_allclose(root, expected_root[selection])
    assert_allclose(rate, expected_rate[selection])
    assert_array_equal(counts, expected_counts[selection])
    assert_allclose(location, expected_location[selection])
    assert_allclose(output, expected_output[selection])
    assert_allclose(rate, 0.25 + host_values(root) / 2.0)
    assert_allclose(location, 1.0 + host_values(counts))
    assert all(
        value.coupled_variable_group is anchor.coupled_variable_group
        for value in (root, rate, counts, location, output)
    )


def test_independent_distribution_parameter_branches_merge_in_downstream_gamma() -> None:
    """Two independent generators become coupled through one compound distribution."""
    n_sims = 8
    seed = 72
    generator = np.random.default_rng(seed)

    shape = Gamma(alpha=2.0, theta=1.2).generate(n_sims, generator)
    scale = LogNormal(mu=-1.5, sigma=0.2).generate(n_sims, generator)
    assert shape.coupled_variable_group is not scale.coupled_variable_group

    output = Gamma(alpha=shape, theta=scale).generate(n_sims, generator)

    reference = np.random.default_rng(seed)
    expected_shape = reference.gamma(2.0, 1.2, size=n_sims)
    expected_scale = np.exp(-1.5 + 0.2 * scipy.special.ndtri(reference.uniform(size=n_sims)))
    expected_output = reference.gamma(expected_shape, expected_scale, size=n_sims)
    assert_allclose(shape, expected_shape)
    assert_allclose(scale, expected_scale)
    assert_allclose(output, expected_output)
    assert output.coupled_variable_group is shape.coupled_variable_group
    assert output.coupled_variable_group is scale.coupled_variable_group

    scores = [5, 2, 7, 4, 0, 6, 1, 3]
    selection = _copula_permutation(expected_output, scores)
    driver = StochasticScalar(np.arange(n_sims))
    apply_copula(
        [driver, output],
        [StochasticScalar(np.arange(n_sims)), StochasticScalar(scores)],
    )
    assert_allclose(shape, expected_shape[selection])
    assert_allclose(scale, expected_scale[selection])
    assert_allclose(output, expected_output[selection])
    assert all(
        value.coupled_variable_group is driver.coupled_variable_group
        for value in (shape, scale, output)
    )


def test_chained_frequency_and_severity_distributions_preserve_both_ancestry_branches() -> None:
    """Independent upstream chains merge through compound losses, including empty sims."""
    n_sims = 8
    seed = 1134
    rng = np.random.default_rng(seed)

    frequency_driver = Gamma(alpha=1.5, theta=0.8).generate(n_sims, rng)
    rate = Gamma(alpha=2.0, theta=frequency_driver + 0.1).generate(n_sims, rng)
    severity_driver = LogNormal(mu=-0.7, sigma=0.3).generate(n_sims, rng)
    scale = Gamma(alpha=2.0, theta=severity_driver).generate(n_sims, rng)
    assert rate.coupled_variable_group is not scale.coupled_variable_group

    model = FrequencySeverityModel(Poisson(mean=rate), Gamma(alpha=2.0, theta=scale))
    losses = model.generate(n_sims, rng)
    aggregate = losses.aggregate()

    reference = np.random.default_rng(seed)
    expected_frequency_driver = reference.gamma(1.5, 0.8, size=n_sims)
    expected_rate = reference.gamma(2.0, expected_frequency_driver + 0.1, size=n_sims)
    expected_severity_driver = np.exp(
        -0.7 + 0.3 * scipy.special.ndtri(reference.uniform(size=n_sims))
    )
    expected_scale = reference.gamma(2.0, expected_severity_driver, size=n_sims)
    expected_counts = reference.poisson(expected_rate, size=n_sims)
    original_indices = np.repeat(np.arange(n_sims), expected_counts)
    expected_losses = reference.gamma(2.0, expected_scale[original_indices], size=len(original_indices))
    expected_aggregate = np.bincount(original_indices, weights=expected_losses, minlength=n_sims)

    assert_allclose(frequency_driver, expected_frequency_driver)
    assert_allclose(rate, expected_rate)
    assert_allclose(severity_driver, expected_severity_driver)
    assert_allclose(scale, expected_scale)
    assert_array_equal(losses.count(), expected_counts)
    assert_array_equal(losses.sim_index, original_indices)
    assert_allclose(losses.values, expected_losses)
    assert_allclose(aggregate, expected_aggregate)
    assert all(
        value.coupled_variable_group is aggregate.coupled_variable_group
        for value in (frequency_driver, rate, severity_driver, scale, losses)
    )

    scores = [4, 7, 1, 6, 0, 3, 5, 2]
    selection = _copula_permutation(expected_aggregate, scores)
    driver = StochasticScalar(np.arange(n_sims))
    apply_copula(
        [driver, aggregate],
        [StochasticScalar(np.arange(n_sims)), StochasticScalar(scores)],
    )

    assert_allclose(frequency_driver, expected_frequency_driver[selection])
    assert_allclose(rate, expected_rate[selection])
    assert_allclose(severity_driver, expected_severity_driver[selection])
    assert_allclose(scale, expected_scale[selection])
    assert_array_equal(losses.count(), expected_counts[selection])
    assert_allclose(aggregate, expected_aggregate[selection])
    assert_allclose(losses.values, expected_losses)
    assert_array_equal(losses.sim_index, np.argsort(selection)[original_indices])
    assert_allclose(
        host_values(scale)[host_values(losses.sim_index)],
        expected_scale[original_indices],
    )
    assert all(
        value.coupled_variable_group is driver.coupled_variable_group
        for value in (frequency_driver, rate, severity_driver, scale, losses, aggregate)
    )

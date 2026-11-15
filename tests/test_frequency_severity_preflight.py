"""Regression tests for frequency-severity scenario-axis and count validation."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from pal import FrequencySeverityModel, StochasticScalar
from pal._maths import xp
from pal.distributions import DistributionBase, Gamma, Normal, Poisson
from tests._assertions import assert_allclose, assert_array_equal, host_values


class _FixedFrequency(DistributionBase):
    """Return predetermined frequency values, including invalid outputs."""

    def __init__(self, counts: list[float]) -> None:
        super().__init__()
        self.counts = counts

    def _generate(self, n_sims: int, rng: object) -> StochasticScalar:
        return StochasticScalar(xp.asarray(self.counts))


@pytest.mark.parametrize(
    "counts",
    [
        [-1.0, 0.0, 2.0],
        [0.0, 1.5, 2.0],
        [0.0, float("nan"), 2.0],
        [0.0, float("inf"), 2.0],
    ],
)
def test_invalid_frequency_counts_are_rejected_before_severity_generation(counts: list[float]) -> None:
    class _NeverDrawSeverity(DistributionBase):
        def _generate(self, n_sims: int, rng: object) -> StochasticScalar:
            raise AssertionError("Severity sampling must not run after invalid frequencies")

    model = FrequencySeverityModel(_FixedFrequency(counts), _NeverDrawSeverity())
    with pytest.raises(ValueError, match="finite, nonnegative integer counts"):
        model.generate(3, np.random.default_rng(8))


@pytest.mark.parametrize("bad_axis", ["frequency", "severity"])
def test_stochastic_axis_mismatch_is_atomic_and_preserves_rng_state(bad_axis: str) -> None:
    frequency_rate = StochasticScalar([1.0, 2.0, 3.0] if bad_axis == "severity" else [1.0, 2.0])
    severity_scale = StochasticScalar([10.0, 20.0, 30.0] if bad_axis == "frequency" else [10.0, 20.0])
    model = FrequencySeverityModel(Poisson(mean=frequency_rate), Gamma(alpha=2.0, theta=severity_scale))
    rng = np.random.default_rng(31)
    initial_state = deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="n_sims|one value per simulation"):
        model.generate(3, rng)
    assert rng.bit_generator.state == initial_state
    assert frequency_rate.coupled_variable_group is not severity_scale.coupled_variable_group


def test_nonpositive_simulation_count_rejected_before_sampling() -> None:
    rng = np.random.default_rng(13)
    initial_state = deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="n_sims must be >= 1"):
        FrequencySeverityModel(Poisson(2.0), Gamma(2.0, 10.0)).generate(0, rng)
    assert rng.bit_generator.state == initial_state


def test_compound_simulation_preserves_parameter_ancestry_and_event_alignment() -> None:
    n_sims = 5
    rate = StochasticScalar([0.0, 1.0, 2.0, 0.0, 3.0])
    scale = StochasticScalar([10.0, 20.0, 30.0, 40.0, 50.0])
    frequency = Poisson(rate)
    severity = Gamma(alpha=2.0, theta=scale)
    rng = np.random.default_rng(911)
    result = FrequencySeverityModel(frequency, severity).generate(n_sims, rng)

    reference = np.random.default_rng(911)
    expected_counts = reference.poisson(host_values(rate), size=n_sims)
    expected_index = np.repeat(np.arange(n_sims), expected_counts)
    if len(expected_index):
        expected_losses = reference.gamma(2.0, host_values(scale)[expected_index], size=len(expected_index))
        assert_allclose(result.values, expected_losses)
    assert_array_equal(result.sim_index, expected_index)
    assert_array_equal(result.count(), expected_counts)
    assert result.coupled_variable_group is rate.coupled_variable_group
    assert result.coupled_variable_group is scale.coupled_variable_group
    assert severity._params["theta"] is scale


def test_zero_claims_preserve_simulation_level_coupling_without_event_temporaries() -> None:
    rate = StochasticScalar([0.0, 0.0, 0.0])
    scale = StochasticScalar([10.0, 20.0, 30.0])
    result = FrequencySeverityModel(Poisson(rate), Gamma(alpha=2.0, theta=scale)).generate(3, np.random.default_rng(21))
    assert len(result.values) == 0
    assert_array_equal(result.count(), [0, 0, 0])
    assert result.coupled_variable_group is rate.coupled_variable_group
    assert result.coupled_variable_group is scale.coupled_variable_group


def test_fractional_continuous_frequency_distribution_is_rejected() -> None:
    rate = StochasticScalar([1.0, 2.0, 3.0])
    scale = StochasticScalar([10.0, 20.0, 30.0])
    model = FrequencySeverityModel(Normal(mu=rate, sigma=0.1), Gamma(alpha=2.0, theta=scale))
    with pytest.raises(ValueError, match="finite, nonnegative integer counts"):
        model.generate(3, np.random.default_rng(25))
    assert rate.coupled_variable_group is not scale.coupled_variable_group

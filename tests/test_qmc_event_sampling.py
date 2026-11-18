"""Packed event-indexed RQMC must equal the full Sobol claim-ordinal design."""

from __future__ import annotations

import numpy as np
import pytest

from pal._maths import asnumpy, xp
from pal.distributions import Gamma, InverseGaussian, Poisson, Uniform
from pal.frequency_severity import FrequencySeverityModel
from pal.qmc import RandomizedQMCGenerator
from pal.stochastic_scalar import StochasticScalar


@pytest.mark.parametrize("dimensions", [3, 8, 32])
def test_event_coordinates_match_full_draws_across_blocks(dimensions: int) -> None:
    """A packed lookup must preserve every sampled coordinate and next RNG state."""
    n = 128
    counts_host = np.random.default_rng(391).poisson(6.0, size=n).astype(np.int64)
    counts = xp.asarray(counts_host)
    paths = xp.repeat(xp.arange(n), counts)
    starts = xp.cumsum(counts) - counts
    ordinals = xp.arange(len(paths)) - starts[paths]
    maximum = int(counts_host.max())

    sequential = RandomizedQMCGenerator(seed=812, dimensions=dimensions)
    packed = RandomizedQMCGenerator(seed=812, dimensions=dimensions)
    np.testing.assert_array_equal(asnumpy(sequential.random(n)), asnumpy(packed.random(n)))

    expected = np.empty(len(paths))
    path_ids = asnumpy(paths)
    claim_ids = asnumpy(ordinals)
    for ordinal in range(maximum):
        coordinate = asnumpy(sequential.random(n))
        active = claim_ids == ordinal
        expected[active] = coordinate[path_ids[active]]

    actual = asnumpy(packed._event_uniforms(n, paths, ordinals, maximum))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(asnumpy(sequential.random(n)), asnumpy(packed.random(n)))


@pytest.mark.parametrize("stochastic_scale", [False, True])
def test_packed_frequency_severity_matches_original_ordinal_sampling(stochastic_scale: bool) -> None:
    """Packed Gamma severities match full-scenario generation claim for claim."""
    n = 512
    scale = StochasticScalar(xp.linspace(30, 50, n)) if stochastic_scale else 40.0
    frequency = Poisson(2.0)
    severity = Gamma(alpha=2.5, theta=scale)
    model = FrequencySeverityModel(frequency, severity)
    packed = RandomizedQMCGenerator(seed=2026, dimensions=32)
    reference = RandomizedQMCGenerator(seed=2026, dimensions=32)

    result = model.generate(n, rng=packed)
    counts = asnumpy(frequency.generate(n, rng=reference).values).astype(int)
    offsets = np.cumsum(counts) - counts
    expected = np.empty(int(counts.sum()))
    for ordinal in range(int(counts.max())):
        draws = asnumpy(severity.generate(n, rng=reference).values)
        active = counts > ordinal
        expected[offsets[active] + ordinal] = draws[active]

    np.testing.assert_allclose(asnumpy(result.values), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(asnumpy(packed.random(n)), asnumpy(reference.random(n)))
    if stochastic_scale:
        assert scale.coupled_variable_group is result.coupled_variable_group


def test_frequency_severity_legacy_inverse_gaussian_fallback() -> None:
    """Severities without an inverse CDF retain the established generator."""
    model = FrequencySeverityModel(Poisson(2.0), InverseGaussian(mu=2.0, lambda_=4.0))
    result = model.generate(128, rng=RandomizedQMCGenerator(seed=31, dimensions=8))
    assert len(result.values) == len(result.sim_index)
    assert np.all(np.isfinite(asnumpy(result.values)))


def test_packed_date_severity_and_no_claims() -> None:
    """Non-numeric severity types and the zero-event result remain supported."""
    from datetime import date

    severity = Uniform(date(2026, 1, 1), date(2026, 1, 31))
    result = FrequencySeverityModel(Poisson(2.0), severity).generate(
        128, rng=RandomizedQMCGenerator(seed=100, dimensions=8)
    )
    assert result.values.dtype.kind == "M"
    empty = FrequencySeverityModel(Poisson(0), severity).generate(
        128, rng=RandomizedQMCGenerator(seed=100, dimensions=8)
    )
    assert len(empty.values) == 0
    assert empty.values.dtype.kind == "M"

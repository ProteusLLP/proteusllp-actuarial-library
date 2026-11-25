"""Constant one-element StochasticScalar regression tests on CPU and GPU.

The dedicated GPU workflow runs these same tests with PAL_USE_GPU=1. MC and
RQMC must share the same singleton broadcasting and coupling semantics.
"""

from __future__ import annotations

import numpy as np
import pytest

from pal import StochasticScalar
from pal._maths import asnumpy, xp
from pal.config import config, set_random_seed, set_sampling_method
from pal.distributions import Gamma, Poisson
from pal.frequency_severity import FrequencySeverityModel
from pal.types import SamplingMethod


@pytest.fixture(params=[SamplingMethod.MC, SamplingMethod.RQMC])
def sampling(request):
    """Exercise both sampling engines and restore the global configuration."""
    old_method, old_dimensions, old_seed, old_rng = (
        config.sampling_method,
        config.qmc_dimensions,
        config.seed,
        config.rng,
    )
    set_sampling_method(request.param, dimensions=8)
    set_random_seed(14031)
    try:
        yield request.param
    finally:
        config.sampling_method = old_method
        config.qmc_dimensions = old_dimensions
        config.seed = old_seed
        config.rng = old_rng


def test_singleton_arithmetic_and_numpy_preserve_scenario_ancestry(sampling) -> None:
    """A singleton constant remains length one while the result has all paths."""
    constant = StochasticScalar([5.0])
    losses = Gamma(2.0, 3.0).generate(128)
    result = constant + losses
    assert result.n_sims == 128
    np.testing.assert_allclose(asnumpy(result.values), asnumpy(losses.values) + 5.0)
    assert result.coupled_variable_group is losses.coupled_variable_group
    assert constant.coupled_variable_group is not losses.coupled_variable_group
    assert constant.n_sims == 1
    np.testing.assert_allclose(asnumpy(constant.values), [5.0])

    result2 = np.maximum(losses, constant)
    assert result2.n_sims == 128
    np.testing.assert_allclose(asnumpy(result2.values), np.maximum(asnumpy(losses.values), 5.0))
    assert result2.coupled_variable_group is losses.coupled_variable_group

    comparison = constant > 0
    selected = np.where(comparison, losses, 0.0)
    assert selected.n_sims == 128
    assert selected.coupled_variable_group is losses.coupled_variable_group
    np.testing.assert_allclose(asnumpy(selected.values), asnumpy(losses.values))
    assert isinstance(result.values, type(losses.values))


def test_singleton_ufunc_output_and_coupling(sampling) -> None:
    """A full output may be updated, but a singleton output may not expand."""
    constant = StochasticScalar([2.5])
    path = StochasticScalar(xp.arange(16, dtype=float))
    out = StochasticScalar(xp.zeros(16))
    np.add(constant, path, out=out)
    np.testing.assert_allclose(asnumpy(out.values), np.arange(16) + 2.5)
    assert out.coupled_variable_group is path.coupled_variable_group
    assert constant.coupled_variable_group is not path.coupled_variable_group

    bad = StochasticScalar([0.0])
    group = bad.coupled_variable_group
    with pytest.raises(ValueError, match="out= destination"):
        np.add(constant, path, out=bad)
    assert bad.coupled_variable_group is group
    np.testing.assert_allclose(asnumpy(bad.values), [0.0])


def test_singleton_distribution_parameters_under_mc_and_rqmc(sampling) -> None:
    constant_shape = StochasticScalar([2.0])
    varying_scale = StochasticScalar(xp.linspace(1.0, 3.0, 128))
    distribution = Gamma(alpha=constant_shape, theta=varying_scale)
    losses = distribution.generate(128)
    assert losses.n_sims == 128
    assert bool(xp.all(xp.isfinite(losses.values)))
    assert losses.coupled_variable_group is varying_scale.coupled_variable_group
    assert constant_shape.coupled_variable_group is not losses.coupled_variable_group
    assert constant_shape.n_sims == 1

    cdf = distribution.cdf(4.0)
    assert cdf.n_sims == 128
    assert cdf.coupled_variable_group is varying_scale.coupled_variable_group
    assert bool(xp.all((cdf.values >= 0) & (cdf.values <= 1)))


def test_singleton_parameters_in_frequency_severity(sampling) -> None:
    """Frequency and claim severity can use length-one stochastic parameters."""
    frequency_mean = StochasticScalar([2.0])
    severity_shape = StochasticScalar([2.0])
    stochastic_scale = StochasticScalar(xp.linspace(0.75, 1.25, 128))
    model = FrequencySeverityModel(Poisson(frequency_mean), Gamma(severity_shape, stochastic_scale))
    events = model.generate(128)
    aggregate = events.aggregate()
    assert aggregate.n_sims == 128
    assert bool(xp.all(xp.isfinite(events.values)))
    assert bool(xp.all(aggregate.values >= 0))
    assert events.coupled_variable_group is stochastic_scale.coupled_variable_group
    assert frequency_mean.n_sims == severity_shape.n_sims == 1
    assert isinstance(events.values, xp.ndarray)


def test_non_singleton_mismatched_axes_still_rejected_atomically(sampling) -> None:
    a = StochasticScalar(xp.arange(3.0))
    b = StochasticScalar(xp.arange(2.0))
    group_a, group_b = a.coupled_variable_group, b.coupled_variable_group
    with pytest.raises(ValueError, match="matching n_sims"):
        np.add(a, b)
    assert a.coupled_variable_group is group_a
    assert b.coupled_variable_group is group_b

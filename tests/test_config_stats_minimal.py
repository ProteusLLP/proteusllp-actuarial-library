"""Tests for config and stats modules to improve coverage."""

import numpy as np
import pytest

from pal import config
from pal._maths import create_random_generator, xp
from pal.config import SamplingMethod, set_sampling_method
from pal.qmc import RandomizedQMCGenerator
from pal.stats import tvar
from pal.types import Config


def test_set_default_n_sims():
    """Test set_default_n_sims function (config.py line 18)."""
    from pal.config import set_default_n_sims

    original = config.n_sims
    try:
        # Set a new default
        set_default_n_sims(5000)
        assert config.n_sims == 5000

        # Set another value
        set_default_n_sims(10000)
        assert config.n_sims == 10000
    finally:
        # Restore original
        config.n_sims = original


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("mc", SamplingMethod.MC),
        ("pseudo", SamplingMethod.MC),
        ("rqmc", SamplingMethod.RQMC),
        (SamplingMethod.MC, SamplingMethod.MC),
        (SamplingMethod.RQMC, SamplingMethod.RQMC),
    ],
)
def test_set_sampling_method_accepts_literals_and_enum(method, expected):
    """String names and enum members select the same generators."""
    original = (config.seed, config.sampling_method, config.qmc_dimensions, config.rng)
    try:
        set_sampling_method(method, dimensions=8)
        assert config.sampling_method is expected
        assert isinstance(config.rng, RandomizedQMCGenerator) is (expected is SamplingMethod.RQMC)
    finally:
        config.seed, config.sampling_method, config.qmc_dimensions, config.rng = original


@pytest.mark.parametrize("method", ["", "invalid", "MC", "random"])
def test_set_sampling_method_rejects_invalid_literals(method):
    """Invalid strings fail without modifying sampling configuration."""
    original = (config.seed, config.sampling_method, config.qmc_dimensions, config.rng)
    with pytest.raises(ValueError, match="method must be"):
        set_sampling_method(method)
    assert (config.seed, config.sampling_method, config.qmc_dimensions, config.rng) == original


def test_sampling_method_enum_descriptions():
    """Both sampling methods provide understandable descriptions."""
    assert "Monte Carlo" in SamplingMethod.MC.description
    assert "Sobol" in SamplingMethod.RQMC.description


def test_config_initializes_rng_from_seed():
    """The configured seed determines the initial random-number stream."""
    seed = 987654321
    configured = Config(seed=seed)
    expected = create_random_generator(seed)

    assert bool(xp.array_equal(configured.rng.uniform(size=10), expected.uniform(size=10)))


def test_tvar_high_percentile_error():
    """Test tvar raises error for impossible percentile (stats.py line 74)."""
    values = np.array([1, 2, 3, 4, 5])

    # Requesting 200th percentile should raise ValueError
    with pytest.raises(ValueError, match="requires more data points"):
        tvar(values, 200)


def test_tvar_exact_boundary():
    """Test tvar at exact boundary conditions."""
    values = np.array([1, 2, 3, 4, 5])

    # Test at 100th percentile (edge case)
    result = tvar(values, 100)
    assert result == 5.0  # Should return max value

    # Test at 0th percentile
    result = tvar(values, 0)
    assert isinstance(result, float)

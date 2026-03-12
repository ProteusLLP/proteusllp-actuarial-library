"""Tests for config and stats modules to improve coverage."""

import numpy as np
import pytest
from pal import config
from pal.stats import tvar


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

"""Minimal tests for stats module to improve coverage."""

import numpy as np
import pytest
from pal import FreqSevSims
from pal.stats import loss_summary, tvar


def test_tvar_empty_array():
    """Test that tvar raises ValueError for empty array."""
    with pytest.raises(ValueError):
        tvar(np.array([]), 50)


def test_tvar_single_value():
    """Test tvar with single value."""
    result = tvar(np.array([42]), 50)
    assert result == 42.0


def test_tvar_100_percentile():
    """Test tvar at 100th percentile."""
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = tvar(values, 100)
    assert result == 10.0


def test_tvar_list_of_percentiles():
    """Test tvar with list of percentiles."""
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    results = tvar(values, [50, 75, 90, 95])
    assert isinstance(results, list)
    assert len(results) == 4


def test_tvar_all_identical():
    """Test tvar when all values are identical."""
    result = tvar(np.array([5, 5, 5, 5, 5]), 90)
    assert result == 5.0


def test_tvar_negative_values():
    """Test tvar with negative values."""
    values = np.array([-10, -5, 0, 5, 10])
    result = tvar(values, 80)
    assert isinstance(result, (int, float))


def test_loss_summary_basic():
    """Test basic loss summary."""
    sim_idx = np.array([0, 0, 1, 1, 2])
    losses = np.array([100, 200, 300, 400, 500])
    freq_sev = FreqSevSims(sim_idx, losses, n_sims=3)
    summary = loss_summary(freq_sev)
    assert "Occurrence" in summary
    assert "Aggregate" in summary


def test_loss_summary_single_claim():
    """Test loss summary with single claim."""
    freq_sev = FreqSevSims(np.array([0]), np.array([1000]), n_sims=1)
    summary = loss_summary(freq_sev)
    assert isinstance(summary, dict)


def test_loss_summary_many_sims():
    """Test loss summary with many simulations."""
    np.random.seed(42)
    sim_idx = np.random.randint(0, 100, size=500)
    losses = np.random.gamma(2, 1000, size=500)
    freq_sev = FreqSevSims(sim_idx, losses, n_sims=100)
    summary = loss_summary(freq_sev)
    assert len(summary["Occurrence"]) > 0
    assert len(summary["Aggregate"]) > 0


def test_loss_summary_large_losses():
    """Test loss summary with very large loss values."""
    sim_idx = np.array([0, 1, 2, 3, 4])
    losses = np.array([1e9, 1e10, 1e11, 1e12, 1e13])
    freq_sev = FreqSevSims(sim_idx, losses, n_sims=5)
    summary = loss_summary(freq_sev)
    assert np.max(summary["Occurrence"]) > 0

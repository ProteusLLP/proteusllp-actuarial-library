"""Tests for frequency_severity module to improve coverage."""

import numpy as np
import pytest
from pal import FreqSevSims
from pal.variables import StochasticScalar


def test_freqsevsims_init_length_mismatch():
    """Test FreqSevSims raises ValueError for mismatched lengths (line 201)."""
    sim_idx = np.array([0, 1, 2])
    losses = np.array([100, 200])  # Mismatched length

    with pytest.raises(ValueError, match="Length mismatch"):
        FreqSevSims(sim_idx, losses, n_sims=3)


def test_freqsevsims_getitem_non_int():
    """Test FreqSevSims __getitem__ with non-int raises NotImplementedError (line 237)."""
    sim_idx = np.array([0, 0, 1, 1, 2])
    losses = np.array([100, 200, 300, 400, 500])
    freq_sev = FreqSevSims(sim_idx, losses, n_sims=3)

    with pytest.raises(NotImplementedError):
        _ = freq_sev["invalid"]  # type: ignore


def test_freqsevsims_len_and_iter():
    """Test __len__ and __iter__ methods (lines 241-242)."""
    sim_idx = np.array([0, 0, 1, 1, 2])
    losses = np.array([100, 200, 300, 400, 500])
    freq_sev = FreqSevSims(sim_idx, losses, n_sims=3)

    # Test __len__
    assert len(freq_sev) == 3

    # Test __iter__
    sims_list = list(freq_sev)
    assert len(sims_list) == 3
    assert all(isinstance(s, StochasticScalar) for s in sims_list)

    # Verify the actual values in each simulation
    # Sim 0 has losses at indices where sim_idx == 0
    np.testing.assert_array_equal(sims_list[0].values, [100, 200])
    # Sim 1 has losses at indices where sim_idx == 1
    np.testing.assert_array_equal(sims_list[1].values, [300, 400])
    # Sim 2 has losses at indices where sim_idx == 2
    np.testing.assert_array_equal(sims_list[2].values, [500])


def test_freqsevsims_deep_copy():
    """Test deep_copy method (line 318)."""
    sim_idx = np.array([0, 0, 1, 1, 2])
    losses = np.array([100, 200, 300, 400, 500])
    freq_sev = FreqSevSims(sim_idx, losses, n_sims=3)

    # Create deep copy
    copied = freq_sev.deep_copy()

    # Verify it's a copy with same values
    assert isinstance(copied, FreqSevSims)
    assert len(copied.values) == len(freq_sev.values)
    np.testing.assert_array_equal(copied.values, freq_sev.values)

    # Modify original - copy should not be affected
    original_value = freq_sev.values[0]
    freq_sev.values[0] = 999
    assert copied.values[0] == original_value
    assert copied.values[0] != 999


def test_freqsevsims_private_is_compatible():
    """Test _is_compatible private method (line 429)."""
    sim_idx = np.array([0, 0, 1, 1, 2])
    losses1 = np.array([100, 200, 300, 400, 500])
    losses2 = np.array([10, 20, 30, 40, 50])

    freq_sev1 = FreqSevSims(sim_idx, losses1, n_sims=3)
    freq_sev2 = FreqSevSims(sim_idx, losses2, n_sims=3)

    # Same sim_index should be compatible
    assert freq_sev1._is_compatible(freq_sev2)

    # Different sim_index should not be compatible
    different_idx = np.array([0, 1, 1, 2, 2])
    freq_sev3 = FreqSevSims(different_idx, losses2, n_sims=3)
    assert not freq_sev1._is_compatible(freq_sev3)

    # Non-FreqSevSims should not be compatible
    assert not freq_sev1._is_compatible(123)  # type: ignore


def test_freqsevsims_upsample_same_size():
    """Test upsample when n_sims equals current size (line 444)."""
    sim_idx = np.array([0, 0, 1, 1, 2])
    losses = np.array([100, 200, 300, 400, 500])
    freq_sev = FreqSevSims(sim_idx, losses, n_sims=3)

    # Upsample to same size should return a copy
    upsampled = freq_sev.upsample(3)
    assert isinstance(upsampled, FreqSevSims)
    assert upsampled.n_sims == 3
    # Modify original to verify it's a copy
    freq_sev.values[0] = 999
    assert upsampled.values[0] != 999


def test_freqsevsims_upsample_with_modulo():
    """Test upsample with non-divisible n_sims (lines 448-451)."""
    sim_idx = np.array([0, 0, 1])
    losses = np.array([100, 200, 300])
    freq_sev = FreqSevSims(sim_idx, losses, n_sims=2)

    # Upsample to 5 (not evenly divisible by 2)
    upsampled = freq_sev.upsample(5)
    assert isinstance(upsampled, FreqSevSims)
    assert upsampled.n_sims == 5
    assert len(upsampled.values) > len(freq_sev.values)

    # Verify the upsampled values contain the original values
    # Original has [100, 200, 300], upsampled should repeat them
    original_set = set(freq_sev.values)
    upsampled_set = set(upsampled.values)
    assert original_set.issubset(upsampled_set)


def test_freqsevsims_array_ufunc_with_out():
    """Test __array_ufunc__ with out parameter (line 359)."""
    sim_idx = np.array([0, 0, 1, 1, 2])
    losses = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    freq_sev = FreqSevSims(sim_idx, losses, n_sims=3)

    # Create another FreqSevSims for out parameter
    out_losses = np.zeros_like(losses)
    out_freq_sev = FreqSevSims(sim_idx, out_losses, n_sims=3)

    # Perform operation with out parameter on FreqSevSims object
    result = np.add(freq_sev, 100, out=(out_freq_sev,))

    assert result is not None
    assert isinstance(result, FreqSevSims)
    # Verify the result has the correct values (original + 100)
    expected = losses + 100
    np.testing.assert_array_equal(result.values, expected)

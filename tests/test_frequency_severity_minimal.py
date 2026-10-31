"""Tests for frequency_severity module to improve coverage."""

import numpy as np
import pytest

from pal.frequency_severity import FreqSevSims
from pal.variables import StochasticScalar
from tests._assertions import assert_array_equal, host_values


def test_freqsevsims_init_length_mismatch():
    """Test FreqSevSims raises ValueError for mismatched lengths (line 201)."""
    sim_idx = np.array([0, 1, 2])
    losses = np.array([100, 200])  # Mismatched length

    with pytest.raises(ValueError, match="Length mismatch"):
        FreqSevSims(sim_idx, losses, n_sims=3)


def test_freqsevsims_getitem_non_int():
    """Test FreqSevSims __getitem__ with non-int raises."""
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

    assert len(freq_sev) == 3
    sims_list = list(freq_sev)
    assert len(sims_list) == 3
    assert all(isinstance(s, StochasticScalar) for s in sims_list)
    assert_array_equal(sims_list[0].values, [100, 200])
    assert_array_equal(sims_list[1].values, [300, 400])
    assert_array_equal(sims_list[2].values, [500])


def test_freqsevsims_deep_copy():
    """Test deep_copy method (line 318)."""
    sim_idx = np.array([0, 0, 1, 1, 2])
    losses = np.array([100, 200, 300, 400, 500])
    freq_sev = FreqSevSims(sim_idx, losses, n_sims=3)
    copied = freq_sev.deep_copy()
    assert isinstance(copied, FreqSevSims)
    assert len(copied.values) == len(freq_sev.values)
    assert_array_equal(copied.values, freq_sev.values)
    original_value = host_values(freq_sev)[0]
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
    assert freq_sev1._is_compatible(freq_sev2)
    different_idx = np.array([0, 1, 1, 2, 2])
    freq_sev3 = FreqSevSims(different_idx, losses2, n_sims=3)
    assert not freq_sev1._is_compatible(freq_sev3)
    assert not freq_sev1._is_compatible(123)  # type: ignore


def test_freqsevsims_upsample_same_size():
    """Test upsample when n_sims equals current size (line 444)."""
    sim_idx = np.array([0, 0, 1, 1, 2])
    losses = np.array([100, 200, 300, 400, 500])
    freq_sev = FreqSevSims(sim_idx, losses, n_sims=3)
    upsampled = freq_sev.upsample(3)
    assert isinstance(upsampled, FreqSevSims)
    assert upsampled.n_sims == 3
    freq_sev.values[0] = 999
    assert upsampled.values[0] != 999


def test_freqsevsims_upsample_with_modulo():
    """Test upsample with non-divisible n_sims (lines 448-451)."""
    sim_idx = np.array([0, 0, 1])
    losses = np.array([100, 200, 300])
    freq_sev = FreqSevSims(sim_idx, losses, n_sims=2)
    upsampled = freq_sev.upsample(5)
    assert isinstance(upsampled, FreqSevSims)
    assert upsampled.n_sims == 5
    assert len(upsampled.values) > len(freq_sev.values)
    original_set = set(host_values(freq_sev))
    upsampled_set = set(host_values(upsampled))
    assert original_set.issubset(upsampled_set)

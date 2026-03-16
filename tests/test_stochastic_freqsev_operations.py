"""Comprehensive tests for StochasticScalar and FreqSevSims operations.

Tests all arithmetic operations between StochasticScalar and FreqSevSims in both
directions, verifies correct results, and ensures coupling groups are maintained.
"""

import numpy as np
import pytest
from pal.frequency_severity import FreqSevSims
from pal.variables import StochasticScalar

# =============================================================================
# Test Setup Fixtures
# =============================================================================


@pytest.fixture
def simple_freqsev():
    """Create a simple FreqSevSims for testing."""
    sim_index = np.array([0, 0, 1, 1, 2])
    values = np.array([10, 20, 30, 40, 50])
    return FreqSevSims(sim_index, values, n_sims=3)


@pytest.fixture
def simple_scalar():
    """Create a simple StochasticScalar for testing."""
    return StochasticScalar([1, 2, 3])


# =============================================================================
# Forward Operations: FreqSevSims op StochasticScalar
# =============================================================================


def test_freqsev_add_scalar(simple_freqsev, simple_scalar):
    """Test FreqSevSims + StochasticScalar."""
    fs = simple_freqsev
    sc = simple_scalar
    result = fs + sc

    # Verify result type and values
    assert isinstance(result, FreqSevSims)
    # Values should be: [10+1, 20+1, 30+2, 40+2, 50+3]
    expected = np.array([11, 21, 32, 42, 53])
    np.testing.assert_array_equal(result.values, expected)

    # Verify coupling groups are merged
    assert fs.coupled_variable_group is sc.coupled_variable_group
    assert result.coupled_variable_group is fs.coupled_variable_group


def test_freqsev_sub_scalar(simple_freqsev, simple_scalar):
    """Test FreqSevSims - StochasticScalar."""
    fs = simple_freqsev
    sc = simple_scalar
    result = fs - sc

    # Verify result type and values
    assert isinstance(result, FreqSevSims)
    # Values should be: [10-1, 20-1, 30-2, 40-2, 50-3]
    expected = np.array([9, 19, 28, 38, 47])
    np.testing.assert_array_equal(result.values, expected)

    # Verify coupling groups are merged
    assert fs.coupled_variable_group is sc.coupled_variable_group
    assert result.coupled_variable_group is fs.coupled_variable_group


def test_freqsev_mul_scalar(simple_freqsev, simple_scalar):
    """Test FreqSevSims * StochasticScalar."""
    fs = simple_freqsev
    sc = simple_scalar
    result = fs * sc

    # Verify result type and values
    assert isinstance(result, FreqSevSims)
    # Values should be: [10*1, 20*1, 30*2, 40*2, 50*3]
    expected = np.array([10, 20, 60, 80, 150])
    np.testing.assert_array_equal(result.values, expected)

    # Verify coupling groups are merged
    assert fs.coupled_variable_group is sc.coupled_variable_group
    assert result.coupled_variable_group is fs.coupled_variable_group


def test_freqsev_div_scalar(simple_freqsev, simple_scalar):
    """Test FreqSevSims / StochasticScalar."""
    fs = simple_freqsev
    sc = simple_scalar
    result = fs / sc

    # Verify result type and values
    assert isinstance(result, FreqSevSims)
    # Values should be: [10/1, 20/1, 30/2, 40/2, 50/3]
    expected = np.array([10.0, 20.0, 15.0, 20.0, 50.0 / 3.0])
    np.testing.assert_allclose(result.values, expected)

    # Verify coupling groups are merged
    assert fs.coupled_variable_group is sc.coupled_variable_group
    assert result.coupled_variable_group is fs.coupled_variable_group


def test_freqsev_pow_scalar(simple_freqsev, simple_scalar):
    """Test FreqSevSims ** StochasticScalar."""
    fs = simple_freqsev
    sc = simple_scalar
    result = fs**sc

    # Verify result type and values
    assert isinstance(result, FreqSevSims)
    # Values should be: [10**1, 20**1, 30**2, 40**2, 50**3]
    expected = np.array([10, 20, 900, 1600, 125000])
    np.testing.assert_array_equal(result.values, expected)

    # Verify coupling groups are merged
    assert fs.coupled_variable_group is sc.coupled_variable_group
    assert result.coupled_variable_group is fs.coupled_variable_group


# =============================================================================
# Reverse Operations: StochasticScalar op FreqSevSims
# =============================================================================


def test_scalar_radd_freqsev(simple_freqsev, simple_scalar):
    """Test StochasticScalar + FreqSevSims (reverse add)."""
    fs = simple_freqsev
    sc = simple_scalar
    result = sc + fs

    # Verify result type and values
    assert isinstance(result, FreqSevSims)
    # Addition is commutative: [1+10, 1+20, 2+30, 2+40, 3+50]
    expected = np.array([11, 21, 32, 42, 53])
    np.testing.assert_array_equal(result.values, expected)

    # Verify coupling groups are merged
    assert fs.coupled_variable_group is sc.coupled_variable_group
    assert result.coupled_variable_group is fs.coupled_variable_group


def test_scalar_rsub_freqsev(simple_freqsev, simple_scalar):
    """Test StochasticScalar - FreqSevSims (reverse subtract)."""
    fs = simple_freqsev
    sc = simple_scalar
    result = sc - fs

    # Verify result type and values
    assert isinstance(result, FreqSevSims)
    # Subtraction is not commutative: [1-10, 1-20, 2-30, 2-40, 3-50]
    expected = np.array([-9, -19, -28, -38, -47])
    np.testing.assert_array_equal(result.values, expected)

    # Verify coupling groups are merged
    assert fs.coupled_variable_group is sc.coupled_variable_group
    assert result.coupled_variable_group is fs.coupled_variable_group


def test_scalar_rmul_freqsev(simple_freqsev, simple_scalar):
    """Test StochasticScalar * FreqSevSims (reverse multiply)."""
    fs = simple_freqsev
    sc = simple_scalar
    result = sc * fs

    # Verify result type and values
    assert isinstance(result, FreqSevSims)
    # Multiplication is commutative: [1*10, 1*20, 2*30, 2*40, 3*50]
    expected = np.array([10, 20, 60, 80, 150])
    np.testing.assert_array_equal(result.values, expected)

    # Verify coupling groups are merged
    assert fs.coupled_variable_group is sc.coupled_variable_group
    assert result.coupled_variable_group is fs.coupled_variable_group


def test_scalar_rtruediv_freqsev(simple_freqsev, simple_scalar):
    """Test StochasticScalar / FreqSevSims (reverse divide)."""
    fs = simple_freqsev
    sc = simple_scalar
    result = sc / fs

    # Verify result type and values
    assert isinstance(result, FreqSevSims)
    # Division is not commutative: [1/10, 1/20, 2/30, 2/40, 3/50]
    expected = np.array([0.1, 0.05, 2.0 / 30.0, 2.0 / 40.0, 3.0 / 50.0])
    np.testing.assert_allclose(result.values, expected)

    # Verify coupling groups are merged
    assert fs.coupled_variable_group is sc.coupled_variable_group
    assert result.coupled_variable_group is fs.coupled_variable_group


def test_scalar_rpow_freqsev(simple_freqsev, simple_scalar):
    """Test StochasticScalar ** FreqSevSims (reverse power)."""
    # Use smaller values to avoid integer overflow
    fs = FreqSevSims(np.array([0, 0, 1, 1, 2]), np.array([2, 3, 4, 5, 6]), n_sims=3)
    sc = StochasticScalar([2, 3, 4])
    result = sc**fs

    # Verify result type and values
    assert isinstance(result, FreqSevSims)
    # Power is not commutative: [2**2, 2**3, 3**4, 3**5, 4**6]
    expected = np.array([4, 8, 81, 243, 4096])
    np.testing.assert_array_equal(result.values, expected)

    # Verify coupling groups are merged
    assert fs.coupled_variable_group is sc.coupled_variable_group
    assert result.coupled_variable_group is fs.coupled_variable_group


# =============================================================================
# Coupling Group Preservation Across Multiple Operations
# =============================================================================


def test_coupling_chain_operations():
    """Test that coupling groups are maintained across a chain of operations."""
    fs1 = FreqSevSims(np.array([0, 0, 1]), np.array([10, 20, 30]), n_sims=2)
    fs2 = FreqSevSims(np.array([0, 1, 1]), np.array([5, 15, 25]), n_sims=2)
    sc1 = StochasticScalar([2, 3])
    sc2 = StochasticScalar([10, 20])

    # Chain operations
    result1 = fs1 + sc1  # fs1, sc1, result1 coupled
    result2 = result1 * fs2  # All coupled
    result3 = result2 - sc2  # All coupled

    # Verify all are in the same coupling group
    assert fs1.coupled_variable_group is sc1.coupled_variable_group
    assert fs1.coupled_variable_group is fs2.coupled_variable_group
    assert fs1.coupled_variable_group is sc2.coupled_variable_group
    assert fs1.coupled_variable_group is result1.coupled_variable_group
    assert fs1.coupled_variable_group is result2.coupled_variable_group
    assert fs1.coupled_variable_group is result3.coupled_variable_group

    # Verify at least 6 objects are in the coupling group (may have intermediates)
    assert len(fs1.coupled_variable_group) >= 6


def test_coupling_independent_groups():
    """Test that independent variables maintain separate coupling groups."""
    fs1 = FreqSevSims(np.array([0, 0, 1]), np.array([10, 20, 30]), n_sims=2)
    sc1 = StochasticScalar([2, 3])

    fs2 = FreqSevSims(np.array([0, 1, 1]), np.array([5, 15, 25]), n_sims=2)
    sc2 = StochasticScalar([10, 20])

    # Create two independent operations
    result1 = fs1 + sc1  # fs1, sc1, result1 in one group
    result2 = fs2 * sc2  # fs2, sc2, result2 in another group

    # Groups should be different
    assert fs1.coupled_variable_group is not fs2.coupled_variable_group
    assert sc1.coupled_variable_group is not sc2.coupled_variable_group

    # Each group should have 3 variables
    assert len(fs1.coupled_variable_group) == 3
    assert len(fs2.coupled_variable_group) == 3

    # Now merge them
    _ = result1 + result2

    # All should now be in the same group
    assert fs1.coupled_variable_group is fs2.coupled_variable_group
    assert len(fs1.coupled_variable_group) == 7


# =============================================================================
# Edge Cases and Special Values
# =============================================================================


def test_freqsev_scalar_with_zeros():
    """Test operations with zero values."""
    fs = FreqSevSims(np.array([0, 0, 1]), np.array([0, 10, 20]), n_sims=2)
    sc = StochasticScalar([0, 5])

    # Addition with zeros
    result = fs + sc
    expected = np.array([0, 10, 25])
    np.testing.assert_array_equal(result.values, expected)

    # Multiplication with zeros
    result = fs * sc
    expected = np.array([0, 0, 100])
    np.testing.assert_array_equal(result.values, expected)


def test_freqsev_scalar_with_negatives():
    """Test operations with negative values."""
    fs = FreqSevSims(np.array([0, 0, 1]), np.array([-10, 20, -30]), n_sims=2)
    sc = StochasticScalar([2, -3])

    # Addition with negatives
    result = fs + sc
    expected = np.array([-8, 22, -33])
    np.testing.assert_array_equal(result.values, expected)

    # Multiplication with negatives
    result = fs * sc
    expected = np.array([-20, 40, 90])
    np.testing.assert_array_equal(result.values, expected)


def test_freqsev_scalar_single_event_per_sim():
    """Test with single event per simulation."""
    fs = FreqSevSims(np.array([0, 1, 2]), np.array([100, 200, 300]), n_sims=3)
    sc = StochasticScalar([10, 20, 30])

    # Each simulation has exactly one event
    result = fs * sc
    expected = np.array([1000, 4000, 9000])
    np.testing.assert_array_equal(result.values, expected)

    # Verify coupling
    assert fs.coupled_variable_group is sc.coupled_variable_group
    assert result.coupled_variable_group is fs.coupled_variable_group


def test_freqsev_scalar_many_events_one_sim():
    """Test with many events in one simulation."""
    # Sim 0 has 5 events, sim 1 has 0 events, sim 2 has 1 event
    fs = FreqSevSims(np.array([0, 0, 0, 0, 0, 2]), np.array([1, 2, 3, 4, 5, 10]), n_sims=3)
    sc = StochasticScalar([10, 20, 30])

    result = fs + sc
    # Events in sim 0 get +10, events in sim 2 get +30
    expected = np.array([11, 12, 13, 14, 15, 40])
    np.testing.assert_array_equal(result.values, expected)


# =============================================================================
# Operations with Scalar Constants (not StochasticScalar)
# =============================================================================


def test_freqsev_add_constant():
    """Test FreqSevSims + constant creates new result in same coupling."""
    fs = FreqSevSims(np.array([0, 0, 1]), np.array([10, 20, 30]), n_sims=2)

    result = fs + 5
    expected = np.array([15, 25, 35])
    np.testing.assert_array_equal(result.values, expected)

    # Result should be coupled with original (though may be different group object)
    assert fs in result.coupled_variable_group
    assert result in fs.coupled_variable_group


def test_constant_radd_freqsev():
    """Test constant + FreqSevSims creates new result in same coupling."""
    fs = FreqSevSims(np.array([0, 0, 1]), np.array([10, 20, 30]), n_sims=2)

    result = 5 + fs
    expected = np.array([15, 25, 35])
    np.testing.assert_array_equal(result.values, expected)

    # Result should be coupled with original
    assert fs in result.coupled_variable_group
    assert result in fs.coupled_variable_group


def test_freqsev_mul_constant():
    """Test FreqSevSims * constant creates new result in same coupling."""
    fs = FreqSevSims(np.array([0, 0, 1]), np.array([10, 20, 30]), n_sims=2)

    result = fs * 2
    expected = np.array([20, 40, 60])
    np.testing.assert_array_equal(result.values, expected)

    # Result should be coupled with original
    assert fs in result.coupled_variable_group
    assert result in fs.coupled_variable_group


def test_constant_rmul_freqsev():
    """Test constant * FreqSevSims creates new result in same coupling."""
    fs = FreqSevSims(np.array([0, 0, 1]), np.array([10, 20, 30]), n_sims=2)

    result = 2 * fs
    expected = np.array([20, 40, 60])
    np.testing.assert_array_equal(result.values, expected)

    # Result should be coupled with original
    assert fs in result.coupled_variable_group
    assert result in fs.coupled_variable_group

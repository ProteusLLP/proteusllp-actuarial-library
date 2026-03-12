"""Tests for additional numeric operations on stochastic variables.

This module tests floor division, modulo, divmod, unary operators,
rounding, and other numeric operations for StochasticScalar and FreqSevSims.
"""

import math

import numpy as np
import pytest
from pal.frequency_severity import FreqSevSims
from pal.stochastic_scalar import StochasticScalar

# =============================================================================
# Floor Division Tests
# =============================================================================


def test_stochastic_scalar_floordiv():
    """Test floor division between two StochasticScalars."""
    x = StochasticScalar([10, 21, 32, 43, 54])
    y = StochasticScalar([3, 4, 5, 6, 7])
    result = x // y

    assert isinstance(result, StochasticScalar)
    np.testing.assert_array_equal(result.values, [3, 5, 6, 7, 7])
    assert x.coupled_variable_group is result.coupled_variable_group


def test_stochastic_scalar_floordiv_scalar():
    """Test floor division of StochasticScalar by scalar."""
    x = StochasticScalar([10, 21, 32, 43, 54])
    result = x // 3

    assert isinstance(result, StochasticScalar)
    np.testing.assert_array_equal(result.values, [3, 7, 10, 14, 18])
    assert x.coupled_variable_group is result.coupled_variable_group


def test_scalar_rfloordiv_stochastic_scalar():
    """Test reverse floor division of scalar by StochasticScalar."""
    x = StochasticScalar([2, 3, 4, 5, 6])
    result = 20 // x

    assert isinstance(result, StochasticScalar)
    np.testing.assert_array_equal(result.values, [10, 6, 5, 4, 3])
    assert x.coupled_variable_group is result.coupled_variable_group


def test_freqsev_floordiv_scalar():
    """Test floor division of FreqSevSims by scalar."""
    # sim_index: [0, 0, 1, 1, 2] - sim 0 has 2 events, sim 1 has 2 events, sim 2 has 1 event
    fs = FreqSevSims(
        sim_index=np.array([0, 0, 1, 1, 2]),
        values=np.array([11, 22, 33, 44, 55]),
        n_sims=3,
    )
    result = fs // 5

    assert isinstance(result, FreqSevSims)
    np.testing.assert_array_equal(result.values, [2, 4, 6, 8, 11])
    assert fs.coupled_variable_group is result.coupled_variable_group


# =============================================================================
# Modulo Tests
# =============================================================================


def test_stochastic_scalar_mod():
    """Test modulo operation between two StochasticScalars."""
    x = StochasticScalar([10, 21, 32, 43, 54])
    y = StochasticScalar([3, 4, 5, 6, 7])
    result = x % y

    assert isinstance(result, StochasticScalar)
    np.testing.assert_array_equal(result.values, [1, 1, 2, 1, 5])
    assert x.coupled_variable_group is result.coupled_variable_group


def test_stochastic_scalar_mod_scalar():
    """Test modulo of StochasticScalar by scalar."""
    x = StochasticScalar([10, 21, 32, 43, 54])
    result = x % 7

    assert isinstance(result, StochasticScalar)
    np.testing.assert_array_equal(result.values, [3, 0, 4, 1, 5])
    assert x.coupled_variable_group is result.coupled_variable_group


def test_scalar_rmod_stochastic_scalar():
    """Test reverse modulo of scalar by StochasticScalar."""
    x = StochasticScalar([3, 4, 5, 6, 7])
    result = 20 % x

    assert isinstance(result, StochasticScalar)
    np.testing.assert_array_equal(result.values, [2, 0, 0, 2, 6])
    assert x.coupled_variable_group is result.coupled_variable_group


def test_freqsev_mod_scalar():
    """Test modulo of FreqSevSims by scalar."""
    fs = FreqSevSims(
        sim_index=np.array([0, 0, 1, 1, 2]),
        values=np.array([11, 22, 33, 44, 55]),
        n_sims=3,
    )
    result = fs % 10

    assert isinstance(result, FreqSevSims)
    np.testing.assert_array_equal(result.values, [1, 2, 3, 4, 5])
    assert fs.coupled_variable_group is result.coupled_variable_group


# =============================================================================
# Divmod Tests
# =============================================================================


def test_stochastic_scalar_divmod():
    """Test divmod operation between two StochasticScalars."""
    x = StochasticScalar([10, 21, 32, 43, 54])
    y = StochasticScalar([3, 4, 5, 6, 7])
    quotient, remainder = divmod(x, y)

    assert isinstance(quotient, StochasticScalar)
    assert isinstance(remainder, StochasticScalar)
    np.testing.assert_array_equal(quotient.values, [3, 5, 6, 7, 7])
    np.testing.assert_array_equal(remainder.values, [1, 1, 2, 1, 5])


def test_stochastic_scalar_divmod_scalar():
    """Test divmod of StochasticScalar by scalar."""
    x = StochasticScalar([10, 21, 32, 43, 54])
    quotient, remainder = divmod(x, 7)

    assert isinstance(quotient, StochasticScalar)
    assert isinstance(remainder, StochasticScalar)
    np.testing.assert_array_equal(quotient.values, [1, 3, 4, 6, 7])
    np.testing.assert_array_equal(remainder.values, [3, 0, 4, 1, 5])


def test_scalar_rdivmod_stochastic_scalar():
    """Test reverse divmod of scalar by StochasticScalar."""
    x = StochasticScalar([3, 4, 5, 6, 7])
    quotient, remainder = divmod(20, x)

    assert isinstance(quotient, StochasticScalar)
    assert isinstance(remainder, StochasticScalar)
    np.testing.assert_array_equal(quotient.values, [6, 5, 4, 3, 2])
    np.testing.assert_array_equal(remainder.values, [2, 0, 0, 2, 6])


# =============================================================================
# Unary Operator Tests
# =============================================================================


def test_stochastic_scalar_positive():
    """Test unary positive operator."""
    x = StochasticScalar([-5, -2, 0, 3, 7])
    result = +x

    assert isinstance(result, StochasticScalar)
    np.testing.assert_array_equal(result.values, [-5, -2, 0, 3, 7])
    # Should create new object but not merge coupling groups
    assert result is not x


def test_stochastic_scalar_abs():
    """Test absolute value operation."""
    x = StochasticScalar([-5, -2, 0, 3, 7])
    result = abs(x)

    assert isinstance(result, StochasticScalar)
    np.testing.assert_array_equal(result.values, [5, 2, 0, 3, 7])


def test_stochastic_scalar_negative():
    """Test unary negative operator (already exists but include for completeness)."""
    x = StochasticScalar([1, 2, 3, 4, 5])
    result = -x

    assert isinstance(result, StochasticScalar)
    np.testing.assert_array_equal(result.values, [-1, -2, -3, -4, -5])


def test_freqsev_abs():
    """Test absolute value of FreqSevSims."""
    fs = FreqSevSims(
        sim_index=np.array([0, 0, 1, 1, 2]),
        values=np.array([-10, 20, -30, 40, 50]),
        n_sims=3,
    )
    result = abs(fs)

    assert isinstance(result, FreqSevSims)
    np.testing.assert_array_equal(result.values, [10, 20, 30, 40, 50])


# =============================================================================
# Rounding Tests
# =============================================================================


def test_stochastic_scalar_round_no_digits():
    """Test round operation with no digits specified."""
    x = StochasticScalar([1.4, 2.5, 3.6, 4.7, 5.1])
    result = round(x)

    assert isinstance(result, StochasticScalar)
    np.testing.assert_array_equal(result.values, [1, 2, 4, 5, 5])


def test_stochastic_scalar_round_with_digits():
    """Test round operation with specified digits."""
    x = StochasticScalar([1.234, 2.567, 3.891, 4.456, 5.123])
    result = round(x, 1)

    assert isinstance(result, StochasticScalar)
    np.testing.assert_array_almost_equal(result.values, [1.2, 2.6, 3.9, 4.5, 5.1])


def test_stochastic_scalar_round_negative_digits():
    """Test round operation with negative digits (round to tens, hundreds, etc)."""
    x = StochasticScalar([123, 456, 789, 1234, 5678])
    result = round(x, -1)

    assert isinstance(result, StochasticScalar)
    np.testing.assert_array_equal(result.values, [120, 460, 790, 1230, 5680])


def test_freqsev_round():
    """Test round operation on FreqSevSims."""
    fs = FreqSevSims(
        sim_index=np.array([0, 0, 1, 1, 2]),
        values=np.array([1.23, 4.56, 7.89, 10.11, 12.34]),
        n_sims=3,
    )
    result = round(fs, 1)

    assert isinstance(result, FreqSevSims)
    np.testing.assert_array_almost_equal(result.values, [1.2, 4.6, 7.9, 10.1, 12.3])


# =============================================================================
# Floor, Ceil, Trunc Tests
# =============================================================================


def test_stochastic_scalar_floor():
    """Test floor operation using math.floor."""
    x = StochasticScalar([1.9, 2.1, 3.5, 4.9, 5.1])
    result = math.floor(x)

    assert isinstance(result, StochasticScalar)
    np.testing.assert_array_equal(result.values, [1, 2, 3, 4, 5])


def test_stochastic_scalar_ceil():
    """Test ceiling operation using math.ceil."""
    x = StochasticScalar([1.1, 2.1, 3.5, 4.9, 5.9])
    result = math.ceil(x)

    assert isinstance(result, StochasticScalar)
    np.testing.assert_array_equal(result.values, [2, 3, 4, 5, 6])


def test_stochastic_scalar_trunc():
    """Test truncation operation using math.trunc."""
    x = StochasticScalar([1.9, 2.1, -3.5, -4.9, 5.1])
    result = math.trunc(x)

    assert isinstance(result, StochasticScalar)
    np.testing.assert_array_equal(result.values, [1, 2, -3, -4, 5])


def test_freqsev_floor():
    """Test floor operation on FreqSevSims."""
    fs = FreqSevSims(
        sim_index=np.array([0, 0, 1, 1, 2]),
        values=np.array([1.9, 2.1, 3.5, 4.9, 5.1]),
        n_sims=3,
    )
    result = math.floor(fs)

    assert isinstance(result, FreqSevSims)
    np.testing.assert_array_equal(result.values, [1, 2, 3, 4, 5])


def test_freqsev_ceil():
    """Test ceiling operation on FreqSevSims."""
    fs = FreqSevSims(
        sim_index=np.array([0, 0, 1, 1, 2]),
        values=np.array([1.1, 2.1, 3.5, 4.9, 5.9]),
        n_sims=3,
    )
    result = math.ceil(fs)

    assert isinstance(result, FreqSevSims)
    np.testing.assert_array_equal(result.values, [2, 3, 4, 5, 6])


# =============================================================================
# Edge Cases and Coupling Tests
# =============================================================================


def test_floor_division_coupling_maintained():
    """Test that floor division maintains coupling groups."""
    x = StochasticScalar([10, 20, 30])
    y = StochasticScalar([3, 4, 5])
    result = x // y

    # All should be in same coupling group
    assert x.coupled_variable_group is y.coupled_variable_group
    assert y.coupled_variable_group is result.coupled_variable_group


def test_modulo_coupling_maintained():
    """Test that modulo maintains coupling groups."""
    x = StochasticScalar([10, 20, 30])
    y = StochasticScalar([3, 4, 7])
    result = x % y

    # All should be in same coupling group
    assert x.coupled_variable_group is y.coupled_variable_group
    assert y.coupled_variable_group is result.coupled_variable_group


def test_divmod_coupling_maintained():
    """Test that divmod maintains coupling groups."""
    x = StochasticScalar([10, 20, 30])
    y = StochasticScalar([3, 4, 7])
    quotient, remainder = divmod(x, y)

    # All should be in same coupling group
    assert x.coupled_variable_group is y.coupled_variable_group
    assert y.coupled_variable_group is quotient.coupled_variable_group
    assert quotient.coupled_variable_group is remainder.coupled_variable_group


def test_combined_operations():
    """Test combination of multiple operations."""
    x = StochasticScalar([10.7, 21.3, 32.9, 43.1, 54.6])
    y = StochasticScalar([3, 4, 5, 6, 7])

    # Complex expression using multiple new operators
    result = abs(round(x) // y + x % y)

    assert isinstance(result, StochasticScalar)
    # round(x) = [11, 21, 33, 43, 55]
    # round(x) // y = [3, 5, 6, 7, 7]
    # x % y = [1.7, 1.3, 2.9, 1.1, 5.6]
    # round(x) // y + x % y = [4.7, 6.3, 8.9, 8.1, 12.6]
    # abs(...) = [4.7, 6.3, 8.9, 8.1, 12.6]
    expected = np.array([4.7, 6.3, 8.9, 8.1, 12.6])
    np.testing.assert_array_almost_equal(result.values, expected, decimal=10)


def test_floor_division_with_negative_numbers():
    """Test floor division with negative numbers."""
    x = StochasticScalar([-10, -21, 32, -43, 54])
    y = StochasticScalar([3, 4, 5, 6, 7])
    result = x // y

    assert isinstance(result, StochasticScalar)
    # Python's floor division rounds toward negative infinity
    np.testing.assert_array_equal(result.values, [-4, -6, 6, -8, 7])


def test_modulo_with_negative_numbers():
    """Test modulo with negative numbers."""
    x = StochasticScalar([-10, -21, 32, -43, 54])
    y = StochasticScalar([3, 4, 5, 6, 7])
    result = x % y

    assert isinstance(result, StochasticScalar)
    # Python's modulo has same sign as divisor
    np.testing.assert_array_equal(result.values, [2, 3, 2, 5, 5])


def test_unhashable():
    """Test that StochasticScalar is unhashable."""
    x = StochasticScalar([1, 2, 3])

    with pytest.raises(TypeError, match="unhashable"):
        hash(x)


def test_cannot_use_in_set():
    """Test that StochasticScalar cannot be added to a set."""
    x = StochasticScalar([1, 2, 3])

    with pytest.raises(TypeError):
        {x}


def test_cannot_use_as_dict_key():
    """Test that StochasticScalar cannot be used as dict key."""
    x = StochasticScalar([1, 2, 3])

    with pytest.raises(TypeError):
        {x: "value"}

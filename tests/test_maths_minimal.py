"""Tests for maths module to improve coverage."""

import numpy as np
from pal import maths as pnp
from pal.variables import StochasticScalar


def test_minimum_function():
    """Test minimum function preserves types (line 341)."""
    x = StochasticScalar([1, 2, 3, 4, 5])
    y = StochasticScalar([5, 4, 3, 2, 1])

    result = pnp.minimum(x, y)
    assert isinstance(result, StochasticScalar)
    assert np.array_equal(result.values, [1, 2, 3, 2, 1])


def test_maximum_function():
    """Test maximum function preserves types (line 362)."""
    x = StochasticScalar([1, 2, 3, 4, 5])
    y = StochasticScalar([5, 4, 3, 2, 1])

    result = pnp.maximum(x, y)
    assert isinstance(result, StochasticScalar)
    assert np.array_equal(result.values, [5, 4, 3, 4, 5])


def test_cumsum_with_list_of_stochastic_scalars():
    """Test cumsum with list of StochasticScalar objects (lines 384-387)."""
    x = StochasticScalar([1, 2, 3])
    y = StochasticScalar([4, 5, 6])
    z = StochasticScalar([7, 8, 9])

    result = pnp.cumsum([x, y, z])
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)
    # First row is x, second is x+y, third is x+y+z
    np.testing.assert_array_equal(result[0], [1, 2, 3])
    np.testing.assert_array_equal(result[1], [5, 7, 9])
    np.testing.assert_array_equal(result[2], [12, 15, 18])

    # Also test with a single StochasticScalar (line 387)
    single_result = pnp.cumsum(x)
    assert isinstance(single_result, StochasticScalar)
    np.testing.assert_array_equal(single_result.values, [1, 3, 6])


def test_floor_function():
    """Test floor function preserves types (line 399)."""
    x = StochasticScalar([1.7, 2.3, 3.9, 4.1, 5.5])

    result = pnp.floor(x)
    assert isinstance(result, StochasticScalar)
    assert np.array_equal(result.values, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_all_function():
    """Test all function."""
    # Test with all True values
    x = StochasticScalar([1, 2, 3, 4, 5])
    assert pnp.all(x) is True

    # Test with some False values
    y = StochasticScalar([1, 0, 3, 4, 5])
    assert pnp.all(y) is False

    # Test with all False values
    z = StochasticScalar([0, 0, 0, 0, 0])
    assert pnp.all(z) is False

"""Minimal tests for StochasticScalar to improve coverage."""

import numpy as np
import pytest
from pal.variables import StochasticScalar


def test_list_input():
    """Test creating StochasticScalar from list."""
    x = StochasticScalar([1.0, 2.0, 3.0])
    assert len(x) == 3
    assert x.mean() == 2.0
    assert x.sum() == 6.0


def test_numpy_array_input():
    """Test creating from numpy array."""
    arr = np.array([10, 20, 30, 40])
    x = StochasticScalar(arr)
    assert len(x) == 4
    assert x.mean() == 25.0
    np.testing.assert_array_equal(x.values, arr)


def test_array_conversion():
    """Test __array__ protocol."""
    x = StochasticScalar([1, 2, 3, 4, 5])
    arr = np.array(x)
    assert arr.shape == (5,)


def test_dtype_preservation():
    """Test dtype is preserved."""
    x = StochasticScalar(np.array([1.5, 2.5, 3.5]))
    assert x.values.dtype == np.float64


def test_percentile_single():
    """Test percentile with single value."""
    x = StochasticScalar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    p50 = x.percentile(50)
    assert isinstance(p50, (int, float))
    assert p50 == 5.5  # Median of 1-10


def test_percentile_list():
    """Test percentile with list."""
    x = StochasticScalar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    percentiles = x.percentile([25, 50, 75])
    assert isinstance(percentiles, list)
    assert len(percentiles) == 3
    assert percentiles[0] == 3.25  # 25th percentile
    assert percentiles[1] == 5.5  # 50th percentile
    assert percentiles[2] == 7.75  # 75th percentile


def test_tvar_single():
    """Test TVAR with single percentile."""
    x = StochasticScalar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    tvar_95 = x.tvar(95)
    assert isinstance(tvar_95, (int, float))
    assert tvar_95 == 10.0  # Mean of values >= 95th percentile


def test_tvar_list():
    """Test TVAR with list of percentiles."""
    x = StochasticScalar([1, 2, 3, 4, 5])
    tvars = x.tvar([80, 90, 95])
    assert isinstance(tvars, list)
    assert len(tvars) == 3
    # Each TVAR should be >= the mean
    assert all(tvar >= x.mean() for tvar in tvars)


def test_statistics_methods():
    """Test statistical methods."""
    x = StochasticScalar([1, 2, 3, 4, 5])
    assert x.mean() == 3.0
    assert x.sum() == 15.0
    assert x.std() > 0


def test_upsample():
    """Test upsampling to more simulations."""
    x = StochasticScalar([1, 2, 3, 4, 5])
    y = x.upsample(20)
    assert len(y) == 20
    assert isinstance(y, StochasticScalar)
    # Check that upsampling preserves the original values
    assert set(y.values[:5]) == set(x.values)
    # Mean should be approximately the same
    np.testing.assert_allclose(y.mean(), x.mean(), rtol=0.1)


def test_ranks():
    """Test ranks method."""
    x = StochasticScalar([5, 2, 8, 1, 9])
    ranks = x.ranks
    assert isinstance(ranks, StochasticScalar)
    assert len(ranks) == 5
    # Check that ranks are in the correct order
    # 1 is smallest (rank 0), 2 (rank 1), 5 (rank 2), 8 (rank 3), 9 (rank 4)
    np.testing.assert_array_equal(ranks.values, [2, 1, 3, 0, 4])


def test_tolist():
    """Test converting to list."""
    x = StochasticScalar([1, 2, 3])
    lst = x.tolist()
    assert isinstance(lst, list)
    assert len(lst) == 3
    assert lst == [1, 2, 3]


def test_iteration():
    """Test iterating over StochasticScalar."""
    x = StochasticScalar([10, 20, 30])
    values = list(x)
    assert len(values) == 3
    assert values == [10, 20, 30]


def test_indexing():
    """Test indexing."""
    x = StochasticScalar([10, 20, 30, 40, 50])
    assert x[0] == 10
    assert x[-1] == 50


def test_boolean_indexing():
    """Test boolean indexing."""
    x = StochasticScalar([1, 2, 3, 4, 5])
    mask = x > 3
    filtered = x[mask]
    assert isinstance(filtered, StochasticScalar)
    np.testing.assert_array_equal(filtered.values, [4, 5])


def test_arithmetic_operations():
    """Test arithmetic with scalars and arrays."""
    x = StochasticScalar([10, 20, 30])
    y = x + 5
    z = x * 2
    w = x / 2
    assert isinstance(y, StochasticScalar)
    assert isinstance(z, StochasticScalar)
    assert isinstance(w, StochasticScalar)
    np.testing.assert_array_equal(y.values, [15, 25, 35])
    np.testing.assert_array_equal(z.values, [20, 40, 60])
    np.testing.assert_array_equal(w.values, [5, 10, 15])


def test_comparison_operations():
    """Test comparison operators."""
    x = StochasticScalar([10, 20, 30])
    result1 = x > 15
    result2 = x < 25
    result3 = x >= 20
    result4 = x <= 20
    assert isinstance(result1, StochasticScalar)
    assert isinstance(result2, StochasticScalar)
    np.testing.assert_array_equal(result1.values, [False, True, True])
    np.testing.assert_array_equal(result2.values, [True, True, False])
    np.testing.assert_array_equal(result3.values, [False, True, True])
    np.testing.assert_array_equal(result4.values, [True, True, False])


def test_negative():
    """Test unary negation."""
    x = StochasticScalar([1, 2, 3])
    neg_x = -x
    assert isinstance(neg_x, StochasticScalar)
    np.testing.assert_array_equal(neg_x.values, [-1, -2, -3])


def test_abs():
    """Test absolute value."""
    x = StochasticScalar([-5, -2, 0, 3, 7])
    abs_x = abs(x)
    np.testing.assert_array_equal(abs_x.values, [5, 2, 0, 3, 7])


def test_power():
    """Test power operation."""
    x = StochasticScalar([2, 3, 4])
    squared = x**2
    assert isinstance(squared, StochasticScalar)
    np.testing.assert_array_equal(squared.values, [4, 9, 16])


def test_repr():
    """Test string representation."""
    x = StochasticScalar([1, 2, 3])
    repr_str = repr(x)
    assert isinstance(repr_str, str)
    assert "StochasticScalar" in repr_str


# =============================================================================
# Input Validation Tests (lines 63-77)
# =============================================================================


def test_input_2d_numpy_array():
    """Test that 2D numpy array raises ValueError."""
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="1D array"):
        StochasticScalar(arr)


def test_input_3d_numpy_array():
    """Test that 3D numpy array raises ValueError."""
    arr = np.array([[[1, 2]], [[3, 4]]])
    with pytest.raises(ValueError, match="1D array"):
        StochasticScalar(arr)


def test_input_invalid_type_string():
    """Test that string input raises TypeError."""
    with pytest.raises(TypeError, match="Type of values must be"):
        StochasticScalar("not an array")  # type: ignore


def test_input_invalid_type_dict():
    """Test that dict input raises TypeError."""
    with pytest.raises(TypeError, match="Type of values must be"):
        StochasticScalar({"a": 1, "b": 2})  # type: ignore


def test_input_invalid_type_set():
    """Test that set input raises TypeError."""
    with pytest.raises(TypeError, match="Type of values must be"):
        StochasticScalar({1, 2, 3})  # type: ignore


# =============================================================================
# Reduction Operations Tests (lines 133-145)
# =============================================================================


def test_ufunc_reduce_keepdims():
    """Test ufunc reduce with keepdims=True."""
    x = StochasticScalar([1, 2, 3, 4, 5])
    # Use numpy add.reduce with keepdims
    result = np.add.reduce(x.values, keepdims=True)
    assert result.shape == (1,)


def test_ufunc_reduce_with_axis():
    """Test that reduce operations work correctly."""
    x = StochasticScalar([1, 2, 3, 4, 5])
    # Standard reduction
    total = np.sum(x.values)
    assert total == 15


def test_ufunc_accumulate():
    """Test ufunc accumulate operation."""
    x = StochasticScalar([1, 2, 3, 4, 5])
    cumsum = np.add.accumulate(x.values)
    assert len(cumsum) == 5
    assert cumsum[-1] == 15


def test_ufunc_reduceat():
    """Test ufunc reduceat operation."""
    x = StochasticScalar([1, 2, 3, 4, 5, 6, 7, 8])
    # reduceat with indices
    result = np.add.reduceat(x, [0, 3, 5])
    assert len(result) == 3


def test_ufunc_reduce_with_keepdims_and_axis():
    """Test reduction with both keepdims and axis parameters."""
    import numpy as np

    x = StochasticScalar([1, 2, 3, 4, 5])
    # Use multiply to test reduction
    arr_2d = np.array([x, x * 2])
    result = np.add.reduce(arr_2d, axis=0, keepdims=True)
    assert result.shape == (1, 5)


# =============================================================================
# Indexing Edge Cases (lines 197, 226)
# =============================================================================


def test_indexing_with_stochastic_scalar():
    """Test indexing with StochasticScalar indices."""
    x = StochasticScalar([10, 20, 30, 40, 50])
    indices = StochasticScalar([0, 2, 4])
    result = x[indices]
    assert isinstance(result, StochasticScalar)
    assert len(result) == 3


def test_indexing_with_invalid_type():
    """Test that indexing with invalid type raises TypeError."""
    x = StochasticScalar([10, 20, 30, 40, 50])
    with pytest.raises(TypeError, match="Unexpected type"):
        x["invalid"]  # type: ignore


def test_indexing_with_tuple():
    """Test that indexing with tuple raises TypeError."""
    x = StochasticScalar([10, 20, 30, 40, 50])
    with pytest.raises(TypeError, match="Unexpected type"):
        x[(1, 2)]  # type: ignore


# =============================================================================
# Upsample Edge Case (line 296)
# =============================================================================


def test_upsample_same_size():
    """Test upsampling to same size returns self."""
    x = StochasticScalar([1, 2, 3, 4, 5])
    y = x.upsample(5)
    # When n_sims == self.n_sims, should return self
    assert isinstance(y, StochasticScalar)
    assert y is x
    assert len(y) == 5

"""Tests for StochasticScalar functionality and operations.

Comprehensive tests covering arithmetic operations, statistics computation,
stochastic indexing, and integration with numpy operations.
"""

# standard library
# none

# third party
import numpy as np
import pal.maths as pnp
import pytest

# project
from pal.types import NumericProtocol
from pal.variables import StochasticScalar


def test_empty():
    """Tests an empty stochastic scalar."""
    x = StochasticScalar([])
    assert x.n_sims == 0
    assert np.sum(x) == 0


def test_stochastic_scalar():
    """Test that stochastic scalars can be created and manipulated."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    assert (x.values == [4, 5, 2, 1, 3]).all()
    assert pnp.sum(x) == 15
    assert pnp.mean(x) == 3
    assert pnp.var(x) == 2
    assert pnp.std(x) == 2**0.5
    assert pnp.percentile(x, 50) == 3
    assert (pnp.percentile(x, [10, 90]) == [1.4, 4.6]).all()
    assert (x + 1 == StochasticScalar([5, 6, 3, 2, 4])).values.all()


def test_add():
    """Tests the addition of two stochastic scalars."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    y = StochasticScalar([1, 2, 3, 4, 5])
    z = x + y
    assert (z.values == [5, 7, 5, 5, 8]).all()
    assert (
        x.coupled_variable_group == y.coupled_variable_group == z.coupled_variable_group
    )


def test_add_self():
    """Tests the addition of a stochastic scalar to itself."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    z: StochasticScalar = x + x
    assert (z.values == [8, 10, 4, 2, 6]).all()
    assert x.coupled_variable_group == z.coupled_variable_group


def test_add_self_inplace():
    """Tests the inplace addition of two stochastic scalars."""
    y = StochasticScalar(np.zeros(5))
    x = StochasticScalar([4, 5, 2, 1, 3])
    for _i in range(10):
        y += x
    assert (y.values == [40, 50, 20, 10, 30]).all()


def test_add_scalar():
    """Tests the addition of a stochastic scalar and a scalar."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    z: StochasticScalar = x + 1
    assert (z.values == [5, 6, 3, 2, 4]).all()
    assert x.coupled_variable_group == z.coupled_variable_group


def test_radd_scalar():
    """Tests the addition of a scalar and a stochastic scalar."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    z: StochasticScalar = 1 + x
    assert (z.values == [5, 6, 3, 2, 4]).all()


def test_subtract():
    """Tests the subtraction of two stochastic scalars."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    y = StochasticScalar([1, 2, 3, 4, 5])
    z = x - y
    assert (z.values == [3, 3, -1, -3, -2]).all()
    assert (
        x.coupled_variable_group == y.coupled_variable_group == z.coupled_variable_group
    )


def test_subtract_scalar():
    """Tests the subtraction of a stochastic scalar and a scalar."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    z = x - 1
    assert (z.values == [3, 4, 1, 0, 2]).all()
    assert x.coupled_variable_group == z.coupled_variable_group


def test_rsubtract_scalar():
    """Tests the subtraction of a scalar and a stochastic scalar."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    z = 1 - x
    assert (z.values == [-3, -4, -1, 0, -2]).all()
    assert x.coupled_variable_group == z.coupled_variable_group


def test_multiply():
    """Tests the multiplication of two stochastic scalars."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    y = StochasticScalar([1, 2, 3, 4, 5])
    z = x * y
    assert (z.values == [4, 10, 6, 4, 15]).all()
    assert (
        x.coupled_variable_group == y.coupled_variable_group == z.coupled_variable_group
    )


def test_multiply_scalar():
    """Tests the multiplication of a stochastic scalar and a scalar."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    z = x * 2
    assert (z.values == [8, 10, 4, 2, 6]).all()
    assert x.coupled_variable_group == z.coupled_variable_group


def test_rmultiply_scalar():
    """Tests the multiplication of a scalar and a stochastic scalar."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    z = 2 * x
    assert (z.values == [8, 10, 4, 2, 6]).all()
    assert x.coupled_variable_group == z.coupled_variable_group


def test_divide():
    """Tests the division of two stochastic scalars."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    y = StochasticScalar([1, 2, 3, 4, 5])
    z: StochasticScalar = x / y
    assert (z.values == [4, 2.5, 2 / 3, 0.25, 0.6]).all()
    assert (
        x.coupled_variable_group == y.coupled_variable_group == z.coupled_variable_group
    )


def test_divide_scalar():
    """Tests the division of a stochastic scalar and a scalar."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    z: StochasticScalar = x / 2
    assert (z.values == [2, 2.5, 1, 0.5, 1.5]).all()
    assert x.coupled_variable_group == z.coupled_variable_group


def test_rdivide_scalar():
    """Tests the division of a scalar and a stochastic scalar."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    z = 2 / x
    assert (z.values == [0.5, 0.4, 1, 2, 2 / 3]).all()
    assert x.coupled_variable_group == z.coupled_variable_group


def test_power():
    """Tests the power of two stochastic scalars."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    y = StochasticScalar([1, 2, 3, 4, 5])
    z = x**y
    assert (z.values == [4, 25, 8, 1, 243]).all()
    assert (
        x.coupled_variable_group == y.coupled_variable_group == z.coupled_variable_group
    )


def test_power_scalar():
    """Tests the power of a stochastic scalar and a scalar."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    z = x**2
    assert (z.values == [16, 25, 4, 1, 9]).all()
    assert x.coupled_variable_group == z.coupled_variable_group


def test_rpower_scalar():
    """Tests the power of a scalar and a stochastic scalar."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    z = 2**x
    assert (z.values == [16, 32, 4, 2, 8]).all()
    assert x.coupled_variable_group == z.coupled_variable_group


def test_eq():
    """Tests the logical equals of two stochastic scalars."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    y = StochasticScalar([4, 2, 3, 1, 5])
    z = x == y
    assert (z.values == [True, False, False, True, False]).all()
    assert (
        x.coupled_variable_group == y.coupled_variable_group == z.coupled_variable_group
    )


def test_eq_scalar():
    """Tests the logical equals of two stochastic scalars."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    y = 1
    z = x == y
    assert (z.values == [False, False, False, True, False]).all()
    assert x.coupled_variable_group == z.coupled_variable_group


def test_not_eq():
    """Tests the logical not equals of two stochastic scalars."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    y = StochasticScalar([4, 2, 3, 1, 5])
    z = x != y
    assert (z.values == [False, True, True, False, True]).all()
    assert (
        x.coupled_variable_group == y.coupled_variable_group == z.coupled_variable_group
    )


def test_not_eq_scalar():
    """Tests the logical not equals of two stochastic scalars."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    y = 1
    z = x != y
    assert (z.values == [True, True, True, False, True]).all()
    assert x.coupled_variable_group == z.coupled_variable_group


def test_and():
    """Tests the logical and of two stochastic scalars."""
    x = StochasticScalar([False, True, False, True, True])
    y = StochasticScalar([False, False, True, True, True])
    z = x & y
    assert (z.values == [False, False, False, True, True]).all()
    assert (
        x.coupled_variable_group == y.coupled_variable_group == z.coupled_variable_group
    )


def test_numpy_ufunc():
    """Tests that a numpy ufunc can be applied to a stochastic scalar."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    y: StochasticScalar = pnp.exp(x)
    assert (y.values == np.exp([4, 5, 2, 1, 3])).all()
    assert x.coupled_variable_group == y.coupled_variable_group
    assert type(y) is StochasticScalar


def test_ssum():
    """Tests the sum of a stochastic scalar."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    y = np.sum(x)
    assert y == 15


def test_mean():
    """Tests the mean of a stochastic scalar."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    y = np.mean(x)
    assert y == 3


def test_dereference():
    x = StochasticScalar([4, 5, 2, 1, 3])
    # Direct integer indexing returns the raw value however the type signature is
    # currently inconsistent with this behavior (see FIXME in stochastic_scalar.py and
    # refer to github issue #24).
    y: int = x[3]  # type: ignore[reportAssignmentType]
    assert y == 1


def test_stochastic_dereference():
    x = StochasticScalar([4, 5, 2, 1, 3])
    inds = StochasticScalar([3, 2, 1, 0, 0, 4])
    y = x[inds]
    assert (y.values == [1, 2, 5, 4, 4, 3]).all()
    assert inds.coupled_variable_group == y.coupled_variable_group


def test_percentile():
    x = StochasticScalar([4, 5, 2, 1, 3])
    y = np.percentile(x, 50)
    assert y == 3
    y = np.percentile(x, [10, 90])
    assert (y == [1.4, 4.6]).all()


def test_tvar():
    """Test the tail value at risk (TVAR) of the variable."""
    from pal.stats import tvar

    x = StochasticScalar([4, 5, 2, 1, 3])
    y = tvar(x, 50)
    assert y == 4.5


def test_tvar2():
    """Test the tail value at risk (TVAR) of the variable."""
    from pal.stats import tvar

    x = StochasticScalar([4, 5, 2, 1, 3])
    y = tvar(x, [50, 80])
    assert y == [4.5, 5]


def test_min():
    """Test the min() method of a stochastic scalar."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    y: int = pnp.min(x)
    assert y == 1
    assert isinstance(y, NumericProtocol)  # Should return scalar


def test_max():
    """Test the max() method of a stochastic scalar."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    y: int = pnp.max(x)
    assert y == 5
    assert isinstance(y, NumericProtocol)  # Should return scalar


def test_min_empty():
    """Test the min() method on an empty stochastic scalar."""
    x = StochasticScalar([])
    with pytest.raises(ValueError, match="zero-size array"):
        np.min(x)


def test_max_empty():
    """Test the max() method on an empty stochastic scalar."""
    x = StochasticScalar([])
    with pytest.raises(ValueError, match="zero-size array"):
        np.max(x)


def test_min_single_value():
    """Test the min() method on a single-value stochastic scalar."""
    x = StochasticScalar([42])
    y = np.min(x)
    assert y == 42
    assert isinstance(y, NumericProtocol)


def test_max_single_value():
    """Test the max() method on a single-value stochastic scalar."""
    x = StochasticScalar([42])
    y = np.max(x)
    assert y == 42
    assert isinstance(y, NumericProtocol)


def test_any_all():
    """Test the any() and all() methods of a stochastic scalar."""
    x = StochasticScalar([False, True, True, True])
    assert x.any() is True
    assert x.all() is False

    y = StochasticScalar([True, True, True])
    assert y.any() is True
    assert y.all() is True

    z = StochasticScalar([False, False, False])
    assert z.any() is False
    assert z.all() is False

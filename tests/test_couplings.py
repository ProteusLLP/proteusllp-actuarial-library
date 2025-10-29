from pal import copulas
from pal.variables import StochasticScalar
import pytest  # noqa


def test_copula_reordering():
    """A check that the copula reordering works as expected."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    y = StochasticScalar([1, 2, 3, 4, 5])
    copula_samples = [
        StochasticScalar([0, 4, 3, 1, 2]),
        StochasticScalar([1, 3, 2, 0, 4]),
    ]
    copulas.apply_copula([x, y], copula_samples)
    assert (x.values == [4, 5, 2, 1, 3]).all()
    assert (y.values == [3, 4, 1, 2, 5]).all()


def test_coupled_variable_reordering():
    """Test that coupled variables are reordered correctly."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    y = StochasticScalar([1, 2, 3, 4, 5])
    z = y + 1  # y and z are now coupled
    copula_samples = [
        StochasticScalar([0, 4, 3, 1, 2]),
        StochasticScalar([1, 3, 2, 0, 4]),
    ]
    copulas.apply_copula([x, y], copula_samples)
    assert (x.values == [4, 5, 2, 1, 3]).all()
    assert (y.values == [3, 4, 1, 2, 5]).all()
    assert (z.values == [4, 5, 2, 3, 6]).all()
    assert (
        x.coupled_variable_group == y.coupled_variable_group == z.coupled_variable_group
    )


def test_coupled_variable_reordering2():
    """Test that coupled variables are reordered correctly."""
    x = StochasticScalar([4, 5, 2, 1, 3])
    y = StochasticScalar([1, 2, 3, 4, 5])
    z = StochasticScalar([7, 3, 1, 9, 0])
    a = y + z  # a, y, and z are now coupled
    copula_samples = [
        StochasticScalar([0, 4, 3, 1, 2]),
        StochasticScalar([1, 3, 2, 0, 4]),
    ]
    copulas.apply_copula([x, y], copula_samples)
    assert (x.values == [4, 5, 2, 1, 3]).all()
    assert (y.values == [3, 4, 1, 2, 5]).all()
    assert (z.values == [1, 9, 7, 3, 0]).all()
    assert (a.values == [4, 13, 8, 5, 5]).all()
    assert (
        x.coupled_variable_group
        == y.coupled_variable_group
        == z.coupled_variable_group
        == a.coupled_variable_group
    )

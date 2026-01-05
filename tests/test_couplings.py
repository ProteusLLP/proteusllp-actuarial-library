"""Tests for stochastic variable coupling and reordering.

Tests covering copula-based coupling mechanisms and simulation reordering
for dependency modeling between stochastic variables.
"""

from pal import copulas
from pal.variables import StochasticScalar


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


def test_coupled_variable_groups():
    """Test that coupled variables use identity comparison correctly."""
    x = StochasticScalar([1.0, 2.0, 3.0])
    y = StochasticScalar([4.0, 5.0, 6.0])

    # Performing an operation should merge coupling groups
    z = x + y

    # All should be in the same coupling group now
    assert x.coupled_variable_group is y.coupled_variable_group
    assert y.coupled_variable_group is z.coupled_variable_group

    # The internal dictionary in the coupling group should contain all three
    assert len(x.coupled_variable_group) == 3

    assert x in x.coupled_variable_group
    assert y in x.coupled_variable_group
    assert z in x.coupled_variable_group

    y2 = StochasticScalar([4.0, 5.0, 6.0])
    assert y2.coupled_variable_group is not x.coupled_variable_group
    assert y2 not in x.coupled_variable_group


def test_variable_membership_in_own_coupling_group() -> None:
    """Test that a variable can be identified as member of its coupling group.

    Users should be able to check if a variable is in its coupling group.
    """
    x = StochasticScalar([1.0, 2.0, 3.0])

    # Should be able to check membership
    assert x in x.coupled_variable_group

"""Tests for couplings module to improve coverage."""

from pal.couplings import CouplingGroup
from pal.variables import StochasticScalar


def test_coupling_group_discard():
    """Test discard method removes variable from coupling group (line 41)."""
    x = StochasticScalar([1, 2, 3])
    y = StochasticScalar([4, 5, 6])

    # Create group with first variable
    group = CouplingGroup(x)
    group.add(y)
    assert len(group) == 2

    # Discard one variable
    group.discard(x)
    assert len(group) == 1
    assert x not in group
    assert y in group

    # Discard non-existent variable (should not raise error)
    z = StochasticScalar([7, 8, 9])
    group.discard(z)  # Should not raise
    assert len(group) == 1


def test_proteus_stochastic_variable_any():
    """Test any() method on ProteusStochasticVariable (line 221)."""
    # Test with some True values
    x = StochasticScalar([0, 1, 2])
    assert x.any() is True

    # Test with all False values
    y = StochasticScalar([0, 0, 0])
    assert y.any() is False

    # Test with all True values
    z = StochasticScalar([1, 2, 3])
    assert z.any() is True

"""Tests for stochastic variable coupling and reordering.

Tests covering copula-based coupling mechanisms and simulation reordering
for dependency modeling between stochastic variables.
"""

import copy

import numpy as np
from pal import copulas
from pal.frequency_severity import FreqSevSims
from pal.variables import ProteusVariable, StochasticScalar


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


def test_variable_membership_in_own_coupling_group() -> None:
    """Test that a variable can be identified as member of its coupling group.

    Users should be able to check if a variable is in its coupling group.
    """
    x = StochasticScalar([1.0, 2.0, 3.0])

    # Should be able to check membership
    assert x in x.coupled_variable_group


def test_freqsevsims_membership_in_coupling_group() -> None:
    """Test that FreqSevSims can be checked for coupling group membership."""
    sim_index = np.array([1, 2, 2, 3, 4], dtype=int)
    fs = FreqSevSims(
        sim_index=sim_index,
        values=np.array([500.0, 700.0, 600.0, 800.0, 550.0]),
        n_sims=5,
    )

    # Should be able to check membership
    assert fs in fs.coupled_variable_group


def test_deepcopy_with_operations() -> None:
    """Test that deep copy works when operations are performed after copying.

    Deep copy creates new CouplingGroups, and when we later try to merge them
    during operations, the old WeakSet implementation would fail because
    WeakSet.add() uses __eq__ which returns arrays for numpy-like objects.
    """
    # Create FreqSevSims with shared sim_index (coupled variables)
    sim_index = np.array([1, 2, 2, 3, 4], dtype=int)
    losses: ProteusVariable[FreqSevSims] = ProteusVariable(
        "item",
        {
            "asset1": FreqSevSims(
                sim_index=sim_index,
                values=np.array([500.0, 700.0, 600.0, 800.0, 550.0]),
                n_sims=5,
            ),
            "asset2": FreqSevSims(
                sim_index=sim_index,
                values=np.array([1500.0, 2000.0, 1800.0, 2200.0, 1600.0]),
                n_sims=5,
            ),
        },
    )

    # Step 1: Use original (triggers coupling group merge)
    _ = losses["asset1"] * 0.5 + losses["asset2"] * 0.5

    # Step 2: Deep copy (creates new CouplingGroups)
    copied_losses = copy.deepcopy(losses)

    # Step 3: Try using copied FreqSevSims
    # This triggers merge() which would call WeakSet.add() and fail
    combined_copy: FreqSevSims = (
        copied_losses["asset1"] * 0.5 + copied_losses["asset2"] * 0.5
    )
    assert combined_copy is not None

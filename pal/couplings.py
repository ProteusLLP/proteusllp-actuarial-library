"""Stochastic variable coupling and dependency management.

Provides coupling mechanisms for stochastic variables, allowing them to maintain
dependency relationships during reordering and copula applications. Key classes
include CouplingGroup for managing variable groups and ProteusStochasticVariable
as the base class for all stochastic types.
"""

from __future__ import annotations

import typing as t
import weakref
from abc import ABC

import numpy as np
import numpy.typing as npt
from numpy.lib.mixins import NDArrayOperatorsMixin


class CouplingGroup:
    """A class to represent a group of variables that are coupled together."""

    variables: weakref.WeakSet[ProteusStochasticVariable]

    @property
    def id(self) -> int:
        """Get the unique identifier for this coupling group."""
        return id(self)

    def __init__(self, variable: ProteusStochasticVariable):
        """Initialize coupling group with a single variable.

        Args:
            variable: The initial variable to add to the group.
        """
        # Start the group with a single variable, stored as a weak reference.
        self.variables: weakref.WeakSet[ProteusStochasticVariable] = weakref.WeakSet(
            [variable]
        )

    def merge(self, other: CouplingGroup) -> None:
        """Merge another coupling group into this one.

        Args:
            other: The other coupling group to merge.
        """
        if self is other:
            return
        # Merge the other group's variables into this one, updating their pointer.
        for var in list(other.variables):
            var.coupled_variable_group = self
            self.variables.add(var)
        return


class ProteusStochasticVariable(ABC, NDArrayOperatorsMixin):
    """A class to represent a stochastic variable in a simulation."""

    n_sims: int | None = None
    values: npt.NDArray[np.floating]

    def __init__(self) -> None:
        """Initialize stochastic variable with new coupling group."""
        self.coupled_variable_group = CouplingGroup(self)

    def __array__(self) -> npt.NDArray[np.floating]:
        """Return the underlying numpy array for compatibility with numpy functions."""
        return self.values

    def all(self) -> bool:
        """Return True if all values are True."""
        return t.cast(bool, self.values.all())

    def upsample(self, n_sims: int) -> t.Self:
        """Upsample the variable to match the specified number of simulations.

        Args:
            n_sims: The number of simulations to upsample to.

        Returns:
            A new instance of self with the upsampled values.
        """
        raise NotImplementedError

    def astype(self, dtype: np.dtype[t.Any] | type[t.Any]) -> npt.NDArray[t.Any]:
        """Convert the underlying values to a specified dtype.

        Args:
            dtype: The data type to convert to.

        Returns:
            A new numpy array with the specified dtype.
        """
        return self.values.astype(dtype)

    # Private methods for internal use
    def _reorder_sims(self, new_order: t.Sequence[int]) -> None:
        raise NotImplementedError

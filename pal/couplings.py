from __future__ import annotations

import weakref
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from numpy.lib.mixins import NDArrayOperatorsMixin


class CouplingGroup:
    """A class to represent a group of variables that are coupled together."""

    variables: weakref.WeakSet[ProteusStochasticVariable]

    @property
    def id(self):
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

    def merge(self, other: CouplingGroup):
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

    values: npt.NDArray[np.float64]
    n_sims: int | None = None

    def __init__(self):
        """Initialize stochastic variable with new coupling group."""
        self.coupled_variable_group = CouplingGroup(self)

    @abstractmethod
    def _reorder_sims(self, new_order: npt.NDArray[np.int64]):
        pass

    def all(self):
        """Return True if all values are True."""
        return self.values.all()

    def any(self):
        """Return True if any value is True."""
        return self.values.any()

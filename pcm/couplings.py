from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from typing import Union
from abc import ABC, abstractmethod
import weakref


class CouplingGroup:
    """A class to represent a group of variables that are coupled together."""

    variables: weakref.WeakSet[ProteusStochasticVariable]

    @property
    def id(self):
        return id(self)

    def __init__(self, variable: ProteusStochasticVariable):
        # Start the group with a single variable, stored as a weak reference.
        self.variables: weakref.WeakSet[ProteusStochasticVariable] = weakref.WeakSet(
            [variable]
        )

    def merge(self, other: CouplingGroup):
        if self is other:
            return
        # Merge the other group's variables into this one, updating their pointer.
        for var in list(other.variables):
            var.coupled_variable_group = self
            self.variables.add(var)
        return


class ProteusStochasticVariable(ABC):
    """A class to represent a stochastic variable in a simulation."""

    values: np.ndarray
    n_sims: int

    def __init__(self):
        self.coupled_variable_group = CouplingGroup(self)

    @abstractmethod
    def _reorder_sims(self, new_order: np.ndarray):
        pass

    @abstractmethod
    def __add__(
        self, other: Union[ProteusStochasticVariable, ArrayLike]
    ) -> ProteusStochasticVariable:
        pass

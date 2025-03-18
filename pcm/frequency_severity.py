from typing import Union, Callable
import numpy
from .couplings import ProteusStochasticVariable
from .stochastic_scalar import (
    StochasticScalar,
)
from .config import config, xp as np
from . import distributions

ProteusCompatibleTypes = Union["FreqSevSims", StochasticScalar, int, float, np.ndarray]


def _get_sims_of_events(n_events_by_sim: np.ndarray):
    """Given the number of events in each simulation, returns the simulation index for each event.

    >>> n_events_by_sim = np.array([1, 0, 3])
    >>> _get_sims_of_events(n_events_by_sim)
    array([0, 2, 2, 2])

    Parameters:
    - n_events_by_sim (np.ndarray): Array of the number of events in each simulation.

    Returns:
    - np.ndarray: Array of simulation indices for each event.
    """
    cumulative_n_events = n_events_by_sim.cumsum()
    total_events = cumulative_n_events[-1]
    event_no = np.arange(total_events)
    return cumulative_n_events.searchsorted(event_no + 1)


class FrequencySeverityModel:
    """A class for constructing and simulating from Frequency-Severity, or Compound distributions"""

    def __init__(
        self,
        freq_dist: distributions.DistributionBase,
        sev_dist: distributions.DistributionBase,
    ):
        self.freq_dist = freq_dist
        self.sev_dist = sev_dist

    def generate(
        self, n_sims=None, rng: np.random.Generator = config.rng
    ) -> "FreqSevSims":
        """
        Generate simulations from the Frequency-Severity model.

        Parameters:
        - n_sims (int): Number of simulations to generate. If None, uses the default value from the config.
        - rng (np.random.Generator): Random number generator. Defaults to the value from the config.

        Returns:
        - FreqSevSims: Object containing the generated simulations.
        """
        if n_sims is None:
            n_sims = config.n_sims
        n_events = self.freq_dist.generate(n_sims, rng)
        total_events = n_events.ssum()
        sev = self.sev_dist.generate(int(total_events), rng)
        return FreqSevSims(_get_sims_of_events(n_events.values), sev.values, n_sims)


class FreqSevSims(ProteusStochasticVariable):
    """A class for storing and manipulating Frequency-Severity simulations.
    FreqSevSims objects provide convenience methods for aggregating and summarizing the simulations.

    >>> sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    >>> values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> n_sims = 3
    >>> fs = FreqSevSims(sim_index, values, n_sims)
    >>> fs.aggregate()
    StochasticScalar([ 3., 12., 30.])
    >>> fs.occurrence()
    StochasticScalar([2., 5., 9.])

    They can be operated on using standard mathematical operations, as well as as numpy ufuncs and functions.

    >>> fs + 1
    FreqSevSims(array([0, 0, 1, 1, 1, 2, 2, 2, 2]), array([ 2,  3,  4,  5,  6,  7,  8,  9, 10]), 3)
    >>> np.maximum(fs, 5)
    FreqSevSims(array([0, 0, 1, 1, 1, 2, 2, 2, 2]), array([5, 5, 5, 5, 5, 6, 7, 8, 9]), 3)
    >>> np.where(fs > 5, 1, 0)
    FreqSevSims(array([0, 0, 1, 1, 1, 2, 2, 2, 2]), array([0, 0, 0, 0, 0, 1, 1, 1, 1]), 3)

    FreqSevSims objects can be multiplied, added, subtracted, divided, and compared with other FreqSevSims objects,
    provided that the simulation indices match.

    >>> fs1 = FreqSevSims(sim_index, values, n_sims)
    >>> fs2 = FreqSevSims(sim_index, values, n_sims)
    >>> fs1 + fs2
    FreqSevSims(array([0, 0, 1, 1, 1, 2, 2, 2, 2]), array([ 2,  4,  6,  8, 10, 12, 14, 16, 18]), 3)
    """

    def __init__(
        self,
        sim_index: np.ndarray | list[int],
        values: np.ndarray | list[int],
        n_sims: int,
    ):
        """
        Create a new FreqSevSims object out the list of simulation indices, and the list of values corresponding to
        each simulation index. Note, the simulation indices are assumed to be ordered and 0-indexed.


        Parameters:
        sim_index (np.ndarray|list): Array of simulation indices.
        values (np.ndarray|list): Array of values.
        n_sims (int): Number of simulations.

        Raises:
        AssertionError: If lengths of values and sim_index don't match.


        """
        super().__init__()
        self.sim_index = np.asarray(sim_index)
        self.values = np.asarray(values)
        self.n_sims = n_sims

        assert len(self.sim_index) == len(self.values)

    def __hash__(self):
        return id(self)

    def __str__(self):
        return (
            "Simulation Index\n"
            + str(self.sim_index)
            + "\n Values\n"
            + str(self.values)
        )

    def _reorder_sims(self, ordering: np.ndarray) -> None:
        """Reorder the simulations of the FreqSevSims object according to the given order."""
        reverse_ordering = np.empty(len(ordering), dtype=int)
        reverse_ordering[ordering] = np.arange(len(ordering), dtype=int)
        self.sim_index = reverse_ordering[self.sim_index]

    def __getitem__(self, sim_index: int) -> StochasticScalar:
        """Returns the values of the simulation with the given simulation index."""
        # get the positions of the given simulation index
        ints = np.where(self.sim_index == sim_index)
        return StochasticScalar(self.values[ints])

    def _reduce_over_events(self, operation) -> StochasticScalar:
        result = np.zeros(self.n_sims)
        operation(result, self.sim_index, self.values)
        result = StochasticScalar(result)
        result.coupled_variable_group.merge(self.coupled_variable_group)
        return result

    def aggregate(self) -> StochasticScalar:
        """Calculates the aggregate loss for each simulation.

        >>> sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
        >>> values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> n_sims = 3
        >>> fs = FreqSevSims(sim_index, values, n_sims)
        >>> fs.aggregate()
        array([ 3., 12., 30.])

        Returns:
            numpy.ndarray: An array containing the aggregate loss for each simulation.
        """
        return self._reduce_over_events(np.add.at)

    def occurrence(self) -> StochasticScalar:
        """Calculates the maximum occurrence loss for each simulation.

        >>> sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
        >>> values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> n_sims = 3
        >>> fs = FreqSevSims(sim_index, values, n_sims)
        >>> fs.occurrence()
        array([2., 5., 9.])

        Returns:
            numpy.ndarray: An array containing the aggregate loss for each simulation.
        """
        return self._reduce_over_events(np.maximum.at)

    def deep_copy(self) -> "FreqSevSims":
        """Creates a deep copy of the FreqSevSims object."""
        return FreqSevSims(self.sim_index, self.values.copy(), self.n_sims)

    def copy(self) -> "FreqSevSims":
        """Creates a copy of the FreqSevSims object."""
        result = FreqSevSims(self.sim_index, self.values.copy(), self.n_sims)
        result.coupled_variable_group.merge(self.coupled_variable_group)
        return result

    def apply(self, func) -> "FreqSevSims":
        """Applies a function to the values of the FreqSevSims object."""
        result = FreqSevSims(self.sim_index, func(self.values), self.n_sims)
        result.coupled_variable_group.merge(self.coupled_variable_group)
        return result

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs, **kwargs
    ) -> "FreqSevSims":
        inputs = tuple(
            (
                x.values
                if isinstance(x, FreqSevSims)
                else (
                    x[self.sim_index]
                    if isinstance(x, np.ndarray)
                    else (
                        x.values[self.sim_index]
                        if isinstance(x, StochasticScalar)
                        else x
                    )
                )  # promote an input ndarray to match the simulation index
            )
            for x in inputs
        )
        out = kwargs.get("out", ())
        if out:
            kwargs["out"] = tuple(x.values for x in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)
        result = FreqSevSims(self.sim_index, result, self.n_sims)
        result.coupled_variable_group.merge(self.coupled_variable_group)

        return result

    def __array_function__(self, func: Callable, types, args, kwargs):
        if func not in (numpy.where, numpy.sum):
            raise NotImplementedError
        args = tuple(x.values if isinstance(x, FreqSevSims) else x for x in args)
        result = func(*args, **kwargs)
        if isinstance(result, np.number):
            return result
        result = FreqSevSims(self.sim_index, result, self.n_sims)
        result.coupled_variable_group.merge(self.coupled_variable_group)
        return result

    def _binary_operation(self, x, operation) -> "FreqSevSims":
        if self._is_compatible(x):
            assert isinstance(x, FreqSevSims)
            result = FreqSevSims(
                self.sim_index, operation(self.values, x.values), self.n_sims
            )
            result.coupled_variable_group.merge(self.coupled_variable_group)
            return result
        elif isinstance(x, int) or isinstance(x, float):
            result = FreqSevSims(self.sim_index, operation(self.values, x), self.n_sims)
            result.coupled_variable_group.merge(self.coupled_variable_group)
            return result
        elif isinstance(x, StochasticScalar):
            result = FreqSevSims(
                self.sim_index,
                operation(self.values, x.values[self.sim_index]),
                self.n_sims,
            )
            result.coupled_variable_group.merge(self.coupled_variable_group)
            result.coupled_variable_group.merge(x.coupled_variable_group)
            return result
        elif isinstance(x, np.ndarray):
            return FreqSevSims(
                self.sim_index, operation(self.values, x[self.sim_index]), self.n_sims
            )
        else:
            raise NotImplementedError(
                f"Cannot perform operation {operation} on {type(x)} and {type(self)}"
            )

    def __add__(self, x: ProteusCompatibleTypes) -> "FreqSevSims":
        return self._binary_operation(x, operation=lambda a, b: a + b)

    def __radd__(self, x: ProteusCompatibleTypes):
        return self.__add__(x)

    def __sub__(self, x: ProteusCompatibleTypes):
        return self._binary_operation(x, operation=lambda a, b: a - b)

    def __rsub__(self, x: ProteusCompatibleTypes):
        return -self.__sub__(x)

    def __neg__(self):
        result = FreqSevSims(self.sim_index, -self.values, self.n_sims)
        result.coupled_variable_group.merge(self.coupled_variable_group)
        return result

    def __mul__(self, other: ProteusCompatibleTypes):
        return self._binary_operation(other, operation=lambda a, b: a * b)

    def __rmul__(self, other: ProteusCompatibleTypes):
        return self.__mul__(other)

    def __truediv__(self, other: ProteusCompatibleTypes):
        return self._binary_operation(other, operation=lambda a, b: a / b)

    def __rtruediv__(self, other: ProteusCompatibleTypes):
        return self.__mul__(1 / other)

    def __pow__(self, other: ProteusCompatibleTypes) -> "FreqSevSims":
        return self._binary_operation(other, operation=lambda a, b: a**b)

    def __rpow__(self, other: ProteusCompatibleTypes) -> "FreqSevSims":
        return self._binary_operation(other, operation=lambda a, b: b**a)

    def __lt__(self, other: ProteusCompatibleTypes):
        return self._binary_operation(other, operation=lambda a, b: a < b)

    def __le__(self, other: ProteusCompatibleTypes):
        return self._binary_operation(other, operation=lambda a, b: a <= b)

    def __gt__(self, other: ProteusCompatibleTypes):
        return self._binary_operation(other, operation=lambda a, b: a > b)

    def __ge__(self, other: ProteusCompatibleTypes):
        return self._binary_operation(other, operation=lambda a, b: a >= b)

    def __eq__(self, other: ProteusCompatibleTypes):
        return self._binary_operation(other, operation=lambda a, b: a == b)

    def __and__(self, other: ProteusCompatibleTypes):
        return self._binary_operation(other, operation=lambda a, b: a & b)

    def __rand__(self, other: ProteusCompatibleTypes):
        return self.__and__(other)

    def __or__(self, other: ProteusCompatibleTypes):
        return self._binary_operation(other, operation=lambda a, b: a | b)

    def __ror__(self, other: ProteusCompatibleTypes):
        return self.__or__(other)

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self.values)

    def _is_compatible(self, other: ProteusCompatibleTypes):
        """Check if two FreqSevSims objects are compatible for mathematical operations."""
        return isinstance(other, FreqSevSims) and self.sim_index is other.sim_index

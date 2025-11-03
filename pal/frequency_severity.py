"""Frequency-severity modeling for actuarial applications.

This module provides classes and functions for modeling compound distributions
commonly used in insurance and actuarial science, where claims are modeled as
the sum of a random number (frequency) of random amounts (severity).

Key components:
- FrequencySeverityModel: Main class for compound distribution modeling
- FreqSevSims: Container for frequency-severity simulation results
- Utility functions for simulation index management

The frequency-severity approach is fundamental in actuarial modeling for:
- Aggregate claims modeling
- Risk assessment and capital modeling
- Insurance pricing and reserving

Example:
    >>> from pal.distributions import Poisson, LogNormal
    >>> freq_dist = Poisson(5.0)  # Expected 5 claims
    >>> sev_dist = LogNormal(mean=10000, sigma=0.5)  # Claim amounts
    >>> model = FrequencySeverityModel(freq_dist, sev_dist)
    >>> simulations = model.simulate(n_sims=10000)
"""

from __future__ import annotations

import typing as t
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from . import distributions
from .config import config
from .couplings import ProteusStochasticVariable
from .stochastic_scalar import (
    StochasticScalar,
)

# Type aliases for frequency-severity modeling

# Types that can be used in mathematical operations with FreqSevSims objects
ProteusCompatibleTypes = t.Union[
    "FreqSevSims", StochasticScalar, int, float, npt.NDArray[t.Any]
]

# Function signature for numpy ufunc operations that reduce events to simulations
ReductionOperation = t.Callable[
    [
        npt.NDArray[np.floating],  # result: array to store reduced values
        npt.NDArray[np.int64],  # indices: simulation indices for each event
        npt.NDArray[np.floating],  # values: event values to reduce
    ],
    None,
]

# Function signature for transforming arrays element-wise
ArrayTransform = t.Callable[
    [npt.NDArray[np.floating]],  # input_array: array to transform
    npt.NDArray[np.floating],  # return: transformed array
]


def _get_sims_of_events(
    n_events_by_sim: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Get the simulation index for each event.

    Given the number of events in each simulation, returns the simulation
    index for each event.

    >>> n_events_by_sim = np.array([1, 0, 3])
    >>> _get_sims_of_events(n_events_by_sim)
    array([0, 2, 2, 2])

    Args:
        n_events_by_sim (np.ndarray): Array of the number of events in each simulation.

    Returns:
        np.ndarray: Array of simulation indices for each event.
    """
    cumulative_n_events = n_events_by_sim.cumsum()
    total_events = cumulative_n_events[-1]
    event_no = t.cast(npt.NDArray[np.int64], np.arange(total_events))
    return cumulative_n_events.searchsorted(event_no + 1)


class FrequencySeverityModel:
    """Constructs and simulates from Frequency-Severity, or Compound distributions."""

    def __init__(
        self,
        freq_dist: distributions.DistributionBase,
        sev_dist: distributions.DistributionBase,
    ):
        """Initialize a frequency-severity model.

        Args:
            freq_dist: Distribution for frequency component.
            sev_dist: Distribution for severity component.
        """
        self.freq_dist = freq_dist
        self.sev_dist = sev_dist

    def generate(
        self, n_sims: int | None = None, rng: np.random.Generator = config.rng
    ) -> FreqSevSims:
        """Generate simulations from the Frequency-Severity model.

        Parameters:
        - n_sims (int): Number of simulations to generate. If None, uses the
            default value from the config.
        - rng (np.random.Generator): Random number generator. Defaults to the
            value from the config.

        Returns:
        - FreqSevSims: Object containing the generated simulations.
        """
        if n_sims is None:
            n_sims = config.n_sims
        n_events = self.freq_dist.generate(n_sims, rng)
        total_events = np.sum(n_events)
        sev = self.sev_dist.generate(int(total_events), rng)
        # Convert n_events to integers since _get_sims_of_events expects integer counts
        # but n_events.values comes from distributions which return floating arrays
        result = FreqSevSims(
            _get_sims_of_events(n_events.values.astype(np.int64)), sev.values, n_sims
        )
        result.coupled_variable_group.merge(n_events.coupled_variable_group)
        result.coupled_variable_group.merge(sev.coupled_variable_group)
        return result


class FreqSevSims(ProteusStochasticVariable):
    """A class for storing and manipulating Frequency-Severity simulations.

    FreqSevSims objects provide convenience methods for aggregating and
    summarizing the simulations.

    >>> sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    >>> values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> n_sims = 3
    >>> fs = FreqSevSims(sim_index, values, n_sims)
    >>> fs.aggregate()
    StochasticScalar([ 3., 12., 30.])
    >>> fs.occurrence()
    StochasticScalar([2., 5., 9.])

    They can be operated on using standard mathematical operations, as well as
    as numpy ufuncs and functions.

    >>> fs + 1  # doctest: +ELLIPSIS
    FreqSevSims(...)
    >>> np.maximum(fs, 5)  # doctest: +ELLIPSIS
    FreqSevSims(...)
    >>> np.where(fs > 5, 1, 0)  # doctest: +ELLIPSIS
    FreqSevSims(...)

    FreqSevSims objects can be multiplied, added, subtracted, divided, and
    compared with other FreqSevSims objects,
    provided that the simulation indices match.

    >>> fs1 = FreqSevSims(sim_index, values, n_sims)
    >>> fs2 = FreqSevSims(sim_index, values, n_sims)
    >>> fs1 + fs2  # doctest: +ELLIPSIS
    FreqSevSims(...)
    """

    def __init__(
        self,
        sim_index: npt.ArrayLike,
        values: npt.ArrayLike,
        n_sims: int | None = None,
    ):
        """Create a new FreqSevSims object out the list of simulation indices.

        Creates a FreqSevSims object from simulation indices and corresponding values.
        Note, the simulation indices are assumed to be ordered and 0-indexed.


        Parameters:
            sim_index: simulation indices.
            values: the values.
            n_sims: Number of simulations.

        Raises:
            AssertionError: If lengths of values and sim_index don't match.


        """
        super().__init__()
        self.sim_index = np.asarray(sim_index)
        self.values = np.asarray(values)
        self.n_sims = n_sims

        if len(self.sim_index) != len(self.values):
            raise ValueError(
                f"Length mismatch: sim_index has {len(self.sim_index)} elements "
                f"but values has {len(self.values)} elements"
            )

    def __hash__(self):
        return id(self)

    def __str__(self):
        return (
            "Simulation Index\n"
            + str(self.sim_index)
            + "\n Values\n"
            + str(self.values)
        )

    def _reorder_sims(self, new_order: t.Sequence[int]) -> None:
        """Reorder simulations according to the given order.

        This method updates the simulation indices to match the new order.

        Args:
            new_order: A sequence of integers representing the new order of simulations.
        """
        reverse_ordering = np.empty(len(new_order), dtype=int)
        reverse_ordering[new_order] = np.arange(len(new_order), dtype=int)
        self.sim_index = reverse_ordering[self.sim_index]

    def __getitem__(self, sim_index: int) -> StochasticScalar:
        """Returns the values of the simulation with the given simulation index."""
        # get the positions of the given simulation index
        if isinstance(sim_index, int):  # type: ignore[unecessary-check]
            ints = np.where(self.sim_index == sim_index)
            return StochasticScalar(self.values[ints])
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of simulations."""
        return self.n_sims if self.n_sims is not None else 0

    def __iter__(self) -> t.Iterator[StochasticScalar]:
        """Iterate over the simulations."""
        if self.n_sims is None:
            raise ValueError("Cannot iterate. n_sims not specified.")
        for i in range(self.n_sims):
            yield self[i]

    def _reduce_over_events(self, operation: ReductionOperation) -> StochasticScalar:
        """Apply a reduction operation over events for each simulation.

        Groups events by simulation index and applies the specified operation
        to combine values within each simulation.

        Args:
            operation: Numpy ufunc operation to apply (e.g., np.add.at, np.maximum.at)

        Returns:
            A StochasticScalar containing the reduced values for each simulation

        Raises:
            ValueError: If n_sims is not set
        """
        if self.n_sims is None:
            raise ValueError("n_sims must be set before reducing over events.")
        _result = np.zeros(self.n_sims)
        operation(_result, self.sim_index, self.values)
        result = StochasticScalar(_result)
        result.coupled_variable_group.merge(self.coupled_variable_group)
        return result

    def aggregate(self) -> StochasticScalar:
        """Calculates the aggregate loss for each simulation.

        Sums all individual event losses within each simulation to get the total
        loss per simulation. This converts event-level FreqSevSims data to
        simulation-level StochasticScalar data suitable for statistical analysis.

        Example:
            >>> sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
            >>> values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
            >>> n_sims = 3
            >>> fs = FreqSevSims(sim_index, values, n_sims)
            >>> aggregate_losses: StochasticScalar = fs.aggregate()
            >>> aggregate_losses
            StochasticScalar([3., 12., 30.])
            >>> # Now you can apply statistical methods
            >>> aggregate_losses.mean()
            15.0

        Returns:
            StochasticScalar: Array containing the aggregate loss for each simulation.
                Use this for statistical analysis (mean, std, percentiles, etc.).
        """
        return self._reduce_over_events(np.add.at)

    def occurrence(self) -> StochasticScalar:
        """Calculates the maximum occurrence loss for each simulation.

        Finds the largest individual event loss within each simulation. This
        converts event-level FreqSevSims data to simulation-level StochasticScalar
        data suitable for statistical analysis.

        Example:
            >>> sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
            >>> values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
            >>> n_sims = 3
            >>> fs = FreqSevSims(sim_index, values, n_sims)
            >>> max_losses: StochasticScalar = fs.occurrence()
            >>> max_losses
            StochasticScalar([2., 5., 9.])
            >>> # Now you can apply statistical methods
            >>> max_losses.mean()
            5.33

        Returns:
            StochasticScalar: Array containing the maximum occurrence loss for each
                simulation. Use this for statistical analysis (mean, std,
                percentiles, etc.).
        """
        return self._reduce_over_events(np.maximum.at)

    def deep_copy(self) -> FreqSevSims:
        """Creates a deep copy of the FreqSevSims object."""
        return FreqSevSims(self.sim_index, self.values.copy(), self.n_sims)

    def copy(self) -> FreqSevSims:
        """Creates a copy of the FreqSevSims object."""
        result = FreqSevSims(self.sim_index, self.values.copy(), self.n_sims)
        result.coupled_variable_group.merge(self.coupled_variable_group)
        return result

    def apply(self, func: ArrayTransform) -> FreqSevSims:
        """Applies a function to the values of the FreqSevSims object."""
        result = FreqSevSims(self.sim_index, func(self.values), self.n_sims)
        result.coupled_variable_group.merge(self.coupled_variable_group)
        return result

    def _extract_array_for_ufunc(self, x: t.Any) -> npt.NDArray[np.floating]:
        """Extract array values from various input types for ufunc operations.

        Args:
            x: Input value that could be FreqSevSims, StochasticScalar, ndarray,
               or scalar

        Returns:
            Array values aligned with simulation indices
        """
        if isinstance(x, FreqSevSims):
            return x.values
        elif isinstance(x, StochasticScalar):
            return x.values[self.sim_index]
        elif isinstance(x, np.ndarray):
            # Type ignore: Pyright can't infer the exact dtype of indexed arrays
            return x[self.sim_index]  # type: ignore[misc]
        else:
            # Scalar value - return as-is
            return x

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: t.Any, **kwargs: t.Any
    ) -> FreqSevSims:
        _inputs = tuple(self._extract_array_for_ufunc(x) for x in inputs)
        out = kwargs.get("out", ())
        if out:
            kwargs["out"] = tuple(x.values for x in out)
        result = getattr(ufunc, method)(*_inputs, **kwargs)
        result = FreqSevSims(self.sim_index, result, self.n_sims)
        for input in inputs:
            if isinstance(input, ProteusStochasticVariable):
                input.coupled_variable_group.merge(self.coupled_variable_group)
        result.coupled_variable_group.merge(self.coupled_variable_group)

        return result

    def __array_function__(
        self, func: Callable[..., t.Any], _: t.Any, args: t.Any, kwargs: t.Any
    ) -> np.number[t.Any] | FreqSevSims:
        """Handle numpy array functions for FreqSevSims objects.

        Args:
            func: The numpy function being called
            types: Types involved in the operation
            args: Arguments passed to the function
            kwargs: Keyword arguments passed to the function

        Returns:
            Either a scalar result or new FreqSevSims object

        Raises:
            NotImplementedError: If the function is not supported
        """
        if func not in (
            np.where,
            np.sum,
            np.array_equal,
            np.minimum,
            np.maximum,
            np.mean,
        ):
            raise NotImplementedError(f"Function {func.__name__} not supported")

        # Special handling for mean - aggregate first, then take mean
        if func is np.mean and len(args) == 1 and args[0] is self:
            return func(self.aggregate(), **kwargs)

        # Extract values from FreqSevSims objects, leave others as-is
        processed_args = tuple(
            x.values if isinstance(x, FreqSevSims) else x for x in args
        )

        result = func(*processed_args, **kwargs)

        # If result is a scalar, return it directly
        # Type ignore: Pyright can't infer the exact numpy scalar type
        if isinstance(result, np.number | np.bool_ | bool) or np.isscalar(result):
            return result  # type: ignore[misc]

        # Otherwise create a new FreqSevSims object with the result
        new_freq_sev = FreqSevSims(self.sim_index, result, self.n_sims)
        new_freq_sev.coupled_variable_group.merge(self.coupled_variable_group)
        return new_freq_sev

    def __repr__(self):
        return f"{type(self).__name__}({self.values!r})"

    def _is_compatible(self, other: ProteusCompatibleTypes) -> bool:
        """Check if two FreqSevSims objects are compatible for mathematical operations.

        Args:
            other: Another FreqSevSims object or compatible type

        Returns:
            True if compatible, False otherwise
        """
        return isinstance(other, FreqSevSims) and self.sim_index is other.sim_index

    def upsample(self, n_sims: int) -> FreqSevSims:
        """Upsamples the FreqSevSims object to the given number of simulations.

        Args:
            n_sims: Target number of simulations

        Returns:
            New FreqSevSims object with upsampled data

        Raises:
            ValueError: If self.n_sims is None
        """
        if self.n_sims is None:
            raise ValueError("Cannot upsample: n_sims is None")
        if n_sims == self.n_sims:
            return self.copy()
        sim_index = np.repeat(self.sim_index, n_sims // self.n_sims)
        values = np.repeat(self.values, n_sims // self.n_sims)
        if n_sims % self.n_sims > 0:
            sim_index = np.concatenate(
                (sim_index, self.sim_index[self.sim_index < n_sims % self.n_sims])
            )
            values = np.concatenate(
                (values, self.values[self.sim_index < n_sims % self.n_sims])
            )
        sim_index = sim_index + np.arange(len(sim_index)) % n_sims
        return FreqSevSims(sim_index, values, n_sims)

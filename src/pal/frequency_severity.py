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
from functools import lru_cache

import numpy as np
import numpy.typing as npt

from . import distributions
from ._maths import asnumpy, scalar_or_array, to_backend, xp
from ._value_arrays import host_values, stochastic_array
from .config import config
from .couplings import ProteusStochasticVariable as _ProteusStochasticVariable
from .stochastic_scalar import (
    StochasticScalar,
)
from .types import RandomGenerator

# Type aliases for frequency-severity modeling

# Types that can be used in mathematical operations with FreqSevSims objects
ProteusCompatibleTypes = t.Union["FreqSevSims", StochasticScalar, int, float, npt.NDArray[t.Any]]

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


_GPU_EVENT_INDEX_FILL_SOURCE = r"""
extern "C" __global__
void fill_cooperative(const long long* counts, const long long* ends,
                      long long* indices, const long long n_sims) {
    __shared__ int heavy_lanes[256];
    __shared__ int n_heavy;
    if (threadIdx.x == 0) n_heavy = 0;
    __syncthreads();

    const long long sim = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long count = sim < n_sims ? counts[sim] : 0;
    const long long start = sim < n_sims ? ends[sim] - count : 0;
    // Each lane handles normal frequencies directly. Large segments are
    // finished cooperatively by the whole block, rather than one CUDA lane.
    const long long cap = count < 64 ? count : 64;
    for (long long j = 0; j < cap; ++j) indices[start + j] = sim;
    if (count > 64) {
        const int slot = atomicAdd(&n_heavy, 1);
        heavy_lanes[slot] = threadIdx.x;
    }
    __syncthreads();

    for (int h = 0; h < n_heavy; ++h) {
        const long long heavy_sim =
            (long long)blockIdx.x * blockDim.x + heavy_lanes[h];
        const long long heavy_count = counts[heavy_sim];
        const long long heavy_start = ends[heavy_sim] - heavy_count;
        for (long long j = 64 + threadIdx.x; j < heavy_count;
             j += blockDim.x) {
            indices[heavy_start + j] = heavy_sim;
        }
    }
}
"""


@lru_cache(maxsize=1)
def _gpu_event_index_fill_kernel() -> t.Any:
    """Compile the CUDA event-index kernel only on the GPU backend."""
    return xp.RawKernel(_GPU_EVENT_INDEX_FILL_SOURCE, "fill_cooperative")


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
    if n_events_by_sim.size == 0:
        # The CUDA path cannot read the final cumulative count of an empty array.
        return xp.empty(0, dtype=xp.int64)

    if xp is np:
        # Avoid binary searches for each event on CPU. The repeated simulation
        # indices are the same as the cumulative-count search, including gaps.
        return np.repeat(np.arange(len(n_events_by_sim), dtype=np.int64), n_events_by_sim)

    # The GPU path assigns short segments to individual threads. Within each
    # block, threads cooperate to fill unusually large segments. This avoids
    # an event-wise binary search and also handles concentrated claim counts.
    cumulative_n_events = n_events_by_sim.cumsum(dtype=xp.int64)
    total_events = int(cumulative_n_events[-1])
    indices = xp.empty(total_events, dtype=xp.int64)
    if total_events:
        threads = 256
        blocks = (n_events_by_sim.size + threads - 1) // threads
        _gpu_event_index_fill_kernel()(
            (blocks,),
            (threads,),
            (n_events_by_sim, cumulative_n_events, indices, np.int64(n_events_by_sim.size)),
        )
    return indices


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
        self,
        n_sims: int | None = None,
        rng: RandomGenerator | None = None,
    ) -> FreqSevSims:
        """Generate simulations from the Frequency-Severity model.

        Parameters:
        - n_sims (int): Number of simulations to generate. If None, uses the
            default value from the config.
        - rng (RandomGenerator, optional): Random number generator. Uses
            config.rng if None.

        Returns:
            Object containing the generated simulations.
        """
        if n_sims is None:
            n_sims = config.n_sims
        if rng is None:
            rng = config.rng
        n_events = self.freq_dist.generate(n_sims, rng)
        total_events = n_events.sum()
        # Preserve the event-value type even when all simulated frequencies are zero.
        sev = self.sev_dist.generate(max(1, int(total_events)), rng)
        if total_events == 0:
            sev = StochasticScalar(sev.values[:0])
        # Convert n_events to integers since _get_sims_of_events expects integer counts
        # but n_events.values comes from distributions which return floating arrays
        result = FreqSevSims(_get_sims_of_events(n_events.values.astype(np.int64)), sev.values, n_sims)
        result.coupled_variable_group.merge(n_events.coupled_variable_group)
        result.coupled_variable_group.merge(sev.coupled_variable_group)
        return result


class FreqSevSims(_ProteusStochasticVariable):
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

    n_sims: int
    """Number of simulations."""

    def __init__(
        self,
        sim_index: np.ndarray | list[int],
        values: npt.ArrayLike,
        n_sims: int,
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
        self.sim_index = xp.asarray(sim_index)
        # NumPy infers float for an empty list, but empty event indices
        # must remain suitable for indexed reductions and bincount.
        if self.sim_index.size == 0:
            self.sim_index = self.sim_index.astype(int)
        self.values = stochastic_array(values)
        self.n_sims = n_sims  # type: ignore

        if len(self.sim_index) != len(self.values):
            raise ValueError(
                f"Length mismatch: sim_index has {len(self.sim_index)} elements "
                f"but values has {len(self.values)} elements"
            )

    def __str__(self):
        return "Simulation Index\n" + str(self.sim_index) + "\n Values\n" + str(self.values)

    def _reorder_sims(self, new_order: t.Sequence[int]) -> None:
        """Reorder simulations according to the given order.

        This method updates the simulation indices to match the new order.

        Args:
            new_order: A sequence of integers representing the new order of simulations.
        """
        reverse_ordering = xp.empty(len(new_order), dtype=int)
        reverse_ordering[new_order] = xp.arange(len(new_order), dtype=int)
        self.sim_index = reverse_ordering[self.sim_index]

    def __getitem__(self, sim_index: int) -> StochasticScalar:
        """Returns the values of the simulation with the given simulation index."""
        # get the positions of the given simulation index
        if isinstance(sim_index, int):  # type: ignore[unecessary-check]
            indices = asnumpy(self.sim_index) if host_values(self.values) else self.sim_index
            ints = (indices == sim_index).nonzero()
            return StochasticScalar(self.values[ints])
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of simulations."""
        return self.n_sims

    def __iter__(self) -> t.Iterator[StochasticScalar]:
        """Iterate over the simulations, preserving each simulation's event order."""
        # __getitem__ scans the entire event-index array on each access.
        # Group once rather than repeating that scan for every simulation.
        indices = self.sim_index.astype(xp.int64, copy=False)
        counts = xp.bincount(indices, minlength=self.n_sims)
        offsets = xp.cumsum(counts)
        # Python iteration requires host integer slice offsets. On CUDA, copy
        # only these counts, never the event values.
        ends = asnumpy(offsets) if xp is not np else offsets

        if indices.size < 2 or bool(xp.all(indices[1:] >= indices[:-1])):
            grouped_values = self.values
        else:
            # Copula reordering may relabel the simulations without sorting
            # the events. A stable sort retains their original within-sim order.
            order = xp.argsort(indices, kind="stable")
            grouped_values = self.values[asnumpy(order) if host_values(self.values) else order]

        start = 0
        for end in ends:
            end = int(end)
            yield StochasticScalar(grouped_values[start:end].copy())
            start = end

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
        if host_values(self.values):
            raise TypeError("Event values must be numeric for aggregation")
        _result = xp.zeros(self.n_sims)
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

    def count(self) -> StochasticScalar:
        """Counts the number of losses (events) in each simulation.

        Returns the frequency count of how many individual losses occurred
        within each simulation. This is useful for analyzing frequency
        distributions and understanding claim counts.

        Example:
            >>> sim_index = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
            >>> values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
            >>> n_sims = 3
            >>> fs = FreqSevSims(sim_index, values, n_sims)
            >>> loss_counts: StochasticScalar = fs.count()
            >>> loss_counts
            StochasticScalar([2., 3., 4.])
            >>> # Now you can apply statistical methods
            >>> loss_counts.mean()
            3.0
            >>> loss_counts.quantile(0.5)
            3.0

        Returns:
            StochasticScalar: Array containing the number of losses for each
                simulation. Use this for analyzing frequency distributions.
        """
        counts = xp.bincount(self.sim_index, minlength=self.n_sims).astype(float)
        result = StochasticScalar(counts)
        result.coupled_variable_group.merge(self.coupled_variable_group)
        return result

    def deep_copy(self) -> FreqSevSims:
        """Creates a deep copy of the FreqSevSims object."""
        return FreqSevSims(self.sim_index.copy(), self.values.copy(), self.n_sims)

    def copy(self) -> FreqSevSims:
        """Creates a copy of the FreqSevSims object."""
        result = FreqSevSims(self.sim_index.copy(), self.values.copy(), self.n_sims)
        result.coupled_variable_group.merge(self.coupled_variable_group)
        return result

    def apply(self, func: ArrayTransform) -> FreqSevSims:
        """Applies a function to the values of the FreqSevSims object."""
        result = FreqSevSims(self.sim_index, func(self.values), self.n_sims)
        result.coupled_variable_group.merge(self.coupled_variable_group)
        return result

    def _extract_array_for_ufunc(self, x: t.Any, host: bool | None = None) -> npt.NDArray[np.floating]:
        """Extract array values from various input types for ufunc operations.

        Args:
            x: Input value that could be FreqSevSims, StochasticScalar, ndarray,
               or scalar
            host: Whether operands should be extracted as NumPy host arrays.

        Returns:
            Array values aligned with simulation indices
        """
        if host is None:
            host = host_values(self.values) or host_values(
                x.values if isinstance(x, (FreqSevSims, StochasticScalar)) else x
            )
        indices = asnumpy(self.sim_index) if host else self.sim_index
        if isinstance(x, FreqSevSims):
            return asnumpy(x.values) if host else x.values
        elif isinstance(x, StochasticScalar):
            values = asnumpy(x.values) if host else x.values
            return values[indices]
        elif isinstance(x, (np.ndarray, xp.ndarray)):
            backend_array = asnumpy(x) if host else to_backend(x)
            return backend_array[indices]  # type: ignore[misc]
        else:
            # Scalar value - return as-is
            return x

    def _validate_event_layout(self, *operands: t.Any) -> None:
        """Reject event-wise operations on different scenario/event assignments."""
        for other in operands:
            if not isinstance(other, FreqSevSims) or other is self:
                continue
            if self.n_sims != other.n_sims or (
                self.sim_index is not other.sim_index and not bool(xp.array_equal(self.sim_index, other.sim_index))
            ):
                raise ValueError("Event simulation indices must match for event-wise operations")

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: t.Any, **kwargs: t.Any) -> FreqSevSims:
        where_mask = kwargs.get("where")
        self._validate_event_layout(*inputs, *(kwargs.get("out") or ()), where_mask)
        host = (
            host_values(self.values)
            or any(host_values(x.values if isinstance(x, (FreqSevSims, StochasticScalar)) else x) for x in inputs)
            or host_values(kwargs.get("where"))
        )
        _inputs = tuple(self._extract_array_for_ufunc(x, host) for x in inputs)
        if host:
            _inputs = tuple(asnumpy(x) if isinstance(x, xp.ndarray) else x for x in _inputs)
            ufunc = getattr(np, ufunc.__name__, ufunc)
        out = kwargs.get("out", ())
        out_arrays: tuple[t.Any, ...] = ()
        if out:
            out_arrays = tuple(x.values if isinstance(x, FreqSevSims) else x for x in out)
            kwargs["out"] = (
                tuple(asnumpy(x).copy() if isinstance(x, xp.ndarray) else x for x in out_arrays) if host else out_arrays
            )
        if "where" in kwargs:
            if isinstance(where_mask, (FreqSevSims, StochasticScalar)):
                kwargs["where"] = self._extract_array_for_ufunc(where_mask, host)
            elif host:
                kwargs["where"] = asnumpy(where_mask)
        if xp is not np and not host and method == "__call__" and "where" in kwargs:
            # CuPy ufuncs do not accept the NumPy "where" keyword. Keep the
            # mask and the output on the device, preserving unselected values.
            mask = xp.asarray(kwargs["where"])
            if isinstance(where_mask, (FreqSevSims, StochasticScalar, np.ndarray, xp.ndarray)) and not np.can_cast(
                mask.dtype, np.bool_, casting="safe"
            ):
                raise TypeError("where mask cannot be safely cast to bool")
            unmasked_kwargs = {key: value for key, value in kwargs.items() if key not in ("where", "out")}
            calculated = getattr(ufunc, method)(*_inputs, **unmasked_kwargs)
            if out_arrays and out_arrays[0] is not None:
                destination = out_arrays[0]
                xp.copyto(destination, calculated, casting=kwargs.get("casting", "same_kind"), where=mask)
                result = destination
            else:
                result = xp.where(mask, calculated, xp.empty_like(calculated))
        else:
            result = getattr(ufunc, method)(*_inputs, **kwargs)
        if out and host:
            for destination, buffer in zip(out_arrays, kwargs["out"]):
                if destination is not buffer:
                    destination[...] = xp.asarray(buffer) if isinstance(destination, xp.ndarray) else buffer
        scalar_result = scalar_or_array(result)
        if scalar_result is not result:
            return scalar_result  # type: ignore[return-value]
        result = FreqSevSims(self.sim_index, result, self.n_sims)
        for input in (*inputs, where_mask, *out):
            if isinstance(input, _ProteusStochasticVariable):
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
            np.round,
        ):
            raise NotImplementedError(f"Function {func.__name__} not supported")

        if func is not np.array_equal:
            self._validate_event_layout(*args)

        # Special handling for mean - aggregate first, then take mean
        if func is np.mean and len(args) == 1 and args[0] is self:
            return func(self.aggregate(), **kwargs)

        # Extract values from FreqSevSims objects, leave others as-is
        host = host_values(self.values) or any(host_values(x.values if isinstance(x, FreqSevSims) else x) for x in args)
        processed_args = tuple(
            (asnumpy(x.values) if isinstance(x, FreqSevSims) else asnumpy(x) if isinstance(x, xp.ndarray) else x)
            if host
            else (x.values if isinstance(x, FreqSevSims) else to_backend(x))
            for x in args
        )
        if func is np.where:
            processed_args = ((np if host else xp).asarray(processed_args[0]), *processed_args[1:])

        result = func(*processed_args, **kwargs)

        scalar_result = scalar_or_array(result)
        if scalar_result is not result:
            return scalar_result  # type: ignore[return-value]
        if isinstance(result, (np.number, np.bool_, bool)) or np.isscalar(result):
            return result  # type: ignore[misc]

        # Otherwise create a new FreqSevSims object with the result
        new_freq_sev = FreqSevSims(self.sim_index, result, self.n_sims)
        new_freq_sev.coupled_variable_group.merge(self.coupled_variable_group)
        for argument in args:
            if isinstance(argument, _ProteusStochasticVariable):
                new_freq_sev.coupled_variable_group.merge(argument.coupled_variable_group)
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
        return (
            isinstance(other, FreqSevSims)
            and self.n_sims == other.n_sims
            and bool(xp.array_equal(self.sim_index, other.sim_index))
        )

    def upsample(self, n_sims: int) -> FreqSevSims:
        """Resample complete event lists by cycling through source simulations.

        The events belonging to each source simulation are copied as a unit.
        Within-simulation event order is preserved, including for unsorted inputs.
        """
        if n_sims < 1 or self.n_sims < 1:
            raise ValueError("n_sims must be positive for frequency-severity resampling")
        if n_sims == self.n_sims:
            return self.copy()

        source_sims = xp.arange(n_sims, dtype=int) % self.n_sims
        source_counts = xp.bincount(self.sim_index, minlength=self.n_sims)
        target_counts = source_counts[source_sims]
        sim_index = xp.repeat(xp.arange(n_sims, dtype=int), target_counts)
        source_event_order = xp.argsort(self.sim_index, kind="stable")
        source_starts = xp.cumsum(source_counts) - source_counts
        target_starts = xp.cumsum(target_counts) - target_counts
        within_event = xp.arange(len(sim_index)) - xp.repeat(target_starts, target_counts)
        source_positions = source_starts[source_sims[sim_index]] + within_event
        event_positions = source_event_order[source_positions]
        values = self.values[asnumpy(event_positions) if host_values(self.values) else event_positions]
        return FreqSevSims(sim_index, values, n_sims)

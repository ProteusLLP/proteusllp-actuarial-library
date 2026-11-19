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
from copy import copy

import numpy as np
import numpy.typing as npt

from . import distributions
from ._maths import asnumpy, scalar_or_array, to_backend, xp
from ._value_arrays import host_values, stochastic_array
from .config import config
from .couplings import ProteusStochasticVariable as _ProteusStochasticVariable
from .couplings import couple_stochastic_results, validate_stochastic_axes
from .qmc import RandomizedQMCGenerator
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
    if xp is np:
        # Avoid binary searches for each event on CPU. The repeated simulation
        # indices are the same as the cumulative-count search, including gaps.
        return np.repeat(np.arange(len(n_events_by_sim), dtype=np.int64), n_events_by_sim)

    cumulative_n_events = n_events_by_sim.cumsum()
    total_events = cumulative_n_events[-1]
    event_no = t.cast(npt.NDArray[np.int64], xp.arange(total_events))
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

    def _qmc_severities_ordinal(
        self, n_sims: int, counts: t.Any, total_events: int, rng: RandomizedQMCGenerator
    ) -> t.Any:
        """Sample full scenario vectors for severities without inverse CDFs.

        Event-list rows are not independent QMC paths. Instead, generate each
        claim position across simulation paths to retain the intended
        within-simulation joint severity distribution.
        """
        offsets = xp.cumsum(counts) - counts
        max_count = int(xp.max(counts))
        values: t.Any = None
        # One draw also preserves string/date dtypes when no claims occur.
        for claim in range(max(1, max_count)):
            draw = self.sev_dist.generate(n_sims, rng).values
            if values is None:
                backend = np if host_values(draw) else xp
                values = backend.empty(total_events, dtype=draw.dtype)
            if claim >= max_count:
                continue
            active = counts > claim
            positions = offsets[active] + claim
            if host_values(draw):
                values[asnumpy(positions)] = draw[asnumpy(active)]
            else:
                values[positions] = draw[active]
        return values

    def _qmc_severities(
        self,
        n_sims: int,
        counts: t.Any,
        event_indices: t.Any,
        total_events: int,
        rng: RandomizedQMCGenerator,
    ) -> t.Any:
        """Use indexed Sobol claim coordinates and one packed inverse CDF.

        The severity assigned to the k-th claim in simulation i is the
        k-th QMC severity coordinate for path i. Reconstructing just these
        observed pairs preserves exactly the same scrambled Sobol design as
        drawing an entire coordinate for every claim ordinal.
        """
        # The inverse Gaussian currently has no inverse-CDF implementation.
        # Keep its established two-coordinate sampling construction.
        if total_events == 0 or isinstance(self.sev_dist, distributions.InverseGaussian):
            return self._qmc_severities_ordinal(n_sims, counts, total_events, rng)

        maximum = int(xp.max(counts))
        offsets = xp.cumsum(counts) - counts
        ordinals = xp.arange(total_events, dtype=xp.int64) - offsets[event_indices]
        uniforms = rng._event_uniforms(n_sims, event_indices, ordinals, maximum)
        uniforms = xp.clip(uniforms, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))

        severity_distribution = self.sev_dist
        if any(isinstance(value, StochasticScalar) for value in self.sev_dist._params.values()):
            # Each claim receives the parameter value of its simulation.
            # Temporary claim-indexed scalars must not be coupled with the
            # original simulation-indexed parameters.
            severity_distribution = copy(self.sev_dist)
            severity_distribution._params = {
                name: (
                    StochasticScalar(
                        value.values[asnumpy(event_indices) if host_values(value.values) else event_indices]
                    )
                    if isinstance(value, StochasticScalar)
                    else value
                )
                for name, value in self.sev_dist._params.items()
            }

        quantiles = severity_distribution.invcdf(StochasticScalar(uniforms))
        return quantiles.values if isinstance(quantiles, StochasticScalar) else stochastic_array(quantiles)

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

        Stochastic severity parameters have one value per simulation. Every
        claim in a simulation uses that simulation's parameter values.
        """
        if n_sims is None:
            n_sims = config.n_sims
        if n_sims < 1:
            raise ValueError(f"n_sims must be >= 1, got {n_sims}")

        frequency_params = getattr(self.freq_dist, "_params", {})
        validate_stochastic_axes(
            (param for param in frequency_params.values() if isinstance(param, StochasticScalar)),
            n_sims=n_sims,
        )
        if rng is None:
            rng = config.rng
        severity_params = getattr(self.sev_dist, "_params", {})
        stochastic_params = {
            name: param for name, param in severity_params.items() if isinstance(param, StochasticScalar)
        }
        for name, param in stochastic_params.items():
            if param.n_sims != n_sims:
                raise ValueError(
                    f"Stochastic severity parameter {name!r} must have one value per simulation "
                    f"({n_sims} values), got {param.n_sims}"
                )

        n_events = self.freq_dist.generate(n_sims, rng)
        counts = n_events.values
        # A frequency distribution must produce whole, nonnegative claim counts.
        # Casting first would silently truncate fractional values, and negative
        # or nonfinite values can corrupt event indexing.
        if counts.dtype.kind not in "iuf" or bool(
            xp.any(~xp.isfinite(counts) | (counts < 0) | (counts != xp.floor(counts)))
        ):
            raise ValueError("Frequency simulations must contain finite, nonnegative integer counts")
        total_events = int(n_events.sum())
        event_indices = _get_sims_of_events(counts.astype(np.int64))

        if isinstance(rng, RandomizedQMCGenerator):
            values = self._qmc_severities(n_sims, counts.astype(np.int64), event_indices, total_events, rng)
            result = FreqSevSims(event_indices, values, n_sims)
            couple_stochastic_results((result,), (n_events, *stochastic_params.values()), n_sims=n_sims)
            return result

        severity_distribution = self.sev_dist
        if stochastic_params:
            # Generate one severity per claim, using the parameter values of that
            # claim's simulation. Do not modify the reusable distribution instance.
            severity_distribution = copy(self.sev_dist)
            sample_indices = event_indices if total_events else xp.asarray([0], dtype=int)
            severity_distribution._params = {
                name: (
                    StochasticScalar(
                        param.values[asnumpy(sample_indices) if host_values(param.values) else sample_indices]
                    )
                    if isinstance(param, StochasticScalar)
                    else param
                )
                for name, param in severity_params.items()
            }

        # Draw once even when there are no events to preserve nonnumeric severity types.
        sev = severity_distribution.generate(max(1, total_events), rng)
        values = sev.values[:total_events]
        result = FreqSevSims(event_indices, values, n_sims)
        # Link original simulation-level parameters, never expanded claim-level temporaries.
        couple_stochastic_results((result,), (n_events, *stochastic_params.values()), n_sims=n_sims)
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
        couple_stochastic_results((result,), (self,))
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
        couple_stochastic_results((result,), (self,))
        return result

    def deep_copy(self) -> FreqSevSims:
        """Creates a deep copy of the FreqSevSims object."""
        return FreqSevSims(self.sim_index.copy(), self.values.copy(), self.n_sims)

    def copy(self) -> FreqSevSims:
        """Creates a copy of the FreqSevSims object."""
        result = FreqSevSims(self.sim_index.copy(), self.values.copy(), self.n_sims)
        couple_stochastic_results((result,), (self,))
        return result

    def apply(self, func: ArrayTransform) -> FreqSevSims:
        """Applies a function to the values of the FreqSevSims object."""
        result = FreqSevSims(self.sim_index, func(self.values), self.n_sims)
        couple_stochastic_results((result,), (self,))
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

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: t.Any, **kwargs: t.Any) -> t.Any:
        """Apply event-wise ufuncs with atomic output and simulation-axis coupling."""
        if method == "at":
            raise NotImplementedError("In-place ufunc.at is unsupported on frequency-severity results")
        if method not in ("__call__", "reduce", "reduceat", "accumulate", "outer"):
            return NotImplemented

        kwargs = dict(kwargs)
        raw_out = kwargs.pop("out", None)
        destinations = raw_out if isinstance(raw_out, tuple) else (raw_out,) if raw_out is not None else ()
        if destinations and len(destinations) != (ufunc.nout if method == "__call__" else 1):
            raise ValueError("Number of ufunc output destinations does not match the operation")
        where_mask = kwargs.pop("where", True)
        self._validate_event_layout(*inputs, *destinations, where_mask)
        sources = (
            self,
            *(value for value in (*inputs, where_mask) if isinstance(value, _ProteusStochasticVariable)),
        )
        targets = tuple(value for value in destinations if isinstance(value, _ProteusStochasticVariable))
        validate_stochastic_axes((*sources, *targets), n_sims=self.n_sims)
        if any(isinstance(value, StochasticScalar) for value in destinations):
            raise TypeError("Frequency-severity event outputs require event-aligned FreqSevSims or raw arrays")
        if method != "__call__" and any(isinstance(value, FreqSevSims) for value in destinations):
            raise ValueError("A frequency-severity out= destination requires event-wise elementwise results")

        raw_mask = (
            self._extract_array_for_ufunc(where_mask)
            if isinstance(where_mask, (FreqSevSims, StochasticScalar))
            else where_mask
        )
        if not isinstance(raw_mask, (bool, np.bool_)):
            mask_shape = raw_mask.shape if hasattr(raw_mask, "shape") else np.shape(raw_mask)
            mask_dtype = raw_mask.dtype if hasattr(raw_mask, "dtype") else np.asarray(raw_mask).dtype
            if mask_dtype.kind != "b":
                raise TypeError("where must be a boolean mask")
            if mask_shape not in ((), (len(self.values),)):
                raise ValueError("Event where mask has incompatible event count")
        if method == "__call__" and not (isinstance(raw_mask, (bool, np.bool_)) and bool(raw_mask)):
            if not destinations or any(destination is None for destination in destinations):
                raise ValueError("Masked event-wise ufuncs require out= for every output to define unselected events")

        host = (
            host_values(self.values)
            or any(host_values(x.values if isinstance(x, (FreqSevSims, StochasticScalar)) else x) for x in inputs)
            or host_values(raw_mask)
        )
        prepared_inputs = tuple(self._extract_array_for_ufunc(value, host) for value in inputs)
        if host:
            prepared_inputs = tuple(
                asnumpy(value) if isinstance(value, xp.ndarray) else value for value in prepared_inputs
            )
        operation = getattr(np, ufunc.__name__, ufunc) if host else ufunc

        if method == "reduce" and not (isinstance(raw_mask, (bool, np.bool_)) and bool(raw_mask)):
            kwargs["where"] = asnumpy(raw_mask) if host else to_backend(raw_mask)
        elif method not in ("__call__", "reduce") and not (isinstance(raw_mask, (bool, np.bool_)) and bool(raw_mask)):
            raise TypeError("where is only supported for elementwise ufuncs or reduce")
        calculated = getattr(operation, method)(*prepared_inputs, **kwargs)
        results = calculated if isinstance(calculated, tuple) else (calculated,)

        if method != "__call__":
            if destinations:
                destination = destinations[0]
                if destination is None:
                    return scalar_or_array(calculated)
                raw_destination = destination
                if not isinstance(raw_destination, (np.ndarray, xp.ndarray)):
                    raise TypeError("Reduction out= must be a raw NumPy/CuPy array")
                if raw_destination.shape != np.shape(calculated):
                    raise ValueError("out= has an incompatible result shape")
                backend = np if isinstance(raw_destination, np.ndarray) else xp
                values = asnumpy(calculated) if backend is np else xp.asarray(calculated)
                backend.copyto(raw_destination, values, casting=kwargs.get("casting", "same_kind"))
                return destination
            return scalar_or_array(calculated)

        if destinations and len(destinations) != len(results):
            raise ValueError("Unexpected number of ufunc outputs")
        if not destinations:
            destinations = (None,) * len(results)

        # Preflight *every* output before writing the first destination.
        casting = kwargs.get("casting", "same_kind")
        for value, destination in zip(results, destinations):
            if getattr(value, "ndim", None) != 1 or len(value) != len(self.values):
                raise ValueError("Event-wise ufunc result has incompatible event count")
            if destination is None:
                continue
            raw_destination = destination.values if isinstance(destination, FreqSevSims) else destination
            if not isinstance(raw_destination, (np.ndarray, xp.ndarray)):
                raise TypeError("out= must contain raw arrays or event-aligned FreqSevSims")
            if raw_destination.shape != np.shape(value):
                raise ValueError("out= has an incompatible result shape")
            if not np.can_cast(value.dtype, raw_destination.dtype, casting=casting):
                raise TypeError("out= cannot represent the event-wise result")

        outputs: list[t.Any] = []
        for value, destination in zip(results, destinations):
            if destination is None:
                outputs.append(FreqSevSims(self.sim_index, value, self.n_sims))
                continue
            raw_destination = destination.values if isinstance(destination, FreqSevSims) else destination
            backend = np if isinstance(raw_destination, np.ndarray) else xp
            copy_value = asnumpy(value) if backend is np else xp.asarray(value)
            copy_mask = (
                asnumpy(raw_mask)
                if backend is np and not isinstance(raw_mask, (bool, np.bool_))
                else xp.asarray(raw_mask)
                if backend is xp and not isinstance(raw_mask, (bool, np.bool_))
                else raw_mask
            )
            backend.copyto(raw_destination, copy_value, where=copy_mask, casting=casting)
            outputs.append(destination)

        coupled_outputs = tuple(value for value in outputs if isinstance(value, FreqSevSims))
        if coupled_outputs:
            couple_stochastic_results(coupled_outputs, sources, n_sims=self.n_sims)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    def __array_function__(self, func: Callable[..., t.Any], _: t.Any, args: t.Any, kwargs: t.Any) -> t.Any:
        """Dispatch supported array functions without mixing claim and scenario axes."""
        supported = (np.where, np.sum, np.array_equal, np.minimum, np.maximum, np.mean, np.round)
        if func not in supported:
            raise NotImplementedError(f"Function {func.__name__} not supported")

        kwargs = dict(kwargs)
        output_index = 2 if func is np.round else 3 if func in (np.sum, np.mean) else None
        has_keyword_out = "out" in kwargs
        output = kwargs.pop("out", None)
        if output_index is not None and len(args) > output_index:
            if has_keyword_out:
                raise TypeError("out= was supplied both positionally and by keyword")
            output = args[output_index]
            args = (*args[:output_index], None, *args[output_index + 1 :])
        if func is not np.array_equal:
            self._validate_event_layout(*args, output)
        sources = (
            self,
            *(value for value in (*args, *kwargs.values()) if isinstance(value, _ProteusStochasticVariable)),
        )
        target = (output,) if isinstance(output, _ProteusStochasticVariable) else ()
        if func is not np.array_equal:
            validate_stochastic_axes((*sources, *target), n_sims=self.n_sims)
        preserving = func in (np.where, np.minimum, np.maximum, np.round)
        if func is np.where and len(args) != 3:
            preserving = False
        if isinstance(output, _ProteusStochasticVariable) and (not preserving or not isinstance(output, FreqSevSims)):
            raise ValueError("Event-wise out= requires an event-aligned FreqSevSims result")

        if func is np.mean and args and args[0] is self and all(value is None for value in args[1:]):
            mean_result = func(self.aggregate(), **kwargs)
            if output is None:
                return mean_result
            if not isinstance(output, (np.ndarray, xp.ndarray)) or output.shape != ():
                raise ValueError("Mean reduction requires a scalar raw-array destination")
            backend = np if isinstance(output, np.ndarray) else xp
            mean_value = asnumpy(mean_result) if backend is np else xp.asarray(mean_result)
            if not np.can_cast(mean_value.dtype, output.dtype, casting="same_kind"):
                raise TypeError("out= cannot represent the mean")
            backend.copyto(output, mean_value, casting="same_kind")
            return output

        host = host_values(self.values) or any(
            host_values(value.values if isinstance(value, (FreqSevSims, StochasticScalar)) else value)
            for value in (*args, *kwargs.values())
        )

        def prepare(value: t.Any) -> t.Any:
            if isinstance(value, (FreqSevSims, StochasticScalar)):
                return self._extract_array_for_ufunc(value, host)
            return asnumpy(value) if host and isinstance(value, xp.ndarray) else value if host else to_backend(value)

        parsed_args = tuple(prepare(value) for value in args)
        parsed_kwargs = {key: prepare(value) for key, value in kwargs.items()}
        if func is np.where:
            parsed_args = ((np if host else xp).asarray(parsed_args[0]), *parsed_args[1:])
        result = func(*parsed_args, **parsed_kwargs)
        if preserving and getattr(result, "ndim", None) == 1 and len(result) != len(self.values):
            raise ValueError("Event-wise result has incompatible event count")
        if output is not None:
            destination = output.values if isinstance(output, FreqSevSims) else output
            if not isinstance(destination, (np.ndarray, xp.ndarray)):
                raise TypeError("out= must be a raw array or event-aligned FreqSevSims")
            if destination.shape != np.shape(result):
                raise ValueError("out= has incompatible event count")
            if not np.can_cast(result.dtype, destination.dtype, casting="same_kind"):
                raise TypeError("out= cannot represent the result")
            backend = np if isinstance(destination, np.ndarray) else xp
            copy_value = asnumpy(result) if backend is np else xp.asarray(result)
            backend.copyto(destination, copy_value, casting="same_kind")
            if isinstance(output, FreqSevSims):
                couple_stochastic_results((output,), sources, n_sims=self.n_sims)
            return output
        scalar_result = scalar_or_array(result)
        if scalar_result is not result:
            return scalar_result
        if isinstance(result, (np.number, np.bool_, bool)) or np.isscalar(result):
            return result
        if getattr(result, "ndim", None) != 1:
            return result
        if not preserving:
            return result
        new_freq_sev = FreqSevSims(self.sim_index, result, self.n_sims)
        couple_stochastic_results((new_freq_sev,), sources, n_sims=self.n_sims)
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

"""Stochastic scalar variables for Monte Carlo simulation.

Provides the StochasticScalar class for representing and manipulating
scalar-valued stochastic variables in actuarial and risk modeling applications.
Supports arithmetic operations, statistical functions, and numpy integration.
"""

from __future__ import annotations

import os
import typing as t

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go  # type: ignore

from pal import stats  # type: ignore

from ._compat import Self
from ._maths import asnumpy, scalar_or_array, to_backend, xp
from ._value_arrays import host_values, stochastic_array
from .couplings import CouplingGroup, ProteusStochasticVariable, couple_stochastic_results, validate_stochastic_axes
from .stats import NumberOrList
from .types import Numeric, ScipyNumeric


class StochasticScalar(ProteusStochasticVariable):
    """A class to represent a single scalar variable in a simulation."""

    coupled_variable_group: CouplingGroup
    n_sims: int  # pyright: ignore[reportIncompatibleVariableOverride]
    """The number of simulations in the variable."""

    # ===================
    # DUNDER METHODS
    # ===================

    def __init__(self, values: npt.ArrayLike):
        """Initialize a stochastic scalar.

        Args:
            values: An array-like sequence of values that describe the distribution
                for the scalar variable.
        """
        super().__init__()

        if isinstance(values, StochasticScalar):
            self.values = values.values
            self.n_sims = values.n_sims
            couple_stochastic_results((self,), (values,))
            return

        invalid_type_message = f"Type of values must be a sequence or array. Found {type(values).__name__}"
        if isinstance(values, (str, dict, set)):
            raise TypeError(invalid_type_message)

        try:
            array = stochastic_array(values)
        except TypeError:
            raise
        except ValueError as exc:
            raise TypeError(invalid_type_message) from exc

        if array.ndim != 1:
            raise ValueError("Values must be a 1D array.")

        self.values = array
        self.n_sims = len(array)

    def __repr__(self) -> str:
        try:
            return f"{type(self).__name__}(values={self.values}, n_sims={self.n_sims})"
        except AttributeError:
            return f"{type(self).__name__}(values=..., n_sims=...)"

    def __array_ufunc__(
        self,
        ufunc: t.Any,
        method: str,
        *inputs: t.Any,
        **kwargs: t.Any,
    ) -> t.Any:
        """Apply NumPy ufuncs while respecting scenario axes and output ownership.

        Elementwise outputs retain their scenario ancestry. Accumulations,
        reductions and indexed reductions are order-dependent summaries, so
        their outputs are detached. With `out=`, NumPy's original destination
        object is returned and its coupling is updated only after a successful
        calculation and write.
        """
        other_variable = next(
            (
                value
                for value in inputs
                if type(value).__name__ == "ProteusVariable"
                or (isinstance(value, ProteusStochasticVariable) and not isinstance(value, StochasticScalar))
            ),
            None,
        )
        if other_variable is not None:
            return t.cast(t.Any, other_variable).__array_ufunc__(ufunc, method, *inputs, **kwargs)

        if method == "at":
            raise NotImplementedError("In-place ufunc.at is unsupported on stochastic variables; use explicit values")
        if method not in ("__call__", "reduce", "reduceat", "accumulate", "outer"):
            return NotImplemented

        raw_out = kwargs.pop("out", None)
        destinations = raw_out if isinstance(raw_out, tuple) else (raw_out,) if raw_out is not None else ()
        if destinations and len(destinations) != (ufunc.nout if method == "__call__" else 1):
            raise ValueError("Number of ufunc output destinations does not match the operation")

        where = kwargs.pop("where", True)
        stochastic_inputs = tuple(value for value in inputs if isinstance(value, StochasticScalar))
        mask_source = (where,) if isinstance(where, StochasticScalar) else ()
        output_sources = tuple(value for value in destinations if isinstance(value, StochasticScalar))
        validate_stochastic_axes((self, *stochastic_inputs, *mask_source, *output_sources))

        # Only elementwise operations preserve scenario alignment in the output.
        if method != "__call__" and output_sources:
            raise ValueError("A stochastic out= destination requires a scenario-preserving elementwise operation")

        raw_mask = where.values if isinstance(where, StochasticScalar) else where
        if not isinstance(raw_mask, (bool, np.bool_)):
            mask_shape = raw_mask.shape if hasattr(raw_mask, "shape") else np.shape(raw_mask)
            mask_dtype = raw_mask.dtype if hasattr(raw_mask, "dtype") else np.asarray(raw_mask).dtype
            if mask_dtype.kind != "b":
                raise TypeError("where must be a boolean mask")
            if mask_shape not in ((), (self.n_sims,)):
                raise ValueError("where mask has incompatible n_sims")
        if (
            method == "__call__"
            and not destinations
            and not (isinstance(raw_mask, (bool, np.bool_)) and bool(raw_mask))
        ):
            raise ValueError("Masked stochastic ufuncs require out= to define unselected simulation values")

        is_host = (
            host_values(self.values)
            or any(host_values(x.values if isinstance(x, StochasticScalar) else x) for x in inputs)
            or host_values(raw_mask)
        )
        operation = getattr(np, ufunc.__name__, ufunc) if is_host else ufunc
        def prepare_input(value: t.Any) -> t.Any:
            if isinstance(value, StochasticScalar):
                return asnumpy(value.values) if is_host else value.values
            if is_host:
                return asnumpy(value) if isinstance(value, xp.ndarray) else value
            return to_backend(value)

        prepared_inputs = tuple(prepare_input(value) for value in inputs)
        if method == "reduce" and "where" in kwargs:
            raise TypeError("Duplicate where argument")
        if method == "reduce" and not (isinstance(raw_mask, (bool, np.bool_)) and bool(raw_mask)):
            kwargs["where"] = asnumpy(raw_mask) if is_host else to_backend(raw_mask)
        if method not in ("reduce", "__call__") and not (isinstance(raw_mask, (bool, np.bool_)) and bool(raw_mask)):
            raise TypeError("where is only supported for elementwise ufuncs or reduce")

        # Calculate without touching destinations. Invalid shapes or ufunc errors
        # must not mutate data or join coupling groups.
        calculated = getattr(operation, method)(*prepared_inputs, **kwargs)
        values = calculated if isinstance(calculated, tuple) else (calculated,)

        if destinations:
            if any(value is not None for value in destinations) and method == "outer":
                # Outer products have no one-dimensional scenario result.
                if any(isinstance(value, StochasticScalar) for value in destinations):
                    raise ValueError("outer outputs cannot be stochastic variables")
            if method == "__call__" and len(values) != len(destinations):
                raise ValueError("Unexpected number of ufunc results")
            casting = kwargs.get("casting", "same_kind")
            for value, destination in zip(values, destinations):
                if destination is None:
                    continue
                raw_dest = destination.values if isinstance(destination, StochasticScalar) else destination
                if not isinstance(raw_dest, (np.ndarray, xp.ndarray)):
                    raise TypeError("out= must contain NumPy/CuPy arrays or stochastic variables")
                if raw_dest.shape != np.shape(value):
                    raise ValueError("out= has an incompatible result shape")
                if not np.can_cast(value.dtype, raw_dest.dtype, casting=casting):
                    raise TypeError("out= cannot represent the result with the requested casting")

            result_leaves: list[t.Any] = []
            for value, destination in zip(values, destinations):
                if destination is None:
                    result_leaves.append(StochasticScalar(value) if method == "__call__" else value)
                    continue
                raw_dest = destination.values if isinstance(destination, StochasticScalar) else destination
                copy_backend = np if isinstance(raw_dest, np.ndarray) else xp
                copy_value = asnumpy(value) if copy_backend is np else xp.asarray(value)
                copy_mask = (
                    asnumpy(raw_mask)
                    if copy_backend is np and not isinstance(raw_mask, (bool, np.bool_))
                    else xp.asarray(raw_mask)
                    if copy_backend is xp and not isinstance(raw_mask, (bool, np.bool_))
                    else raw_mask
                )
                copy_backend.copyto(raw_dest, copy_value, where=copy_mask, casting=casting)
                result_leaves.append(destination)
            coupled_results = tuple(value for value in result_leaves if isinstance(value, StochasticScalar))
            if coupled_results:
                couple_stochastic_results(coupled_results, (self, *stochastic_inputs, *mask_source), n_sims=self.n_sims)
            result = tuple(result_leaves) if len(result_leaves) > 1 else result_leaves[0]
            return result

        if method == "__call__":
            return self._wrap_result_with_coupling(calculated, (self, *stochastic_inputs, *mask_source))
        if method == "outer":
            return calculated
        if method == "reduce":
            # Even keepdims=True does not retain the scenario dimension.
            return scalar_or_array(calculated)
        if method == "reduceat":
            # Its output positions denote reduction segments, not simulations.
            return StochasticScalar(calculated) if getattr(calculated, "ndim", None) == 1 else calculated
        # accumulate depends on sequence order; it cannot be re-ordered with inputs.
        return StochasticScalar(calculated) if getattr(calculated, "ndim", None) == 1 else calculated

    def __array_function__(self, func: t.Callable[..., t.Any], _: t.Any, args: t.Any, kwargs: t.Any) -> t.Any:
        """Apply NumPy functions with conservative scenario-alignment semantics.

        Only elementwise functions keep coupling. Sorting, rolling, cumulative
        functions and selections of a different shape create detached results,
        even if the final array happens to have `n_sims` entries.
        """
        if func in (np.copyto, np.put, np.putmask, np.place):
            raise NotImplementedError("Use a controlled stochastic operation instead of raw NumPy mutation")

        kwargs = dict(kwargs)
        output = kwargs.pop("out", None)
        preserving = func in (np.where, np.clip, np.round, np.around, np.nan_to_num, np.isclose)
        # The one-argument form of where returns positions, not scenario values.
        if func is np.where and len(args) != 3:
            preserving = False
        if isinstance(output, StochasticScalar) and not preserving:
            raise ValueError("A stochastic out= destination requires a scenario-preserving function")

        sources = (self, *(arg for arg in (*args, *kwargs.values()) if isinstance(arg, StochasticScalar)))
        targets = (output,) if isinstance(output, StochasticScalar) else ()
        validate_stochastic_axes((*sources, *targets), n_sims=self.n_sims)
        is_host = host_values(self.values) or any(
            host_values(value.values if isinstance(value, StochasticScalar) else value)
            for value in (*args, *kwargs.values())
        )

        def unwrapped(value: t.Any) -> t.Any:
            if isinstance(value, StochasticScalar):
                return asnumpy(value.values) if is_host else value.values
            if is_host and isinstance(value, xp.ndarray):
                return asnumpy(value)
            return value if is_host else to_backend(value)

        parsed_args = tuple(unwrapped(value) for value in args)
        parsed_kwargs = {key: unwrapped(value) for key, value in kwargs.items()}
        if func is np.where:
            parsed_args = ((np if is_host else xp).asarray(parsed_args[0]), *parsed_args[1:])
        calculated = func(*parsed_args, **parsed_kwargs)

        if output is not None:
            destination = output.values if isinstance(output, StochasticScalar) else output
            if not isinstance(destination, (np.ndarray, xp.ndarray)):
                raise TypeError("out= must be a NumPy/CuPy array or StochasticScalar")
            if destination.shape != np.shape(calculated):
                raise ValueError("out= has an incompatible result shape")
            if not np.can_cast(calculated.dtype, destination.dtype, casting="same_kind"):
                raise TypeError("out= cannot represent the result")
            destination_backend = np if isinstance(destination, np.ndarray) else xp
            values = asnumpy(calculated) if destination_backend is np else xp.asarray(calculated)
            destination_backend.copyto(destination, values, casting="same_kind")
            if isinstance(output, StochasticScalar):
                couple_stochastic_results((output,), sources, n_sims=self.n_sims)
            return output

        scalar_result = scalar_or_array(calculated)
        if scalar_result is not calculated:
            return scalar_result
        if getattr(calculated, "ndim", None) != 1:
            return calculated
        if preserving:
            if len(calculated) != self.n_sims:
                raise ValueError("Stochastic result has incompatible n_sims")
            return self._wrap_result_with_coupling(calculated, sources)
        return StochasticScalar(calculated)

    def __array__(self, dtype: t.Any = None, copy: bool | None = None) -> npt.NDArray[t.Any]:
        """Convert the StochasticScalar to a numpy array.

        Args:
            dtype: The desired data type of the output array.
            copy: Whether NumPy requires a copy of the host array.

        Returns:
            A numpy array representation of the StochasticScalar values.
        """
        result = asnumpy(self.values, dtype=dtype)
        if copy is True:
            return result.copy()
        return result

    def __getitem__(self, index: ScipyNumeric | StochasticScalar) -> StochasticScalar:
        # FIXME: Type signature inconsistent with SequenceLike protocol and runtime
        # - SequenceLike expects __getitem__(int) -> T_co (should return float)
        # - Runtime: int indexing returns scalar, StochasticScalar returns
        #   StochasticScalar.
        # - Current signature claims all indexing returns StochasticScalar (wrong)
        # Need overloads to match runtime behavior and protocol expectations
        # See: https://github.com/ProteusLLP/proteusllp-actuarial-library/issues/24
        # handle an actual numeric index...
        if isinstance(index, ScipyNumeric):
            return self.values[int(index)]  # type: ignore[return-value]

        if isinstance(index, type(self)):
            # Check if index contains boolean values for masking
            if xp.issubdtype(index.values.dtype, xp.bool_):
                # Use boolean indexing directly - no conversion needed
                # Type ignore: Runtime type checking ensures boolean indexing is valid
                indices = asnumpy(index.values) if host_values(self.values) else index.values
                result = type(self)(self.values[indices])  # type: ignore[arg-type]
            else:
                # Convert numeric indices to integers for positional indexing
                indices = index.values.astype(int)
                if host_values(self.values):
                    indices = asnumpy(indices)
                result = type(self)(self.values[indices])

            if result.n_sims == index.n_sims:
                couple_stochastic_results((result,), (index,))
            return result

        raise TypeError(
            f"Unexpected type {type(index).__name__}. Index must be an integer, float, or StochasticScalar."
        )

    def __len__(self) -> int:
        """Return the number of simulations."""
        return len(self.values)

    def __iter__(self) -> t.Iterator[t.Any]:
        """Iterate over the values."""
        return iter(self.values)

    # ===================
    # PUBLIC PROPERTIES
    # ===================

    @property
    def ranks(self) -> StochasticScalar:
        """Return the ranks of the variable."""
        result = xp.empty(self.n_sims, dtype=int)
        order = np.argsort(self.values) if host_values(self.values) else xp.argsort(self.values)
        result[order] = xp.arange(self.n_sims)
        return StochasticScalar(result)

    # ===================
    # PUBLIC METHODS
    # ===================

    def tolist(self) -> list[t.Any]:
        """Convert the values to a Python list."""
        return self.values.tolist()

    def mean(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: t.Any = None,
        out: t.Any = None,
        keepdims: bool = False,
    ) -> t.Any:
        """Return the mean across the simulations."""
        result = xp.mean(self.values, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
        return self._wrap_reduction_result(result)

    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: t.Any = None,
        out: t.Any = None,
        keepdims: bool = False,
    ) -> t.Any:
        """Return the sum across the simulations."""
        result = xp.sum(self.values, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
        return self._wrap_reduction_result(result)

    def std(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: t.Any = None,
        out: t.Any = None,
        ddof: int = 0,
        keepdims: bool = False,
    ) -> t.Any:
        """Return the standard deviation across the simulations."""
        result = xp.std(self.values, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)
        return self._wrap_reduction_result(result)

    def var(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: t.Any = None,
        out: t.Any = None,
        ddof: int = 0,
        keepdims: bool = False,
    ) -> t.Any:
        """Return the variance across the simulations."""
        result = xp.var(self.values, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)
        return self._wrap_reduction_result(result)

    def percentile(self, p: NumberOrList) -> NumberOrList:
        """Return the percentile of the variable across the simulation dimension.

        Args:
            p: The percentile level (between 0 and 100).

        Returns:
            The percentile value.

        """
        if isinstance(p, list):
            return t.cast(list[Numeric], xp.percentile(self.values, p).tolist())  # type: ignore
        return float(xp.percentile(self.values, p))  # type: ignore

    def tvar(self, p: NumberOrList) -> NumberOrList:
        """Calculate the Tail Value at Risk (TVaR) at a given percentile.

        Args:
            p: The percentile level (between 0 and 100) to calculate TVaR.

        Returns:
            The TVaR value as a float.
        """
        return stats.tvar(self.values, p)

    def upsample(self, n_sims: int) -> Self:
        """Increase the number of simulations in the variable."""
        if n_sims == self.n_sims:
            return self
        backend = np if host_values(self.values) else xp
        return type(self)(self.values[backend.arange(n_sims) % self.n_sims])

    def histogram_plot(self, title: str | None = None) -> go.Figure:
        """Return a Plotly histogram of the simulated values.

        Args:
            title: Optional title for the histogram.

        Returns:
            A Plotly figure. Call ``.show()`` to display it, or use any of
            Plotly's figure export methods to save it.
        """
        return go.Figure(go.Histogram(x=self), layout={"title": title})

    def show_histogram(self, title: str | None = None) -> go.Figure:
        """Show and return a histogram of the simulated values.

        This method is retained for backwards compatibility. New code can use
        :meth:`histogram_plot` and explicitly call ``.show()`` when required.

        Args:
            title: Optional title for the histogram.

        Returns:
            The Plotly figure that was displayed.
        """
        fig = self.histogram_plot(title=title)
        if os.getenv("PAL_SUPPRESS_PLOTS", "").lower() != "true":
            # Type ignore: plotly-stubs has incomplete type information
            fig.show()  # type: ignore[misc]
        return fig

    def cdf_plot(self, title: str | None = None) -> go.Figure:
        """Return a Plotly empirical cumulative distribution function plot.

        Args:
            title: Optional title for the CDF plot.

        Returns:
            A Plotly figure. Call ``.show()`` to display it, or use any of
            Plotly's figure export methods to save it.
        """
        sorted_values = StochasticScalar(xp.sort(self.values))
        cumulative_probabilities = StochasticScalar(xp.arange(self.n_sims) / self.n_sims)
        fig = go.Figure(
            go.Scatter(
                x=sorted_values,
                y=cumulative_probabilities,
            ),
            layout={"title": title},
        )
        # Type ignore: plotly-stubs has incomplete type information
        fig.update_xaxes({"title": "Value"})  # type: ignore[misc]
        fig.update_yaxes({"title": "Cumulative Probability"})  # type: ignore[misc]
        return fig

    def show_cdf(self, title: str | None = None) -> go.Figure:
        """Show and return the empirical cumulative distribution function.

        This method is retained for backwards compatibility. New code can use
        :meth:`cdf_plot` and explicitly call ``.show()`` when required.

        Args:
            title: Optional title for the CDF plot.

        Returns:
            The Plotly figure that was displayed.
        """
        fig = self.cdf_plot(title=title)
        if os.getenv("PAL_SUPPRESS_PLOTS", "").lower() != "true":
            # Type ignore: plotly-stubs has incomplete type information
            fig.show()  # type: ignore[misc]
        return fig

    # ===================
    # PRIVATE METHODS
    # ===================

    def _reorder_sims(self, new_order: t.Sequence[int]) -> None:
        """Reorder the simulations in the variable."""
        order = asnumpy(new_order) if host_values(self.values) else new_order
        self.values = self.values[order]

    def _wrap_reduction_result(self, result: t.Any) -> t.Any:
        """Return scalar reductions directly and preserve one-dimensional results."""
        scalar_result = scalar_or_array(result)
        if scalar_result is not result:
            return scalar_result
        if getattr(result, "ndim", None) == 1:
            wrapped = StochasticScalar(result)
            if wrapped.n_sims == self.n_sims:
                couple_stochastic_results((wrapped,), (self,))
            return wrapped
        return result

    def _wrap_result_with_coupling(self, result_array: t.Any, inputs: tuple[t.Any, ...]) -> t.Any:
        """Wrap all scenario-valued result leaves, then link them atomically.

        A multi-output ufunc must not couple its first output if a later
        output has an incompatible simulation axis.
        """

        def prepare(value: t.Any) -> t.Any:
            scalar_result = scalar_or_array(value)
            if scalar_result is not value:
                return scalar_result
            if getattr(value, "ndim", None) != 1:
                return value
            if len(value) != self.n_sims:
                raise ValueError(
                    f"Stochastic result has incompatible n_sims (expected {self.n_sims}, got {len(value)})"
                )
            return StochasticScalar(value)

        if isinstance(result_array, tuple):
            prepared = tuple(prepare(value) for value in result_array)
        else:
            prepared = prepare(result_array)
        leaves = prepared if isinstance(prepared, tuple) else (prepared,)
        result_leaves = tuple(value for value in leaves if isinstance(value, StochasticScalar))
        if result_leaves:
            stochastic_inputs = (self, *(value for value in inputs if isinstance(value, ProteusStochasticVariable)))
            couple_stochastic_results(result_leaves, stochastic_inputs, n_sims=self.n_sims)
        return prepared

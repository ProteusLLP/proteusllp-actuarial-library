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
from ._maths import asnumpy, generate_upsample_indices, scalar_or_array, to_backend, xp
from .config import config
from .couplings import CouplingGroup, ProteusStochasticVariable
from .stats import NumberOrList
from .types import Numeric, NumericLike, ScipyNumeric


class StochasticScalar(ProteusStochasticVariable):
    """A class to represent a single scalar variable in a simulation."""

    coupled_variable_group: CouplingGroup
    n_sims: int
    """The number of simulations in the variable."""

    # ===================
    # DUNDER METHODS
    # ===================

    def __init__(self, values: npt.ArrayLike):
        """Initialize a stochastic scalar.

        Args:
            values: An array of values that describe the distribution for the scalar
                variable.
        """
        super().__init__()

        if isinstance(values, StochasticScalar):
            self.values = values.values
            self.n_sims = values.n_sims
            self.coupled_variable_group.merge(values.coupled_variable_group)
            return

        if isinstance(values, list):
            # Type ignore: Generic list type inference limitation
            self.values = xp.array(values)  # type: ignore[misc]
            self.n_sims = len(values)  # type: ignore[misc]
            return

        if isinstance(values, xp.ndarray):
            if values.ndim == 1:
                self.values = values
                # Type ignore: Generic array type inference limitation
                self.n_sims = len(values)  # type: ignore[misc]
                return
            raise ValueError("Values must be a 1D array.")

        if isinstance(values, np.ndarray):
            if values.ndim == 1:
                self.values = xp.asarray(
                    values,
                    dtype=values.dtype,  # type: ignore
                )
                # Type ignore: Generic array type inference limitation
                self.n_sims = len(values)  # type: ignore[misc]
                return
            raise ValueError("Values must be a 1D array.")

        # Type ignore: Generic ArrayLike type inference limitation
        raise TypeError("Type of values must be a sequence or array. Found " + type(values).__name__)  # type: ignore[misc]

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
    ) -> StochasticScalar:
        """Override the __array_ufunc__ method to apply standard numpy functions.

        If there's a mix of different variable types in the inputs, delegate to the
        more specialized variable type to handle the operation. Otherwise, extract
        values from StochasticScalar objects and apply the ufunc directly.

        Returns:
            When delegating to another object's __array_ufunc__, the return type depends
            on that object's implementation. When handling the operation directly,
            returns a new StochasticScalar.
        """
        # check if the input types to the function are types of ProteusVariables
        # other than StochasticScalar
        var_not_stochastic_scalar = [
            type(x).__name__ == "ProteusVariable"
            or isinstance(x, ProteusStochasticVariable)
            and not isinstance(x, StochasticScalar)
            for x in inputs
        ]

        if any(var_not_stochastic_scalar):
            # call the __array_ufunc__ method of variable which is not StochasticScalar
            #
            var_pos = var_not_stochastic_scalar.index(True)
            return inputs[var_pos].__array_ufunc__(ufunc, method, *inputs, **kwargs)
        _inputs = tuple(x.values if isinstance(x, StochasticScalar) else to_backend(x) for x in inputs)
        out = kwargs.get("out", ())
        if out:
            kwargs["out"] = tuple(x.values if isinstance(x, StochasticScalar) else x for x in out)

        # Handle reduction operations - return scalars directly
        if method == "reduce":
            result = getattr(ufunc, method)(*_inputs, **kwargs)

            # Check if result should be wrapped (keepdims=True or axis specified)
            keepdims = kwargs.get("keepdims", False)
            axis = kwargs.get("axis", None)

            if keepdims or (axis is not None and hasattr(result, "shape") and result.shape):
                return self._wrap_result_with_coupling(result, inputs)

            # Standard reduction returns scalar directly
            return scalar_or_array(result)

        # Handle reduceat/accumulate operations - return wrapped arrays
        if method in ("reduceat", "accumulate"):
            result = getattr(ufunc, method)(*_inputs, **kwargs)
            return self._wrap_result_with_coupling(result, inputs)

        # Handle regular element-wise operations
        result = getattr(ufunc, method)(*_inputs, **kwargs)
        return self._wrap_result_with_coupling(result, inputs)

    def __array_function__(
        self, func: t.Callable[..., t.Any], _: t.Any, args: t.Any, kwargs: t.Any
    ) -> np.number[t.Any] | StochasticScalar:
        """Handle numpy array functions for StochasticScalar objects.

        Args:
            func: The numpy function being called
            types: Types involved in the operation
            args: Arguments passed to the function
            kwargs: Keyword arguments passed to the function

        Returns:
            Either a scalar result or new StochasticScalar object

        Raises:
            NotImplementedError: If the function is not supported
        """
        # Extract PAL values and move any NumPy operands to the active backend.
        processed_args = tuple(x.values if isinstance(x, StochasticScalar) else to_backend(x) for x in args)
        if func is np.where:
            processed_args = (xp.asarray(processed_args[0]), *processed_args[1:])
        result = func(*processed_args, **kwargs)

        scalar_result = scalar_or_array(result)
        if scalar_result is not result:
            return scalar_result  # type: ignore[return-value]
        if isinstance(result, (np.number, np.bool_, bool)) or np.isscalar(result):
            return result  # type: ignore[return-value]

        # A StochasticScalar is one-dimensional. Functions such as stack return
        # higher-dimensional backend arrays and should remain arrays.
        if getattr(result, "ndim", None) != 1:
            return result
        return self._wrap_result_with_coupling(result, args)

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
                result = type(self)(self.values[index.values])  # type: ignore[arg-type]
            else:
                # Convert numeric indices to integers for positional indexing
                indices = index.values.astype(int)
                result = type(self)(self.values[indices])

            result.coupled_variable_group.merge(index.coupled_variable_group)
            return result

        raise TypeError(
            f"Unexpected type {type(index).__name__}. Index must be an integer, float, or StochasticScalar."
        )

    def __len__(self) -> int:
        """Return the number of simulations."""
        return len(self.values)

    def __iter__(self) -> t.Iterator[NumericLike]:
        """Iterate over the values."""
        return iter(self.values)

    # ===================
    # PUBLIC PROPERTIES
    # ===================

    @property
    def ranks(self) -> StochasticScalar:
        """Return the ranks of the variable."""
        result = xp.empty(self.n_sims, dtype=int)
        result[xp.argsort(self.values)] = xp.arange(self.n_sims)
        return StochasticScalar(result)

    # ===================
    # PUBLIC METHODS
    # ===================

    def tolist(self) -> list[Numeric]:
        """Convert the values to a Python list."""
        return t.cast(list[Numeric], self.values.tolist())

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

    def upsample(
        self,
        n_sims: int,
        rng: np.random.Generator | None = None,
        method: str = "random",
    ) -> Self:
        """Increase or decrease the number of simulations in the variable.

        Args:
            n_sims: Target number of simulations.
            rng: Random number generator. Uses config.rng if None
                (only used with method="random").
            method: Upsampling method to use:
                - "random" (default): Random resampling that preserves coupling groups
                  and independence between different coupling groups. First chunk is
                  ordered, remaining chunks are random permutations.
                - "cyclic": Deterministic cycling through existing simulations. Faster
                  and deterministic. Creates a new instance without preserving coupling
                  groups. When used across multiple variables, induces synchronized
                  resampling (all variables cycle together).

        Returns:
            New StochasticScalar with target number of simulations.
        """
        if n_sims == self.n_sims:
            return self

        if method == "cyclic":
            from ._maths import generate_cyclic_indices

            indices = generate_cyclic_indices(n_sims, self.n_sims)
            # Create new instance without preserving coupling
            return type(self)(self.values[indices])
        elif method == "random":
            if rng is None:
                rng = config.rng
            indices = generate_upsample_indices(n_sims, self.n_sims, rng=rng)
            # Use __getitem__ to preserve coupling
            result = self[type(self)(indices)]
            return t.cast(Self, result)
        else:
            raise ValueError(
                f"Invalid method '{method}'. Must be 'random' or 'cyclic'."
            )

    def show_histogram(self, title: str | None = None) -> None:
        """Show a histogram of the variable.

        Args:
            title (optional): Title of the histogram plot. Defaults to None.
        """
        if os.getenv("PAL_SUPPRESS_PLOTS", "").lower() == "true":
            return
        fig = go.Figure(go.Histogram(x=self.values), layout={"title": title})
        # Type ignore: plotly-stubs has incomplete type information
        fig.show()  # type: ignore[misc]

    def show_cdf(self, title: str | None = None) -> None:
        """Show a plot of the cumulative distribution function (cdf) of the variable.

        Args:
            title (optional): Title of the cdf plot. Defaults to None.
        """
        if os.getenv("PAL_SUPPRESS_PLOTS", "").lower() == "true":
            return

        fig = go.Figure(
            go.Scatter(
                x=xp.sort(self.values).tolist(),
                y=(xp.arange(self.n_sims) / self.n_sims).tolist(),
            ),
            layout={"title": title},
        )
        # Type ignore: plotly-stubs has incomplete type information
        fig.update_xaxes({"title": "Value"})  # type: ignore[misc]
        fig.update_yaxes({"title": "Cumulative Probability"})  # type: ignore[misc]
        fig.show()  # type: ignore[misc]

    # ===================
    # PRIVATE METHODS
    # ===================

    def _reorder_sims(self, new_order: t.Sequence[int]) -> None:
        """Reorder the simulations in the variable."""
        self.values = self.values[new_order]

    def _wrap_reduction_result(self, result: t.Any) -> t.Any:
        """Return scalar reductions directly and preserve one-dimensional results."""
        scalar_result = scalar_or_array(result)
        if scalar_result is not result:
            return scalar_result
        if getattr(result, "ndim", None) == 1:
            wrapped = StochasticScalar(result)
            wrapped.coupled_variable_group.merge(self.coupled_variable_group)
            return wrapped
        return result

    def _wrap_result_with_coupling(self, result_array: t.Any, inputs: tuple[t.Any, ...]) -> t.Any:
        """Wrap result in StochasticScalar and merge coupling groups.

        Args:
            result_array: The numpy array result to wrap.
            inputs: The input arguments from __array_ufunc__.

        Returns:
            A new StochasticScalar with proper coupling group merging.
        """
        if isinstance(result_array, tuple):
            return tuple(self._wrap_result_with_coupling(item, inputs) for item in result_array)
        scalar_result = scalar_or_array(result_array)
        if scalar_result is not result_array:
            return scalar_result
        if getattr(result_array, "ndim", None) != 1:
            return result_array

        wrapped_result = StochasticScalar(result_array)
        for input in inputs:
            if isinstance(input, ProteusStochasticVariable):
                input.coupled_variable_group.merge(self.coupled_variable_group)
        wrapped_result.coupled_variable_group.merge(self.coupled_variable_group)
        return wrapped_result

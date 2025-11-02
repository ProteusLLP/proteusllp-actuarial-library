"""Stochastic scalar variables for Monte Carlo simulation.

Provides the StochasticScalar class for representing and manipulating
scalar-valued stochastic variables in actuarial and risk modeling applications.
Supports arithmetic operations, statistical functions, and numpy integration.
"""

from __future__ import annotations

import os
import typing as t

import numpy.typing as npt
import plotly.graph_objects as go  # type: ignore

from pal import stats  # type: ignore

from ._maths import xp as np
from .couplings import CouplingGroup, ProteusStochasticVariable
from .types import Numeric, NumericLike, ScipyNumeric


class StochasticScalar(ProteusStochasticVariable):
    """A class to represent a single scalar variable in a simulation."""

    coupled_variable_group: CouplingGroup

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
            self.values = np.array(values)  # type: ignore[misc]
            self.n_sims = len(values)  # type: ignore[misc]
            return

        if isinstance(values, np.ndarray):
            if values.ndim == 1:
                self.values = values
                # Type ignore: Generic array type inference limitation
                self.n_sims = len(values)  # type: ignore[misc]
                return
            raise ValueError("Values must be a 1D array.")

        # Type ignore: Generic ArrayLike type inference limitation
        raise TypeError(
            "Type of values must be a sequence or array. Found " + type(values).__name__
        )  # type: ignore[misc]

    def __hash__(self) -> int:
        # FIXME: this hash function is not robust - defining a hash implies that this
        # object is immutable, but it is not. The hash implies that two objects of this
        # class with the same values are equal, but this is not the case if they are
        # coupled to different variable groups.
        return id(self)

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
        _inputs = tuple(
            (
                x.values
                if isinstance(x, StochasticScalar)
                else x  # promote an input ndarray to match the simulation index
            )
            for x in inputs
        )
        out = kwargs.get("out", ())
        if out:
            kwargs["out"] = tuple(x.values for x in out)

        # Handle reduction operations - return scalars directly
        if method == "reduce":
            result = getattr(ufunc, method)(*_inputs, **kwargs)

            # Check if result should be wrapped (keepdims=True or axis specified)
            keepdims = kwargs.get("keepdims", False)
            axis = kwargs.get("axis", None)

            if keepdims or (
                axis is not None and hasattr(result, "shape") and result.shape
            ):
                return self._wrap_result_with_coupling(result, inputs)

            # Standard reduction returns scalar directly
            return result

        # Handle reduceat/accumulate operations - return wrapped arrays
        if method in ("reduceat", "accumulate"):
            result = getattr(ufunc, method)(*_inputs, **kwargs)
            return self._wrap_result_with_coupling(result, inputs)

        # Handle regular element-wise operations
        result = getattr(ufunc, method)(*_inputs, **kwargs)
        return self._wrap_result_with_coupling(result, inputs)

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
            if np.issubdtype(index.values.dtype, np.bool_):
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
            f"Unexpected type {type(index).__name__}. Index must be an integer, "
            "float, or StochasticScalar."
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
        if self.n_sims is None:
            raise ValueError("Cannot compute ranks for an uninitialized variable.")
        result = np.empty(self.n_sims, dtype=int)
        result[np.argsort(self.values)] = np.arange(self.n_sims)
        return StochasticScalar(result)

    # ===================
    # PUBLIC METHODS
    # ===================

    def tolist(self) -> list[Numeric]:
        """Convert the values to a Python list."""
        return t.cast(list[Numeric], self.values.tolist())

    def tvar(self, percentile: stats.NumberOrList) -> stats.NumberOrList:
        """Calculate the Tail Value at Risk (TVaR) at a given percentile.

        Args:
            percentile: The percentile level (between 0 and 100) to calculate TVaR.

        Returns:
            The TVaR value as a float.
        """
        return stats.tvar(self.values, percentile)

    def upsample(self, n_sims: int) -> t.Self:
        """Increase the number of simulations in the variable."""
        if self.n_sims is None:
            raise ValueError("Cannot upsample an uninitialized variable.")
        if n_sims == self.n_sims:
            return self
        return type(self)(self.values[np.arange(n_sims) % self.n_sims])

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

        if self.n_sims is None:
            raise ValueError("Cannot compute CDF for an uninitialized variable.")

        fig = go.Figure(
            go.Scatter(x=np.sort(self.values), y=np.arange(self.n_sims) / self.n_sims),
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

    def _wrap_result_with_coupling(
        self, result_array: t.Any, inputs: tuple[t.Any, ...]
    ) -> StochasticScalar:
        """Wrap result in StochasticScalar and merge coupling groups.

        Args:
            result_array: The numpy array result to wrap.
            inputs: The input arguments from __array_ufunc__.

        Returns:
            A new StochasticScalar with proper coupling group merging.
        """
        wrapped_result = StochasticScalar(result_array)
        for input in inputs:
            if isinstance(input, ProteusStochasticVariable):
                input.coupled_variable_group.merge(self.coupled_variable_group)
        wrapped_result.coupled_variable_group.merge(self.coupled_variable_group)
        return wrapped_result

from __future__ import annotations

import math
import os
import typing as t

import numpy.typing as npt
import plotly.graph_objects as go  # type: ignore

from ._maths import xp as np
from .couplings import CouplingGroup, ProteusStochasticVariable
from .types import Numeric, NumericLike, ScipyNumeric

NumberOrList = Numeric | list[Numeric]
NumericOrStochasticScalar = t.TypeVar(
    "NumericOrStochasticScalar", Numeric, "StochasticScalar"
)


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
    ) -> t.Any:
        """Override the __array_ufunc__ method to apply standard numpy functions.

        If there's a mix of different variable types in the inputs, delegate to the
        more specialized variable type to handle the operation. Otherwise, extract
        values from StochasticScalar objects and apply the ufunc directly.

        Returns:
            When delegating to another object's __array_ufunc__, the return type depends
            on that object's implementation. When handling the operation directly,
            returns a new StochasticScalar.
        """
        # check if the input types to the function are types of ProteusVariables other than StochasticScalar
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
        result = StochasticScalar(getattr(ufunc, method)(*_inputs, **kwargs))
        for input in inputs:
            if isinstance(input, ProteusStochasticVariable):
                input.coupled_variable_group.merge(self.coupled_variable_group)
        result.coupled_variable_group.merge(self.coupled_variable_group)

        return result

    def __getitem__(self, index: int | float | StochasticScalar) -> NumericLike:
        # handle an actual numeric index...
        if isinstance(index, int | float):
            return t.cast(NumericLike, self.values[int(index)])

        if isinstance(index, type(self)):
            # Convert floating point indices to integers for array indexing
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

    def ssum(self) -> ScipyNumeric:
        """Sum the values of the variable across the simulation dimension."""
        return t.cast(ScipyNumeric, np.sum(self.values))

    def mean(self) -> ScipyNumeric:
        """Return the mean of the variable across the simulation dimension."""
        return t.cast(ScipyNumeric, np.mean(self.values))

    def skew(self) -> ScipyNumeric:
        """Return the coefficient of skewness of the variable across the simulation dimension."""
        return t.cast(
            ScipyNumeric, np.mean((self.values - self.mean()) ** 3) / self.std() ** 3
        )

    def kurt(self) -> ScipyNumeric:
        """Return the kurtosis of the variable across the simulation dimension."""
        return t.cast(
            ScipyNumeric, np.mean((self.values - self.mean()) ** 4) / self.std() ** 4
        )

    def std(self) -> ScipyNumeric:
        """Return the standard deviation of the variable across the simulation dimension."""
        return t.cast(ScipyNumeric, np.std(self.values))

    def percentile(self, p: ScipyNumeric) -> ScipyNumeric:
        """Return the percentile of the variable across the simulation dimension."""
        return t.cast(ScipyNumeric, np.percentile(self.values, p))

    def tvar(self, p: NumberOrList) -> NumberOrList:
        """Return the tail value at risk (TVAR) of the variable."""
        if self.n_sims is None:
            raise ValueError("Cannot compute TVAR for an uninitialized variable.")

        # get the rank of the variable
        rank_positions = np.argsort(self.values)
        if isinstance(p, list):
            # Type ignore: Generic list type inference limitation
            result = []  # type: ignore[misc]
            for perc in p:
                result.append(  # type: ignore[misc]
                    self.values[
                        rank_positions[math.ceil(perc / 100 * self.n_sims) :]
                    ].mean()
                )
            return result  # type: ignore[misc]
        idx = math.ceil(p / 100 * self.n_sims)
        result = self.values[rank_positions[idx:]].mean()
        return t.cast(NumberOrList, result)

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

from __future__ import annotations

import math
import os
import typing as t

import numpy.typing as npt
import plotly.graph_objects as go  # type: ignore

from ._maths import xp as np
from .couplings import CouplingGroup, ProteusStochasticVariable
from .types import Numeric, NumericLike, SequenceLike

NumberOrList = Numeric | list[Numeric]


class StochasticScalar(ProteusStochasticVariable):
    """A class to represent a single scalar variable in a simulation."""

    coupled_variable_group: CouplingGroup

    def __init__(self, values: SequenceLike[float]) -> None:
        """Initialize a stochastic scalar.

        Args:
            values: An array of values that describe the distribution for the scalar
                variable.
        """
        super().__init__()

        if not isinstance(values, SequenceLike):
            raise TypeError(f"Values must be a sequence object. Got {type(values)}")

        if isinstance(values, StochasticScalar):
            self.values = values.values
            self.n_sims = values.n_sims
            self.coupled_variable_group.merge(values.coupled_variable_group)
        else:
            if isinstance(values, list):
                self.values = np.array(values)
                self.n_sims = len(values)
            elif isinstance(values, np.ndarray):
                if values.ndim == 1:
                    self.values = values
                    self.n_sims = len(values)
                else:
                    raise ValueError("Values must be a 1D array.")
            else:
                raise ValueError(
                    "Values must be a list or numpy array. Found " + str(type(values))
                )

    def __hash__(self) -> int:
        # FIXME: this hash function is not robust - defining a hash implies that this
        # object is immutable, but it is not. The hash implies that two objects of this
        # class with the same values are equal, but this is not the case if they are
        # coupled to different variable groups.
        return id(self)

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
        # Check for inputs that have __array_ufunc__ capability but are not
        # StochasticScalar so we can delegate to them if necessary.
        delegate_to: t.Any = None
        for x in inputs:
            if hasattr(x, "__array_ufunc__") and not isinstance(x, StochasticScalar):
                delegate_to = x
                break

        if delegate_to is not None:
            # Find the first specialized variable and let it handle the operation
            return delegate_to.__array_ufunc__(ufunc, method, *inputs, **kwargs)
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

    @property
    def ranks(self) -> StochasticScalar:
        """Return the ranks of the variable."""
        if self.n_sims is None:
            raise ValueError("Cannot compute ranks for an uninitialized variable.")
        result = np.empty(self.n_sims, dtype=int)
        result[np.argsort(self.values)] = np.arange(self.n_sims)
        return StochasticScalar(result)

    def tolist(self) -> list[Numeric]:
        """Convert the values to a Python list."""
        return t.cast(list[Numeric], self.values.tolist())

    def ssum(self) -> Numeric:
        """Sum the values of the variable across the simulation dimension."""
        return np.sum(self.values)

    def mean(self) -> Numeric:
        """Return the mean of the variable across the simulation dimension."""
        return np.mean(self.values)

    def skew(self) -> Numeric:
        """Return the coefficient of skewness of the variable across the simulation dimension."""
        return np.mean((self.values - self.mean()) ** 3) / self.std() ** 3

    def kurt(self) -> Numeric:
        """Return the kurtosis of the variable across the simulation dimension."""
        return np.mean((self.values - self.mean()) ** 4) / self.std() ** 4

    def std(self) -> Numeric:
        """Return the standard deviation of the variable across the simulation dimension."""
        return np.std(self.values)

    def percentile(self, p: Numeric) -> Numeric:
        """Return the percentile of the variable across the simulation dimension."""
        return np.percentile(self.values, p)

    def tvar(self, p: NumberOrList) -> NumberOrList:
        """Return the tail value at risk (TVAR) of the variable."""
        if self.n_sims is None:
            raise ValueError("Cannot compute TVAR for an uninitialized variable.")

        # get the rank of the variable
        rank_positions = np.argsort(self.values)
        if isinstance(p, list):
            result = []
            for perc in p:
                result.append(
                    self.values[
                        rank_positions[math.ceil(perc / 100 * self.n_sims) :]
                    ].mean()
                )
            return result
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

    def __repr__(self) -> str:
        return f"{type(self).__name__}(values={self.values}\nn_sims={self.n_sims})"

    # implement the index referencing
    def __getitem__(self, index: Numeric | t.Self) -> Numeric | StochasticScalar:
        # handle an actual numeric index...
        if isinstance(index, int | float):
            return t.cast(Numeric, self.values[int(index)])

        if isinstance(index, type(self)):
            result = type(self)(self.values[index.values])
            result.coupled_variable_group.merge(index.coupled_variable_group)
            return result

        raise TypeError(
            f"Unexpected type {type(index).__name__}. Index must be an integer, "
            "StochasticScalar or numpy array."
        )

    def __len__(self) -> int:
        """Return the number of simulations."""
        return len(self.values)

    def __iter__(self) -> t.Iterator[NumericLike]:
        """Iterate over the values."""
        return iter(self.values)

    def show_histogram(self, title: str | None = None) -> None:
        """Show a histogram of the variable.

        Args:
            title (optional): Title of the histogram plot. Defaults to None.
        """
        if os.getenv("PAL_SUPPRESS_PLOTS", "").lower() == "true":
            return
        fig = go.Figure(go.Histogram(x=self.values), layout={"title": title})
        fig.show()

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
        fig.update_xaxes({"title": "Value"})
        fig.update_yaxes({"title": "Cumulative Probability"})
        fig.show()

    def _reorder_sims(self, new_order: npt.NDArray[np.int64]) -> None:
        """Reorder the simulations in the variable."""
        self.values = self.values[new_order]

from __future__ import annotations

import math
from typing import TypeVar, Union

import plotly.graph_objects as go  # type: ignore
from numpy.typing import ArrayLike

from ._maths import xp as np
from .couplings import CouplingGroup, ProteusStochasticVariable

Numeric = Union[int, float]
NumberOrList = TypeVar("NumberOrList", Numeric, list[Numeric])
NumericOrStochasticScalar = TypeVar(
    "NumericOrStochasticScalar", Numeric, "StochasticScalar"
)
NumberOrList = Numeric | list[Numeric]


class StochasticScalar(ProteusStochasticVariable):
    """A class to represent a single scalar variable in a simulation."""

    coupled_variable_group: CouplingGroup

    def __init__(self, values: ArrayLike):
        """Initialize a stochastic scalar.

        Args:
            values: An array of values that describe the distribution for the scalar
                variable.
        """
        super().__init__()
        assert hasattr(values, "__getitem__"), "Values must be an array-like object."
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

    def __repr__(self):
        return f"StochasticScalar(values={self.values}\nn_sims={self.n_sims})"

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs, **kwargs
    ) -> StochasticScalar:
        """Override the __array_ufunc__ method means that you can apply standard numpy functions"""
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

    def __getitem__(
        self, index: NumericOrStochasticScalar
    ) -> NumericOrStochasticScalar:
        if isinstance(index, (int, float)):
            return self.values[int(index)]
        elif isinstance(index, StochasticScalar):
            result = StochasticScalar(self.values[index.values])
            result.coupled_variable_group.merge(index.coupled_variable_group)
            return result
        raise ValueError("Index must be an integer, StochasticScalar or numpy array.")

    @property
    def ranks(self) -> StochasticScalar:
        """Return the ranks of the variable."""
        result = np.empty(self.n_sims, dtype=int)
        result[np.argsort(self.values)] = np.arange(self.n_sims)
        return StochasticScalar(result)

    def tolist(self):
        return self.values.tolist()

    def ssum(self) -> float:
        """Sum the values of the variable across the simulation dimension."""
        return np.sum(self.values)

    def mean(self) -> float:
        """Return the mean of the variable across the simulation dimension."""
        return np.mean(self.values)

    def skew(self) -> float:
        """Return the coefficient of skewness of the variable across the simulation dimension."""
        return float(np.mean((self.values - self.mean()) ** 3) / self.std() ** 3)

    def kurt(self) -> float:
        """Return the kurtosis of the variable across the simulation dimension."""
        return float(np.mean((self.values - self.mean()) ** 4) / self.std() ** 4)

    def std(self) -> float:
        """Return the standard deviation of the variable across the simulation dimension."""
        return np.std(self.values)

    def percentile(self, p: NumberOrList) -> NumberOrList:
        """Return the percentile of the variable across the simulation dimension."""
        return np.percentile(self.values, p)

    def tvar(self, p: NumberOrList) -> NumberOrList:
        """Return the tail value at risk (TVAR) of the variable."""
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
        return self.values[rank_positions[math.ceil(p / 100 * self.n_sims) :]].mean()

    def upsample(self, n_sims: int) -> StochasticScalar:
        """Increase the number of simulations in the variable."""
        if n_sims == self.n_sims:
            return self
        return StochasticScalar(self.values[np.arange(n_sims) % self.n_sims])

    def show_histogram(self, title: str | None = None):
        """Show a histogram of the variable.

        Args:
            title (str | None): Title of the histogram plot. Defaults to None.

        """
        fig = go.Figure(go.Histogram(x=self.values), layout=dict(title=title))
        fig.show()

    def show_cdf(self, title: str | None = None):
        """Show a plot of the cumulative distribution function (cdf) of the variable.

        Args:
            title (str | None): Title of the cdf plot. Defaults to None.

        """
        fig = go.Figure(
            go.Scatter(x=np.sort(self.values), y=np.arange(self.n_sims) / self.n_sims),
            layout=dict(title=title),
        )
        fig.update_xaxes(dict(title="Value"))
        fig.update_yaxes(dict(title="Cumulative Probability"))
        fig.show()

    def _reorder_sims(self, new_order) -> None:
        """Reorder the simulations in the variable."""
        self.values = self.values[new_order]

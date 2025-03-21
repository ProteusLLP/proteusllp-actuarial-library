from __future__ import annotations
from .config import xp as np
from numpy.typing import ArrayLike
from .couplings import ProteusStochasticVariable, CouplingGroup
from typing import Union, TypeVar
import math
import plotly.graph_objects as go

Numeric = Union[int, float]
NumberOrList = TypeVar("NumberOrList", Numeric, list[Numeric])
NumericOrStochasticScalar = TypeVar(
    "NumericOrStochasticScalar", Numeric, "StochasticScalar"
)


class StochasticScalar(ProteusStochasticVariable):
    """A class to represent a single scalar variable in a simulation."""

    coupled_variable_group: CouplingGroup

    @property
    def ranks(self) -> StochasticScalar:
        """Return the ranks of the variable."""
        result = np.empty(self.n_sims, dtype=int)
        result[np.argsort(self.values)] = np.arange(self.n_sims)
        return StochasticScalar(result)

    def __init__(self, values: ArrayLike):
        super().__init__()
        assert hasattr(values, "__getitem__"), "Values must be an array-like object."
        if isinstance(values, StochasticScalar):
            self.values = values.values
            self.n_sims = values.n_sims
            self.coupled_variable_group = values.coupled_variable_group
        else:
            self.values = np.asarray(values)
            self.n_sims = len(self.values)

    def __hash__(self):
        return id(self)

    def tolist(self):
        return self.values.tolist()

    def _reorder_sims(self, new_order) -> None:
        """Reorder the simulations in the variable."""
        self.values = self.values[new_order]

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs, **kwargs
    ) -> StochasticScalar:
        """Override the __array_ufunc__ method means that you can apply standard numpy functions"""
        inputs = tuple(
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
        result = StochasticScalar(getattr(ufunc, method)(*inputs, **kwargs))
        for input in inputs:
            if isinstance(input, ProteusStochasticVariable):
                self.coupled_variable_group.merge(input.coupled_variable_group)
        result.coupled_variable_group = self.coupled_variable_group

        return result

    def _binary_operation(self, other, operation, is_reversible=True):
        if isinstance(other, StochasticScalar):
            if self.n_sims != other.n_sims:
                if self.n_sims != 1 and other.n_sims != 1:
                    raise ValueError("Number of simulations do not match.")
            result = StochasticScalar(operation(self.values, other.values))
            self.coupled_variable_group.merge(other.coupled_variable_group)
            result.coupled_variable_group.merge(self.coupled_variable_group)
            return result
        elif isinstance(other, (int, float)):
            result = StochasticScalar(operation(self.values, other))
            result.coupled_variable_group.merge(self.coupled_variable_group)
            return result
        elif is_reversible:
            # try the reverse operation on the other object
            result = operation(other, self)
            return result
        else:
            raise ValueError(
                f"Operation not supported on {type(self)} and {type(other)}."
            )

    def __add__(self, other):
        return self._binary_operation(other, lambda x, y: x + y)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary_operation(other, lambda x, y: x - y, False)

    def __rsub__(self, other):
        result = StochasticScalar(other - self.values)
        result.coupled_variable_group.merge(self.coupled_variable_group)
        return result

    def __mul__(self, other):
        return self._binary_operation(other, lambda x, y: x * y)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binary_operation(other, lambda x, y: x / y, False)

    def __rtruediv__(self, other):
        result = StochasticScalar(other / self.values)
        result.coupled_variable_group.merge(self.coupled_variable_group)
        return result

    def __pow__(self, other):
        return self._binary_operation(other, lambda x, y: x**y, False)

    def __rpow__(self, other):
        result = StochasticScalar(other**self.values)
        result.coupled_variable_group.merge(self.coupled_variable_group)
        return result

    def __eq__(self, other):
        return self._binary_operation(other, lambda x, y: x == y)

    def __ne__(self, other):
        return self._binary_operation(other, lambda x, y: x != y)

    def __lt__(self, other):
        return self._binary_operation(other, lambda x, y: x < y, False)

    def __le__(self, other):
        return self._binary_operation(other, lambda x, y: x <= y, False)

    def __gt__(self, other):
        return self._binary_operation(other, lambda x, y: x > y, False)

    def __ge__(self, other):
        return self._binary_operation(other, lambda x, y: x >= y, False)

    def _req__(self, other):
        return self.__eq__(other)

    def _rne__(self, other):
        return self.__ne__(other)

    def _rlt__(self, other):
        return self.__ge__(other)

    def _rle__(self, other):
        return self.__gt__(other)

    def _rgt__(self, other):
        return self.__le__(other)

    def _rge__(self, other):
        return self.__lt__(other)

    def __and__(self, other):
        return self._binary_operation(other, lambda x, y: x & y)

    def __rand__(self, other):
        return self.__and__(other)

    def __or__(self, other):
        return self._binary_operation(other, lambda x, y: x | y)

    def __ror__(self, other):
        return self.__or__(other)

    def __neg__(self):
        result = StochasticScalar(-self.values)
        result.coupled_variable_group.merge(self.coupled_variable_group)
        return result

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

    def __repr__(self):
        return f"StochasticScalar(values={self.values}\nn_sims={self.n_sims})"

    # implement the index referencing
    def __getitem__(
        self, index: NumericOrStochasticScalar
    ) -> NumericOrStochasticScalar:
        if isinstance(index, int):
            return self.values[index]
        elif isinstance(index, StochasticScalar):
            result = StochasticScalar(self.values[index.values])
            result.coupled_variable_group.merge(index.coupled_variable_group)
            return result
        raise ValueError("Index must be an integer, StochasticScalar or numpy array.")

    def show_histogram(self):
        fig = go.Figure(go.Histogram(x=self.values))
        fig.show()

    def show_cdf(self):
        fig = go.Figure(
            go.Scatter(x=np.sort(self.values), y=np.arange(self.n_sims) / self.n_sims)
        )
        fig.update_xaxes(dict(title="Value"))
        fig.update_yaxes(dict(title="Cumulative Probability"))
        fig.show()

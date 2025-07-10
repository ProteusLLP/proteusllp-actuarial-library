from __future__ import annotations

import os
import typing as t

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore
import plotly.io as pio  # type: ignore
import scipy.stats

from .couplings import ProteusStochasticVariable
from .frequency_severity import FreqSevSims
from .stochastic_scalar import StochasticScalar
from .types import Numeric, NumericLike, ProteusLike

pio.templates.default = "none"

__all__ = [
    "ProteusVariable",
]


class ProteusVariable(ProteusLike):
    """A class to hold a multivariate variable in a simulation.

    A Proteus Variable is a hierarchical structure that can hold multiple
    scalar variables. The purpose of this class is to allow
    for the creation of more complex variables that can be used in
    simulations.

    Each level of a Proteus Variable can be a list or dictionary of scalar
    variables or other ProteusVariable objects. Each level can have a different
    number of elements.
    Each level has a name that can be used to access the level in the hierarchy.

    Sub elements of a ProteusVariable can be accessed using the [] notation.

    """

    dim_name: str
    # not really sure what this thing should hold?
    values: dict[str, Numeric | ProteusLike]
    dimensions: list[str]

    def __init__(
        self,
        dim_name: str,
        values: dict[str, Numeric | ProteusLike],
    ):
        """Initialize a ProteusVariable.

        Args:
            dim_name: Name of the dimension.
            values: A dictionary which will contain variables that must support
                PAL variable operations.
        """
        self.dim_name: str = dim_name
        self.values = values
        self.dimensions = [dim_name]
        self._dimension_set = set(self.dimensions)
        # check the number of simulations in each variable
        self.n_sims = None
        for value in self:
            if isinstance(value, ProteusVariable):
                if (
                    self._dimension_set.intersection(value._dimension_set)
                    or self.dim_name == value.dim_name
                ):
                    raise ValueError(
                        "Duplicate dimension names in ProteusVariable hierarchy."
                    )
                self._dimension_set.intersection_update(value.dimensions)
                self.dimensions.extend(value.dimensions)

            if self.n_sims is None:
                if isinstance(value, ProteusStochasticVariable):
                    self.n_sims = value.n_sims
                else:
                    self.n_sims = 1
            elif isinstance(value, ProteusStochasticVariable):
                if value.n_sims != self.n_sims:
                    if self.n_sims == 1:
                        self.n_sims = value.n_sims
                    else:
                        raise ValueError("Number of simulations do not match.")

    def __len__(self) -> int:
        return len(self.values)

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: t.Any, **kwargs: t.Any
    ) -> ProteusVariable:
        """Handle numpy universal functions applied to ProteusVariable objects.

        This method enables ProteusVariable objects to work with numpy ufuncs by
        recursively applying the ufunc to the hierarchical structure of values.

        Args:
            ufunc: The numpy universal function to apply.
            method: The method name (only "__call__" is supported).
            *inputs: Input arguments to the ufunc.
            **kwargs: Keyword arguments to pass to the ufunc.

        Returns:
            A new ProteusVariable with the ufunc applied to its values.

        Raises:
            NotImplementedError: If method is not "__call__".
        """
        if method != "__call__":
            raise NotImplementedError(
                f"Method {method} not implemented for ProteusVariable."
            )

        def recursive_apply(*items: t.Any, **kwargs: t.Any) -> t.Any:
            # If none of the items is a ProteusVariable (i.e. a container), then
            # assume they are leaf nodes (e.g., numbers or stochastic types) and
            # simply call ufunc.
            if not any(isinstance(item, ProteusVariable) for item in items):
                # For stochastic types that implement __array_ufunc__, this call will
                # automatically delegate to their own __array_ufunc__.
                return ufunc(*items, **kwargs)

            # Otherwise, at least one of the items is a container.
            # We assume that the container structure is consistent across items.

            first_container = items[
                [
                    i
                    for i, item in enumerate(items)
                    if isinstance(item, ProteusVariable)
                ][0]
            ]

            # if the first container is a ProteusVariable, we can assume that
            # all other items are also ProteusVariables or compatible types.
            if not isinstance(first_container, ProteusVariable):
                raise TypeError(
                    f"No {type(self).__name__} found in inputs, cannot apply ufunc."
                )

            # Process dictionary containers.
            if isinstance(first_container.values, dict):
                new_data: dict[str, t.Any] = {}
                # Iterate over each key in the container.
                for key in first_container.values:
                    new_items: list[t.Any] = []
                    for item in items:
                        # Assumes that data types are homogeneous across nodes ie. if
                        # the parent ProteusVariable contains dicts, then children
                        # should also contain dicts.
                        if isinstance(item, ProteusVariable):
                            if not isinstance(item.values, dict):
                                raise TypeError(
                                    f"Expected dict values in {type(self).__name__}, "
                                    f"but got {type(item.values).__name__}."
                                )
                            new_items.append(item.values[key])
                        else:
                            new_items.append(item)
                    new_data[key] = recursive_apply(*new_items, **kwargs)
                return ProteusVariable(first_container.dim_name, new_data)

            # In case data is not a dict, try applying ufunc directly.
            return t.cast(ProteusVariable, ufunc(first_container.values, **kwargs))

        return t.cast(ProteusVariable, recursive_apply(*inputs, **kwargs))

    def __array_function__(
        self,
        func: t.Any,
        _: tuple[type, ...],
        args: tuple[t.Any, ...],
        kwargs: dict[str, t.Any],
    ) -> ProteusVariable:
        """Handle numpy array functions applied to ProteusVariable objects.

        This method enables ProteusVariable objects to work with numpy array functions
        by extracting the underlying values, applying the function, and reconstructing
        the ProteusVariable with the result.

        Args:
            func: The numpy array function to apply.
            _: Tuple of types involved in the operation (unused).
            args: Positional arguments to the function.
            kwargs: Keyword arguments to pass to the function.

        Returns:
            A new ProteusVariable with the function applied to its values.
        """
        parsed_args = [
            (list(arg.values.values()) if isinstance(arg, ProteusVariable) else arg)
            for arg in args
        ]
        temp = func(*parsed_args, **kwargs)
        return ProteusVariable(
            self.dim_name,
            {key: temp[i] for i, key in enumerate(self.values.keys())},
        )

    def sum(self, dimensions: list[str] | None = None) -> NumericLike:
        """Sum the variables across the specified dimensions.

        Returns a new ProteusVariable with the summed values.
        """
        if dimensions is None:
            dimensions = []
        if dimensions == []:
            result = self.sum()
            return result

        return self

        # FIXME: This always evaluates to false and so the code never executes in this
        # block. Basically, self.dimensions is always a list of strings and so is
        # dimensions therefore dimensions would have to be a list of lists for this to
        # evaluate to true - perhaps the intention was to check if the set of dimensions
        # in self is a subset of the dimensions passed in? ie.
        # if set(self.dimensions) <= set(dimensions): ...
        # Here is the original code...
        # if self.dimensions in dimensions:
        #     # Also, the values here could be a dict?
        #     result = ProteusVariable(dim_name=self.values[0].dimensions, values=0)
        #     for value in self.values:
        #         if isinstance(value, (ProteusVariable, StochasticScalar)):
        #             result += value.sum(dimensions)
        #         else:
        #             result += value
        #     return result
        # else:
        #     return self

    def __iter__(self) -> t.Iterator[NumericLike]:
        return iter(self.values.values())

    def _binary_operation(
        self,
        other: object,
        operation: t.Callable[[t.Any, t.Any], t.Any],
    ) -> t.Any:
        """Perform a binary operation between this ProteusVariable and another value.

        This method applies a binary operation element-wise between this
        ProteusVariable and another ProteusVariable or scalar value, preserving the
        hierarchical structure.

        Args:
            other: The other operand, which can be another ProteusVariable or a
                numeric value.
            operation: A callable that takes two arguments and returns the result
                of the operation.

        Returns:
            The result of the binary operation.

        Raises:
            ValueError: If the other operand is a ProteusVariable with mismatched
                dimensions.
        """
        if isinstance(other, ProteusVariable):
            # Assumed that the values and dimensions on self and other are homogeneous.
            if self.dimensions != other.dimensions:
                raise ValueError("Dimensions of the two variables do not match.")

            # strange syntax here, but this is to ensure that the type of the
            # returned ProteusVariable is the same as the type of self and therefore
            # satisfies typechecks.
            return type(self)(
                dim_name=self.dim_name,
                values={
                    key: operation(value, other.values[key])
                    for key, value in self.values.items()
                },
            )

        raise TypeError(
            f"Unsupported type for binary operation: {type(self.values).__name__}"
        )

    # Arithmetic operations
    def __add__(self, other: t.Any) -> t.Self:
        return t.cast(t.Self, self._binary_operation(other, lambda a, b: a + b))

    def __radd__(self, other: t.Any) -> t.Self:
        return self.__add__(other)

    def __sub__(self, other: t.Any) -> t.Self:
        return t.cast(t.Self, self._binary_operation(other, lambda a, b: a - b))

    def __rsub__(self, other: t.Any) -> t.Self:
        return t.cast(t.Self, self._binary_operation(other, lambda a, b: b - a))

    def __mul__(self, other: t.Any) -> t.Self:
        return t.cast(t.Self, self._binary_operation(other, lambda a, b: a * b))

    def __rmul__(self, other: t.Any) -> t.Self:
        return self.__mul__(other)

    def __truediv__(self, other: t.Any) -> t.Self:
        return t.cast(t.Self, self._binary_operation(other, lambda a, b: a / b))

    def __rtruediv__(self, other: t.Any) -> t.Self:
        return t.cast(t.Self, self._binary_operation(other, lambda a, b: b / a))

    def __pow__(self, other: t.Any) -> t.Self:
        return t.cast(t.Self, self._binary_operation(other, lambda a, b: a**b))

    def __rpow__(self, other: t.Any) -> t.Self:
        return t.cast(t.Self, self._binary_operation(other, lambda a, b: b**a))

    def __neg__(self) -> t.Self:
        """Return the negation of the variable."""
        return t.cast(t.Self, self._binary_operation(self, lambda a, _: -a))

    # Comparison operations
    def __lt__(self, other: t.Any) -> bool:
        return t.cast(bool, self._binary_operation(other, lambda a, b: a < b))

    def __rlt__(self, other: t.Any) -> bool:
        return self.__ge__(other)

    def __le__(self, other: t.Any) -> bool:
        return t.cast(bool, self._binary_operation(other, lambda a, b: a <= b))

    def __rle__(self, other: t.Any) -> bool:
        return self.__gt__(other)

    def __gt__(self, other: t.Any) -> bool:
        return t.cast(bool, self._binary_operation(other, lambda a, b: a > b))

    def __rgt__(self, other: t.Any) -> bool:
        return self.__le__(other)

    def __ge__(self, other: t.Any) -> bool:
        return t.cast(bool, self._binary_operation(other, lambda a, b: a >= b))

    def __rge__(self, other: t.Any) -> bool:
        return self.__lt__(other)

    # Equality operations
    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __ne__(self, other: object) -> bool:
        raise NotImplementedError

    def __getitem__(self, key: int | str) -> NumericLike:
        if isinstance(key, int):
            return self.values[list(self.values.keys())[key]]
        else:
            return self.values[key]

    def _get_value_at_sim_helper(
        self, x: NumericLike, sim_no: int | StochasticScalar
    ) -> ProteusLike:
        """Helper method to get value at simulation for a single element."""
        if isinstance(x, ProteusVariable):
            return x.get_value_at_sim(sim_no)

        if isinstance(x, StochasticScalar) or isinstance(x, FreqSevSims):
            # Handle StochasticScalar and FreqSevSims types
            if x.n_sims is None:
                # If n_sims is None, return the value directly
                return x

            if x.n_sims <= 1:
                # If n_sims is 1 or None, return the value directly
                return x

            if isinstance(sim_no, StochasticScalar):
                # Extract all values and return a new StochasticScalar with those indices
                indices = sim_no.values.astype(int)
                return StochasticScalar(x.values[indices])

        if isinstance(x, NumericLike):
            # If x is a scalar, return it directly
            return x

        raise TypeError(
            f"Unsupported type for value at simulation: {type(x).__name__}. "
            "Expected ProteusVariable, StochasticScalar, FreqSevSims, or Numeric."
        )

    def get_value_at_sim(self, sim_no: int | StochasticScalar) -> t.Self:
        """Get values at specific simulation number(s)."""
        # FIXME: this makes a bit of a mess of the interface. Would make sense to just
        # make use of the __getitem__ method instead. Since ProteusVariable is
        # SequenceLike, it should support indexing with integers and strings.
        return type(self)(
            dim_name=self.dim_name,
            values={
                k: self._get_value_at_sim_helper(v, sim_no)
                for k, v in self.values.items()
            },
        )

    def all(self) -> bool:
        """Return True if all values are True.

        Assumes that values also support the `all()` method, such as
        ProteusStochasticVariable or FreqSevSims. If not, just checks for truthiness.

        Returns:
            True if all values are True, False otherwise.
        """

        def _is_truthy(value: t.Any) -> bool:
            try:
                return bool(value.all())
            except AttributeError:
                return bool(value)

        return all(_is_truthy(value) for value in self.values.values())

    def any(self) -> bool:
        """Return True if any value is True."""

        def _is_truthy(value: t.Any) -> bool:
            try:
                return bool(value.any())
            except AttributeError:
                return bool(value)

        return any(_is_truthy(value) for value in self.values.values())

    def percentile(self, p: float | list[float]) -> ProteusVariable:
        """Return the percentile of the variable across the simulation dimension."""
        raise NotImplementedError
        # FIXME: This code is untested and will also raise an AttributeError if it's
        # called. Notice that the ProteusStochasticVariable class does not have a
        # percentile method.
        return ProteusVariable(
            dim_name=self.dim_name,
            values={
                key: (
                    value.percentile(p)
                    if isinstance(value, ProteusStochasticVariable)
                    else value
                )
                for key, value in self.values.items()
            },
        )

    def tvar(self, p: float | list[float]) -> ProteusVariable:
        """Return the tail value at risk (TVAR) of the variable."""
        raise NotImplementedError
        # Again, ProteusStochasticVariable does not have a tvar method, so this code
        # is not expected to work as is.
        return ProteusVariable(
            dim_name=self.dim_name,
            values={
                key: (
                    value.tvar(p)
                    if isinstance(value, ProteusStochasticVariable)
                    else value
                )
                for key, value in self.values.items()
            },
        )

    def mean(self) -> ProteusVariable:
        """Return the mean of the variable across the simulation dimension."""

        def _mean_helper(value: NumericLike) -> t.Any:
            """Helper function to compute mean for different value types."""
            if isinstance(value, FreqSevSims):
                return value.aggregate().mean()
            if isinstance(value, StochasticScalar):
                return value.mean()
            if isinstance(value, ProteusVariable):
                # For nested ProteusVariable, recursively compute mean
                return value.mean()
            if isinstance(value, (int, float, np.number)):
                # If the value is a scalar, return it directly
                return float(value)
            raise TypeError(
                f"{type(value).__name__} cannot be converted to float. "
                "Mean cannot be computed."
            )

        return ProteusVariable(
            dim_name=self.dim_name,
            values={key: _mean_helper(value) for key, value in self.values.items()},
        )

    def upsample(self, n_sims: int) -> ProteusVariable:
        """Upsample the variable to the specified number of simulations."""
        if self.n_sims == n_sims:
            return self
        return ProteusVariable(
            dim_name=self.dim_name,
            values={
                key: (
                    value.upsample(n_sims)
                    if isinstance(value, ProteusStochasticVariable)
                    else value
                )
                for key, value in self.values.items()
            },
        )

    @classmethod
    def from_csv(
        cls,
        file_name: str,
        dim_name: str,
        values_column: str,
        simulation_column: str = "Simulation",
    ) -> ProteusVariable:
        """Import a ProteusVariable from a CSV file.

        Note that only one dimensional variables are supported.
        """
        df: pd.DataFrame = pd.read_csv(file_name)
        pivoted_df = df.pivot(
            index=simulation_column, columns=dim_name, values=values_column
        )
        count = df[dim_name].value_counts()
        pivoted_df.sort_index(inplace=True)

        result = cls(
            dim_name,
            {
                str(label): StochasticScalar(pivoted_df[label].values[: count[label]])
                for label in df[dim_name].unique()
            },
        )
        result.n_sims = max(count)

        return result

    @classmethod
    def from_dict(cls, data: dict[str, list[float]]) -> ProteusVariable:
        """Create a ProteusVariable from a dictionary.

        Note that only one dimensional variables are supported.
        """
        result = cls(
            dim_name="Dim1",
            values={str(label): StochasticScalar(data[label]) for label in data.keys()},
        )
        result.n_sims = max([len(v) for v in data.values()])

        return result

    @classmethod
    def from_series(cls, data: pd.Series) -> ProteusVariable:
        """Create a ProteusVariable from a pandas Series.

        Note that only one dimensional variables are supported.
        """
        result = cls(
            dim_name=str(data.index.name) if data.index.name is not None else "Dim1",
            values={label: data[label] for label in data.index},
        )
        result.n_sims = 1

        return result

    def __repr__(self) -> str:
        return f"ProteusVariable(dim_name={self.dim_name}, values={self.values})"

    def correlation_matrix(
        self, correlation_type: str = "spearman"
    ) -> list[list[float]]:
        """Compute correlation matrix between variables."""
        correlation_type = correlation_type.lower()
        if correlation_type not in ["linear", "spearman", "kendall"]:
            raise ValueError(
                f"Unsupported correlation type: {correlation_type}. "
                "Supported types are 'linear', 'spearman', and 'kendall'."
            )
        if not hasattr(self, "values"):
            raise TypeError(f"{type(self).__name__} does not have 'values' attribute.")
        n = len(self.values)
        result: list[list[float]] = [[0.0] * n] * n
        values = [self[i] for i in range(len(self.values))]

        # Extract underlying arrays for correlation calculations
        numeric_values = []
        for value in values:
            if hasattr(value, "values"):
                numeric_values.append(value.values)
            else:
                numeric_values.append(value)

        if correlation_type.lower() in ["spearman", "kendall"]:
            # Check that all values are supported for ranking
            for _, value in enumerate(values):
                if not isinstance(value, (ProteusVariable | ProteusStochasticVariable)):
                    raise TypeError(f"{value} not supported. Spearman and Kendall")

            # rank the variables first
            ranked_values = [scipy.stats.rankdata(arr) for arr in numeric_values]

            # FIXME: The logic here is unfinished.
            raise NotImplementedError

        elif correlation_type == "kendall":
            for i, value1 in enumerate(numeric_values):
                for j, value2 in enumerate(numeric_values):
                    statistic, _ = scipy.stats.kendalltau(value1, value2)
                    result[i][j] = t.cast(float, statistic)
        else:
            raise ValueError(f"Unsupported correlation type: {correlation_type}")

        return result

    def show_histogram(self, title: str | None = None) -> None:
        """Show a histogram of the variable values.

        Args:
            title (str | None): The title of the histogram. If None, no title is set.

        """
        if os.getenv("PAL_SUPPRESS_PLOTS", "").lower() == "true":
            return
        fig = go.Figure(layout=go.Layout(title=title))
        for label, value in self.values.items():
            fig.add_trace(go.Histogram(x=value.values(), name=label))
        fig.show()

    def show_cdf(self, title: str | None = None) -> None:
        """Plot the cumulative distribution function (cdf) of the variable values.

        Args:
            title: Optional title for the cdf. If None, no title is set.
        """
        if os.getenv("PAL_SUPPRESS_PLOTS", "").lower() == "true":
            return
        fig = go.Figure(layout=go.Layout(title=title))
        for label, value in self.values.items():
            if not isinstance(value, (ProteusVariable | ProteusStochasticVariable)):
                raise TypeError(
                    f"{type(value).__name__} does not support CDF plotting. "
                )
            if value.n_sims is None or value.n_sims <= 1:
                raise ValueError(
                    "CDF can only be plotted for variables with multiple simulations."
                )
            fig.add_trace(
                go.Scatter(
                    x=np.sort(np.array(value.values)),
                    y=np.arange(value.n_sims) / value.n_sims,
                    name=label,
                )
            )
        fig.update_xaxes(title_text="Value")
        fig.update_yaxes(title_text="Cumulative Probability")
        fig.show()

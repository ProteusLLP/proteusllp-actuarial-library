"""Multi-dimensional stochastic variables for actuarial modeling.

This module provides the ProteusVariable class, which represents multi-dimensional
stochastic variables commonly used in actuarial and risk modeling. A ProteusVariable
can contain different types of stochastic objects across multiple dimensions,
enabling complex risk factor modeling.

Key features:
- Multi-dimensional stochastic variables with named dimensions
- Support for various stochastic types (StochasticScalar, FreqSevSims, etc.)
- Mathematical operations across dimensions and simulations
- Correlation analysis and upsampling capabilities
- Export functionality for analysis and reporting

NOTE: The serialization/deserialization methods (from_csv, from_dict, from_series)
      are currently incomplete and have significant limitations. A comprehensive
      codec system is planned to address these issues.
      See: https://github.com/ProteusLLP/proteusllp-actuarial-library/issues/22

The ProteusVariable is designed for actuarial applications such as:
- Multi-factor risk modeling (e.g., frequency, severity, inflation)
- Portfolio-level aggregation across risk dimensions
- Scenario analysis with correlated risk factors
- Capital modeling with interdependent variables

Example:
    >>> from pal.stochastic_scalar import StochasticScalar
    >>> from pal.frequency_severity import FreqSevSims
    >>>
    >>> # Create a multi-dimensional risk variable
    >>> risk_var = ProteusVariable(
    ...     dim_name="insurance_risk",
    ...     values={
    ...         "frequency": StochasticScalar([10, 12, 8, 15]),
    ...         "severity": StochasticScalar([5000, 6000, 4500, 7000]),
    ...         "expense_ratio": StochasticScalar([0.3, 0.32, 0.28, 0.35])
    ...     }
    ... )
    >>> total_cost = (
    ...     risk_var["frequency"]
    ...     * risk_var["severity"]
    ...     * (1 + risk_var["expense_ratio"])
    ... )
"""

# standard library imports
from __future__ import annotations

import os
import typing as t
from numbers import Number

# third-party imports
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go  # type: ignore
import plotly.io as pio  # type: ignore
import scipy.stats

# local imports
from . import maths as pnp
from .couplings import ProteusStochasticVariable
from .frequency_severity import FreqSevSims
from .stochastic_scalar import StochasticScalar
from .types import ProteusLike, VectorLike

pio.templates.default = "none"

__all__ = [
    "ProteusVariable",
]


class ProteusVariable[T]:
    """A generic, homogeneous container for multivariate variables in simulations.

    ProteusVariable is a hierarchical structure that holds multiple variables of
    the SAME type (homogeneous container). Each instance must contain either all
    scalars, all vectors (like StochasticScalar), or all nested ProteusVariables
    - but never a mix of different types.

    Type Parameter:
        T: The type of values stored. By convention, T should be a ScalarOrVector
           type (NumericLike | VectorLike), though the parameter is unconstrained
           to allow flexible type inference. Usage with non-ScalarOrVector types
           may not be fully supported by all operations.

    Key Features:
    - **Homogeneous**: All values in a single instance must be the same type.
      Like List[T], you cannot mix types within one container.
    - **Type Safety**: Operations like mean() return type T, preserving type
      information through the computation.
    - **Nesting**: ProteusVariable containing ProteusVariable enables hierarchical
      data structures (e.g., risks by region by peril)
    - **Dictionary Access**: Sub-elements accessed via [] notation with
      string keys or integer indices

    Examples:
        >>> # Homogeneous scalar container
        >>> scalar_risks = ProteusVariable(
        ...     dim_name="risk_amounts",
        ...     values={"fire": 100000, "flood": 200000}  # All int
        ... )

        >>> # Homogeneous vector container
        >>> vector_risks = ProteusVariable(
        ...     dim_name="stochastic_losses",
        ...     values={
        ...         "fire": StochasticScalar([100, 200, 300]),
        ...         "flood": StochasticScalar([150, 250, 350])
        ...     }  # All StochasticScalar
        ... )

        >>> # Homogeneous nested container
        >>> nested_risks = ProteusVariable(
        ...     dim_name="regions",
        ...     values={
        ...         "north": scalar_risks,
        ...         "south": scalar_risks
        ...     }  # All ProteusVariable instances
        ... )

        >>> # INVALID - mixing types not allowed
        >>> # mixed = ProteusVariable(values={"a": 100, "b": StochasticScalar([1])})
        >>> # This would violate homogeneity and cause type errors

    Note: Statistical operations should be performed using numpy and scipy functions
    directly on ProteusVariable instances. For example:
    - Use np.percentile(variable, p)
    - Use np.mean(variable)
    - Use pal.stats.tvar(variable, p)
    """

    dim_name: str
    values: t.Mapping[str, T]
    dimensions: list[str]

    def __init__(
        self,
        dim_name: str,
        values: t.Mapping[str, T],
    ):
        """Initialize a ProteusVariable.

        Args:
            dim_name: Name of the dimension.
            values: A mapping (dict-like object) containing variables that must
                support PAL variable operations. Keys will be sorted alphabetically
                during initialization to ensure consistent ordering.

        Raises:
            TypeError: If values is not a mapping type.
        """
        self.dim_name: str = dim_name
        # TODO: Clarify whether the values dict is intended to be mutable during the
        # variable's lifetime, or if it should be treated as immutable after
        # initialization. Consider using a frozen dict if immutability is desired.
        # See: https://github.com/ProteusLLP/proteusllp-actuarial-library/issues/20
        self.values = values
        self.dimensions = [dim_name]
        self._dimension_set = set(self.dimensions)
        # Ensure that values is a mapping type
        if not isinstance(values, t.Mapping):  # type: ignore[redundant-expr]
            raise TypeError(
                f"Expected a mapping (dict-like) for 'values', got "
                f"{type(values).__name__}"
            )
        # check the number of simulations in each variable
        self.n_sims = None
        for value in (
            self.values.values() if isinstance(self.values, dict) else self.values
        ):
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
        """Return the number of elements in the variable."""
        return len(self.values)

    def __array__(self, dtype: t.Any = None) -> npt.NDArray[t.Any]:
        """Convert ProteusVariable to numpy array for basic operations.

        This method enables ProteusVariable to work with numpy functions like
        np.sum(), making it VectorLike protocol compliant. Current implementation
        provides basic functionality by concatenating all values into a 1D array.

        NOTE: This is a simplified implementation. Complex nested container
              scenarios and multi-dimensional operations need architectural
              decisions about data representation and operation semantics.
              See: https://github.com/ProteusLLP/proteusllp-actuarial-library/issues/23

        Args:
            dtype: Optional data type for the resulting array.

        Returns:
            A numpy array created by concatenating all values.

        Raises:
            NotImplementedError: For mismatched simulation lengths or other
                               complex scenarios requiring architectural decisions.
        """
        # For basic 1D operations like np.sum(), concatenate all values
        arrays = [np.asarray(value) for value in self.values.values()]

        # If we have any scalars (0D arrays), convert them to 1D arrays with single
        # element
        processed_arrays: list[npt.NDArray[t.Any]] = []
        for arr in arrays:
            if arr.ndim == 0:
                # Convert scalar to 1D array with single element
                processed_arrays.append(np.array([arr.item()]))
            else:
                processed_arrays.append(arr)

        # Now check if all 1D arrays have the same length (simulation dimension)
        lengths = [len(arr) for arr in processed_arrays]
        if len(set(lengths)) > 1:
            raise NotImplementedError(
                "Array conversion not supported for ProteusVariable with "
                "mismatched simulation lengths. Use .upsample() first."
            )

        # Concatenate arrays - this creates a 1D array suitable for np.sum()
        result = np.concatenate(processed_arrays)

        if dtype is not None:
            result = result.astype(dtype)

        return result

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: t.Any, **kwargs: t.Any
    ) -> ProteusVariable[T]:
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
            if isinstance(
                first_container.values,  # type: ignore[reportUnknownMemberType]
                dict,
            ):
                new_data: dict[str, t.Any] = {}
                # Iterate over each key in the container.
                # Type ignore: Runtime type narrowing - we're intentionally checking
                # types at runtime to handle heterogeneous inputs from numpy ufuncs
                for key in first_container.values:  # type: ignore[reportUnknownMemberType]  # noqa: E501
                    new_items: list[t.Any] = []
                    for item in items:
                        # Assumes that data types are homogeneous across nodes ie. if
                        # the parent ProteusVariable contains dicts, then children
                        # should also contain dicts.
                        if isinstance(item, ProteusVariable):
                            # Type ignore: Runtime type checking for structural
                            # validation. We need to verify dict structure at runtime
                            # for ufunc recursion
                            vals = item.values  # type: ignore[reportUnknownMemberType]
                            if not isinstance(vals, dict):
                                raise TypeError(
                                    f"Expected dict values in {type(self).__name__}, "
                                    f"but got {type(vals).__name__}."  # type: ignore[reportArgumentType]  # noqa: E501
                                )
                            new_items.append(vals[key])
                        else:
                            new_items.append(item)
                    new_data[key] = recursive_apply(*new_items, **kwargs)
                # Return ProteusVariable without type parameter: The type is determined
                # at runtime through recursive_apply, not statically knowable
                return ProteusVariable(first_container.dim_name, new_data)

            # In case data is not a dict, try applying ufunc directly.
            # Type ignore: Return type depends on runtime ufunc behavior and value
            # types, not statically determinable in this dynamic dispatch context
            return ufunc(first_container.values, **kwargs)  # type: ignore[return-value]

        # Type ignore: recursive_apply's return type is determined at runtime based on
        # the actual ufunc and input types - this method handles arbitrary numpy
        # operations with dynamic type resolution
        return recursive_apply(*inputs, **kwargs)  # type: ignore[return-value]

    def __array_function__(
        self,
        func: t.Any,
        _: tuple[t.Any, ...],
        args: tuple[t.Any, ...],
        kwargs: dict[str, t.Any],
    ) -> t.Any:
        """Handle numpy array functions applied to ProteusVariable objects.

        This method enables ProteusVariable objects to work with numpy array functions
        by extracting the underlying values, applying the function, and reconstructing
        the ProteusVariable with the result.

        Special handling for mean(): Returns a ProteusVariable where each key's value
        is replaced by its mean, preserving the original structure.

        Args:
            func: The numpy array function to apply.
            _: Tuple of types involved in the operation (unused).
            args: Positional arguments to the function.
            kwargs: Keyword arguments to pass to the function.

        Returns:
            A new ProteusVariable with the function applied to its values.
        """
        # Special handling for mean() to preserve ProteusVariable structure
        if func.__name__ == "mean" and len(args) == 1 and args[0] is self:
            mean_values: dict[str, T] = {}
            for key, value in self.values.items():
                # Use pnp.mean() for all values to ensure consistent behavior
                # across all PAL types (StochasticScalar, FreqSevSims, etc.)
                mean_values[key] = pnp.mean(value)

            return ProteusVariable(dim_name=self.dim_name, values=mean_values)

        parsed_args: list[t.Any] = []
        for arg in args:
            if arg is self:
                # For the ProteusVariable itself, stack its dictionary values as columns
                value_arrays = [np.asarray(value) for value in self.values.values()]
                parsed_args.append(np.column_stack(value_arrays))
            else:
                parsed_args.append(arg)

        if "axis" in kwargs:
            # Nothing to do here as the axis is already specified.
            pass
        else:
            if func.__name__ in ["cumsum", "cumprod", "diff"]:
                # For functions that need axis specification, add axis=1 to kwargs
                kwargs["axis"] = 1
            elif func.__name__ in ["sum", "mean", "std", "var"]:
                # For reduction operations, check the type of values to determine axis
                # behavior...
                first_value = next(iter(self.values.values())) if self.values else None

                # Check if we have vector-like values (StochasticScalar, FreqSevSims,
                # etc.)
                if first_value is not None and isinstance(first_value, VectorLike):
                    # Vector-like values: use axis=1 to sum across dimensions (row-wise)
                    # This preserves simulation structure for StochasticScalar objects
                    kwargs["axis"] = 1

                # In case this if doesn't match, for scalar values, use default (no
                # axis) which will return a scalar result.
            else:
                # In all other cases, do not specify axis and let numpy decide
                pass

        temp = func(*parsed_args, **kwargs)

        # Handle 0D (scalar), 1D and 2D results
        if temp.ndim == 0:
            # If result is scalar (0D), return the scalar directly
            return temp.item()

        if temp.ndim == 1:
            # Check if this is a reduction result from vector-like values
            first_value = next(iter(self.values.values())) if self.values else None

            if (
                func.__name__ in ["sum", "mean", "std", "var"]
                and first_value is not None
                and isinstance(first_value, VectorLike)
            ):
                # Reduction of vector-like values: return a single StochasticScalar
                result = StochasticScalar(temp)

                # Merge coupling groups from all original values
                for value in self.values.values():
                    if hasattr(value, "coupled_variable_group"):
                        # Type ignore: we know that this attribute exists because we've
                        # done a runtime check above.
                        result.coupled_variable_group.merge(
                            value.coupled_variable_group  # type: ignore[attr-defined]
                        )

                return result

            # Other 1D results: distribute evenly across keys
            n_keys = len(self.values.keys())
            chunk_size = len(temp) // n_keys
            return ProteusVariable(
                self.dim_name,
                {
                    key: StochasticScalar(temp[i * chunk_size : (i + 1) * chunk_size])
                    for i, key in enumerate(self.values.keys())
                },
            )

        if temp.ndim == 2:
            # If result is 2D, use columns
            return ProteusVariable(
                self.dim_name,
                {
                    key: StochasticScalar(temp[:, i])
                    for i, key in enumerate(self.values.keys())
                },
            )

        # This should be unreachable - we've handled 0D, 1D, and 2D arrays
        raise NotImplementedError(
            f"Unexpected array dimensionality: {temp.ndim}D array returned by "
            f"{func.__name__}. Only 0D (scalar), 1D, and 2D arrays are supported."
        )

    def __iter__(self) -> t.Iterator[T]:
        """Iterate over the values in the variable."""
        return iter(self.values.values())

    def __contains__(self, value: object) -> bool:
        """Check if value is in the container.

        Required for Sequence protocol compatibility.
        """
        return value in self.values.values()

    def __reversed__(self) -> t.Iterator[T]:
        """Return a reverse iterator over the values.

        Required for Sequence protocol compatibility.
        """
        return reversed(list(self.values.values()))

    def __repr__(self) -> str:
        return f"ProteusVariable(dim_name={self.dim_name}, values={self.values})"

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
    def __lt__(self, other: t.Any) -> t.Self:
        return t.cast(t.Self, self._binary_operation(other, lambda a, b: a < b))

    def __rlt__(self, other: t.Any) -> t.Self:
        return self.__ge__(other)

    def __le__(self, other: t.Any) -> t.Self:
        return t.cast(t.Self, self._binary_operation(other, lambda a, b: a <= b))

    def __rle__(self, other: t.Any) -> t.Self:
        return self.__gt__(other)

    def __gt__(self, other: t.Any) -> t.Self:
        return t.cast(t.Self, self._binary_operation(other, lambda a, b: a > b))

    def __rgt__(self, other: t.Any) -> t.Self:
        return self.__le__(other)

    def __ge__(self, other: t.Any) -> t.Self:
        return t.cast(t.Self, self._binary_operation(other, lambda a, b: a >= b))

    def __rge__(self, other: t.Any) -> t.Self:
        return self.__lt__(other)

    # Equality operations
    def __eq__(self, other: object) -> t.Self:  # type: ignore[override]
        return t.cast(t.Self, self._binary_operation(other, lambda a, b: a == b))

    def __ne__(self, other: object) -> t.Self:  # type: ignore[override]
        return t.cast(t.Self, self._binary_operation(other, lambda a, b: a != b))

    def __getitem__(self, key: int | str) -> T:
        # FIXME: This assumes that the ordering of the values never changes. At the
        # moment, this is not true. The values are stored in mutable container!
        if isinstance(key, int):
            return list(self.values.values())[key]
        if isinstance(key, str):  # type: ignore[redundant-expr]
            return self.values[key]
        raise TypeError(f"Key must be an integer or string, got {type(key).__name__}.")

    def count(self, value: T) -> int:
        """Count occurrences of value in the container.

        Required for Sequence protocol compatibility.
        """
        return list(self.values.values()).count(value)

    def index(self, value: T, start: int = 0, stop: int | None = None) -> int:
        """Return index of first occurrence of value.

        Required for Sequence protocol compatibility.

        Raises:
            ValueError: If value is not found.
        """
        values_list = list(self.values.values())
        if stop is None:
            stop = len(values_list)
        try:
            return values_list.index(value, start, stop)
        except ValueError as error:
            raise ValueError(f"{value!r} is not in ProteusVariable") from error

    def get_value_at_sim(
        self, sim_no: int | VectorLike[int]
    ) -> ProteusVariable[T | VectorLike[T] | ProteusLike[T]]:
        """Get values at specific simulation number(s).

        Args:
            sim_no: Simulation index(es) to extract. Can be a single numeric value,
                a list of integers, or a VectorLike object such as StochasticScalar.

        Returns:
            A new ProteusVariable with values at the specified simulation indices.
        """
        # FIXME: this makes a bit of a mess of the interface. Would make sense to just
        # make use of the __getitem__ method instead. Since ProteusVariable is
        # SequenceLike, it should support indexing with integers and strings.
        # For this to work, we need to be sure that the contents of values is indeed
        # VectorLike. Remember that ProteusVariables may be nested and a ProteusVariable
        # will not be VectorLike.
        return ProteusVariable(
            dim_name=self.dim_name,
            values={
                k: self._get_value_at_sim_helper(v, sim_no)
                for k, v in self.values.items()
            },
        )

    def upsample(self, n_sims: int) -> ProteusVariable[T]:
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

    def sum(self) -> T:
        """Return the sum across the outer dimension."""
        return sum(self)  # type: ignore[arg-type]

    @classmethod
    def from_csv(
        cls,
        file_name: str,
        dim_name: str,
        values_column: str,
        simulation_column: str = "Simulation",
    ) -> ProteusVariable[T]:
        """Import a ProteusVariable from a CSV file.

        This method currently has significant limitations and will be replaced
        with a more comprehensive serialization system.

        Current Limitations:
        - Only supports one-dimensional variables
        - Always creates StochasticScalar values regardless of intended type
        - Cannot preserve generic type information through deserialization
        - No support for nested ProteusVariable structures

        Args:
            file_name: Path to the CSV file to read
            dim_name: Name of the dimension column in the CSV
            values_column: Name of the column containing the values
            simulation_column: Name of the column containing simulation indices

        Returns:
            ProteusVariable with StochasticScalar values loaded from the CSV

        TODO: Implement comprehensive codec system for proper serialization
              See: https://github.com/ProteusLLP/proteusllp-actuarial-library/issues/22
        """
        # Type ignore: pandas-stubs has complex overloads causing Pyright to report
        # the function signature as "partially unknown" despite correct usage
        df: pd.DataFrame = pd.read_csv(file_name)  # type: ignore[misc]
        pivoted_df = df.pivot(
            index=simulation_column, columns=dim_name, values=values_column
        )
        count = df[dim_name].value_counts()
        # Type ignore: pandas-stubs overloads cause "partially unknown" warnings
        pivoted_df.sort_index(inplace=True)  # type: ignore[misc]

        # classmethods can't preserve generic type parameters so we need a type ignore
        # here. When data is loaded, the contents of the ProteusVariable will be
        # whatever was present in the CSV file. It may be necessary to separate these
        # factory functions from ProteusVariable completely.
        result = cls(
            dim_name,
            {
                str(label): StochasticScalar(pivoted_df[label].values[: count[label]])  # type: ignore[misc]
                for label in df[dim_name].unique()  # type: ignore[misc]
            },
        )
        result.n_sims = max(count)

        return result

    @classmethod
    def from_dict(
        cls,
        data: dict[str, list[float]],
    ) -> ProteusVariable[T]:
        """Create a ProteusVariable from a dictionary.

        This method currently has significant limitations and will be replaced
        with a more comprehensive serialization system.

        Current Limitations:
        - Only supports one-dimensional variables
        - Always creates StochasticScalar values from float lists
        - Cannot preserve generic type information
        - No support for nested structures or other value types

        Args:
            data: Dictionary mapping dimension labels to lists of float values

        Returns:
            ProteusVariable with StochasticScalar values created from the data

        TODO: Implement comprehensive codec system for proper serialization
              See: https://github.com/ProteusLLP/proteusllp-actuarial-library/issues/22
        """
        # Type ignore: Classmethods can't preserve generic type parameters.
        # This always creates StochasticScalar values regardless of T.
        result = cls(  # type: ignore[arg-type]
            dim_name="Dim1",
            values={str(label): StochasticScalar(data[label]) for label in data.keys()},  # type: ignore[arg-type]
        )
        result.n_sims = max([len(v) for v in data.values()])

        return result

    @classmethod
    def from_series(cls, data: pd.Series) -> ProteusVariable[T]:
        """Create a ProteusVariable from a pandas Series.

        This method currently has significant limitations and will be replaced
        with a more comprehensive serialization system.

        Current Limitations:
        - Only supports one-dimensional variables
        - Creates scalar values, not StochasticScalar
        - Cannot preserve generic type information
        - Limited to single simulation (n_sims=1)

        Args:
            data: Pandas Series with values to load

        Returns:
            ProteusVariable with scalar values from the Series

        TODO: Implement comprehensive codec system for proper serialization
              See: https://github.com/ProteusLLP/proteusllp-actuarial-library/issues/22
        """
        # Type ignore: Classmethods can't preserve generic type parameters.
        # The values type depends on the Series content, not the generic T.
        result = cls(  # type: ignore[arg-type]
            dim_name=str(data.index.name),
            values={str(label): data[label] for label in data.index},  # type: ignore[arg-type]
        )
        result.n_sims = 1

        return result

    def correlation_matrix(
        self, correlation_type: str = "spearman"
    ) -> list[list[float]]:
        """Compute correlation matrix between variables."""
        # validate type
        correlation_type = correlation_type.lower()
        if correlation_type not in ["linear", "spearman", "kendall"]:
            raise ValueError(
                f"Invalid correlation_type: '{correlation_type}'. "
                f"Must be one of: 'linear', 'spearman', 'kendall'"
            )
        if not hasattr(self[0], "values"):
            raise TypeError(
                f"First element must have 'values' attribute, "
                f"got {type(self[0]).__name__}"
            )
        n = len(self.values)
        result: list[list[float]] = [[0.0] * n] * n
        values: list[npt.NDArray[t.Any]] = [
            t.cast(npt.NDArray[t.Any], self[i]) for i in range(len(self.values))
        ]
        if correlation_type.lower() in ["spearman", "kendall"]:
            # rank the variables first
            for i, value in enumerate(values):
                values[i] = scipy.stats.rankdata(value)  # type: ignore[assignment]

        if correlation_type == "kendall":
            for i, value1 in enumerate(values):
                for j, value2 in enumerate(values):
                    result[i][j] = float(
                        scipy.stats.kendalltau(value1, value2).statistic  # type: ignore[arg-type]
                    )
        else:
            result = np.corrcoef(values).tolist()

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
            try:
                # Type ignore: plotly-stubs has incomplete type information
                fig.add_trace(go.Histogram(x=value.values(), name=label))  # type: ignore[union-attr,misc]
            except AttributeError:
                # not all values are ProteusVariable or StochasticScalar and therefore
                # do not have a values() method.
                pass
        # Type ignore: plotly-stubs has incomplete type information
        fig.show()  # type: ignore[misc]

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
            # Type ignore: plotly-stubs has incomplete type information
            fig.add_trace(  # type: ignore[misc]
                go.Scatter(
                    # Type ignore: value.values is known to exist due to isinstance
                    # check
                    x=np.sort(np.array(value.values)),  # type: ignore[attr-defined]
                    y=np.arange(value.n_sims) / value.n_sims,
                    name=label,
                )
            )
        # Type ignore: plotly-stubs has incomplete type information
        fig.update_xaxes(title_text="Value")  # type: ignore[misc]
        # Type ignore: plotly-stubs has incomplete type information
        fig.update_yaxes(title_text="Cumulative Probability")  # type: ignore[misc]
        # Type ignore: plotly-stubs has incomplete type information
        fig.show()  # type: ignore[misc]

    def _binary_operation(
        self,
        other: object,
        operation: t.Callable[[t.Any, t.Any], t.Any],
    ) -> t.Any:
        if isinstance(other, ProteusVariable):
            if self.dimensions != other.dimensions:
                raise ValueError("Dimensions of the two variables do not match.")
            return ProteusVariable(
                dim_name=self.dim_name,
                values={
                    # Type ignore: Runtime type checking - values is dict-like at this
                    # point. We've had to lean on runtime checks here over static.
                    key: operation(value, other.values[key])  # type: ignore[index]
                    for key, value in self.values.items()
                },
            )
        return ProteusVariable(
            dim_name=self.dim_name,
            values={key: operation(value, other) for key, value in self.values.items()},
        )

    def _get_value_at_sim_helper(
        self,
        x: T,
        sim_no: int | VectorLike[int],
    ) -> T | VectorLike[T] | ProteusLike[T]:
        """Helper method to get value at simulation for a single element."""
        if isinstance(x, ProteusVariable):
            # Type ignore: Private helper method with runtime type checks ensures
            # correct return type based on isinstance branching - static analyzer cannot
            # infer the precise type through the generic parameter T
            return x.get_value_at_sim(sim_no)  # pyright: ignore[reportReturnType, reportUnknownVariableType]

        if isinstance(x, StochasticScalar) or isinstance(x, FreqSevSims):
            # Handle StochasticScalar and FreqSevSims types
            if x.n_sims is None:
                # If n_sims is None, return the value directly
                return x

            if x.n_sims <= 1:
                # If n_sims is 1 or None, return the value directly
                return x

            if isinstance(sim_no, StochasticScalar):
                # Extract all values and return a new StochasticScalar with those
                # indices
                indices = sim_no.values.astype(int)
                return StochasticScalar(x.values[indices])

            # Handle the main case: extract value at specific simulation index
            if isinstance(sim_no, int):
                return x.values[sim_no]

            if isinstance(sim_no, list):
                return StochasticScalar(x.values[sim_no])

            return x

        if isinstance(x, Number):  # type: ignore[uneccesaryIsInstance]
            # If x is a numeric type, return it directly
            return x

        raise TypeError(
            f"Unsupported type for value at simulation: {type(x).__name__}.\n"
            f"Value: {x}\n"
            f"Expected one of: ProteusVariable, StochasticScalar, FreqSevSims, or "
            f"Number."
        )

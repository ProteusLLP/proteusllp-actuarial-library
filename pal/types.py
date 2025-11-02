"""Type definitions and protocols for the PAL library.

Defines common type aliases, protocols, and configuration classes used
throughout the library for type safety and consistency.
"""

# standard library
from __future__ import annotations

import dataclasses
import typing as t

# third party
import numpy.typing as npt

# project
from ._maths import xp as np

__all__ = [
    "ArithmeticProtocol",
    "Config",
    "DistributionLike",
    "Numeric",
    "NumericLike",
    "NumericProtocol",
    "ScipyNumeric",
    "VectorLike",
]

Numeric = float | int | np.number

# Type alias for scipy special functions and numpy random generators
# These functions expect more restrictive types than our general Numeric type.
# They don't accept complex numbers, _NumericProtocol objects, or general
# np.number types.
ScipyNumeric = float | int | np.floating | np.integer

T_value = t.TypeVar("T_value")


@dataclasses.dataclass
class Config:
    """Configuration class for PAL."""

    n_sims: int = 10000
    seed: int = 123456789
    rng: np.random.Generator = np.random.default_rng(seed)


@t.runtime_checkable
class ArithmeticProtocol(t.Protocol):
    """Base protocol for objects that support arithmetic operations.

    Defines common arithmetic operations shared by both scalar and vector types.
    """

    # Arithmetic operations
    def __add__(self, other: t.Any) -> t.Self: ...
    def __radd__(self, other: t.Any) -> t.Self: ...
    def __sub__(self, other: t.Any) -> t.Self: ...
    def __rsub__(self, other: t.Any) -> t.Self: ...
    def __mul__(self, other: t.Any) -> t.Self: ...
    def __rmul__(self, other: t.Any) -> t.Self: ...
    def __truediv__(self, other: t.Any) -> t.Self: ...
    def __rtruediv__(self, other: t.Any) -> t.Self: ...
    def __pow__(self, other: t.Any) -> t.Self: ...
    def __rpow__(self, other: t.Any) -> t.Self: ...
    def __neg__(self) -> t.Any: ...


@t.runtime_checkable
class NumericProtocol(ArithmeticProtocol, t.Protocol):
    """Protocol for scalar-like objects that support numeric operations.

    Comparison operations return bool (scalar semantics).
    """

    # Comparison operations (scalar semantics - return bool)
    def __lt__(self, other: t.Any) -> bool: ...
    def __le__(self, other: t.Any) -> bool: ...
    def __gt__(self, other: t.Any) -> bool: ...
    def __ge__(self, other: t.Any) -> bool: ...

    # Equality operations (scalar semantics - return bool)
    def __eq__(self, other: t.Any) -> bool: ...
    def __ne__(self, other: t.Any) -> bool: ...


@t.runtime_checkable
class SupportsArray(t.Protocol):
    """Protocol for objects that support numpy array operations.

    This protocol defines the complete interface for objects that can interact
    with numpy, including conversion to arrays and handling numpy operations.
    It encompasses all three numpy protocol methods for full numpy integration.

    Array Protocol Methods in PAL:
    -----------------------------

    **__array__ method:**
    - **Purpose**: Converts an object to a numpy array for basic array operations
    - **When called**: When np.asarray(obj) or similar conversion functions are used
    - **Return type**: Always returns a numpy array (npt.NDArray)
    - **Usage**: Simple array conversion, enables basic numpy compatibility

    **__array_function__ method:**
    - **Purpose**: Handles numpy function dispatch - intercepts numpy function calls
    - **When called**: When numpy functions like np.sum(), np.mean(), etc. are called
    - **Return type**: Can return any type (scalars, arrays, custom objects)
    - **Usage**: Custom behavior for numpy functions

    **__array_ufunc__ method:**
    - **Purpose**: Handles numpy universal functions (element-wise operations)
    - **When called**: When ufuncs like np.exp(), np.add() are used, or operators +, -
    - **Return type**: Typically returns same type as input (Self) for type preservation
    - **Usage**: Preserves custom types through element-wise operations

    The key distinctions:
    - __array__ is for conversion to numpy arrays
    - __array_function__ is for array functions (np.sum, np.mean, etc.)
    - __array_ufunc__ is for element-wise ufuncs (np.exp, np.add, +, -, etc.)
    """

    def __array__(self, dtype: t.Any = None) -> npt.NDArray[t.Any]:
        """Convert to numpy array for compatibility with numpy functions."""
        ...

    def __array_function__(
        self,
        func: t.Any,
        _: tuple[type, ...],
        args: tuple[t.Any, ...],
        kwargs: dict[str, t.Any],
    ) -> t.Any:
        """Handle numpy function dispatch to preserve object semantics."""
        ...

    def __array_ufunc__(
        self,
        ufunc: t.Any,
        method: t.Literal[
            "__call__", "reduce", "reduceat", "accumulate", "outer", "at"
        ],
        *inputs: t.Any,
        **kwargs: t.Any,
    ) -> t.Any:
        """Handle numpy ufuncs to preserve type through element-wise operations."""
        ...


@t.runtime_checkable
class VectorOperations(ArithmeticProtocol, t.Protocol):
    """Protocol for objects with vector-like operation semantics.

    This protocol defines vector-style operations where:
    - Comparison operations return Self (element-wise)
    - Support for len() to get the vector size
    - No requirement for numpy integration

    This is the base for both VectorLikeProtocol (which adds numpy support)
    and ProteusLike (container protocol without numpy support).
    """

    # Comparison operations (vector semantics - return Self)
    def __lt__(self, other: t.Any) -> t.Self: ...
    def __le__(self, other: t.Any) -> t.Self: ...
    def __gt__(self, other: t.Any) -> t.Self: ...
    def __ge__(self, other: t.Any) -> t.Self: ...

    # Equality operations (vector semantics - return Self)
    # Note: These override object.__eq__ and object.__ne__ which return bool,
    # but vector types need to return Self for element-wise comparisons
    def __eq__(self, other: t.Any) -> t.Self: ...  # type: ignore[override]
    def __ne__(self, other: t.Any) -> t.Self: ...  # type: ignore[override]

    # Length support
    def __len__(self) -> int: ...


@t.runtime_checkable
class VectorLike[T](VectorOperations, SupportsArray, t.Protocol):
    """Protocol for vector-like objects that support array conversion.

    This protocol combines VectorOperations (vector-style arithmetic and comparisons)
    with SupportsArray (numpy array conversion). Use this for objects that are
    true vector-like types that can be converted to numpy arrays.

    Comparison operations return Self (vectorized semantics).

    Why VectorLikeProtocol vs numpy.ArrayLike?
    --------------------------------------------

    These solve fundamentally different problems:

    **numpy.ArrayLike**: "What can become an array?"
    - Purpose: Defines what inputs numpy functions will accept and convert to arrays
    - It's about data conversion compatibility
    - Just a Union of types that numpy knows how to convert (lists, objects with
      __array__, etc.)
    - Example: np.sum([1, 2, 3]) works because list is ArrayLike

    **VectorLikeProtocol**: "How do math operations behave?"
    - Purpose: Defines the behavioral contract for mathematical operations
    - It's about operation semantics and type preservation
    - Ensures that operations return the same type (Self), not just any array
    - Example: StochasticScalar([1, 2, 3]) + 5 returns StochasticScalar, not ndarray

    Key Differences:
    ----------------
    1. **Type Preservation**: VectorLikeProtocol ensures operations maintain the
       original type (e.g., StochasticScalar + StochasticScalar = StochasticScalar),
       while ArrayLike would lose this information (becoming ndarray).

    2. **Comparison Semantics**: VectorLikeProtocol defines that comparisons return
       vectorized results (Self) for element-wise operations, not scalar bool values.
       Example: StochasticScalar([1, 2, 3]) > 2 returns
       StochasticScalar([False, False, True])

    3. **Operation Contracts**: VectorLikeProtocol defines how mathematical operations
       should behave for custom types, while ArrayLike only cares about convertibility.

    Why Both Are Needed:
    --------------------
    Classes implementing VectorLikeProtocol should also be ArrayLike-compatible by
    implementing __array__() for numpy interoperability. This gives:
    - Type preservation through operations (VectorLikeProtocol benefit)
    - Numpy function compatibility (ArrayLike benefit)
    - Clear semantic distinction between scalar and vector operations

    Example Implementation:
    ----------------------
    class StochasticScalar:
        # VectorLikeProtocol: defines operation behavior
        def __add__(self, other) -> Self: ...  # Returns StochasticScalar
        def __gt__(self, other) -> Self: ...   # Returns StochasticScalar

        # ArrayLike compatibility: allows numpy function usage
        def __array__(self) -> np.ndarray: ... # Enables np.sum(stochastic_scalar)

    Numpy Compatibility:
    -------------------
    VectorLikeProtocol includes numpy compatibility methods:
    - __array__(): Enables conversion to numpy array for use with numpy functions
    - __len__(): Required for many numpy operations
    - __array_ufunc__(): (inherited from ArrayUfuncCapable) Enables proper handling
      of numpy universal functions while preserving type

    These methods ensure VectorLike objects can seamlessly integrate with numpy's
    ecosystem while maintaining their custom type semantics.
    """

    # All methods are inherited from:
    # - VectorOperations: comparison ops, __len__, __array_ufunc__
    # - SupportsArray: __array__
    # - ArithmeticProtocol (via VectorOperations): arithmetic ops


# Union type that includes both the basic numeric types and objects implementing
# the scalar protocol (comparison operations return bool)
NumericLike = Numeric | NumericProtocol


# FIXME: VectorLike should be generic VectorLike[T] to enable proper typing
# of math functions like sum(VectorLike[T]) -> T. This would allow:
# - sum(StochasticScalar) -> float
# - sum(ProteusVariable) -> StochasticScalar (when containing StochasticScalar)
# Currently blocked by need to refactor all VectorLike usage sites.


@t.runtime_checkable
class SequenceLike(t.Protocol[T_value]):
    """Protocol for sequence-like objects."""

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> T_value: ...
    def __iter__(self) -> t.Iterator[T_value]: ...
    def __contains__(self, value: object) -> bool: ...
    def __reversed__(self) -> t.Iterator[T_value]: ...
    def count(self, value: T_value) -> int: ...
    def index(self, value: T_value, start: int = 0, stop: int | None = None) -> int: ...


@t.runtime_checkable
class ProteusLike(VectorOperations, SequenceLike[T_value], t.Protocol):
    """Generic protocol for multi-dimensional stochastic variable containers.

    ProteusLike is a generic protocol that is covariant with respect to the type
    of values it contains. This ensures type safety and predictability:
    - ProteusLike[NumericLike] contains scalar values, operations return scalars
    - ProteusLike[VectorLike] contains vector values, operations return vectors
    - ProteusLike[StochasticScalar] contains stochastic scalars, preserves type

    Key Characteristics:
    --------------------
    1. **Type Preservation**: The type parameter T_value determines what type
       operations like mean() return. If you store scalars, you get scalars
       back. If you store vectors, you get vectors back.

    2. **Container Nature**: ProteusLike objects are containers that hold multiple
       stochastic variables or scalars, indexed by dimension names.

    3. **Mathematical Operations**: They support vectorized arithmetic operations
       that preserve the container type (ProteusLike[T] + ProteusLike[T] =
       ProteusLike[T]).

    4. **Iteration**: When iterated, they yield values of type T_value.

    5. **Nesting Support**: ProteusLike[ProteusLike[T]] enables hierarchical
       structures for multi-dimensional risk modeling.

    Usage Examples:
    --------------
    ```python
    # Type hints show the covariance
    def analyze_scalars(var: ProteusLike[NumericLike]) -> NumericLike:
        return var.mean()  # Returns NumericLike

    def analyze_vectors(var: ProteusLike[VectorLike]) -> VectorLike:
        return var.mean()  # Returns VectorLike

    def combine_risks(
        var1: ProteusLike[T_value],
        var2: ProteusLike[T_value]
    ) -> ProteusLike[T_value]:
        return var1 + var2  # Type preserved

    # Nested structures
    def process_nested(
        var: ProteusLike[ProteusLike[StochasticScalar]]
    ) -> ProteusLike[StochasticScalar]:
        # Work with hierarchical risk structures
        return var["region"]["peril"]
    ```

    Implementation Note:
    -------------------
    Classes should NOT inherit from this protocol. Instead, they should implement
    the required methods and attributes. The protocol is used purely for static
    type checking:

    ```python
    class ProteusVariable:  # Note: NO inheritance from ProteusLike
        def __init__(self, dim_name: str, values: dict[str, T]):
            self.dim_name = dim_name
            self.values = values
            self.n_sims = self._calculate_n_sims()

    ```
    """

    n_sims: int | None
    values: t.Mapping[str, T_value]

    def __getitem__(self, key: int | str) -> T_value:
        """Override SequenceLike.__getitem__ to support both int and str indexing.

        ProteusLike containers support both:
        - Integer indexing: Access by position in iteration order
        - String indexing: Access by dimension name (e.g., obj["region"])

        This override is necessary because SequenceLike protocol only supports
        int indexing, but ProteusLike containers are mapping-like and need
        string key access for dimension names.
        """
        ...

    def upsample(self, n_sims: int) -> ProteusLike[t.Any]:
        """Upsample the variable to match the specified number of simulations.

        Args:
            n_sims: The number of simulations to upsample to.

        Returns:
            A new instance of self with the upsampled values.
        """
        ...


T_distribution = t.TypeVar(
    "T_distribution", bound="ScipyNumeric | npt.NDArray[np.floating]"
)


class DistributionLike(t.Protocol[T_distribution]):
    """Generic protocol for distribution-like objects.

    DistributionLike is generic over the type of values it operates on, ensuring
    type consistency between inputs and outputs for mathematical operations.

    Type Parameter:
        T_distribution: The type of values the distribution operates on, bounded to
                       ScipyNumeric (float | int | np.floating | np.integer) or
                       npt.NDArray[np.floating] for vectorized operations.

    Key Properties:
    ---------------
    1. **Input-Output Consistency**: cdf() and invcdf() preserve input types
       - DistributionLike[float].cdf(x: float) -> float
       - DistributionLike[NDArray].cdf(x: NDArray) -> NDArray

    2. **Scalar and Vector Support**: Same distribution can work with both:
       - Scalar inputs: Individual probability calculations
       - Array inputs: Vectorized calculations across multiple values

    3. **Type Safety**: Generic parameter prevents mixing incompatible types
       and ensures mathematical operations maintain type consistency.

    Usage Examples:
    --------------
    ```python
    # Scalar distribution operations
    def eval_at_point(dist: DistributionLike[float], value: float) -> float:
        return dist.cdf(value)  # Returns float

    # Vectorized distribution operations
    def eval_array(dist: DistributionLike[NDArray], values: NDArray) -> NDArray:
        return dist.cdf(values)  # Returns NDArray

    # Generate always returns VectorLike (StochasticScalar-like object)
    def sample_distribution(dist: DistributionLike[Any]) -> VectorLike:
        return dist.generate(n_sims=1000)
    ```
    """

    def cdf(self, x: T_distribution) -> T_distribution:
        """Compute cumulative distribution function."""
        ...

    def invcdf(self, u: T_distribution) -> T_distribution:
        """Compute inverse cumulative distribution function."""
        ...

    def generate(
        self, n_sims: int | None = None, rng: np.random.Generator | None = None
    ) -> ProteusLike[T_distribution]:
        """Generate random samples from the distribution.

        Parameters:
            n_sims (int, optional): Number of simulations. Uses config.n_sims if None.
            rng (np.random.Generator, optional): Random number generator.

        Returns:
            Generated samples.
        """
        ...

"""Type definitions and protocols for the PAL library.

Defines common type aliases, protocols, and configuration classes used
throughout the library for type safety and consistency.
"""
import dataclasses
import typing as t

import numpy.typing as npt
from numpy.lib.mixins import NDArrayOperatorsMixin

from ._maths import xp as np

__all___ = [
    "Config",
    "DistributionLike",
    "Numeric",
    "NumericLike",
    "NumericLikeNDArrayOperatorsMixin",
    "ScipyNumeric",
]

Numeric = float | int | np.number[t.Any]

# Type alias for scipy special functions and numpy random generators
# These functions expect more restrictive types than our general Numeric type.
# They don't accept complex numbers, _NumericProtocol objects, or general
# np.number types.
ScipyNumeric = float | int | np.floating | np.integer

T_co = t.TypeVar("T_co", covariant=True)


@dataclasses.dataclass
class Config:
    """Configuration class for PAL."""

    n_sims: int = 10000
    seed: int = 123456789
    rng: np.random.Generator = np.random.default_rng(seed)


# maybe don't need this could use numbers.Number instead?
@t.runtime_checkable
class NumericProtocol(t.Protocol):
    """Protocol for objects that support numeric operations.

    Defines arithmetic, comparison, and equality operations.
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

    # Comparison operations
    def __lt__(self, other: t.Any) -> bool: ...
    def __le__(self, other: t.Any) -> bool: ...
    def __gt__(self, other: t.Any) -> bool: ...
    def __ge__(self, other: t.Any) -> bool: ...

    # Equality operations
    def __eq__(self, other: t.Any) -> bool: ...
    def __ne__(self, other: t.Any) -> bool: ...


# Union type that includes both the basic numeric types and objects implementing
# the protocol
NumericLike = Numeric | NumericProtocol


@t.runtime_checkable
class SequenceLike(t.Protocol[T_co]):
    """Protocol for sequence-like objects.

    This follows the same interface as collections.abc.Sequence, which requires
    __getitem__ to accept integer indices for positional access.
    """

    def __getitem__(self, key: int) -> T_co: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> t.Iterator[T_co]: ...


@t.runtime_checkable
class ArrayUfuncCapable(t.Protocol):
    """Protocol for objects that support numpy's __array_ufunc__ interface."""

    def __array_ufunc__(
        self,
        ufunc: t.Any,
        method: t.Literal[
            "__call__", "reduce", "reduceat", "accumulate", "outer", "at"
        ],
        *inputs: t.Any,
        **kwargs: t.Any,
    ) -> t.Any: ...


class ProteusLike(
    NumericProtocol, SequenceLike[NumericLike], ArrayUfuncCapable, t.Protocol
):
    """Protocol for ProteusVariable-like objects that support simulation operations."""

    n_sims: int | None
    values: t.Any

    @t.overload
    def sum(self) -> NumericLike: ...
    @t.overload
    def sum(self, dimensions: list[str]) -> NumericLike: ...

    def sum(self, dimensions: list[str] | None = None) -> NumericLike:
        """Sum the variables across the specified dimensions."""
        ...

    def mean(self) -> NumericLike:
        """Return the mean of the variable across the simulation dimension."""
        ...

    def upsample(self, n_sims: int) -> t.Self:
        """Upsample the variable to match the specified number of simulations.

        Args:
            n_sims: The number of simulations to upsample to.

        Returns:
            A new instance of self with the upsampled values.
        """
        ...


class DistributionLike(t.Protocol):
    """Protocol for distribution-like objects."""

    @t.overload
    def cdf(self, x: ScipyNumeric) -> ScipyNumeric: ...

    @t.overload
    def cdf(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]: ...

    @t.overload
    def invcdf(self, u: ScipyNumeric) -> ScipyNumeric: ...

    @t.overload
    def invcdf(self, u: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]: ...

    def generate(
        self, n_sims: int | None = None, rng: np.random.Generator | None = None
    ) -> ProteusLike:
        """Generate random samples from the distribution.

        Parameters:
            n_sims (int, optional): Number of simulations. Uses config.n_sims if None.
            rng (np.random.Generator, optional): Random number generator.

        Returns:
            Generated samples.
        """
        ...


# Type annotation to tell mypy that NDArrayOperatorsMixin provides NumericLike interface
if t.TYPE_CHECKING:

    class NumericLikeNDArrayOperatorsMixin(NDArrayOperatorsMixin, NumericProtocol):
        """Type stub to tell mypy that NDArrayOperatorsMixin satisfies NumericLike."""

        pass
else:
    NumericLikeNDArrayOperatorsMixin = NDArrayOperatorsMixin

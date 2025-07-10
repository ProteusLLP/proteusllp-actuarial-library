import dataclasses
import typing as t

from ._maths import xp as np

# Type annotation to tell mypy that NDArrayOperatorsMixin provides NumericLike interface
if t.TYPE_CHECKING:
    from numpy.lib.mixins import NDArrayOperatorsMixin

    class _NumericLikeNDArrayOperatorsMixin(NDArrayOperatorsMixin, NumericLike):
        """Type stub to tell mypy that NDArrayOperatorsMixin satisfies NumericLike."""

        pass
else:
    pass

Numeric = t.Union[float, int, np.floating[t.Any], np.integer[t.Any]]

T_co = t.TypeVar("T_co", covariant=True)


@dataclasses.dataclass
class Config:
    """Configuration class for PAL."""

    n_sims: int = 10000
    seed: int = 123456789
    rng: np.random.Generator = np.random.default_rng(seed)


@t.runtime_checkable
class NumericLike(t.Protocol):
    """Protocol for objects that support numeric operations (arithmetic, comparison, equality)."""

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


@t.runtime_checkable
class SequenceLike(t.Protocol[T_co]):
    """Protocol for sequence-like objects."""

    def __getitem__(self, key: int) -> T_co: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> t.Iterator[T_co]: ...


@t.runtime_checkable
class ArrayUfuncCapable(t.Protocol[T_co]):
    """Protocol for objects that support numpy's __array_ufunc__ interface."""

    def __array_ufunc__(
        self, ufunc: t.Any, method: str, *inputs: t.Any, **kwargs: t.Any
    ) -> T_co: ...


class ProteusLike(
    NumericLike, SequenceLike[NumericLike], ArrayUfuncCapable[t.Any], t.Protocol
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

    def mean(self) -> t.Self:
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

    def cdf(self, x: Numeric) -> Numeric:
        """Compute the cumulative distribution function at x."""
        ...

    def invcdf(self, u: Numeric) -> Numeric:
        """Compute the inverse cumulative distribution function at u."""
        ...

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

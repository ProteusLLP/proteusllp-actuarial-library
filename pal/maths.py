"""Type-safe math functions that preserve PAL custom types.

This module provides wrappers around numpy math functions that preserve
PAL's custom types (StochasticScalar, etc.) with explicit type information
for type checkers. Import as 'pnp' to mimic numpy usage patterns.
"""

import typing as t
from numbers import Number

import numpy.typing as npt

from ._maths import xp

# Import types only during type checking to avoid circular imports
# (StochasticScalar may import from _maths, which would create a cycle)
if t.TYPE_CHECKING:
    from .stochastic_scalar import StochasticScalar
    from .frequency_severity import FreqSevSims

T = t.TypeVar("T")


@t.overload
def exp[T](x: T) -> T: ...


def exp(x: t.Any) -> t.Any:
    """Exponential function that preserves custom PAL types."""
    return xp.exp(x)


# Reducing functions - these aggregate to scalars
@t.overload
def sum(x: "StochasticScalar") -> float: ...


@t.overload
def sum(x: t.Any) -> t.Any: ...


def sum(x: t.Any) -> t.Any:
    """Sum function that works with PAL types."""
    return xp.sum(x)


@t.overload
def mean(x: "StochasticScalar") -> float: ...


@t.overload
def mean(x: "FreqSevSims") -> float: ...


@t.overload
def mean(x: t.Any) -> t.Any: ...


def mean(x: t.Any) -> t.Any:
    """Mean function that works with PAL types."""
    # Special handling for FreqSevSims - aggregate first, then take mean
    if hasattr(x, 'aggregate') and hasattr(x, '__class__') and 'FreqSevSims' in str(x.__class__):
        return xp.mean(x.aggregate())
    return xp.mean(x)


@t.overload
def std(x: "StochasticScalar") -> float: ...


@t.overload
def std(x: t.Any) -> t.Any: ...


def std(x: t.Any) -> t.Any:
    """Standard deviation function that works with PAL types."""
    return xp.std(x)


@t.overload
def var(x: "StochasticScalar") -> float: ...


@t.overload
def var(x: t.Any) -> t.Any: ...


def var(x: t.Any) -> t.Any:
    """Variance function that works with PAL types."""
    return xp.var(x)


@t.overload
def percentile(x: "StochasticScalar", q: float) -> float: ...


@t.overload
def percentile(x: "StochasticScalar", q: t.Sequence[float]) -> t.Any: ...


@t.overload
def percentile(x: t.Any, q: t.Any) -> t.Any: ...


def percentile(x: t.Any, q: t.Any) -> t.Any:
    """Percentile function that works with PAL types."""
    return xp.percentile(x, q)


@t.overload
def min(x: "StochasticScalar") -> float: ...


@t.overload
def min(x: t.Any) -> t.Any: ...


def min(x: t.Any) -> t.Any:
    """Min function that works with PAL types."""
    return xp.min(x)


@t.overload
def max(x: "StochasticScalar") -> Number: ...


@t.overload
def max(x: t.Any) -> t.Any: ...


def max(x: t.Any) -> t.Any:
    """Max function that works with PAL types."""
    return xp.max(x)


@t.overload
def where(
    condition: t.Any, x: "StochasticScalar", y: "StochasticScalar"
) -> "StochasticScalar": ...


@t.overload
def where(
    condition: t.Any, x: "StochasticScalar", y: float | int
) -> "StochasticScalar": ...


@t.overload
def where(
    condition: t.Any, x: float | int, y: "StochasticScalar"
) -> "StochasticScalar": ...


@t.overload
def where(
    condition: t.Any, x: npt.NDArray[t.Any], y: npt.NDArray[t.Any]
) -> npt.NDArray[t.Any]: ...


@t.overload
def where(condition: t.Any, x: t.Any, y: t.Any) -> t.Any: ...


def where(condition: t.Any, x: t.Any, y: t.Any) -> t.Any:
    """Conditional selection that preserves PAL types."""
    return xp.where(condition, x, y)


# Additional functions for contracts.py and other modules
@t.overload
def minimum(x: "StochasticScalar", y: "StochasticScalar") -> "StochasticScalar": ...


@t.overload
def minimum(x: "StochasticScalar", y: float | int) -> "StochasticScalar": ...


@t.overload
def minimum(x: float | int, y: "StochasticScalar") -> "StochasticScalar": ...


@t.overload
def minimum(x: t.Any, y: t.Any) -> t.Any: ...


def minimum(x: t.Any, y: t.Any) -> t.Any:
    """Element-wise minimum that preserves PAL types."""
    return xp.minimum(x, y)


@t.overload
def maximum(x: "StochasticScalar", y: "StochasticScalar") -> "StochasticScalar": ...


@t.overload
def maximum(x: "StochasticScalar", y: float | int) -> "StochasticScalar": ...


@t.overload
def maximum(x: float | int, y: "StochasticScalar") -> "StochasticScalar": ...


@t.overload
def maximum(x: t.Any, y: t.Any) -> t.Any: ...


def maximum(x: t.Any, y: t.Any) -> t.Any:
    """Element-wise maximum that preserves PAL types."""
    return xp.maximum(x, y)


@t.overload
def cumsum(x: "StochasticScalar") -> "StochasticScalar": ...


@t.overload
def cumsum(x: t.Any) -> t.Any: ...


def cumsum(x: t.Any) -> t.Any:
    """Cumulative sum that preserves PAL types."""
    return xp.cumsum(x)


@t.overload
def floor(x: "StochasticScalar") -> "StochasticScalar": ...


@t.overload
def floor(x: t.Any) -> t.Any: ...


def floor(x: t.Any) -> t.Any:
    """Floor function that preserves PAL types."""
    return xp.floor(x)

"""Type-safe math functions that preserve PAL custom types.

This module provides wrappers around numpy math functions that preserve
PAL's custom types (StochasticScalar, etc.) with explicit type information
for type checkers. Import as 'pnp' to mimic numpy usage patterns.
"""

import typing as t
from numbers import Number

from ._maths import xp

# Import types only during type checking to avoid circular imports
# (StochasticScalar may import from _maths, which would create a cycle)
if t.TYPE_CHECKING:
    from .stochastic_scalar import StochasticScalar

T = t.TypeVar("T")


@t.overload
def exp(x: T) -> T: ...


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
def mean(x: t.Any) -> t.Any: ...


def mean(x: t.Any) -> t.Any:
    """Mean function that works with PAL types."""
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

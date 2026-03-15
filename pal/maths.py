"""Math functions that preserve PAL custom types.

This module provides wrappers around numpy math functions that preserve
PAL's custom types (StochasticScalar, etc.). Import as 'pnp' to mimic
numpy usage patterns.

Type signatures are in maths.pyi.
"""

from __future__ import annotations

import typing as t

# third party
import numpy as np


def exp(x: t.Any) -> t.Any:
    """Exponential function that preserves custom PAL types."""
    return np.exp(x)


def sum(x: t.Any) -> t.Any:
    """Sum function that works with PAL types."""
    return np.sum(x)


def mean(x: t.Any) -> t.Any:
    """Mean function that works with PAL types.

    All PAL types implement the numpy array protocol, so this just
    delegates to numpy's mean function which will dispatch to the
    appropriate __array_function__ or __array__ method.
    """
    return np.mean(x)  # type: ignore[reportUnknownVariableType]


def std(x: t.Any) -> t.Any:
    """Standard deviation function that works with PAL types."""
    return np.std(x)


def var(x: t.Any) -> t.Any:
    """Variance function that works with PAL types."""
    return np.var(x)


def percentile(x: t.Any, q: t.Any) -> t.Any:
    """Percentile function that works with PAL types."""
    return np.percentile(x, q)  # type: ignore[reportUnknownVariableType]


def min(x: t.Any) -> t.Any:
    """Min function that works with PAL types."""
    return np.min(x)


def max(x: t.Any) -> t.Any:
    """Max function that works with PAL types."""
    return np.max(x)


def where(condition: t.Any, x: t.Any, y: t.Any) -> t.Any:
    """Conditional selection that preserves PAL types."""
    return np.where(condition, x, y)  # pyright: ignore[reportUnknownVariableType]


def safe_divide(
    numerator: t.Any,
    denominator: t.Any,
    default: t.Any,
) -> t.Any:
    """Divide numerator by denominator, returning default where denominator is 0.

    Works with PAL types (StochasticScalar, FreqSevSims) and plain
    numpy arrays/scalars.

    Args:
        numerator: The numerator value(s).
        denominator: The denominator value(s).
        default: Value to use where denominator equals zero.

    Returns:
        The result of the division, with default substituted
        where division by zero would occur.

    Examples:
        >>> from pal.stochastic_scalar import StochasticScalar
        >>> import pal.maths as pnp
        >>> x = StochasticScalar([10., 20., 30.])
        >>> y = StochasticScalar([2., 0., 5.])
        >>> result = pnp.safe_divide(x, y, 0.0)
        >>> result.values
        array([5., 0., 6.])
    """
    return where(denominator != 0, numerator / denominator, default)


def minimum(x: t.Any, y: t.Any) -> t.Any:
    """Element-wise minimum that preserves PAL types."""
    return np.minimum(x, y)


def maximum(x: t.Any, y: t.Any) -> t.Any:
    """Element-wise maximum that preserves PAL types."""
    return np.maximum(x, y)


def cumsum(x: t.Any) -> t.Any:
    """Cumulative sum that preserves PAL types.

    When given a list of StochasticScalar objects, stacks their values
    into a 2D array and computes cumsum along axis 0.
    """
    if isinstance(x, list) and len(x) > 0 and hasattr(x[0], "values"):  # type: ignore[reportUnknownMemberType]
        return np.cumsum(  # type: ignore[reportUnknownVariableType]
            np.stack([item.values for item in x], axis=0),  # type: ignore[reportUnknownMemberType, reportUnknownVariableType]
            axis=0,
        )
    return np.cumsum(x)  # type: ignore[reportUnknownVariableType]


def floor(x: t.Any) -> t.Any:
    """Floor function that preserves PAL types."""
    return np.floor(x)


def all(x: t.Any) -> bool:
    """Check if all elements are True."""
    return bool(np.all(x))

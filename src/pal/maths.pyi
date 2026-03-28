"""Type stubs for maths module."""

from __future__ import annotations

import typing as t

import numpy.typing as npt

from .frequency_severity import FreqSevSims
from .stochastic_scalar import StochasticScalar
from .variables import ProteusVariable

T = t.TypeVar("T")

# exp
@t.overload
def exp(
    x: T,
) -> T: ...
@t.overload
def exp(x: t.Any) -> t.Any: ...

# sum
@t.overload
def sum(x: StochasticScalar) -> float: ...
@t.overload
def sum(x: t.Any) -> t.Any: ...

# mean
@t.overload
def mean(x: StochasticScalar) -> float: ...
@t.overload
def mean(x: FreqSevSims) -> float: ...
@t.overload
def mean(x: ProteusVariable[T]) -> ProteusVariable[T]: ...
@t.overload
def mean(x: t.Any) -> t.Any: ...

# std
@t.overload
def std(x: StochasticScalar) -> float: ...
@t.overload
def std(x: t.Any) -> t.Any: ...

# var
@t.overload
def var(x: StochasticScalar) -> float: ...
@t.overload
def var(x: t.Any) -> t.Any: ...

# percentile
@t.overload
def percentile(x: StochasticScalar, q: float) -> float: ...
@t.overload
def percentile(x: StochasticScalar, q: t.Sequence[float]) -> t.Any: ...
@t.overload
def percentile(x: t.Any, q: t.Any) -> t.Any: ...

# min
@t.overload
def min(x: StochasticScalar) -> float | int: ...
@t.overload
def min(x: t.Any) -> t.Any: ...

# max
@t.overload
def max(x: StochasticScalar) -> float | int: ...
@t.overload
def max(x: t.Any) -> t.Any: ...

# where
@t.overload
def where(condition: t.Any, x: StochasticScalar, y: StochasticScalar) -> StochasticScalar: ...
@t.overload
def where(condition: t.Any, x: StochasticScalar, y: float | int) -> StochasticScalar: ...
@t.overload
def where(condition: t.Any, x: float | int, y: StochasticScalar) -> StochasticScalar: ...
@t.overload
def where(condition: t.Any, x: t.Any, y: t.Any) -> t.Any: ...

# safe_divide
@t.overload
def safe_divide(
    numerator: StochasticScalar,
    denominator: StochasticScalar,
    default: StochasticScalar | float | int,
) -> StochasticScalar: ...
@t.overload
def safe_divide(
    numerator: StochasticScalar,
    denominator: float | int,
    default: StochasticScalar | float | int,
) -> StochasticScalar: ...
@t.overload
def safe_divide(
    numerator: float | int,
    denominator: StochasticScalar,
    default: StochasticScalar | float | int,
) -> StochasticScalar: ...
@t.overload
def safe_divide(
    numerator: FreqSevSims,
    denominator: FreqSevSims | float | int,
    default: FreqSevSims | float | int,
) -> FreqSevSims: ...
@t.overload
def safe_divide(
    numerator: float | int,
    denominator: FreqSevSims,
    default: FreqSevSims | float | int,
) -> FreqSevSims: ...
@t.overload
def safe_divide(
    numerator: t.Any,
    denominator: t.Any,
    default: t.Any,
) -> t.Any: ...

# minimum
@t.overload
def minimum(x: StochasticScalar, y: StochasticScalar) -> StochasticScalar: ...
@t.overload
def minimum(x: FreqSevSims, y: float | int) -> FreqSevSims: ...
@t.overload
def minimum(x: float | int, y: FreqSevSims) -> FreqSevSims: ...
@t.overload
def minimum(x: StochasticScalar, y: FreqSevSims) -> FreqSevSims: ...
@t.overload
def minimum(x: float | int, y: StochasticScalar) -> StochasticScalar: ...
@t.overload
def minimum(x: ProteusVariable[T], y: float | int) -> ProteusVariable[T]: ...
@t.overload
def minimum(x: float | int, y: ProteusVariable[T]) -> ProteusVariable[T]: ...
@t.overload
def minimum(x: t.Any, y: t.Any) -> t.Any: ...

# maximum
@t.overload
def maximum(x: StochasticScalar, y: StochasticScalar) -> StochasticScalar: ...
@t.overload
def maximum(x: StochasticScalar, y: float | int) -> StochasticScalar: ...
@t.overload
def maximum(x: float | int, y: StochasticScalar) -> StochasticScalar: ...
@t.overload
def maximum(x: FreqSevSims, y: float | int) -> FreqSevSims: ...
@t.overload
def maximum(x: float | int, y: FreqSevSims) -> FreqSevSims: ...
@t.overload
def maximum(x: ProteusVariable[T], y: float | int) -> ProteusVariable[T]: ...
@t.overload
def maximum(x: float | int, y: ProteusVariable[T]) -> ProteusVariable[T]: ...
@t.overload
def maximum(x: t.Any, y: t.Any) -> t.Any: ...

# cumsum
@t.overload
def cumsum(x: StochasticScalar) -> StochasticScalar: ...
@t.overload
def cumsum(
    x: list[StochasticScalar],
) -> npt.NDArray[t.Any]: ...
@t.overload
def cumsum(x: t.Any) -> t.Any: ...

# floor
@t.overload
def floor(x: StochasticScalar) -> StochasticScalar: ...
@t.overload
def floor(x: t.Any) -> t.Any: ...

# all
@t.overload
def all(x: StochasticScalar) -> bool: ...
@t.overload
def all(x: ProteusVariable[t.Any]) -> bool: ...
@t.overload
def all(x: t.Any) -> bool: ...

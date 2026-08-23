"""Backend-neutral array assertions for CPU and GPU test runs."""

from __future__ import annotations

import typing as t

import numpy as np

from pal._maths import asnumpy
from pal.couplings import ProteusStochasticVariable


def _host(value: t.Any) -> t.Any:
    """Copy stochastic/backend arrays to the host before NumPy assertions."""
    if isinstance(value, ProteusStochasticVariable):
        value = value.values
    if isinstance(value, tuple):
        return tuple(_host(item) for item in value)
    if isinstance(value, list):
        return [_host(item) for item in value]
    return asnumpy(value)


def host_values(value: t.Any) -> np.ndarray[t.Any, t.Any]:
    """Return a host array for a PAL value or backend array."""
    return asnumpy(getattr(value, "values", value))


def array_equal(actual: t.Any, expected: t.Any, *args: t.Any, **kwargs: t.Any) -> bool:
    """Return whether two backend-neutral arrays are equal."""
    return bool(np.array_equal(_host(actual), _host(expected), *args, **kwargs))


def allclose(actual: t.Any, expected: t.Any, *args: t.Any, **kwargs: t.Any) -> bool:
    """Return whether two backend-neutral arrays are close."""
    return bool(np.allclose(_host(actual), _host(expected), *args, **kwargs))


def assert_array_equal(actual: t.Any, expected: t.Any, *args: t.Any, **kwargs: t.Any) -> None:
    """Assert equality after copying backend arrays to the host."""
    np.testing.assert_array_equal(_host(actual), _host(expected), *args, **kwargs)


def assert_allclose(actual: t.Any, expected: t.Any, *args: t.Any, **kwargs: t.Any) -> None:
    """Assert closeness after copying backend arrays to the host."""
    np.testing.assert_allclose(_host(actual), _host(expected), *args, **kwargs)


def assert_array_almost_equal(actual: t.Any, expected: t.Any, *args: t.Any, **kwargs: t.Any) -> None:
    """Assert approximate equality after copying backend arrays to the host."""
    np.testing.assert_array_almost_equal(_host(actual), _host(expected), *args, **kwargs)

"""Regression tests for NumPy dispatch to PAL's active array backend."""

import numpy as np

from pal import maths as pnp
from pal._maths import xp
from pal.frequency_severity import FreqSevSims
from pal.stochastic_scalar import StochasticScalar
from tests._assertions import array_equal


def test_numpy_ufunc_normalizes_mixed_array_backends() -> None:
    """A NumPy ufunc keeps stochastic results on the active backend."""
    variable = StochasticScalar([1.0, 2.0, 3.0])

    result = np.add(variable, np.array([10.0, 20.0, 30.0]))

    assert isinstance(result, StochasticScalar)
    assert isinstance(result.values, xp.ndarray)
    assert array_equal(result, [11.0, 22.0, 33.0])


def test_numpy_array_function_normalizes_mixed_array_backends() -> None:
    """A NumPy array function accepts mixed stochastic and NumPy operands."""
    variable = StochasticScalar([1.0, 2.0, 3.0])

    assert np.array_equal(variable, np.array([1.0, 2.0, 3.0]))


def test_numpy_where_normalizes_scalar_condition() -> None:
    """NumPy where promotes scalar conditions before CuPy dispatch."""
    variable = StochasticScalar([1.0, 2.0, 3.0])

    result = np.where(True, variable, 0.0)

    assert isinstance(result, StochasticScalar)
    assert isinstance(result.values, xp.ndarray)
    assert array_equal(result, variable)


def test_numpy_array_protocol_returns_host_array() -> None:
    """The NumPy array protocol always returns a genuine NumPy array."""
    result = np.asarray(StochasticScalar([1.0, 2.0, 3.0]))

    assert isinstance(result, np.ndarray)
    assert array_equal(result, [1.0, 2.0, 3.0])


def test_numpy_reduction_returns_scalar() -> None:
    """A one-dimensional reduction is returned as a scalar, not re-wrapped."""
    result = np.sum(StochasticScalar([1.0, 2.0, 3.0]))

    assert result == 6.0


def test_freqsev_ufunc_normalizes_numpy_simulation_array() -> None:
    """Per-simulation NumPy operands are promoted and expanded per event."""
    simulations = FreqSevSims([0, 0, 1, 2], [1.0, 2.0, 3.0, 4.0], 3)

    result = simulations + np.array([10.0, 20.0, 30.0])

    assert isinstance(result.values, xp.ndarray)
    assert array_equal(result, [11.0, 12.0, 23.0, 34.0])


def test_cumsum_list_stays_on_active_backend() -> None:
    """Array construction for a stochastic cumsum stays on the backend."""
    result = pnp.cumsum([StochasticScalar([1, 2]), StochasticScalar([3, 4])])

    assert isinstance(result, xp.ndarray)
    assert array_equal(result, [[1, 2], [4, 6]])

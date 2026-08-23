"""Tests for analytical MBBEFD moments."""

import math

import pytest

from pal import MBBEFD
from pal._maths import xp as np
from pal.config import set_random_seed
from pal.stochastic_scalar import StochasticScalar


def test_mbbefd_mean_main_parameter_region() -> None:
    """The analytical mean should match the Bernegger formula."""
    g = 25.0
    b = 4.0
    expected = (1 - b) * math.log(g * b) / ((1 - g * b) * math.log(b))

    assert MBBEFD(g, b).mean() == pytest.approx(expected)


def test_mbbefd_mean_limiting_cases() -> None:
    """The analytical mean should use the continuous boundary limits."""
    assert MBBEFD(5.0, 1.0).mean() == pytest.approx(math.log(5.0) / 4.0)
    assert MBBEFD(4.0, 0.25).mean() == pytest.approx(0.75 / -math.log(0.25))
    assert MBBEFD(1.0, 2.0).mean() == 1.0
    assert MBBEFD(2.0, 0.0).mean() == 1.0


def test_mbbefd_mean_matches_simulation() -> None:
    """Random generation should reproduce the analytical mean."""
    set_random_seed(12345678910)
    dist = MBBEFD.from_c(3.0)
    sims = dist.generate(250_000)

    assert sims.mean() == pytest.approx(dist.mean(), abs=0.003)


def test_mbbefd_mean_preserves_stochastic_parameter_coupling() -> None:
    """A stochastic-parameter mean should stay in the same coupling group."""
    g = StochasticScalar([20.0, 25.0, 30.0])
    b = StochasticScalar([2.0, 3.0, 4.0])
    result = MBBEFD(g=g, b=b).mean()

    assert isinstance(result, StochasticScalar)
    assert np.all(result.values > 0)
    assert result.coupled_variable_group is g.coupled_variable_group
    assert result.coupled_variable_group is b.coupled_variable_group

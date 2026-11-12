"""Regression tests for stochastic distribution probability methods."""

import numpy as np
import pytest
from scipy import stats

from pal import StochasticScalar
from pal._maths import asnumpy
from pal.distributions import Gamma, LogNormal, Normal


@pytest.mark.parametrize("family", ["gamma", "normal", "lognormal"])
@pytest.mark.parametrize("method", ["cdf", "invcdf"])
@pytest.mark.parametrize("stochastic_argument", [False, True])
def test_probability_method_preserves_parameter_ancestry(family: str, method: str, stochastic_argument: bool) -> None:
    """Preserve both parameter groups and any stochastic evaluation argument."""
    first = StochasticScalar([1.0, 2.0, 3.0])
    second = StochasticScalar([0.5, 1.0, 1.5])
    if family == "gamma":
        dist = Gamma(alpha=first, theta=second)
        reference = stats.gamma(a=[1.0, 2.0, 3.0], scale=[0.5, 1.0, 1.5])
    elif family == "normal":
        dist = Normal(mu=first, sigma=second)
        reference = stats.norm(loc=[1.0, 2.0, 3.0], scale=[0.5, 1.0, 1.5])
    else:
        dist = LogNormal(mu=first, sigma=second)
        reference = stats.lognorm(s=[0.5, 1.0, 1.5], scale=np.exp([1.0, 2.0, 3.0]))

    value = 0.5 if method == "invcdf" else 2.0
    argument = StochasticScalar([value] * 3) if stochastic_argument else value
    result = dist.cdf(argument) if method == "cdf" else dist.invcdf(argument)
    expected = reference.cdf(value) if method == "cdf" else reference.ppf(value)
    assert isinstance(result, StochasticScalar)
    assert result.coupled_variable_group is first.coupled_variable_group
    assert result.coupled_variable_group is second.coupled_variable_group
    if isinstance(argument, StochasticScalar):
        assert result.coupled_variable_group is argument.coupled_variable_group
    assert asnumpy(result.values) == pytest.approx(expected)

"""Regression tests for Burr and Weibull probability methods."""

import numpy as np
import pytest
from scipy import stats

from pal import StochasticScalar
from pal._maths import asnumpy
from pal.distributions import Burr, Weibull


@pytest.mark.parametrize("family", ["burr", "weibull"])
@pytest.mark.parametrize("power", [0.5, 2.0, 10.0])
def test_probability_methods(family, power):
    if family == "burr":
        dist = Burr(power, 0.7, 2.0, 0.0)
        reference = stats.burr12(power, 0.7, scale=2)
    else:
        dist = Weibull(power, 2.0)
        reference = stats.weibull_min(power, scale=2)
    for x in [-np.inf, -1.0, 0.0, 1e-20, 0.1, 2.0, 1e100, np.inf]:
        assert dist.cdf(x) == pytest.approx(reference.cdf(x), rel=1e-10, abs=1e-100)
    for u in [0.0, 1e-20, 1e-12, 0.5, 1 - 1e-12, 1.0]:
        assert dist.invcdf(u) == pytest.approx(reference.ppf(u), rel=1e-10, abs=1e-100)


def test_burr_intermediate_overflow():
    dist = Burr(10.0, 0.001, 1.0, 0.0)
    expected = np.exp((-np.log1p(-0.9) / 0.001) / 10)
    assert dist.invcdf(0.9) == pytest.approx(expected, rel=1e-12)
    assert dist.cdf(expected) == pytest.approx(0.9)


@pytest.mark.parametrize("family", ["burr", "weibull"])
@pytest.mark.parametrize("method", ["cdf", "invcdf"])
@pytest.mark.parametrize("stochastic_argument", [False, True])
def test_stochastic_probability_methods(family, method, stochastic_argument):
    powers = np.array([0.5, 2.0, 3.0])
    scales = np.array([1.0, 2.0, 3.0])
    power, scale = (StochasticScalar(v) for v in (powers, scales))
    if family == "burr":
        dist = Burr(power, 0.7, scale, 0.0)
        reference = stats.burr12(powers, 0.7, scale=scales)
    else:
        dist = Weibull(power, scale)
        reference = stats.weibull_min(powers, scale=scales)
    value = 0.2
    argument = StochasticScalar([value] * 3) if stochastic_argument else value
    result = getattr(dist, method)(argument)
    expected = reference.cdf(value) if method == "cdf" else reference.ppf(value)
    assert isinstance(result, StochasticScalar)
    np.testing.assert_allclose(asnumpy(result.values), expected, rtol=1e-12)
    for param in (power, scale, argument):
        if isinstance(param, StochasticScalar):
            assert result.coupled_variable_group is param.coupled_variable_group


@pytest.mark.parametrize("family", ["burr", "weibull"])
@pytest.mark.parametrize("method", ["cdf", "invcdf"])
def test_incompatible_axes(family, method):
    power = StochasticScalar([1.0, 2.0, 3.0])
    scale = StochasticScalar([1.0, 2.0, 3.0])
    argument = StochasticScalar([0.2, 0.5])
    groups = [v.coupled_variable_group for v in (power, scale, argument)]
    dist = Burr(power, 0.7, scale, 0.0) if family == "burr" else Weibull(power, scale)
    with pytest.raises(ValueError):
        getattr(dist, method)(argument)
    for value, group in zip((power, scale, argument), groups):
        assert value.coupled_variable_group is group
    assert power.coupled_variable_group is not scale.coupled_variable_group

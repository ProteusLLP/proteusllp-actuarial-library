"""Regression tests for GPD and GEV probability methods."""

import numpy as np
import pytest
from scipy import stats

from pal import StochasticScalar
from pal._maths import asnumpy
from pal.distributions import GEV, GPD


@pytest.mark.parametrize("family", [GPD, GEV])
@pytest.mark.parametrize("shape", [-0.4, -1e-12, 0.0, 1e-12, 0.4])
def test_scalar_probability_methods(family, shape):
    dist = family(shape, 2.0, 1.0)
    reference = stats.genpareto(shape, loc=1, scale=2) if family is GPD else stats.genextreme(-shape, loc=1, scale=2)
    for x in [-np.inf, -10.0, 1.0, 2.0, 10.0, np.inf]:
        assert dist.cdf(x) == pytest.approx(reference.cdf(x), abs=1e-14)
    for u in [0.0, 1e-12, 0.1, 0.5, 0.999999999999, 1.0]:
        assert dist.invcdf(u) == pytest.approx(reference.ppf(u), rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("family", [GPD, GEV])
@pytest.mark.parametrize("method", ["cdf", "invcdf"])
@pytest.mark.parametrize("stochastic_argument", [False, True])
def test_mixed_shapes_and_coupling(family, method, stochastic_argument):
    shapes = np.array([-0.4, 0.0, 0.4, 1e-12])
    scales = np.array([1.0, 2.0, 3.0, 4.0])
    locations = np.array([0.0, 1.0, 2.0, 3.0])
    shape, scale, loc = (StochasticScalar(v) for v in (shapes, scales, locations))
    dist = family(shape, scale, loc)
    value = 0.99 if method == "invcdf" else 4.0
    argument = StochasticScalar([value] * 4) if stochastic_argument else value
    reference = (
        stats.genpareto(shapes, loc=locations, scale=scales)
        if family is GPD
        else stats.genextreme(-shapes, loc=locations, scale=scales)
    )
    result = getattr(dist, method)(argument)
    expected = reference.ppf(value) if method == "invcdf" else reference.cdf(value)
    assert isinstance(result, StochasticScalar)
    np.testing.assert_allclose(asnumpy(result.values), expected, rtol=1e-10, atol=1e-12)
    for param in (shape, scale, loc, argument):
        if isinstance(param, StochasticScalar):
            assert result.coupled_variable_group is param.coupled_variable_group


@pytest.mark.parametrize("family", [GPD, GEV])
@pytest.mark.parametrize("method", ["cdf", "invcdf"])
def test_incompatible_axes_do_not_merge_groups(family, method):
    shape = StochasticScalar([-0.4, 0.0, 0.4])
    scale = StochasticScalar([1.0, 2.0, 3.0])
    argument = StochasticScalar([0.2, 0.5])
    groups = [v.coupled_variable_group for v in (shape, scale, argument)]
    with pytest.raises(ValueError):
        getattr(family(shape, scale, 0.0), method)(argument)
    for value, group in zip((shape, scale, argument), groups):
        assert value.coupled_variable_group is group
    assert shape.coupled_variable_group is not scale.coupled_variable_group

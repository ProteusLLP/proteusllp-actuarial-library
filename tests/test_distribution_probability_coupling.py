"""Regression tests for stochastic distribution probability methods."""

import numpy as np
import pytest
from scipy import special, stats

from pal import StochasticScalar
from pal._maths import asnumpy
from pal.distributions import (
    Gamma,
    InverseGamma,
    InverseGaussian,
    Logistic,
    LogNormal,
    Normal,
    Pareto,
    Poisson,
    StudentsT,
)


@pytest.mark.parametrize("family", ["gamma", "normal", "lognormal", "logistic", "inversegamma", "pareto"])
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
    elif family == "logistic":
        dist = Logistic(mu=first, sigma=second)
        reference = stats.logistic(loc=[1.0, 2.0, 3.0], scale=[0.5, 1.0, 1.5])
    elif family == "inversegamma":
        dist = InverseGamma(alpha=first, theta=second)
        reference = stats.invgamma(a=[1.0, 2.0, 3.0], scale=[0.5, 1.0, 1.5])
    elif family == "pareto":
        dist = Pareto(shape=first, scale=second)
        reference = stats.pareto(b=[1.0, 2.0, 3.0], scale=[0.5, 1.0, 1.5])
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


@pytest.mark.parametrize("method", ["cdf", "invcdf"])
@pytest.mark.parametrize("stochastic_argument", [False, True])
def test_poisson_probability_preserves_parameter_ancestry(method: str, stochastic_argument: bool) -> None:
    """Preserve the stochastic mean and evaluation argument, including the SciPy fallback."""
    mean = StochasticScalar([1.0, 2.0, 3.0])
    dist = Poisson(mean)
    value = 2.0 if method == "cdf" else 0.75
    argument = StochasticScalar([value] * 3) if stochastic_argument else value
    result = dist.cdf(argument) if method == "cdf" else dist.invcdf(argument)
    expected = special.pdtr(value, [1.0, 2.0, 3.0]) if method == "cdf" else stats.poisson.ppf(value, [1.0, 2.0, 3.0])
    assert isinstance(result, StochasticScalar)
    assert result.coupled_variable_group is mean.coupled_variable_group
    if isinstance(argument, StochasticScalar):
        assert result.coupled_variable_group is argument.coupled_variable_group
    assert asnumpy(result.values) == pytest.approx(expected)


@pytest.mark.parametrize("method", ["cdf", "invcdf"])
@pytest.mark.parametrize("stochastic_argument", [False, True])
def test_students_t_probability_preserves_parameter_ancestry(method: str, stochastic_argument: bool) -> None:
    """Preserve all three parameter groups in Student's t probability methods."""
    nu = StochasticScalar([3.0, 5.0, 8.0])
    mu = StochasticScalar([1.0, 2.0, 3.0])
    sigma = StochasticScalar([0.5, 1.0, 1.5])
    dist = StudentsT(nu, mu, sigma)
    reference = stats.t(df=[3.0, 5.0, 8.0], loc=[1.0, 2.0, 3.0], scale=[0.5, 1.0, 1.5])
    value = 0.75 if method == "invcdf" else 2.0
    argument = StochasticScalar([value] * 3) if stochastic_argument else value
    result = dist.cdf(argument) if method == "cdf" else dist.invcdf(argument)
    expected = reference.cdf(value) if method == "cdf" else reference.ppf(value)
    assert isinstance(result, StochasticScalar)
    for param in (nu, mu, sigma, argument):
        if isinstance(param, StochasticScalar):
            assert result.coupled_variable_group is param.coupled_variable_group
    assert asnumpy(result.values) == pytest.approx(expected)


@pytest.mark.parametrize("stochastic_argument", [False, True])
def test_inverse_gaussian_cdf_preserves_parameter_ancestry(stochastic_argument: bool) -> None:
    """Preserve both inverse Gaussian parameter groups."""
    mu = StochasticScalar([1.0, 2.0, 3.0])
    shape = StochasticScalar([2.0, 3.0, 4.0])
    dist = InverseGaussian(mu, shape)
    reference = stats.invgauss(mu=np.array([1.0, 2.0, 3.0]) / [2.0, 3.0, 4.0], scale=[2.0, 3.0, 4.0])
    argument = StochasticScalar([2.0] * 3) if stochastic_argument else 2.0
    result = dist.cdf(argument)
    assert isinstance(result, StochasticScalar)
    for param in (mu, shape, argument):
        if isinstance(param, StochasticScalar):
            assert result.coupled_variable_group is param.coupled_variable_group
    assert asnumpy(result.values) == pytest.approx(reference.cdf(2.0))

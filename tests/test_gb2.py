"""Tests of the GB2 claims severity distribution."""

import math

import pytest
from scipy.special import beta as beta_function
from scipy.stats import betaprime

from pal import StochasticScalar, set_random_seed
from pal._maths import xp
from pal.distributions import GB2, Burr, ContinuousDistributionGenerator, LogLogistic


@pytest.mark.parametrize(
    "a,b,p,q",
    [(2.0, 100000.0, 1.5, 3.0), (0.75, 10.0, 3.0, 2.5), (3.0, 750.0, 0.65, 0.9)],
)
def test_gb2_matches_independent_beta_prime_reference(a: float, b: float, p: float, q: float) -> None:
    """Compare probabilities and quantiles with the independently parameterised beta-prime family."""
    dist = GB2(a=a, b=b, p=p, q=q)
    for u in (0.001, 0.1, 0.5, 0.9, 0.99, 0.999):
        expected = b * betaprime.ppf(u, p, q) ** (1 / a)
        actual = dist.invcdf(u)
        assert actual == pytest.approx(expected, rel=1e-7)
        assert dist.cdf(expected) == pytest.approx(betaprime.cdf((expected / b) ** a, p, q), rel=1e-8)


@pytest.mark.parametrize("q", [0.1, 0.3, 0.9])
def test_gb2_extreme_upper_quantile(q: float) -> None:
    """Even small q must return a finite high quantile when mathematically finite."""
    dist = GB2(a=2.0, b=100.0, p=1.5, q=q)
    probability = 1 - 1e-6
    expected = 100.0 * betaprime.ppf(probability, 1.5, q) ** 0.5
    actual = dist.invcdf(probability)
    assert math.isfinite(float(actual))
    assert actual == pytest.approx(expected, rel=1e-7)


def test_gb2_boundaries_and_special_cases() -> None:
    """Check positive support and reduction to existing severity distributions."""
    dist = GB2(a=2.0, b=100.0, p=1.5, q=3.0)
    assert dist.cdf(-100.0) == 0.0
    assert dist.cdf(0.0) == 0.0
    assert dist.invcdf(0.0) == 0.0
    assert math.isinf(float(dist.invcdf(1.0)))
    assert dist.cdf(1e12) == pytest.approx(1.0)

    burr = Burr(power=2.0, shape=3.0, scale=100.0, loc=0.0)
    loglogistic = LogLogistic(shape=2.0, scale=100.0)
    for x in (5.0, 50.0, 150.0, 1000.0):
        assert GB2(a=2.0, b=100.0, p=1.0, q=3.0).cdf(x) == pytest.approx(burr.cdf(x))
        assert GB2(a=2.0, b=100.0, p=1.0, q=1.0).cdf(x) == pytest.approx(loglogistic.cdf(x))


def test_gb2_simulated_moments() -> None:
    """Check direct gamma-ratio simulation against analytical mean and variance."""
    a, b, p, q = 2.0, 100000.0, 2.0, 4.0
    base = beta_function(p, q)
    mean = b * beta_function(p + 1 / a, q - 1 / a) / base
    second_moment = b**2 * beta_function(p + 2 / a, q - 2 / a) / base
    variance = second_moment - mean**2

    set_random_seed(12345678)
    sims = GB2(a, b, p, q).generate(100000)
    assert sims.n_sims == 100000
    assert float(xp.mean(sims.values)) == pytest.approx(mean, rel=0.015)
    assert float(xp.var(sims.values)) == pytest.approx(variance, rel=0.05)


def test_gb2_stochastic_parameters_preserve_coupling() -> None:
    """CDF, quantiles and samples remain coupled to all stochastic parameters."""
    a = StochasticScalar([1.5, 2.0, 2.5])
    b = StochasticScalar([100.0, 150.0, 200.0])
    p = StochasticScalar([1.5, 2.0, 2.5])
    q = StochasticScalar([3.0, 3.5, 4.0])
    x = StochasticScalar([80.0, 100.0, 120.0])
    dist = GB2(a, b, p, q)

    cdf = dist.cdf(x)
    assert isinstance(cdf, StochasticScalar)
    for parameter in (a, b, p, q, x):
        assert cdf.coupled_variable_group is parameter.coupled_variable_group

    u = StochasticScalar([0.25, 0.5, 0.75])
    quantiles = dist.invcdf(u)
    assert isinstance(quantiles, StochasticScalar)
    for parameter in (a, b, p, q, x, u):
        assert quantiles.coupled_variable_group is parameter.coupled_variable_group

    sims = dist.generate(3)
    assert isinstance(sims, StochasticScalar)
    assert bool(xp.all(sims.values > 0))
    for parameter in (a, b, p, q, x, u):
        assert sims.coupled_variable_group is parameter.coupled_variable_group


@pytest.mark.parametrize("param", ["a", "b", "p", "q"])
@pytest.mark.parametrize("invalid", [-1.0, 0.0, float("nan"), float("inf")])
def test_gb2_rejects_invalid_parameters(param: str, invalid: float) -> None:
    """All GB2 parameters must be strictly positive and finite."""
    parameters = {"a": 2.0, "b": 100.0, "p": 1.5, "q": 3.0}
    parameters[param] = invalid
    with pytest.raises(ValueError, match=param):
        GB2(**parameters)


def test_gb2_rejects_invalid_stochastic_parameter() -> None:
    with pytest.raises(ValueError, match="q"):
        GB2(a=2.0, b=100.0, p=1.5, q=StochasticScalar([2.0, 0.0, 3.0]))


def test_gb2_named_generator() -> None:
    generator = ContinuousDistributionGenerator("gb2", [2.0, 100.0, 1.5, 3.0])
    assert generator.invcdf(0.5) == pytest.approx(GB2(2.0, 100.0, 1.5, 3.0).invcdf(0.5))
    assert generator.generate(20).n_sims == 20

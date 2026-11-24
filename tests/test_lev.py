"""Independent tests for limited expected values across numeric families."""

import numpy as np
import pytest
from scipy import integrate, stats

from pal.distributions import (
    GEV,
    GPD,
    MBBEFD,
    Bernoulli,
    Beta,
    Binomial,
    Burr,
    Exponential,
    Gamma,
    GeneralizedBetaPrime,
    GeneralizedInverseGaussian,
    HyperGeometric,
    InverseBurr,
    InverseExponential,
    InverseGamma,
    InverseGaussian,
    InverseParalogistic,
    InverseWeibull,
    LogGamma,
    Logistic,
    LogLogistic,
    LogNormal,
    NegBinomial,
    NonCentralChiSquared,
    Normal,
    Paralogistic,
    Pareto,
    Poisson,
    StudentsT,
    Uniform,
    Weibull,
    WeightedDistribution,
)


@pytest.mark.parametrize(
    ("model", "reference"),
    [
        (Pareto(shape=0.7, scale=2), stats.pareto(0.7, scale=2)),
        (GPD(shape=1, scale=2, loc=1), stats.genpareto(1, scale=2, loc=1)),
        (GPD(shape=-0.2, scale=2, loc=1), stats.genpareto(-0.2, scale=2, loc=1)),
        (Gamma(alpha=2, theta=3, loc=1), stats.gamma(2, scale=3, loc=1)),
        (Exponential(scale=2, loc=1), stats.expon(scale=2, loc=1)),
        (LogNormal(mu=0.5, sigma=1.2), stats.lognorm(1.2, scale=np.exp(0.5))),
        (Weibull(shape=0.6, scale=2, loc=1), stats.weibull_min(0.6, scale=2, loc=1)),
        (Burr(power=0.5, shape=1.5, scale=2, loc=1), stats.burr12(0.5, 1.5, scale=2, loc=1)),
        (LogLogistic(shape=2, scale=2, loc=1), stats.fisk(2, scale=2, loc=1)),
        (Paralogistic(shape=0.8, scale=2, loc=1), stats.burr12(0.8, 0.8, scale=2, loc=1)),
        (InverseBurr(power=2, shape=3, scale=2, loc=1), stats.burr(2, 3, scale=2, loc=1)),
        (InverseParalogistic(shape=0.8, scale=2, loc=1), stats.burr(0.8, 0.8, scale=2, loc=1)),
        (InverseGamma(alpha=0.7, theta=2, loc=1), stats.invgamma(0.7, scale=2, loc=1)),
        (InverseGamma(alpha=3, theta=2, loc=1), stats.invgamma(3, scale=2, loc=1)),
        (InverseWeibull(shape=0.8, scale=2, loc=1), stats.invweibull(0.8, scale=2, loc=1)),
        (InverseWeibull(shape=2, scale=2, loc=1), stats.invweibull(2, scale=2, loc=1)),
        (InverseExponential(scale=2, loc=1), stats.invweibull(1, scale=2, loc=1)),
        (GeneralizedBetaPrime(a=1, b=2, p=2, q=3), stats.betaprime(2, 3, scale=2)),
        (GeneralizedBetaPrime(a=1, b=2, p=2, q=0.7), stats.betaprime(2, 0.7, scale=2)),
        (InverseGaussian(mu=2, lambda_=3), stats.invgauss(2 / 3, scale=3)),
        (GeneralizedInverseGaussian(p=0.5, chi=4, psi=1), stats.geninvgauss(0.5, 2, scale=2)),
        (Beta(alpha=2, beta=3, loc=1, scale=2), stats.beta(2, 3, loc=1, scale=2)),
        (Uniform(a=1, b=5), stats.uniform(loc=1, scale=4)),
        (Normal(mu=1, sigma=2), stats.norm(1, 2)),
        (Logistic(mu=1, sigma=2), stats.logistic(1, 2)),
        (StudentsT(nu=3, mu=1, sigma=2), stats.t(3, loc=1, scale=2)),
        (GEV(shape=0.2, loc=1, scale=2), stats.genextreme(-0.2, loc=1, scale=2)),
        (GEV(shape=1.2, loc=1, scale=2), stats.genextreme(-1.2, loc=1, scale=2)),
        (GEV(shape=-0.2, loc=1, scale=2), stats.genextreme(0.2, loc=1, scale=2)),
        (GEV(shape=0, loc=1, scale=2), stats.gumbel_r(loc=1, scale=2)),
        (NonCentralChiSquared(df=4, nonc=3), stats.ncx2(4, 3)),
    ],
)
def test_continuous_lev_against_independent_integral(model, reference):
    limits = np.array([[20.0, 0.0, -2.0], [2.0, 4.0, 2.0]])
    lower, _ = reference.support()

    def expected(d):
        if d <= lower:
            return d
        if np.isfinite(lower):
            return lower + integrate.quad(reference.sf, lower, d, epsabs=1e-10)[0]
        return d - integrate.quad(reference.cdf, -np.inf, d, epsabs=1e-10)[0]

    ref = np.array([expected(d) for d in limits.flat]).reshape(limits.shape)
    np.testing.assert_allclose(model.lev(limits), ref, rtol=2e-8, atol=1e-9)
    assert isinstance(model.lev(2.0), float)
    assert model.lev(2.0) == pytest.approx(ref[1, 0], rel=2e-8, abs=1e-9)


@pytest.mark.parametrize(
    ("model", "reference"),
    [
        (Poisson(mean=3), stats.poisson(3)),
        (Binomial(n=7, p=0.4), stats.binom(7, 0.4)),
        (Bernoulli(p=0.4), stats.bernoulli(0.4)),
        (NegBinomial(n=3, p=0.4), stats.nbinom(3, 0.4)),
        (HyperGeometric(ngood=7, nbad=12, n_draws=5), stats.hypergeom(19, 7, 5)),
    ],
)
def test_discrete_lev_noninteger_limits(model, reference):
    limits = np.array([-2, 0, 0.5, 1, 3.7, 100])
    values = np.arange(1000)
    expected = np.sum(np.minimum(limits[:, None], values) * reference.pmf(values), axis=1)
    np.testing.assert_allclose(model.lev(limits), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("beta", [0.2, 1, 2])
def test_loggamma_lev(beta):
    model = LogGamma(alpha=2, beta=beta)
    limits = np.array([0, 1, 2, 10, 100])
    expected = [
        d if d <= 1 else 1 + integrate.quad(lambda x: stats.gamma.sf(np.log(x), 2, scale=beta), 1, d)[0]
        for d in limits
    ]
    np.testing.assert_allclose(model.lev(limits), expected, rtol=1e-10)


@pytest.mark.parametrize(("g", "b"), [(3, 0.2), (3, 1), (3, 2), (1, 0.5), (3, 0)])
def test_mbbefd_lev(g, b):
    model = MBBEFD(g=g, b=b)
    limits = np.array([-1.0, 0, 0.1, 0.5, 1, 2])
    expected = [
        d if d <= 0 else integrate.quad(lambda x: 1 - float(model.cdf(float(x))), 0, min(d, 1))[0]
        for d in limits
    ]
    np.testing.assert_allclose(model.lev(limits), expected, rtol=1e-9, atol=1e-12)


def test_lev_infinite_mean_and_mixture():
    np.testing.assert_allclose(Pareto(shape=1, scale=1).lev(np.array([1, 10, 1e100])), 1 + np.log([1, 10, 1e100]))
    assert StudentsT(nu=1).lev(10) == -np.inf
    mix = WeightedDistribution([Exponential(scale=2), Exponential(scale=4)], [1, 3])
    limits = np.array([0, 1, 10, np.nan, np.inf])
    expected = 0.5 * (-np.expm1(-limits / 2)) + 3 * (-np.expm1(-limits / 4))
    expected[-1] = np.nan
    np.testing.assert_allclose(mix.lev(limits), expected, rtol=1e-12)
    assert mix.lev(np.empty((0, 2))).shape == (0, 2)

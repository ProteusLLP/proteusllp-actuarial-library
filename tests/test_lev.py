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
        d if d <= 1 else 1 + integrate.quad(lambda x: stats.gamma.sf(np.log(x), 2, scale=beta), 1, d)[0] for d in limits
    ]
    np.testing.assert_allclose(model.lev(limits), expected, rtol=1e-10)


@pytest.mark.parametrize(("g", "b"), [(3, 0.2), (3, 1), (3, 2), (1, 0.5), (3, 0)])
def test_mbbefd_lev(g, b):
    model = MBBEFD(g=g, b=b)
    limits = np.array([-1.0, 0, 0.1, 0.5, 1, 2])
    expected = [
        d if d <= 0 else integrate.quad(lambda x: 1 - float(model.cdf(float(x))), 0, min(d, 1))[0] for d in limits
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


@pytest.mark.parametrize(
    ("model", "limit", "reference"),
    [
        (Gamma(alpha=1e-10, theta=1e300), 1e-100, 9.21456779078616646e-108),
        (Gamma(alpha=2, theta=1e300), 1e-100, 1e-100),
        (LogNormal(mu=0, sigma=100), 1.0, 0.50398902398135681),
        (LogNormal(mu=200, sigma=100), 1e100, 3.84925128728399367e99),
        (LogNormal(mu=-300, sigma=30), 1e100, 7.89709726678477296e30),
        (LogNormal(mu=0, sigma=1e-10), 1.0, 0.999999999960105772),
        (InverseBurr(power=2, shape=1e30, scale=1e-15, loc=0), 1.0, 0.910926144109219655),
        (InverseBurr(power=1, shape=1e300, scale=1e-310, loc=0), 1.0, 2.3448635265088924e-9),
        (GeneralizedBetaPrime(a=0.7, b=2, p=3, q=0.8), 1e8, 27925.041062164178),
        (GeneralizedBetaPrime(a=0.5, b=2, p=3, q=2), 1e10, 229.99357634541635),
        (InverseGamma(alpha=0.2, theta=1e-100), 1e100, 1.36140552632292038e60),
        (InverseGamma(alpha=1, theta=1e-200), 1e200, 9.21456821532716741e-198),
        (InverseGamma(alpha=0.9999999999, theta=3), 1e20, 136.12762202768256),
        (InverseWeibull(shape=0.3, scale=1e-100), 1e100, 1.42857142857142857e40),
        (InverseExponential(scale=1e-200), 1e200, 9.21456821532716741e-198),
        (GEV(shape=1.2, loc=0, scale=1), 1e20, 11098.8464837877705),
        (GEV(shape=1, loc=0, scale=1), 1e20, 45.47448619497938),
        (GEV(shape=1 - 1e-12, loc=0, scale=1), 1e20, 45.47448619396449),
        (GEV(shape=1e-8, loc=0, scale=1), 10.0, 0.577170275349920232),
        (GEV(shape=-1e-8, loc=0, scale=1), 10.0, 0.577170255624187101),
        (GEV(shape=-100, loc=0, scale=1), -10.0, -9.33262154439441527e155),
    ],
)
def test_lev_high_precision_extreme_parameters(model, limit, reference):
    """Independent incomplete gamma/beta moments or 70-digit integrals."""
    assert model.lev(limit) == pytest.approx(reference, rel=5e-10, abs=0)
    values = np.array([[limit, np.nan], [limit, 0.5 * limit]])
    result = model.lev(values)
    assert result.shape == values.shape
    assert result[0, 0] == pytest.approx(reference, rel=5e-10, abs=0)
    assert result[1, 0] == result[0, 0]
    assert np.isnan(result[0, 1])
    if limit >= 0:
        assert result[1, 1] <= result[0, 0] * (1 + 1e-12)
    else:
        assert result[1, 1] >= result[0, 0] - 1e-12 * abs(result[0, 0])


@pytest.mark.parametrize(
    ("model", "rv"),
    [
        (Gamma(alpha=3, theta=2), stats.gamma(3, scale=2)),
        (LogNormal(mu=2, sigma=0.5), stats.lognorm(0.5, scale=np.exp(2))),
        (Burr(power=2, shape=3, scale=2, loc=0), stats.burr12(2, 3, scale=2)),
        (InverseBurr(power=2, shape=3, scale=2, loc=0), stats.burr(2, 3, scale=2)),
        (InverseGamma(alpha=2, theta=3), stats.invgamma(2, scale=3)),
        (InverseGaussian(mu=3, lambda_=4), stats.invgauss(0.75, scale=4)),
        (GeneralizedInverseGaussian(p=0.5, chi=4, psi=1), stats.geninvgauss(0.5, 2, scale=2)),
    ],
)
def test_lev_mean_excess_identity(model, rv):
    limits = rv.ppf(np.linspace(0.25, 0.99, 25))
    expected = rv.mean() - rv.sf(limits) * model.mean_excess(limits)
    np.testing.assert_allclose(model.lev(limits), expected, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize(
    "model",
    [
        InverseGamma(alpha=0.7, theta=2),
        InverseWeibull(shape=0.7, scale=2),
        GeneralizedBetaPrime(a=0.7, b=2, p=3, q=0.8),
        GEV(shape=1.2, loc=1, scale=2),
    ],
)
def test_large_lev_batch_is_monotone_and_matches_scalar(model):
    limits = np.geomspace(0.1, 1e8, 10000)
    values = model.lev(limits)
    assert np.all(np.isfinite(values))
    assert np.all(values <= limits * (1 + 1e-10))
    assert np.all(np.diff(values) >= 0)
    indices = np.linspace(0, len(limits) - 1, 12, dtype=int)
    reference = np.array([model.lev(float(limits[i])) for i in indices])
    np.testing.assert_allclose(values[indices], reference, rtol=5e-10, atol=1e-12)
    shuffled = limits[indices[::-1]].reshape(3, 4)
    np.testing.assert_allclose(model.lev(shuffled), reference[::-1].reshape(3, 4), rtol=5e-10, atol=1e-12)


@pytest.mark.parametrize(
    ("g", "b", "limit", "expected"),
    [
        (1e200, 1e200, 0.5, 1.5e-200),
        (2, 1e-200, 1.0, 0.998494850021680094),
        (1e200, 1e-200, 0.5, 0.00217147240951625914),
        (1e200, 1e-300, 1.0, 1 / 3),
        (3, 1.0000000001, 0.5, 0.346573590276565296),
        (1.0000000001, 1e300, 0.5, 0.499999999950144765),
    ],
)
def test_mbbefd_lev_extreme_scale(g, b, limit, expected):
    assert MBBEFD(g=g, b=b).lev(limit) == pytest.approx(expected, rel=5e-10, abs=0)

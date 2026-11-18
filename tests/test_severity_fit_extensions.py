"""Numerical validation of additional PAL severity fitting and mean excess."""

import numpy as np
import pytest
from scipy import integrate, stats

from pal.distributions import (
    GPD,
    Burr,
    Gamma,
    InverseBurr,
    InverseGamma,
    InverseGaussian,
    InverseWeibull,
    LogGamma,
    LogLogistic,
    LogNormal,
    Paralogistic,
    Pareto,
    Weibull,
)


@pytest.mark.parametrize(
    ("cls", "params", "frozen"),
    [
        (Weibull, {"shape": 1.7, "scale": 8.0, "loc": 0.0}, stats.weibull_min(1.7, scale=8.0)),
        (Burr, {"power": 2.5, "shape": 2.1, "scale": 8.0, "loc": 0.0}, stats.burr12(2.5, 2.1, scale=8.0)),
        (LogLogistic, {"shape": 2.8, "scale": 8.0, "loc": 0.0}, stats.fisk(2.8, scale=8.0)),
        (InverseGaussian, {"mu": 8.0, "lambda_": 16.0}, stats.invgauss(0.5, scale=16.0)),
        (Paralogistic, {"shape": 2.0, "scale": 8.0, "loc": 0.0}, stats.burr12(2, 2, scale=8.0)),
        (InverseBurr, {"power": 2.5, "shape": 2.0, "scale": 8.0, "loc": 0.0}, stats.burr(2.5, 2, scale=8.0)),
        (InverseGamma, {"alpha": 4.0, "theta": 24.0, "loc": 0.0}, stats.invgamma(4, scale=24.0)),
        (InverseWeibull, {"shape": 3.0, "scale": 8.0, "loc": 0.0}, stats.invweibull(3, scale=8.0)),
    ],
)
def test_other_fits_match_independent_censored_likelihood(cls, params, frozen) -> None:
    """Every fitted model reports the same truncated likelihood as SciPy."""
    rng = np.random.default_rng(521)
    raw = frozen.rvs(size=250, random_state=rng)
    raw = raw[raw > 4.0]
    censored = raw > 20.0
    observed = np.minimum(raw, 20.0)
    fitted = cls.fit(observed, threshold=4.0, censored=censored)
    assert np.isfinite(fitted.log_likelihood)
    assert np.isfinite(fitted.aic)
    pars = fitted.parameters
    if cls is Weibull:
        ref = stats.weibull_min(pars["shape"], scale=pars["scale"])
    elif cls is Burr:
        ref = stats.burr12(pars["power"], pars["shape"], scale=pars["scale"])
    elif cls is LogLogistic:
        ref = stats.fisk(pars["shape"], scale=pars["scale"])
    elif cls is InverseGaussian:
        ref = stats.invgauss(pars["mu"] / pars["lambda_"], scale=pars["lambda_"])
    elif cls is Paralogistic:
        ref = stats.burr12(pars["shape"], pars["shape"], scale=pars["scale"])
    elif cls is InverseBurr:
        ref = stats.burr(pars["power"], pars["shape"], scale=pars["scale"])
    elif cls is InverseGamma:
        ref = stats.invgamma(pars["alpha"], scale=pars["theta"])
    else:
        ref = stats.invweibull(pars["shape"], scale=pars["scale"])
    expected = (
        np.sum(ref.logpdf(observed[~censored])) + np.sum(ref.logsf(observed[censored])) - len(observed) * ref.logsf(4.0)
    )
    assert fitted.log_likelihood == pytest.approx(expected, rel=1e-10, abs=1e-9)
    assert fitted.aic == pytest.approx(-2 * expected + 2 * (2 if cls not in {Burr, InverseBurr} else 3))
    assert fitted.distribution.cdf(6.0) == pytest.approx(ref.cdf(6.0), rel=1e-9, abs=1e-10)


@pytest.mark.parametrize(
    ("distribution", "reference", "thresholds"),
    [
        (Gamma(alpha=2.0, theta=3.0), stats.gamma(2, scale=3), (0.0, 4.0, 12.0)),
        (Weibull(shape=1.5, scale=6), stats.weibull_min(1.5, scale=6), (1.0, 5.0)),
        (LogNormal(mu=1.5, sigma=0.6), stats.lognorm(0.6, scale=np.exp(1.5)), (1.0, 6.0)),
        (Burr(power=2.0, shape=2.0, scale=6, loc=0), stats.burr12(2, 2, scale=6), (2.0, 10.0)),
        (InverseGaussian(mu=4, lambda_=10), stats.invgauss(0.4, scale=10), (2.0, 6.0)),
        (InverseGamma(alpha=3, theta=6, loc=0), stats.invgamma(3, scale=6), (2.0, 10.0)),
        (InverseWeibull(shape=2.5, scale=6, loc=0), stats.invweibull(2.5, scale=6), (2.0, 10.0)),
        (LogLogistic(shape=2.5, scale=6, loc=0), stats.fisk(2.5, scale=6), (2.0, 10.0)),
        (Paralogistic(shape=1.8, scale=6, loc=0), stats.burr12(1.8, 1.8, scale=6), (2.0, 10.0)),
        (InverseBurr(power=2.5, shape=2, scale=6, loc=0), stats.burr(2.5, 2, scale=6), (2.0, 10.0)),
    ],
)
def test_mean_excess_matches_independent_survival_integration(distribution, reference, thresholds) -> None:
    """Compare theoretical mean excess with directly integrated survival."""
    for d in thresholds:
        survival = reference.sf(d)
        integral, error = integrate.quad(reference.sf, d, np.inf, epsrel=1e-9)
        assert error < 1e-6
        assert distribution.mean_excess(d) == pytest.approx(integral / survival, rel=2e-5, abs=1e-7)
    vector = distribution.mean_excess(np.array(thresholds))
    assert vector.shape == (len(thresholds),)


def test_pareto_and_gpd_mean_excess_closed_forms() -> None:
    pareto = Pareto(shape=2.5, scale=10.0)
    assert pareto.mean_excess(0.0) == pytest.approx(10 * 2.5 / 1.5)
    assert pareto.mean_excess(20.0) == pytest.approx(20 / 1.5)
    assert np.isinf(Pareto(shape=1.0, scale=10).mean_excess(20))
    gpd = GPD(shape=0.3, scale=7.0, loc=2.0)
    assert gpd.mean_excess(5.0) == pytest.approx((7.0 + 0.3 * 3.0) / 0.7)
    assert np.isinf(GPD(shape=1.2, scale=7.0, loc=2.0).mean_excess(10.0))
    bounded = GPD(shape=-0.5, scale=10.0, loc=2.0)
    assert bounded.mean_excess(10.0) == pytest.approx(6.0 / 1.5)
    assert np.isnan(bounded.mean_excess(22.0))


def test_infinite_mean_tails_are_not_numerically_truncated() -> None:
    for severity in [
        Burr(power=0.5, shape=1.0, scale=6, loc=0),
        LogLogistic(shape=0.9, scale=6),
        InverseBurr(power=0.9, shape=2.0, scale=6, loc=0),
        InverseGamma(alpha=0.9, theta=6),
        InverseWeibull(shape=0.9, scale=6),
        Paralogistic(shape=1.0, scale=6),
    ]:
        assert np.isinf(severity.mean_excess(10.0))


def test_loggamma_distribution_and_censored_fit_against_transformed_gamma() -> None:
    rng = np.random.default_rng(987)
    alpha, beta = 2.0, 0.35
    original = np.exp(rng.gamma(shape=alpha, scale=beta, size=1200))
    original = original[original > 1.5]
    censored = original > 8.0
    observed = np.minimum(original, 8.0)
    fitted = LogGamma.fit(observed, threshold=1.5, censored=censored)
    pars = fitted.parameters
    assert fitted.converged
    assert pars["alpha"] == pytest.approx(alpha, abs=0.6)
    assert pars["beta"] == pytest.approx(beta, abs=0.14)
    shape, scale = pars["alpha"], pars["beta"]
    y = np.log(observed)
    expected = (
        np.sum(stats.gamma.logpdf(y[~censored], a=shape, scale=scale) - y[~censored])
        + np.sum(stats.gamma.logsf(y[censored], a=shape, scale=scale))
        - len(y) * stats.gamma.logsf(np.log(1.5), a=shape, scale=scale)
    )
    assert fitted.log_likelihood == pytest.approx(expected)
    model = fitted.distribution
    q = 0.7
    assert model.invcdf(q) == pytest.approx(np.exp(stats.gamma.ppf(q, a=shape, scale=scale)))
    assert model.cdf(1.0) == 0
    assert model.cdf(5.0) == pytest.approx(stats.gamma.cdf(np.log(5.0), a=shape, scale=scale))


def test_loggamma_mean_excess_matches_transformed_gamma_moment() -> None:
    model = LogGamma(alpha=2.0, beta=0.25)
    d = 3.0
    reference = (1 - 0.25) ** (-2) * stats.gamma.sf(np.log(d), a=2, scale=0.25 / (1 - 0.25)) / stats.gamma.sf(
        np.log(d), a=2, scale=0.25
    ) - d
    assert model.mean_excess(d) == pytest.approx(reference)
    assert np.isinf(LogGamma(alpha=2.0, beta=1.0).mean_excess(3.0))


def test_gpd_free_location_matches_truncated_censored_likelihood() -> None:
    rng = np.random.default_rng(756)
    original = stats.genpareto.rvs(c=0.25, loc=2.0, scale=5.0, size=1200, random_state=rng)
    censored = original > 25.0
    observed = np.minimum(original, 25.0)
    fitted = GPD.fit(observed, censored=censored, free_location=True)
    pars = fitted.parameters
    assert fitted.aic == pytest.approx(6 - 2 * fitted.log_likelihood)
    assert pars["loc"] < observed.min()
    ref = stats.genpareto(c=pars["shape"], loc=pars["loc"], scale=pars["scale"])
    expected = (
        np.sum(ref.logpdf(observed[~censored])) + np.sum(ref.logsf(observed[censored])) - len(observed) * ref.logsf(3.0)
    )
    assert fitted.log_likelihood == pytest.approx(expected, abs=1e-7)
    with pytest.raises(ValueError, match="not identifiable"):
        GPD.fit(observed, threshold=3.0, censored=censored, free_location=True)
    assert fitted.distribution.cdf(15.0) == pytest.approx(ref.cdf(15.0), rel=1e-10)

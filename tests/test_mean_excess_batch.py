"""Regression tests for batched and empirical mean residual life."""

import numpy as np
import pytest
from scipy import integrate, stats

from pal.distributions import (
    GEV,
    Beta,
    Exponential,
    GeneralizedInverseGaussian,
    InverseBurr,
    InverseGamma,
    InverseGaussian,
    InverseWeibull,
    LogLogistic,
    NonCentralChiSquared,
    Normal,
    Paralogistic,
)
from pal.fitting import empirical_mean_excess


@pytest.mark.parametrize(
    ("model", "rv"),
    [
        (Exponential(scale=3, loc=2), stats.expon(scale=3, loc=2)),
        (LogLogistic(shape=2.5, scale=3), stats.fisk(2.5, scale=3)),
        (Paralogistic(shape=2.5, scale=3), stats.burr12(2.5, 2.5, scale=3)),
        (InverseBurr(power=2.5, shape=3, scale=2), stats.burr(2.5, 3, scale=2)),
        (InverseGamma(alpha=3, theta=2), stats.invgamma(3, scale=2)),
        (InverseWeibull(shape=2.5, scale=3), stats.invweibull(2.5, scale=3)),
        (InverseGaussian(mu=3, lambda_=4), stats.invgauss(0.75, scale=4)),
        (GEV(shape=0.2, scale=3), stats.genextreme(-0.2, scale=3)),
        (GEV(shape=-0.2, scale=3), stats.genextreme(0.2, scale=3)),
        (GEV(shape=0, scale=3), stats.gumbel_r(scale=3)),
        (GeneralizedInverseGaussian(p=0.5, chi=4, psi=1), stats.geninvgauss(0.5, 2, scale=2)),
        (NonCentralChiSquared(df=4, nonc=3), stats.ncx2(4, 3)),
    ],
)
def test_batched_matches_independent_quadrature(model, rv):
    ds = np.array([[6.0, -1.0, 3.0], [3.0, 0.5, 1.0]])
    upper = rv.support()[1]
    expected = np.array([
        integrate.quad(rv.sf, d, upper, epsabs=1e-10)[0] / rv.sf(d)
        for d in ds.ravel()
    ]).reshape(ds.shape)
    np.testing.assert_allclose(model.mean_excess(ds), expected, rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(
        model.mean_excess(ds),
        np.array([model.mean_excess(float(d)) for d in ds.ravel()]).reshape(ds.shape),
        rtol=2e-6,
    )


def test_empirical_strict_exceedances_and_shape():
    x = np.array([8, 3, 3, 10, 6, 1], dtype=float)
    ds = np.array([[3, 10, 0], [7, 3, np.nan]])
    expected = np.array([[5, np.nan, x.mean()], [2, 5, np.nan]])
    # At 3, exceedances are 8, 6 and 10; at 7 they are 8 and 10.
    np.testing.assert_allclose(empirical_mean_excess(x, ds), expected)
    assert empirical_mean_excess(x, 3) == pytest.approx(5)
    assert np.isnan(empirical_mean_excess(x, 11))
    assert empirical_mean_excess(x, np.empty((0, 2))).shape == (0, 2)


def test_empirical_large_location_small_spacings():
    x = 1e15 + np.arange(10, dtype=float)
    ds = x + 0.125
    expected = [np.mean(x[x > d] - d) if np.any(x > d) else np.nan for d in ds]
    np.testing.assert_allclose(empirical_mean_excess(x, ds), expected, rtol=1e-14)


def test_normal_deep_tail_and_beta_endpoint():
    np.testing.assert_allclose(
        Normal(mu=0, sigma=1).mean_excess(np.array([1e4, 1e8])),
        [1e-4 * (1 - 2e-8), 1e-8], rtol=1e-12,
    )
    d = np.nextafter(1.0, 0.0)
    assert Beta(alpha=1, beta=3).mean_excess(d) == pytest.approx((1 - d) / 4, rel=1e-12, abs=0)
    assert np.isnan(Beta(alpha=1, beta=3).mean_excess(1.0))


def test_inverse_tail_closed_forms():
    # Inverse Burr with power=2 is regularly varying with tail index 2.
    model = InverseBurr(power=2, shape=3, scale=1)
    assert model.mean_excess(1e100) == pytest.approx(1e100, rel=1e-12)
    assert InverseGamma(alpha=3, theta=2).mean_excess(1e150) == pytest.approx(5e149, rel=1e-12)

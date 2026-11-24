"""Regression tests for batched and empirical mean residual life."""

import numpy as np
import pytest
from scipy import integrate, stats

from pal import _mean_excess
from pal.distributions import (
    GEV,
    MBBEFD,
    Beta,
    Exponential,
    Gamma,
    GeneralizedBetaPrime,
    GeneralizedInverseGaussian,
    InverseBurr,
    InverseGamma,
    InverseGaussian,
    InverseWeibull,
    LogGamma,
    LogLogistic,
    LogNormal,
    NonCentralChiSquared,
    Normal,
    Paralogistic,
    Poisson,
    StudentsT,
)
from pal.fitting import empirical_mean_excess


@pytest.mark.parametrize(
    ("model", "rv"),
    [
        (Exponential(scale=3, loc=2), stats.expon(scale=3, loc=2)),
        (LogLogistic(shape=2.5, scale=3), stats.fisk(2.5, scale=3)),
        (Paralogistic(shape=2.5, scale=3), stats.burr12(2.5, 2.5, scale=3)),
        (InverseBurr(power=2.5, shape=3, scale=2, loc=0), stats.burr(2.5, 3, scale=2)),
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
    expected = np.array([integrate.quad(rv.sf, d, upper, epsabs=1e-10)[0] / rv.sf(d) for d in ds.ravel()]).reshape(
        ds.shape
    )
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
        [1e-4 * (1 - 2e-8), 1e-8],
        rtol=1e-12,
    )
    d = np.nextafter(1.0, 0.0)
    assert Beta(alpha=1, beta=3).mean_excess(d) == pytest.approx((1 - d) / 4, rel=1e-12, abs=0)
    assert np.isnan(Beta(alpha=1, beta=3).mean_excess(1.0))


def test_inverse_tail_closed_forms():
    # Inverse Burr with power=2 is regularly varying with tail index 2.
    model = InverseBurr(power=2, shape=3, scale=1, loc=0)
    assert model.mean_excess(1e100) == pytest.approx(1e100, rel=1e-12)
    assert InverseGamma(alpha=3, theta=2).mean_excess(1e150) == pytest.approx(5e149, rel=1e-12)


def test_shifted_gamma_and_large_shape_tail():
    model = Gamma(alpha=2, theta=3, loc=20)
    ds = np.array([10.0, 20.0, 26.0, 2420.0])
    # For shape two, e(d) = theta*(z+2)/(z+1), z >= 0.
    expected = np.array([16.0, 6.0, 4.0, 3 * 802 / 801])
    np.testing.assert_allclose(model.mean_excess(ds), expected, rtol=1e-12)
    # 70-digit conditional-density integral, not a ratio of underflowed CDFs.
    assert Gamma(alpha=1e6, theta=1).mean_excess(100000001.0) == pytest.approx(1.01010099958981759054, rel=1e-12)


def test_lognormal_small_variance_and_large_log_ratio():
    # At exp(mu), the normal conditional moment has a closed form in erf.
    expected = 7.9788456085286535588e-11
    assert LogNormal(mu=0, sigma=1e-10).mean_excess(1.0) == pytest.approx(expected, rel=1e-11, abs=0)
    assert LogNormal(mu=100, sigma=1).mean_excess(1e-300) == pytest.approx(np.exp(100.5), rel=1e-12)


def test_inverse_gaussian_far_tail():
    # 60-digit integral of the conditional density after exponential rescaling.
    assert InverseGaussian(mu=3, lambda_=4).mean_excess(1e8) == pytest.approx(4.499999696250052, rel=1e-12)
    assert InverseGaussian(mu=1, lambda_=1e6).mean_excess(3) == pytest.approx(2.2499966250131307e-6, rel=1e-11, abs=0)


def test_loggamma_near_support_and_underflow():
    assert LogGamma(alpha=2, beta=1e-9).mean_excess(2) == pytest.approx(2.00000000488539e-9, rel=1e-11, abs=0)
    assert LogGamma(alpha=2, beta=0.1).mean_excess(1e100) == pytest.approx(1.1116470443852886e99, rel=1e-11)


def test_gev_underflow_and_finite_endpoint():
    assert GEV(shape=0.1, scale=1, loc=0).mean_excess(1e100) == pytest.approx(1e100 / 9, rel=1e-12)
    d = np.nextafter(10.0, 0.0)
    assert GEV(shape=-0.1, scale=1, loc=0).mean_excess(d) == pytest.approx(0.1 * (10 - d) / 1.1, rel=1e-12, abs=0)


def test_count_and_student_t_underflow():
    assert Poisson(mean=1).mean_excess(1000) == pytest.approx(1.000998999006989, rel=1e-12)
    assert StudentsT(nu=5).mean_excess(1e100) == pytest.approx(2.5e99, rel=1e-12)


def test_mbbefd_atom_and_beta_underflow_near_endpoint():
    d = np.nextafter(1.0, 0.0)
    assert MBBEFD(g=1e20, b=1).mean_excess(d) == pytest.approx(1 - d, rel=1e-12, abs=0)
    assert MBBEFD(g=1e20, b=2).mean_excess(d) == pytest.approx(1 - d, rel=1e-12, abs=0)
    assert Beta(alpha=1, beta=1000).mean_excess(d) == pytest.approx((1 - d) / 1001, rel=1e-12, abs=0)


def test_gumbel_and_central_chisquare_underflow():
    assert GEV(shape=0, scale=3).mean_excess(1e5) == pytest.approx(3.0, rel=1e-12)
    assert NonCentralChiSquared(df=2, nonc=0).mean_excess(2000) == pytest.approx(2.0, rel=1e-12)


def test_noncentral_chisquare_underflow():
    # 60-digit independent Bessel-density moment ratio.
    assert NonCentralChiSquared(df=4, nonc=3).mean_excess(2000) == pytest.approx(2.0810388910582227, rel=1e-11)


@pytest.mark.parametrize(
    ("a", "p", "q", "threshold", "expected"),
    [
        (0.1, 0.1, 1000, 2.0, 0.039351048528653634),
        (0.1, 10, 1000, 2.0, 0.03972305541742264),
        (0.1, 10, 1000, 10000.0, 142.25041770588297),
        (1, 0.1, 100, 10000.0, 101.02011200245536),
        (100, 1, 1000, 1.0, 1.998045855540408e-5),
        (100, 1000, 1000, 1.0, 0.00035699924727082777),
    ],
)
def test_gb2_extreme_parameter_grid(a, p, q, threshold, expected):
    # Independent 50-digit incomplete-beta moment ratios.
    model = GeneralizedBetaPrime(a=a, b=1, p=p, q=q)
    np.testing.assert_allclose(model.mean_excess(np.array([threshold, threshold])), expected, rtol=5e-9, atol=0)


@pytest.mark.parametrize(
    ("alpha", "beta", "threshold", "expected"),
    [
        (1000, 1000, 0.9, 0.00011232464024910889),
        (1000, 1000, 0.999, 9.999980000139827e-7),
        (1000, 1000, 1 - 1e-10, 9.990010817582138e-14),
        (0.1, 1000, 0.5, 0.0004990535842651377),
    ],
)
def test_beta_extreme_parameter_grid(alpha, beta, threshold, expected):
    assert Beta(alpha=alpha, beta=beta).mean_excess(threshold) == pytest.approx(expected, rel=5e-9, abs=0)


@pytest.mark.parametrize(
    ("lam", "threshold", "expected"),
    [
        (1e6, 1.5, 3.5999593932368467e-6),
        (1e6, 2.0, 2.666656592695439e-6),
        (1e4, 2.0, 0.0002665660286177533),
        (1e-4, 1e5, 16238.94762069739),
    ],
)
def test_inverse_gaussian_concentrated_and_diffuse(lam, threshold, expected):
    assert InverseGaussian(mu=1, lambda_=lam).mean_excess(threshold) == pytest.approx(expected, rel=5e-9, abs=0)


@pytest.mark.parametrize(
    ("shape", "threshold", "expected"),
    [
        (1e-5, 5.0, 1.0017454371220431),
        (1e-5, 1.0, 1.0936945396410311),
        (-1e-5, 5.0, 1.0016247966039277),
        (-1e-5, 1.0, 1.0936488527313808),
    ],
)
def test_gev_near_gumbel_transition(shape, threshold, expected):
    assert GEV(shape=shape, scale=1).mean_excess(threshold) == pytest.approx(expected, rel=5e-9, abs=0)


@pytest.mark.parametrize(
    ("mu", "threshold", "expected"),
    [
        (-20, 2.061153621820215e-9, 6.192574469519574e-19),
        (-20, 2.0611536226446694e-9, 1.0823921061848084e-19),
        (20, 485165195.4097903, 0.03871058206540314),
        (20, 485165195.652372, 0.00904855191864086),
    ],
)
def test_lognormal_narrow_shifted_coordinates(mu, threshold, expected):
    # Fixed binary64 attachments; rounding exp(mu) would change this reference.
    model = LogNormal(mu=mu, sigma=1e-10)
    assert model.mean_excess(threshold) == pytest.approx(expected, rel=1e-11, abs=0)


@pytest.mark.parametrize(
    ("model", "threshold", "expected"),
    [
        (Gamma(alpha=1e6, theta=1), 100000001.0, 1.01010099958981759054),
        (LogGamma(alpha=2, beta=1e-9), 2.0, 2.00000000488539e-9),
        (LogNormal(mu=0, sigma=1e-10), 1.0, 7.9788456085286535588e-11),
        (InverseGaussian(mu=3, lambda_=4), 1e8, 4.499999696250052),
        (GeneralizedBetaPrime(a=0.1, b=1, p=10, q=1000), 2.0, 0.03972305541742264),
        (InverseBurr(power=100, shape=1000, scale=1, loc=0), 2.0, 0.020202020202020202),
        (NonCentralChiSquared(df=4, nonc=0), 2000.0, 2 * 1002 / 1001),
    ],
)
def test_extreme_analytical_paths_do_not_integrate(monkeypatch, model, threshold, expected):
    """Continued fractions and moment series must resolve these tails directly."""

    def unexpected_quadrature(*args, **kwargs):
        pytest.fail("Unexpected numerical quadrature")

    monkeypatch.setattr(integrate, "quad", unexpected_quadrature)
    monkeypatch.setattr(_mean_excess, "_laguerre_rule", unexpected_quadrature)
    assert model.mean_excess(threshold) == pytest.approx(expected, rel=1e-11, abs=0)

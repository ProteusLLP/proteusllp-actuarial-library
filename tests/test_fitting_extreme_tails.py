"""Independent extreme-tail regressions for PAL distribution fitting (#128).

Numerical survival underflow is not a finite support endpoint. These tests
compare conditional likelihoods and mean excess against closed forms and
independent special-function references without running the scenario engine.
"""

import numpy as np
import pytest
from scipy import special, stats

from pal.distributions import GPD, Burr, Gamma, GeneralizedBetaPrime, LogNormal, Pareto, Weibull
from pal.fitting import _gamma_logsf, _gb2_terms, _log_likelihood


@pytest.mark.parametrize("shape,scale,threshold", [(1.0, 2.0, 2000.0), (2.0, 2.0, 2000.0)])
def test_gamma_extreme_truncated_censored_loglikelihood(shape, scale, threshold):
    """Shape 1/2 tails have independent exact exponential/Erlang formulae."""
    observations = np.array([threshold + 1, threshold + 3, threshold + 7], dtype=float)
    censored = np.array([False, True, False])
    weights = np.array([1.0, 2.0, 3.0])
    z = observations / scale
    z0 = threshold / scale
    if shape == 1.0:
        log_sf = -z
        log_pdf = -z - np.log(scale)
        log_sf_threshold = -z0
    else:
        log_sf = -z + np.log1p(z)
        log_pdf = np.log(z) - z - np.log(scale)
        log_sf_threshold = -z0 + np.log1p(z0)
    expected = np.sum(weights * (np.where(censored, log_sf, log_pdf) - log_sf_threshold))
    actual = _log_likelihood(
        Gamma,
        {"alpha": shape, "theta": scale},
        observations,
        censored,
        weights,
        threshold,
    )
    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, rel=1e-11, abs=1e-9)


def test_gamma_log_survival_preserves_unrepresentable_probabilities():
    """Both a scalar and an array retain the finite logarithm of survival."""
    assert _gamma_logsf(1.0, 1200.0) == pytest.approx(-1200.0, abs=1e-10)
    z = np.array([0.0, 2.0, 800.0, 2000.0])
    expected = -z + np.log1p(z)
    np.testing.assert_allclose(_gamma_logsf(2.0, z), expected, rtol=1e-12, atol=1e-9)
    for shape in (0.2, 2.5, 10.0):
        assert np.isfinite(_gamma_logsf(shape, 1000.0))
        # In the representable region the continuation leaves SciPy unchanged.
        assert _gamma_logsf(shape, 10.0) == pytest.approx(stats.gamma.logsf(10.0, a=shape), rel=1e-12)


def test_gamma_mean_excess_is_finite_after_survival_underflow():
    """Exponential memorylessness and Erlang shape-two mean excess are exact."""
    np.testing.assert_allclose(
        Gamma(alpha=1.0, theta=1.0).mean_excess(np.array([0.0, 800.0, 1200.0])),
        [1.0, 1.0, 1.0],
        rtol=1e-12,
        atol=1e-12,
    )
    attachments = np.array([0.0, 800.0, 1600.0])
    scale = 2.0
    z = attachments / scale
    expected = scale * (z + 2.0) / (z + 1.0)
    np.testing.assert_allclose(
        Gamma(alpha=2.0, theta=scale).mean_excess(attachments),
        expected,
        rtol=1e-9,
        atol=1e-9,
    )


def test_weibull_mean_excess_uses_stable_incomplete_gamma():
    """Shape one is exponential; shape two has an erfcx tail reference."""
    assert Weibull(shape=1.0, scale=3.0).mean_excess(2500.0) == pytest.approx(3.0)
    d, scale = 800.0, 10.0
    expected = scale * np.sqrt(np.pi) / 2.0 * special.erfcx(d / scale)
    assert Weibull(shape=2.0, scale=scale).mean_excess(d) == pytest.approx(expected, rel=1e-8)
    shifted = Weibull(shape=1.0, scale=3.0, loc=5.0)
    assert shifted.mean_excess(0.0) == pytest.approx(8.0)
    assert shifted.mean_excess(1000.0) == pytest.approx(3.0)


def test_lognormal_mean_excess_matches_independent_log_normal_tail_ratio():
    """Use scipy.stats normal log survival as reference for deep attachments."""
    d, mu, sigma = 800.0, 0.0, 1.0
    z = (np.log(d) - mu) / sigma
    log_ratio = mu + 0.5 * sigma**2 - np.log(d)
    log_ratio += stats.norm.logsf(z - sigma) - stats.norm.logsf(z)
    expected = d * np.expm1(log_ratio)
    assert LogNormal(mu=mu, sigma=sigma).mean_excess(d) == pytest.approx(expected, rel=1e-9)


def test_burr_mean_excess_closed_form_and_deep_tail():
    """Lomax special case, shifted support and far-tail beta expansion."""
    d = 1e8
    assert Burr(power=1.0, shape=3.0, scale=10.0, loc=0.0).mean_excess(d) == pytest.approx((d + 10.0) / 2.0, rel=1e-12)
    assert Burr(power=1.0, shape=3.0, scale=10.0, loc=5.0).mean_excess(0.0) == pytest.approx(10.0, rel=1e-12)
    # Burr XII with power=2 and shape=2 has asymptotic MRL d/(2*2-1).
    assert Burr(power=2.0, shape=2.0, scale=10.0, loc=0.0).mean_excess(1e8) == pytest.approx(1e8 / 3.0, rel=2e-7)


def test_infinite_mean_and_finite_endpoint_are_distinct_from_underflow():
    """Theoretical infinite means and bounded support retain their semantics."""
    assert np.isposinf(Burr(power=2.0, shape=0.4, scale=1.0, loc=0.0).mean_excess(1e10))
    assert np.isposinf(Pareto(shape=0.8, scale=1.0).mean_excess(1e10))
    finite = GPD(shape=-0.5, scale=10.0, loc=2.0)
    assert np.isnan(finite.mean_excess(22.0))


def test_gb2_log_survival_does_not_underflow():
    """For p=1 the GB2 survival is exactly (1+(x/b)**a)**(-q)."""
    params = {"a": 2.0, "b": 1.0, "p": 1.0, "q": 2.0}
    x = np.array([1e20, 1e100, 1e200])
    _, log_sf = _gb2_terms(x, params)
    expected = -2.0 * np.logaddexp(0.0, 2.0 * np.log(x))
    np.testing.assert_allclose(log_sf, expected, rtol=1e-12, atol=1e-10)
    assert np.all(np.isfinite(log_sf))


def test_gb2_mean_excess_deep_tail_is_finite():
    """For p=1, a=1, q=3 the GB2 is Lomax with exact mean excess."""
    distribution = GeneralizedBetaPrime(a=1.0, b=10.0, p=1.0, q=3.0)
    for attachment in (1e5, 1e12, 1e100):
        actual = distribution.mean_excess(attachment)
        assert np.isfinite(actual)
        assert actual == pytest.approx((attachment + 10.0) / 2.0, rel=1e-7)


@pytest.mark.parametrize("survival_coordinate", [0.9, 0.1, 1e-3, 1.1e-6, 0.9e-6, 1e-12, 1e-120])
def test_gb2_mean_excess_matches_independent_beta_polynomial(survival_coordinate):
    """For a=1,p=2,q=4 the conditional mean has an exact rational form."""
    b = 10.0
    z = survival_coordinate
    attachment = b * (1.0 - z) / z
    # I_z(4,2) = z**4*(5-4*z); I_z(3,3) = z**3*(10-15*z+6*z**2).
    # Simplifying the conditional mean minus d avoids numerical cancellation:
    # MRL(d) = b*(5/3-z)/(z*(5-4*z)).
    expected = b * (5.0 / 3.0 - z) / (z * (5.0 - 4.0 * z))
    model = GeneralizedBetaPrime(a=1.0, b=b, p=2.0, q=4.0)
    actual = model.mean_excess(attachment)
    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, rel=2e-10)


@pytest.mark.parametrize("scale", [1e-4, 1.0, 1e20, 1e120])
def test_gb2_censored_truncated_loglikelihood_extreme_tail(scale):
    """Check GB2 ground-up density and conditional survival against beta-polynomial laws."""
    a, b, p, q = 1.0, 10.0, 2.0, 4.0
    params = {"a": a, "b": b, "p": p, "q": q}
    threshold = scale
    losses = scale * np.array([1.1, 1.3, 1.8, 2.5])
    censored = np.array([False, True, True, False])
    weights = np.array([1.0, 2.0, 0.5, 1.0])

    def independent_logsf(x):
        log_u = np.log(x) - np.log(b)
        log_z = -np.logaddexp(0.0, log_u)
        z = np.exp(log_z)
        return q * log_z + np.log1p(q * (1.0 - z))

    def independent_logpdf(x):
        log_u = np.log(x) - np.log(b)
        return (p - 1.0) * log_u - np.log(b) - special.betaln(p, q) - (p + q) * np.logaddexp(0.0, log_u)

    log_pdf, log_sf = _gb2_terms(losses, params)
    threshold_log_sf = float(_gb2_terms(np.array([threshold]), params)[1][0])
    expected_pdf = independent_logpdf(losses)
    expected_sf = independent_logsf(losses)
    expected_threshold = float(independent_logsf(threshold))
    np.testing.assert_allclose(log_pdf, expected_pdf, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(log_sf, expected_sf, rtol=1e-12, atol=1e-10)
    assert threshold_log_sf == pytest.approx(expected_threshold, rel=1e-12, abs=1e-10)

    actual_ll = np.sum(weights * (np.where(censored, log_sf, log_pdf) - threshold_log_sf))
    expected_ll = np.sum(weights * (np.where(censored, expected_sf, expected_pdf) - expected_threshold))
    assert np.isfinite(actual_ll)
    assert actual_ll == pytest.approx(expected_ll, rel=1e-12, abs=1e-9)


def test_gb2_log_likelihood_when_scale_ratio_exceeds_float_range():
    """Log-density and log-survival remain finite when x / scale would overflow."""
    params = {"a": 1.0, "b": 1e-200, "p": 1.0, "q": 3.0}
    x = np.array([1e150, 1e200])
    log_pdf, log_sf = _gb2_terms(x, params)
    log_u = np.log(x) - np.log(params["b"])
    expected_sf = -3.0 * np.logaddexp(0.0, log_u)
    expected_pdf = (
        np.log(3.0)
        - np.log(params["b"])
        - 4.0 * np.logaddexp(0.0, log_u)
    )
    np.testing.assert_allclose(log_sf, expected_sf, rtol=1e-12)
    np.testing.assert_allclose(log_pdf, expected_pdf, rtol=1e-12)
    assert np.all(np.isfinite(log_sf)) and np.all(np.isfinite(log_pdf))


def test_gb2_mean_excess_below_support():
    """Below its positive support, the mean excess is the mean minus attachment."""
    distribution = GeneralizedBetaPrime(a=1.0, b=10.0, p=1.0, q=3.0)
    assert distribution.mean_excess(0.0) == pytest.approx(5.0)
    assert distribution.mean_excess(-3.0) == pytest.approx(8.0)


def test_gb2_large_beta_shape_log_survival_underflow():
    """A large tail shape can underflow betainc before z reaches 1e-8."""
    q = 100.0
    b = 1.0
    x = np.array([1e5, 1e6])
    params = {"a": 1.0, "b": b, "p": 2.0, "q": q}
    _, log_sf = _gb2_terms(x, params)
    z = 1.0 / (1.0 + x / b)
    # I_z(q,2) = z**q * (q + 1 - q*z).
    expected = q * np.log(z) + np.log(q + 1.0 - q * z)
    np.testing.assert_allclose(log_sf, expected, rtol=1e-12, atol=1e-9)
    assert np.all(np.isfinite(log_sf))

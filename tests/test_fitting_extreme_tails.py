"""Independent extreme-tail regressions for PAL distribution fitting (#128).

Numerical survival underflow is not a finite support endpoint. These tests
compare conditional likelihoods and mean excess against closed forms and
independent special-function references without running the scenario engine.
"""

import numpy as np
import pytest
from scipy import special, stats

from pal.distributions import GPD, Burr, Gamma, LogNormal, Pareto, Weibull
from pal.fitting import _gamma_logsf, _log_likelihood


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

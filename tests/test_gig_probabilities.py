"""Regression tests for sorted generalized inverse Gaussian probabilities."""

import numpy as np
import pytest
from scipy.stats import geninvgauss

from pal._gig import _gig_log_probabilities
from pal.distributions import GeneralizedInverseGaussian


@pytest.mark.parametrize(("p", "b"), [(0.5, 2), (-2, 0.1), (10, 1), (0, 0.0001), (1000, 1000)])
def test_piecewise_probabilities_match_independent_reference(p, b):
    mode = np.arcsinh(p / b)
    points = np.exp(mode + np.linspace(-2, 2, 51) / max(1, np.sqrt(np.hypot(p, b))))
    cdf, sf = _gig_log_probabilities(points, p, b)
    # SciPy independently integrates from zero at every threshold.
    np.testing.assert_allclose(np.exp(cdf), geninvgauss.cdf(points, p, b), rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(np.exp(cdf) + np.exp(sf), 1, rtol=1e-14)
    assert np.all(np.diff(cdf) >= 0)
    assert np.all(np.diff(sf) <= 0)


@pytest.mark.parametrize(
    ("p", "b", "point", "upper", "reference"),
    [
        (0.5, 2, 20, True, 1.78902253503270526395e-9),
        (-2, 0.1, 0.005, False, 0.000500527136338483725795),
        (0, 1e6, 1.001, True, 0.158776148584981753805),
        (0, 1e6, 1.01, True, 1.25619692541104084847e-23),
        (0, 1e-4, 1e5, True, 0.0000615624121531899698627),
    ],
)
def test_piecewise_gig_extreme_tails(p, b, point, upper, reference):
    # Independent 70-digit log-density integral.
    cdf, sf = _gig_log_probabilities(np.array([point, 1.0, point]), p, b)
    tail = sf if upper else cdf
    np.testing.assert_allclose(np.exp(tail[[0, 2]]), reference, rtol=2e-10, atol=0)


def test_gig_piecewise_reciprocity():
    points = np.geomspace(0.001, 1000, 101)
    cdf, sf = _gig_log_probabilities(points, 0, 0.01)
    np.testing.assert_allclose(cdf, sf[::-1], rtol=1e-12, atol=1e-12)


def test_gig_public_array_cdf_and_invcdf():
    model = GeneralizedInverseGaussian(p=0.75, chi=2.5, psi=1.25, loc=3)
    points = np.array([[10, 4, 5, 4], [3, np.inf, np.nan, -np.inf]])
    values = model.cdf(points)
    assert isinstance(values, np.ndarray)
    assert values.shape == points.shape
    expected = geninvgauss.cdf(points, 0.75, np.sqrt(2.5 * 1.25), loc=3, scale=np.sqrt(2))
    np.testing.assert_allclose(values, expected, rtol=1e-10, atol=1e-12, equal_nan=True)
    probabilities = np.array([[0.01, 0.5, 0.99], [0.9, 0.25, 0.75]])
    quantiles = model.invcdf(probabilities)
    assert isinstance(quantiles, np.ndarray)
    assert quantiles.shape == probabilities.shape
    np.testing.assert_allclose(model.cdf(quantiles), probabilities, rtol=1e-10)
    assert isinstance(model.cdf(5.0), float)
    empty = model.cdf(np.empty((0, 2)))
    assert isinstance(empty, np.ndarray)
    assert empty.shape == (0, 2)

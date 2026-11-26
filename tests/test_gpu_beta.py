"""GPU accuracy tests for the incomplete beta functions."""

import importlib

import numpy as np
import pytest
import scipy.special

from pal._beta import betainc, betaincinv
from pal._maths import asnumpy, xp

from ._assertions import assert_beta_inverse_tail_accuracy

pytestmark = pytest.mark.skipif(xp.__name__ != "cupy", reason="requires the CuPy backend")


SHAPES = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 10_000.0])
PROBABILITIES = np.array(
    [
        1e-14,
        1e-10,
        1e-6,
        1e-3,
        0.01,
        0.1,
        0.25,
        0.5,
        0.75,
        0.9,
        0.99,
        1 - 1e-6,
        1 - 1e-10,
        1 - 1e-14,
    ]
)


def _parameter_grid(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha, beta, value = np.meshgrid(SHAPES, SHAPES, values, indexing="ij")
    return alpha.ravel(), beta.ravel(), value.ravel()


def test_gpu_beta_cdf_matches_scipy_across_parameter_regimes() -> None:
    """Match SciPy for small, large, symmetric, skewed, and tail cases."""
    alpha, beta, x = _parameter_grid(PROBABILITIES)

    actual = asnumpy(betainc(xp.asarray(alpha), xp.asarray(beta), xp.asarray(x)))
    expected = scipy.special.betainc(alpha, beta, x)

    np.testing.assert_allclose(actual, expected, rtol=5e-10, atol=5e-14)


def test_gpu_beta_inverse_matches_scipy_across_parameter_regimes() -> None:
    """Match SciPy inverse values across central and extreme probabilities."""
    alpha, beta, probability = _parameter_grid(PROBABILITIES)

    actual = asnumpy(betaincinv(xp.asarray(alpha), xp.asarray(beta), xp.asarray(probability)))
    expected = scipy.special.betaincinv(alpha, beta, probability)

    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-13)


def test_gpu_beta_cdf_matches_scipy_in_large_symmetric_transition() -> None:
    """Keep accuracy where a large symmetric beta CDF transitions steeply."""
    probabilities = PROBABILITIES[2:-2]
    for shape in (10.0, 100.0, 10_000.0):
        values = scipy.special.betaincinv(shape, shape, probabilities)
        actual = asnumpy(betainc(shape, shape, xp.asarray(values)))
        np.testing.assert_allclose(actual, probabilities, rtol=5e-10, atol=5e-14)


def test_gpu_beta_cdf_matches_scipy_for_seven_term_series() -> None:
    """Validate the fixed-length series used by the benchmark shape."""
    values = np.linspace(np.finfo(float).eps, 0.95, 16_384)
    actual = asnumpy(betainc(2.5, 7.0, xp.asarray(values)))
    expected = scipy.special.betainc(2.5, 7.0, values)

    np.testing.assert_allclose(actual, expected, rtol=5e-10, atol=5e-14)


def test_gpu_beta_inverse_round_trip_preserves_tail_probabilities() -> None:
    """Recover input probabilities using an independent SciPy tail evaluation."""
    alpha, beta, probability = _parameter_grid(PROBABILITIES)
    device_alpha = xp.asarray(alpha)
    device_beta = xp.asarray(beta)
    device_probability = xp.asarray(probability)

    quantile = betaincinv(device_alpha, device_beta, device_probability)
    assert_beta_inverse_tail_accuracy(alpha, beta, probability, quantile)


def test_gpu_beta_functions_handle_boundaries_and_invalid_inputs() -> None:
    """Return exact endpoints and NaN for invalid parameters or arguments."""
    alpha = xp.asarray([2.0, 2.0, -1.0, 2.0, 2.0])
    beta = xp.asarray([3.0, 3.0, 3.0, 3.0, 3.0])
    values = xp.asarray([0.0, 1.0, 0.5, -0.1, 1.1])

    cdf = asnumpy(betainc(alpha, beta, values))
    inverse = asnumpy(betaincinv(alpha, beta, values))

    assert cdf[0] == 0.0
    assert cdf[1] == 1.0
    assert inverse[0] == 0.0
    assert inverse[1] == 1.0
    assert np.all(np.isnan(cdf[2:]))
    assert np.all(np.isnan(inverse[2:]))


def test_gpu_beta_cdf_is_monotone_and_symmetric() -> None:
    """Preserve monotonicity and the complementary beta identity."""
    # Binary fractions have exact floating-point complements, so the symmetry
    # assertion measures the implementation rather than subtraction rounding.
    values = xp.arange(1, 4096) / 4096
    cdf = betainc(0.3, 12.0, values)
    complement = betainc(12.0, 0.3, 1 - values)

    assert bool(xp.all(xp.diff(cdf) >= 0))
    np.testing.assert_allclose(asnumpy(cdf + complement), 1.0, rtol=0, atol=3e-14)


def test_gpu_beta_shape_one_inverse_retains_tiny_representable_quantile() -> None:
    """Do not round a tiny original quantile to zero through reflection."""
    a, b, probability = 1e-7, 1.0, 1.0 - 1e-5
    actual = float(asnumpy(betaincinv(a, b, probability)))
    expected = scipy.special.betaincinv(a, b, probability)
    assert actual > 0.0
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0.0)


def test_gpu_beta_arcsine_cdf_near_upper_endpoint() -> None:
    """Use the complementary arcsine expression for upper-tail accuracy."""
    x = np.nextafter(1.0, 0.0)
    actual = float(asnumpy(betainc(0.5, 0.5, x)))
    # Independent analytic upper tail. SciPy's direct betainc shortcut also
    # loses precision here through sqrt(x), so it is not a reliable oracle.
    expected_upper_tail = 2.0 * np.arcsin(np.sqrt(1.0 - x)) / np.pi
    np.testing.assert_allclose(1.0 - actual, expected_upper_tail, rtol=1e-8, atol=1e-15)


@pytest.mark.parametrize("shape", [1e6, 1e8, 1e10])
@pytest.mark.parametrize("array_shapes", [False, True])
def test_gpu_beta_large_symmetric_centre_neighbours(shape: float, array_shapes: bool) -> None:
    """Resolve both adjacent floats around the exact symmetric median."""
    values = np.array([np.nextafter(0.5, 0.0), 0.5, np.nextafter(0.5, 1.0)])
    parameter = xp.full(values.size, shape) if array_shapes else shape
    actual = asnumpy(betainc(parameter, parameter, xp.asarray(values)))
    # Gamma(a+1/2)/Gamma(a) has this expansion; the next term is O(a^-3).
    # Higher powers of the adjacent-float displacement are negligible here.
    density = 2 * np.sqrt(shape / np.pi) * (1 - 1 / (8 * shape) + 1 / (128 * shape**2))
    expected = 0.5 + density * (values - 0.5)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2 * np.finfo(float).eps)
    assert actual[0] < actual[1] < actual[2]
    assert actual[1] == 0.5


@pytest.mark.parametrize(("alpha", "beta"), [(1e6, 1e6), (1e8, 1e8), (1e8, 1.01e8)])
@pytest.mark.parametrize("array_shapes", [False, True])
def test_gpu_beta_large_shape_cdf_and_inverse(alpha: float, beta: float, array_shapes: bool) -> None:
    """Check central and tail values against independent SciPy transforms."""
    probability = np.array([1e-6, 0.01, 0.25, 0.49, 0.5, 0.51, 0.75, 0.99, 1 - 1e-6])
    values = scipy.special.betaincinv(alpha, beta, probability)
    a = xp.full(values.size, alpha) if array_shapes else alpha
    b = xp.full(values.size, beta) if array_shapes else beta
    actual = asnumpy(betainc(a, b, xp.asarray(values)))
    expected = scipy.special.betainc(alpha, beta, values)
    np.testing.assert_allclose(actual, expected, rtol=5e-9, atol=1e-11)
    assert np.all(np.diff(actual) > 0)
    quantiles = asnumpy(betaincinv(a, b, xp.asarray(probability)))
    np.testing.assert_allclose(quantiles, values, rtol=0, atol=2e-13)


@pytest.mark.parametrize("shape", [1e4, 1e6, 1e8])
def test_gpu_beta_local_centre_series_boundary(shape: float) -> None:
    """Both sides of the local-series switch agree with the beta CDF."""
    distance = 0.01 / np.sqrt(shape)
    values = 0.5 + distance * np.array([-1.0001, -0.9999, 0.9999, 1.0001])
    actual = asnumpy(betainc(shape, shape, xp.asarray(values)))
    expected = scipy.special.betainc(shape, shape, values)
    np.testing.assert_allclose(actual, expected, rtol=5e-10, atol=5e-12)
    assert np.all(np.diff(actual) > 0)


def test_gpu_beta_fraction_exhaustion_propagates_nan() -> None:
    """An exhausted fraction cannot become a clamped CDF or an inverse bracket."""
    module = importlib.import_module("pal._gpu_beta")
    preamble = module._BETA_PREAMBLE.replace("m <= 256", "m <= 1").replace("m <= 16384", "m <= 1")
    kernel = xp.ElementwiseKernel(
        "float64 a, float64 b, float64 x, float64 p",
        "float64 cdf, float64 inverse",
        "const double lb = pal_log_beta(a, b); cdf = pal_ibeta(a, b, x, lb); inverse = pal_ibeta_inverse(a, b, p, lb);",
        "pal_beta_forced_fraction_exhaustion",
        preamble=preamble,
    )
    cdf, inverse = kernel(
        xp.asarray([3.5, 100.0]), xp.asarray([4.5, 120.0]), xp.asarray([0.4, 0.45]), xp.asarray([0.3, 0.3])
    )
    assert np.all(np.isnan(asnumpy(cdf)))
    assert np.all(np.isnan(asnumpy(inverse)))

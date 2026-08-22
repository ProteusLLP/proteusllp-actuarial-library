"""GPU accuracy tests for the incomplete beta functions."""

import numpy as np
import pytest
import scipy.special

from pal._beta import betainc, betaincinv
from pal._maths import asnumpy, xp

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


def test_gpu_beta_inverse_round_trip_preserves_tail_probabilities() -> None:
    """Recover input probabilities without losing relative tail accuracy."""
    alpha, beta, probability = _parameter_grid(PROBABILITIES)
    device_alpha = xp.asarray(alpha)
    device_beta = xp.asarray(beta)
    device_probability = xp.asarray(probability)

    quantile = betaincinv(device_alpha, device_beta, device_probability)
    recovered = asnumpy(betainc(device_alpha, device_beta, quantile))
    tail_scale = np.minimum(probability, 1 - probability)
    representable = asnumpy((quantile > 0) & (quantile < 1))

    np.testing.assert_array_less(
        np.abs(recovered[representable] - probability[representable]),
        2e-8 * tail_scale[representable] + 5e-15,
    )


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

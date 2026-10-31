"""Backend-neutral array assertions for CPU and GPU test runs."""

from __future__ import annotations

import typing as t

import numpy as np
import scipy.special
import scipy.stats

from pal._maths import asnumpy
from pal.couplings import ProteusStochasticVariable


def _host(value: t.Any) -> t.Any:
    """Copy stochastic/backend arrays to the host before NumPy assertions."""
    if isinstance(value, ProteusStochasticVariable):
        value = value.values
    if isinstance(value, tuple):
        return tuple(_host(item) for item in value)
    if isinstance(value, list):
        return [_host(item) for item in value]
    return asnumpy(value)


def host_values(value: t.Any) -> np.ndarray[t.Any, t.Any]:
    """Return a host array for a PAL value or backend array."""
    return asnumpy(getattr(value, "values", value))


def array_equal(actual: t.Any, expected: t.Any, *args: t.Any, **kwargs: t.Any) -> bool:
    """Return whether two backend-neutral arrays are equal."""
    return bool(np.array_equal(_host(actual), _host(expected), *args, **kwargs))


def allclose(actual: t.Any, expected: t.Any, *args: t.Any, **kwargs: t.Any) -> bool:
    """Return whether two backend-neutral arrays are close."""
    return bool(np.allclose(_host(actual), _host(expected), *args, **kwargs))


def assert_array_equal(actual: t.Any, expected: t.Any, *args: t.Any, **kwargs: t.Any) -> None:
    """Assert equality after copying backend arrays to the host."""
    np.testing.assert_array_equal(_host(actual), _host(expected), *args, **kwargs)


def assert_allclose(actual: t.Any, expected: t.Any, *args: t.Any, **kwargs: t.Any) -> None:
    """Assert closeness after copying backend arrays to the host."""
    np.testing.assert_allclose(_host(actual), _host(expected), *args, **kwargs)


def assert_array_almost_equal(actual: t.Any, expected: t.Any, *args: t.Any, **kwargs: t.Any) -> None:
    """Assert approximate equality after copying backend arrays to the host."""
    np.testing.assert_array_almost_equal(_host(actual), _host(expected), *args, **kwargs)


def assert_beta_inverse_tail_accuracy(alpha: t.Any, beta: t.Any, probability: t.Any, quantile: t.Any) -> None:
    """Check inverse beta probabilities against SciPy, allowing reference ULPs.

    Select representable quantiles from the reference rather than the result
    being tested. Evaluate the smaller tail directly to avoid cancellation.
    """
    alpha, beta, probability, quantile = np.broadcast_arrays(
        *(np.asarray(_host(value), dtype=float) for value in (alpha, beta, probability, quantile))
    )
    reference = scipy.special.betaincinv(alpha, beta, probability)
    assert np.all(np.isfinite(reference))
    assert np.all(np.isfinite(quantile))
    assert np.all((quantile >= 0) & (quantile <= 1))

    def smaller_tail(value: np.ndarray) -> np.ndarray:
        return np.where(
            probability > 0.5,
            scipy.stats.beta.sf(value, alpha, beta),
            scipy.special.betainc(alpha, beta, value),
        )

    target = np.minimum(probability, 1 - probability)
    reference_tail = smaller_tail(reference)
    lower = smaller_tail(np.maximum(0, np.nextafter(reference, -np.inf)))
    upper = smaller_tail(np.minimum(1, np.nextafter(reference, np.inf)))
    resolution = np.maximum(np.abs(lower - reference_tail), np.abs(upper - reference_tail))
    # Some SciPy versions floor underflowed quantiles at the smallest normal
    # float. Require the reference probability to be locally representable.
    interior = (reference > 0) & (reference < 1) & (np.abs(reference_tail - target) <= 2e-8 * target + 2 * resolution)
    assert np.any(interior), "the test must exercise representable interior quantiles"
    assert np.all((quantile[interior] > 0) & (quantile[interior] < 1))
    tolerance = 2e-8 * target + 2 * resolution + np.abs(reference_tail - target)
    residual = np.abs(smaller_tail(quantile) - target)
    np.testing.assert_array_less(residual[interior], tolerance[interior])

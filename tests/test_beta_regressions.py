"""Beta kernel regressions and deterministic distribution integration checks.

These checks also run against the CPU dispatch to validate the reference
calculations and tolerances independently of the CUDA implementation.
"""

import numpy as np
import pytest
import scipy.special
import scipy.stats

from pal import StochasticScalar
from pal._beta import betainc, betaincinv
from pal._maths import asnumpy, xp
from pal.copulas import _student_t_cdf
from pal.distributions import Beta, StudentsT

from ._assertions import assert_beta_inverse_tail_accuracy

PROBABILITIES = np.array([1e-14, 1e-10, 1e-6, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1 - 1e-6, 1 - 1e-10, 1 - 1e-14])
REGIME_SHAPES = [
    (14.999, 0.5),
    (15.0, 0.5),
    (15.001, 0.5),
    (20.0, 31.999),
    (20.0, 32.0),
    (20.0, 32.001),
    (63.999, 64.0),
    (64.0, 64.0),
    (64.001, 64.0),
    (999.999, 1000.0),
    (1000.0, 1000.0),
    (1000.0, 1020.0),
    (1000.0, 1031.0),
    (1000.0, np.nextafter(1000.0 + np.sqrt(1000.0), 0)),
    (1000.0, np.nextafter(1000.0 + np.sqrt(1000.0), np.inf)),
    (1000.0, 1032.0),
    (10_000.0, 10_000.0),
    (10_000.0, 10_099.0),
    (10_000.0, 10_100.0),
]


@pytest.mark.parametrize("alpha", [1.1, 2.5, 10.0, 20.0, 100.0])
@pytest.mark.parametrize("beta", range(2, 34))
def test_beta_integer_shape_cdf_is_accurate_and_monotone(alpha: float, beta: float) -> None:
    """Cover cancellation and the integer-series boundaries over a dense grid."""
    values = np.unique(np.r_[np.linspace(0.01, 0.8, 257), 0.7, 0.75, np.nextafter(0.8, 0), np.nextafter(0.8, 1), 0.9])
    actual = asnumpy(betainc(alpha, beta, xp.asarray(values)))
    expected = scipy.special.betainc(alpha, beta, values)

    np.testing.assert_allclose(actual, expected, rtol=5e-10, atol=5e-14)
    assert np.all(np.diff(actual) >= -3e-14)


@pytest.mark.parametrize("alpha,beta", [(1000.0, 1032.0), (1000.0, 1032.1), (1032.0, 1000.0)])
def test_beta_inverse_retains_converged_solution(alpha: float, beta: float) -> None:
    """Keep a rounded Halley correction from discarding an accurate root."""
    actual = asnumpy(betaincinv(alpha, beta, xp.asarray(PROBABILITIES)))
    expected = scipy.special.betaincinv(alpha, beta, PROBABILITIES)

    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-13)
    assert np.all(np.diff(actual) > 0)
    assert_beta_inverse_tail_accuracy(alpha, beta, PROBABILITIES, actual)


@pytest.mark.parametrize("alpha,beta", REGIME_SHAPES)
def test_beta_regime_transitions_match_independent_reference(alpha: float, beta: float) -> None:
    """Probe the CDF at reference quantiles and close to its steep centre."""
    centre = alpha / (alpha + beta)
    switch = (alpha + 1) / (alpha + beta + 2)
    values = np.unique(
        np.r_[
            scipy.special.betaincinv(alpha, beta, PROBABILITIES),
            centre + np.array([-1e-8, -1e-10, -1e-12, 0, 1e-12, 1e-10, 1e-8]),
            np.nextafter(switch, 0),
            switch,
            np.nextafter(switch, 1),
            np.nextafter(0.5, 0),
            0.5,
            np.nextafter(0.5, 1),
        ]
    )
    actual = asnumpy(betainc(alpha, beta, xp.asarray(values)))
    np.testing.assert_allclose(actual, scipy.special.betainc(alpha, beta, values), rtol=5e-10, atol=5e-14)
    assert np.all(np.diff(actual) >= -3e-14)

    quantiles = betaincinv(alpha, beta, xp.asarray(PROBABILITIES))
    np.testing.assert_allclose(
        asnumpy(quantiles), scipy.special.betaincinv(alpha, beta, PROBABILITIES), rtol=2e-9, atol=2e-13
    )
    assert_beta_inverse_tail_accuracy(alpha, beta, PROBABILITIES, quantiles)


@pytest.mark.parametrize("alpha,beta", [(0.5, 0.5), (2.5, 7.0), (15.0, 0.5), (1000.0, 1031.0), (10_000.0, 0.1)])
def test_beta_extreme_tails_use_independent_cdf_and_inverse(alpha: float, beta: float) -> None:
    """Measure relative tail errors without a fixed absolute probability floor."""
    probabilities = np.unique(np.r_[1e-100, 1e-50, 1e-30, PROBABILITIES])
    quantiles = betaincinv(alpha, beta, xp.asarray(probabilities))
    assert_beta_inverse_tail_accuracy(alpha, beta, probabilities, quantiles)

    reference = scipy.special.betaincinv(alpha, beta, probabilities)
    upper = probabilities > 0.5
    # Reflect upper tails before evaluation, avoiding subtraction from one.
    actual = asnumpy(
        betainc(
            xp.asarray(np.where(upper, beta, alpha)),
            xp.asarray(np.where(upper, alpha, beta)),
            xp.asarray(np.where(upper, 1 - reference, reference)),
        )
    )
    expected = np.where(
        upper, scipy.stats.beta.sf(reference, alpha, beta), scipy.special.betainc(alpha, beta, reference)
    )
    np.testing.assert_allclose(actual, expected, rtol=5e-10, atol=0)


@pytest.mark.parametrize("bad_quantile", [0.0, 1.0, np.nan])
def test_beta_tail_assertion_rejects_invalid_interior_quantiles(bad_quantile: float) -> None:
    """Ensure a broken inverse cannot remove itself from the checked cases."""
    with pytest.raises(AssertionError):
        assert_beta_inverse_tail_accuracy(2.5, 7.0, [1e-14], [bad_quantile])


@pytest.mark.parametrize("probability", [1e-100, 1e-30, 1e-14, 1 - 1e-14])
def test_beta_tail_assertion_detects_relative_tail_errors(probability: float) -> None:
    """Reject large relative errors even below the old absolute tolerance."""
    wrong_probability = 2 * probability if probability < 0.5 else 1 - 2 * (1 - probability)
    wrong_quantile = scipy.special.betaincinv(2.5, 7.0, wrong_probability)
    with pytest.raises(AssertionError):
        assert_beta_inverse_tail_accuracy(2.5, 7.0, [probability], [wrong_quantile])


def test_beta_scalar_and_device_shapes_agree_with_broadcasting() -> None:
    """Compare the scalar-seven specialization with zero-dimensional and array shapes."""
    alpha = np.array([[2.5], [10.0], [20.0]])
    values = np.array([[0.01, 0.1, 0.4, 0.7, 0.75, 0.8]])
    expected = scipy.special.betainc(alpha, 7.0, values)
    for beta in (7.0, xp.asarray(7.0), xp.full((1, 6), 7.0)):
        actual = betainc(xp.asarray(alpha), beta, xp.asarray(values))
        assert isinstance(actual, xp.ndarray)
        np.testing.assert_allclose(asnumpy(actual), expected, rtol=5e-10, atol=5e-14)

    probabilities = np.array([[1e-10, 0.01, 0.25, 0.5, 0.9, 1 - 1e-10]])
    actual = betaincinv(xp.asarray(alpha), xp.asarray(7.0), xp.asarray(probabilities))
    np.testing.assert_allclose(
        asnumpy(actual), scipy.special.betaincinv(alpha, 7.0, probabilities), rtol=2e-9, atol=2e-13
    )


@pytest.mark.skipif(xp.__name__ != "cupy", reason="GPU kernels calculate in float64")
@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_gpu_beta_promotes_device_inputs_to_double(dtype: type) -> None:
    """Check casting, broadcasting, accuracy, and residency of both GPU functions."""
    alpha = xp.asarray([2, 10, 20], dtype=dtype)[:, None]
    beta = xp.asarray(7, dtype=dtype)
    values = xp.asarray([0.01, 0.25, 0.7, 0.75], dtype=xp.float32)[None, :]
    host_alpha, host_beta, host_values = (asnumpy(value).astype(float) for value in (alpha, beta, values))
    for function, reference in ((betainc, scipy.special.betainc), (betaincinv, scipy.special.betaincinv)):
        actual = function(alpha, beta, values)
        assert isinstance(actual, xp.ndarray)
        assert actual.dtype == xp.float64
        np.testing.assert_allclose(
            asnumpy(actual), reference(host_alpha, host_beta, host_values), rtol=2e-9, atol=2e-13
        )


@pytest.mark.parametrize("method", ["cdf", "invcdf"])
def test_beta_transforms_preserve_every_stochastic_coupling(method: str) -> None:
    """Check numerical results and coupling with all parameters stochastic."""
    alpha = StochasticScalar([2.5, 10.0, 20.0, 1000.0])
    beta = StochasticScalar([7.0, 32.0, 32.0, 1032.0])
    scale = StochasticScalar([1.0, 2.0, 3.0, 4.0])
    loc = StochasticScalar([0.0, 1.0, -2.0, 5.0])
    values = StochasticScalar([0.4, 2.5, 0.1, 6.96] if method == "cdf" else [0.1, 0.5, 0.9, 0.01])
    result = getattr(Beta(alpha, beta, scale, loc), method)(values)

    assert isinstance(result, StochasticScalar)
    assert isinstance(result.values, xp.ndarray)
    for parameter in (alpha, beta, scale, loc, values):
        assert result.coupled_variable_group is parameter.coupled_variable_group
    a, b, s, location, v = (asnumpy(value.values) for value in (alpha, beta, scale, loc, values))
    expected = (
        scipy.special.betainc(a, b, (v - location) / s)
        if method == "cdf"
        else scipy.special.betaincinv(a, b, v) * s + location
    )
    np.testing.assert_allclose(asnumpy(result.values), expected, rtol=2e-9, atol=2e-13)


@pytest.mark.parametrize("nu", [1.0, 2.0, 5.0, 10.0, 29.999, 30.0, 30.001, 100.0, 20_000.0])
def test_student_t_distribution_and_copula_transforms_match_scipy(nu: float) -> None:
    """Exercise the beta half-shape and BGRAT boundary through both call sites."""
    values = np.array([-10.0, -5.0, -2.0, -0.1, 0.0, 0.1, 2.0, 5.0, 10.0])
    distribution = StudentsT(nu, 2.0, 3.0)
    actual = distribution.cdf(StochasticScalar(2 + 3 * values))
    np.testing.assert_allclose(asnumpy(actual.values), scipy.special.stdtr(nu, values), rtol=5e-10, atol=5e-14)

    copula_values = xp.asarray(np.stack([values, -values]))
    copula_actual = _student_t_cdf(copula_values, nu)
    assert isinstance(copula_actual, xp.ndarray)
    np.testing.assert_allclose(
        asnumpy(copula_actual), scipy.special.stdtr(nu, np.stack([values, -values])), rtol=5e-10, atol=5e-14
    )

    probabilities = np.array([1e-10, 1e-6, 0.01, 0.1, 0.5, 0.9, 0.99, 1 - 1e-6, 1 - 1e-10])
    quantiles = distribution.invcdf(StochasticScalar(probabilities))
    np.testing.assert_allclose(
        asnumpy(quantiles.values), 2 + 3 * scipy.special.stdtrit(nu, probabilities), rtol=2e-9, atol=2e-10
    )


@pytest.mark.parametrize("method", ["cdf", "invcdf"])
def test_student_t_transforms_preserve_every_stochastic_coupling(method: str) -> None:
    """Retain simulation coupling through CDF and inverse beta dispatch."""
    nu = StochasticScalar([5.0, 10.0, 30.0, 100.0])
    mu = StochasticScalar([0.0, 1.0, -2.0, 5.0])
    sigma = StochasticScalar([1.0, 2.0, 3.0, 4.0])
    values = StochasticScalar([-10.0, 0.8, -1.7, 45.0] if method == "cdf" else [1e-10, 0.1, 0.9, 1 - 1e-10])
    result = getattr(StudentsT(nu, mu, sigma), method)(values)

    assert isinstance(result, StochasticScalar)
    assert isinstance(result.values, xp.ndarray)
    for parameter in (nu, mu, sigma, values):
        assert result.coupled_variable_group is parameter.coupled_variable_group
    n, m, s, v = (asnumpy(value.values) for value in (nu, mu, sigma, values))
    expected = scipy.special.stdtr(n, (v - m) / s) if method == "cdf" else m + s * scipy.special.stdtrit(n, v)
    np.testing.assert_allclose(asnumpy(result.values), expected, rtol=2e-9, atol=2e-10)


@pytest.mark.skipif(xp.__name__ != "cupy", reason="requires the CuPy backend")
def test_beta_and_student_t_array_transforms_stay_on_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise stochastic transforms with host conversion disabled."""
    alpha = StochasticScalar(xp.asarray([2.5, 10.0, 20.0, 1000.0]))
    beta = StochasticScalar(xp.asarray([7.0, 32.0, 32.0, 1032.0]))
    probabilities = StochasticScalar(xp.asarray([0.1, 0.25, 0.75, 0.9]))
    nu = StochasticScalar(xp.asarray([5.0, 10.0, 30.0, 100.0]))

    def reject_host_transfer(*args: object, **kwargs: object) -> None:
        raise AssertionError("unexpected device-to-host array transfer")

    monkeypatch.setattr(xp, "asnumpy", reject_host_transfer)
    for distribution in (Beta(alpha, beta), StudentsT(nu)):
        quantiles = distribution.invcdf(probabilities)
        recovered = distribution.cdf(quantiles)
        assert isinstance(quantiles.values, xp.ndarray)
        assert isinstance(recovered.values, xp.ndarray)
        assert recovered.coupled_variable_group is probabilities.coupled_variable_group
    assert isinstance(_student_t_cdf(nu.values, 30.0), xp.ndarray)

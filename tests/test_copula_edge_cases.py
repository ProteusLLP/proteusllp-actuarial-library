"""Copula sampling at parameters where earlier implementations failed or returned wrong values.

Each case is checked against a closed form or an exact-arithmetic reference
rather than against the sampler's own output.
"""

from __future__ import annotations

from decimal import Decimal, getcontext

import numpy as np
import pytest
from scipy import integrate, stats

from pal import copulas, set_random_seed
from pal._maths import asnumpy, xp
from pal.copulas import _logseries_from_log_complement, _sibuya_gen
from pal.distributions import Gamma
from tests._assertions import host_values


def _samples(copula: copulas.Copula, n_sims: int, seed: int = 7) -> np.ndarray:
    """Generate samples as a (dimension, n_sims) host array with a fixed seed."""
    set_random_seed(seed)
    return np.stack([host_values(margin) for margin in copula.generate(n_sims)])


def _tau(samples: np.ndarray, n: int = 30_000) -> float:
    """Kendall's tau of the first two margins, on the first n simulations."""
    return float(stats.kendalltau(samples[0][:n], samples[1][:n])[0])


# ── Joe ──────────────────────────────────────────────────────────────────


def _joe_tau(theta: float) -> float:
    """Kendall's tau of the Joe copula, 1 - 4 sum_k 1 / (k (theta k + 2) (theta (k - 1) + 2))."""
    k = np.arange(1, 2_000_000, dtype=float)
    return float(1 - 4 * np.sum(1 / (k * (theta * k + 2) * (theta * (k - 1) + 2))))


@pytest.mark.parametrize("theta", [4.0, 5.0, 10.0, 30.0])
def test_joe_samples_at_strong_dependence(theta):
    """The Sibuya sampler once raised 'lam value too large' from theta about 4 upwards."""
    samples = _samples(copulas.JoeCopula(theta, 3), 300_000)
    assert np.all(np.isfinite(samples))
    assert np.all((samples > 0) & (samples < 1))
    assert _tau(samples) == pytest.approx(_joe_tau(theta), abs=0.015)


class _HugeRateGenerator:
    """Random source whose Sibuya Poisson rate is beyond NumPy's sampler limit of about 9.2e18.

    Arrays are returned on the active backend, as PAL's random generators do.
    """

    def __init__(self) -> None:
        self._generator = np.random.default_rng(1)

    def gamma(self, shape, scale=1.0, size=None):
        # The first gamma is Gamma(alpha) for a small alpha, here tiny.
        return xp.asarray(np.full(size, 1e-30 if shape < 0.5 else 1.0))

    def exponential(self, scale=1.0, size=None):
        return xp.asarray(np.ones(size))

    def poisson(self, lam, size=None):
        return xp.asarray(self._generator.poisson(asnumpy(lam), size))


def test_sibuya_rates_beyond_the_poisson_limit_are_used_directly():
    counts = asnumpy(_sibuya_gen(0.2, 4, _HugeRateGenerator()))
    assert counts == pytest.approx(1 + 1e30, rel=1e-12)


def test_joe_inverse_generator_keeps_precision_near_one():
    """1 - u is of order t^(1/theta), so a latent variable of 1e30 must not give exactly one."""
    copula = copulas.JoeCopula(3.0, 2)
    t = np.array([1e-30, 1e-12, 1.0, 50.0])
    expected = 1 - (-np.expm1(-t)) ** (1 / 3.0)
    inverse = asnumpy(copula.generator_inv(xp.asarray(t)))
    assert np.all(inverse <= 1)
    assert inverse == pytest.approx(expected, rel=1e-12, abs=1e-15)
    # The sampled uniforms stay strictly inside the unit interval even then.
    uniform = asnumpy(copula._transform_to_uniform(xp.asarray(-t)))
    assert np.all((uniform > 0) & (uniform < 1))


# ── Frank ────────────────────────────────────────────────────────────────


def _frank_cdf(theta: float, point: tuple[float, ...]) -> float:
    """Frank CDF in 60-digit arithmetic, since double precision cancels at strong dependence."""
    getcontext().prec = 400
    t = Decimal(theta)
    prod = Decimal(1)
    for u in point:
        prod *= (-t * Decimal(u)).exp() - 1
    inner = 1 + prod / ((-t).exp() - 1) ** (len(point) - 1)
    return float(-inner.ln() / t)


def _frank_tau(theta: float) -> float:
    """Kendall's tau 1 + 4 (D1(theta) - 1) / theta, with the Debye function D1 of either sign."""
    d1 = integrate.quad(lambda x: x / np.expm1(x) if x != 0 else 1.0, 0, theta)[0] / theta
    return 1 + 4 * (d1 - 1) / theta


@pytest.mark.parametrize("theta", [-40.0, -5.0, -0.5, 0.5, 5.0, 29.0, 31.0, 100.0, 400.0])
def test_frank_matches_closed_forms_across_theta(theta):
    """Negative theta (two variables), and theta above 36.7 where NumPy's log-series sampler fails."""
    n = 200_000
    samples = _samples(copulas.FrankCopula(theta, 2), n)
    assert np.all((samples > 0) & (samples < 1))
    assert _tau(samples) == pytest.approx(_frank_tau(theta), abs=0.015)
    for point in ((0.5, 0.5), (0.9, 0.9), (0.3, 0.8)):
        p = _frank_cdf(theta, point)
        empirical = np.mean((samples[0] <= point[0]) & (samples[1] <= point[1]))
        assert empirical == pytest.approx(p, abs=4.5 * np.sqrt(p * (1 - p) / n))
    for margin in samples:
        assert stats.kstest(margin, "uniform").statistic < 0.006


def test_frank_three_variables_at_strong_dependence():
    n = 300_000
    samples = _samples(copulas.FrankCopula(50.0, 3), n)
    p = _frank_cdf(50.0, (0.9, 0.9, 0.9))
    empirical = np.mean(np.all(samples <= 0.9, axis=0))
    assert empirical == pytest.approx(p, abs=4.5 * np.sqrt(p * (1 - p) / n))


def test_frank_zero_is_independence():
    samples = _samples(copulas.FrankCopula(0.0, 3), 50_000)
    assert np.all(np.isfinite(samples))
    assert abs(_tau(samples)) < 0.02
    assert np.corrcoef(samples)[np.triu_indices(3, 1)] == pytest.approx(0.0, abs=0.02)


def test_frank_parameter_validation():
    with pytest.raises(ValueError, match="only defined for two variables"):
        copulas.FrankCopula(-2.0, 3)
    with pytest.raises(ValueError, match="at most 700"):
        copulas.FrankCopula(701.0, 2)
    with pytest.raises(ValueError, match="finite"):
        copulas.FrankCopula(float("nan"), 2)
    # A copula without a fixed dimension still checks when it is generated or applied.
    with pytest.raises(RuntimeError, match="dimension"):
        copulas.FrankCopula(-2.0).generate(10)


def test_frank_negative_theta_can_be_applied_to_two_variables():
    first, second = Gamma(2, 50).generate(5000), Gamma(2, 50).generate(5000)
    set_random_seed(3)
    copulas.FrankCopula(-8.0).apply([first, second])
    assert stats.kendalltau(host_values(first), host_values(second))[0] < -0.5


class _BackendUniforms:
    """Uniform source returning arrays on the active backend."""

    def __init__(self, seed: int) -> None:
        self._generator = np.random.default_rng(seed)

    def uniform(self, size=None):
        return xp.asarray(self._generator.uniform(size=size))


@pytest.mark.parametrize("theta", [3.0, 20.0])
def test_log_series_sampler_matches_the_distribution(theta):
    """The rearranged Kemp sampler agrees with the exact distribution where NumPy's also works."""
    n = 1_000_000
    draws = asnumpy(_logseries_from_log_complement(theta, n, _BackendUniforms(11)))
    reference = stats.logser(1 - np.exp(-theta))
    assert draws.min() >= 1
    for level in (1, 2, 3, 10, 1000):
        p = reference.cdf(level)
        assert np.mean(draws <= level) == pytest.approx(p, abs=4.5 * np.sqrt(p * (1 - p) / n) + 1e-9)


# ── Other families ───────────────────────────────────────────────────────


def test_mm1_with_theta_one_has_a_constant_mixing_variable():
    """Theta = 1 once gave values from 0 to infinity and a Kendall's tau of 1 instead of 0.5."""
    samples = _samples(copulas.MM1Copula([[1, 0], [2, 1]], 1), 100_000)
    assert np.all((samples > 0) & (samples <= 1))
    # Without mixing this is a bivariate Gumbel copula with theta = 2, tau = 1 - 1 / 2.
    assert _tau(samples) == pytest.approx(0.5, abs=0.015)


@pytest.mark.parametrize("theta", [0.003, 0.0058, 0.006])
def test_galambos_small_theta_matches_the_closed_form(theta):
    """The gamma functions in the constant overflowed below theta of about 0.0058, giving all zeros."""
    n = 10_000
    samples = _samples(copulas.GalambosCopula(theta, 2), n)
    assert np.all(np.isfinite(samples)) and samples.max() > 0.5

    def cdf(u, v):
        return u * v * np.exp(((-np.log(u)) ** -theta + (-np.log(v)) ** -theta) ** (-1 / theta))

    for point in ((0.5, 0.5), (0.9, 0.9), (0.3, 0.7)):
        p = cdf(*point)
        empirical = np.mean((samples[0] <= point[0]) & (samples[1] <= point[1]))
        assert empirical == pytest.approx(p, abs=4.5 * np.sqrt(p * (1 - p) / n))


def test_huslerreiss_does_not_modify_the_callers_matrix():
    matrix = np.array([[0.0, 200.0], [200.0, 0.0]])
    original = matrix.copy()
    copula = copulas.HuslerReissCopula(matrix)
    assert np.array_equal(matrix, original)
    assert copula.adjusted_lambda_matrix.max() <= 100 + 1e-6


def test_archimedean_copula_without_a_dimension_raises_a_clear_error():
    with pytest.raises(RuntimeError, match="dimension"):
        copulas.GumbelCopula(1.5).generate(10)


# ── Student's t quantile ─────────────────────────────────────────────────


def _student_t_grid() -> np.ndarray:
    """Probabilities near the median, throughout the body and down into the far tails."""
    return np.concatenate(
        [
            np.linspace(0.0001, 0.9999, 2001),
            0.5 + np.linspace(-1e-6, 1e-6, 401),
            10.0 ** -np.arange(5, 300, 7),
            1 - 10.0 ** -np.arange(1, 16),
        ]
    )


def _tail_relative_error(quantile: np.ndarray, probabilities: np.ndarray, dof: float) -> np.ndarray:
    """Relative error of the returned tail probability, measured in the tail it belongs to.

    Near one, the CDF resolves only one rounding step of 2.2e-16, which is 22% of a tail of
    1e-15, so the upper tail is measured with the survival function instead.
    """
    tail = np.minimum(probabilities, 1 - probabilities)
    returned = np.where(probabilities > 0.5, stats.t.sf(quantile, dof), stats.t.cdf(quantile, dof))
    return np.abs(returned - tail) / tail


@pytest.mark.parametrize("dof", [3.0, 30.0, 1000.0])
def test_student_t_quantile_from_beta_is_accurate_near_the_median_and_in_the_tails(dof):
    """The beta formulation, used on GPU, once lost up to 1e-7 of probability near the median.

    It is backend independent, so it is checked here against SciPy's CDF on whichever backend is
    active. The tolerances leave room for a GPU inverse incomplete beta function that is less exact
    than SciPy's, yet are far below the earlier error of 1e-7 near the median and, in the tails,
    relative errors of order one.
    """
    grid = _student_t_grid()
    quantile = asnumpy(copulas._student_t_ppf_from_beta(xp.asarray(grid), dof))
    assert np.all(np.isfinite(quantile))
    centre = np.abs(grid - 0.5) < 0.01
    assert np.max(np.abs(stats.t.cdf(quantile, dof) - grid)[centre]) < 2e-14
    assert np.max(_tail_relative_error(quantile, grid, dof)[~centre]) < 1e-6


@pytest.mark.parametrize("dof", [3.0, 30.0, 1000.0])
def test_student_t_quantile_is_accurate_in_the_far_tails(dof):
    """SciPy's stdtrit saturates at 1e100 and is wrong by large factors below about 1e-155."""
    lower = np.concatenate([10.0 ** -np.arange(100, 300, 7), 10.0 ** -np.arange(150, 300, 7)])
    for probabilities in (lower, 1 - np.array([1e-12, 1e-15])):
        quantile = asnumpy(copulas._student_t_ppf(xp.asarray(probabilities), dof))
        assert np.max(_tail_relative_error(quantile, probabilities, dof)) < 1e-6

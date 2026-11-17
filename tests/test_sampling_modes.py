"""Run the same stochastic contract tests under MC and randomized QMC.

These tests intentionally check distributions and coupling invariants, not
specific draws from NumPy's pseudorandom generator. Exact-sequence reference
tests elsewhere in the suite continue to test the default MC mode.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from scipy import stats

from pal import FrequencySeverityModel, StochasticScalar, config, set_random_seed
from pal._maths import asnumpy, xp
from pal.config import SamplingMethod, set_sampling_method
from pal.copulas import GaussianCopula, GumbelCopula, StudentsTCopula
from pal.distributions import (
    GPD,
    Beta,
    Binomial,
    Gamma,
    GeneralizedBetaPrime,
    HyperGeometric,
    InverseGaussian,
    LogNormal,
    NegBinomial,
    NonCentralChiSquared,
    Normal,
    Poisson,
    StudentsT,
    Uniform,
)


@pytest.fixture(params=[SamplingMethod.MC, SamplingMethod.RQMC], ids=["mc", "rqmc"])
def sampling_method(request: pytest.FixtureRequest):
    """Restore global configuration after testing both sampling algorithms."""
    original = (config.seed, config.sampling_method, config.qmc_dimensions, config.rng)
    method = request.param
    set_sampling_method(method, dimensions=8)
    set_random_seed(20261117)
    try:
        yield method
    finally:
        config.seed, config.sampling_method, config.qmc_dimensions, config.rng = original


@pytest.mark.parametrize(
    ("distribution", "expected_mean", "expected_variance", "relative_tolerance"),
    [
        (Normal(mu=2.0, sigma=3.0), 2.0, 9.0, 0.07),
        (Gamma(alpha=3.0, theta=2.0), 6.0, 12.0, 0.10),
        (Beta(alpha=2.0, beta=3.0), 0.4, 0.04, 0.10),
        (Poisson(mean=4.0), 4.0, 4.0, 0.10),
        (LogNormal(mu=0.0, sigma=0.4), float(np.exp(0.08)), float((np.exp(0.16) - 1) * np.exp(0.16)), 0.13),
        (StudentsT(nu=7, mu=1, sigma=2), 1.0, 4.0 * 7 / 5, 0.16),
        (GPD(shape=0.2, scale=3, loc=0), 3.75, 3**2 / (0.8**2 * 0.6), 0.25),
    ],
)
def test_distribution_moments_in_both_modes(
    sampling_method: SamplingMethod,
    distribution: object,
    expected_mean: float,
    expected_variance: float,
    relative_tolerance: float,
) -> None:
    """Check that QMC changes sampling, not the target probability law."""
    n = 16384
    samples = distribution.generate(n)
    actual = asnumpy(samples.values)
    assert actual.shape == (n,)
    assert np.all(np.isfinite(actual))
    assert np.mean(actual) == pytest.approx(expected_mean, rel=relative_tolerance)
    assert np.var(actual) == pytest.approx(expected_variance, rel=relative_tolerance)


def test_stochastic_chain_coupling_in_both_modes(sampling_method: SamplingMethod) -> None:
    """Generating child distributions must preserve stochastic parameter alignment."""
    n = 4096
    root = Gamma(alpha=3, theta=1.5).generate(n)
    mean = 0.5 + root * 0.7
    frequency = Poisson(mean).generate(n)
    child = Normal(mu=frequency + 1, sigma=0.5).generate(n)

    assert frequency.coupled_variable_group is root.coupled_variable_group
    assert child.coupled_variable_group is root.coupled_variable_group
    assert np.mean(asnumpy(frequency.values)) == pytest.approx(0.5 + 3 * 1.5 * 0.7, rel=0.1)
    assert np.all(asnumpy(frequency.values) >= 0)


@pytest.mark.parametrize("family", ["gaussian", "student_t", "gumbel"])
def test_copula_marginals_dependence_in_both_modes(sampling_method: SamplingMethod, family: str) -> None:
    """Check the joint law using independent rank-correlation references."""
    n = 4096
    if family == "gaussian":
        copula = GaussianCopula([[1.0, 0.65], [0.65, 1.0]])
        expected_tau = 2 / np.pi * np.arcsin(0.65)
    elif family == "student_t":
        copula = StudentsTCopula([[1.0, 0.65], [0.65, 1.0]], dof=6)
        expected_tau = 2 / np.pi * np.arcsin(0.65)
    else:
        copula = GumbelCopula(theta=2.0, dimension=2)
        expected_tau = 0.5

    generated = list(copula.generate(n))
    assert len(generated) == 2
    first, second = (asnumpy(value.values) for value in generated)
    assert np.all((first > 0) & (first < 1))
    assert np.all((second > 0) & (second < 1))
    assert np.mean(first) == pytest.approx(0.5, abs=0.025)
    assert np.mean(second) == pytest.approx(0.5, abs=0.025)
    assert stats.kendalltau(first, second).statistic == pytest.approx(expected_tau, abs=0.06)


def test_frequency_severity_stochastic_parameters_in_both_modes(sampling_method: SamplingMethod) -> None:
    """Frequency/severity parameters remain per-simulation and coupled."""
    n = 2048
    scale = StochasticScalar(xp.linspace(30, 50, n))
    count_driver = StochasticScalar(xp.linspace(1, 2, n))
    model = FrequencySeverityModel(Poisson(mean=count_driver), Gamma(alpha=2, theta=scale))
    losses = model.generate(n)
    counts = asnumpy(losses.count().values)
    aggregate = asnumpy(losses.aggregate().values)

    assert losses.n_sims == n
    assert int(counts.sum()) == len(losses.values)
    assert np.all(aggregate >= 0)
    assert np.mean(counts) == pytest.approx(1.5, abs=0.12)
    assert np.mean(aggregate) == pytest.approx(120.0, rel=0.14)
    assert scale.coupled_variable_group is losses.coupled_variable_group
    assert count_driver.coupled_variable_group is losses.coupled_variable_group


def test_copula_reordering_keeps_chain_in_both_modes(sampling_method: SamplingMethod) -> None:
    n = 1024
    driver = Gamma(alpha=2, theta=5).generate(n)
    derived = driver * 1.15 + 3
    other = LogNormal(mu=1, sigma=0.5).generate(n)
    before = asnumpy(derived.values).copy()
    GaussianCopula([[1.0, 0.5], [0.5, 1.0]]).apply([driver, other])
    assert np.allclose(asnumpy(derived.values), asnumpy(driver.values) * 1.15 + 3)
    assert np.array_equal(np.sort(asnumpy(derived.values)), np.sort(before))
    assert driver.coupled_variable_group is derived.coupled_variable_group
    assert driver.coupled_variable_group is other.coupled_variable_group


def test_date_uniform_generation_in_both_modes(sampling_method: SamplingMethod) -> None:
    """Discrete dates must remain valid when random inputs are Sobol coordinates."""
    n = 1024
    draws = Uniform(date(2026, 1, 1), date(2026, 1, 31)).generate(n)
    days = asnumpy(draws.values).astype("datetime64[D]")
    assert draws.n_sims == n
    assert days.min() >= np.datetime64("2026-01-01")
    assert days.max() <= np.datetime64("2026-01-31")


def test_sampling_dispatch_preserves_numpy_streams():
    """Moving sampling to qmc.py must preserve ordinary seeded MC draws."""
    n, seed = 128, 391
    cases = [
        (Gamma(alpha=2.5, theta=4, loc=3), lambda g: g.gamma(2.5, 4, size=n) + 3),
        (Beta(alpha=3, beta=5, scale=7, loc=2), lambda g: g.beta(3, 5, size=n) * 7 + 2),
        (StudentsT(nu=8, mu=1, sigma=2), lambda g: g.standard_t(8, size=n) * 2 + 1),
        (Poisson(mean=3.5), lambda g: g.poisson(3.5, size=n)),
        (NegBinomial(n=4, p=0.3), lambda g: g.negative_binomial(4, 0.3, size=n)),
        (Binomial(n=12, p=0.6), lambda g: g.binomial(12, 0.6, size=n)),
        (HyperGeometric(ngood=10, nbad=20, n_draws=4), lambda g: g.hypergeometric(10, 20, 4, size=n)),
        (NonCentralChiSquared(df=4, nonc=2), lambda g: g.noncentral_chisquare(4, 2, size=n)),
        (
            GeneralizedBetaPrime(a=2, b=3, p=4, q=5),
            lambda g: 3 * (g.gamma(4, 1.0, size=n) / g.gamma(5, 1.0, size=n)) ** 0.5,
        ),
    ]
    for distribution, reference in cases:
        actual = asnumpy(distribution.generate(n_sims=n, rng=np.random.default_rng(seed)).values)
        expected = reference(np.random.default_rng(seed))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_inverse_gaussian_dispatch_preserves_two_draw_algorithm():
    """The specialized inverse Gaussian construction still uses two draws."""
    n, seed, mu, shape = 128, 937, 2.0, 4.0
    generator = np.random.default_rng(seed)
    nu = generator.normal(0, 1, n) ** 2
    y = mu + mu**2 * nu / (2 * shape) - mu / (2 * shape) * np.sqrt(4 * mu * shape * nu + mu**2 * nu**2)
    uniforms = generator.uniform(0, 1, n)
    expected = np.where(uniforms <= mu / (mu + y), y, mu**2 / y)
    actual = asnumpy(InverseGaussian(mu=mu, lambda_=shape).generate(n, rng=np.random.default_rng(seed)).values)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

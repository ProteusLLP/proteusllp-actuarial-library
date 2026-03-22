"""Tests for probability distribution implementations.

Comprehensive tests for actuarial distributions including parametric tests,
CDF/inverse CDF validation, and simulation accuracy checks.
"""
# pyright: reportUnknownMemberType=false

import math

import pytest  # pyright: ignore[reportUnknownMemberType] - pytest.approx not fully typed
import scipy.special
from pal import distributions
from pal._maths import xp as np
from pal.config import set_random_seed
from scipy.special import gamma


def test_poisson() -> None:
    set_random_seed(12345678910)
    lamda = 3.5
    dist = distributions.Poisson(lamda)
    assert dist.cdf(0) == np.exp(-lamda)
    assert dist.invcdf(0) == 0
    assert np.allclose(
        dist.invcdf(dist.cdf(np.array([0, 2, 5, 10]))),
        np.array([0, 2, 5, 10]),
        1e-8,
    )
    sims = dist.generate(100000)
    assert np.isclose(np.mean(sims), lamda, 1e-3)
    assert np.isclose(np.std(sims) ** 2, lamda, 1e-2)


def test_poisson_gamma() -> None:
    """Tests the Poisson distribution with a gamma distributed lambda."""
    set_random_seed(12345678910)
    alpha = 0.5
    beta = 3
    lamda = distributions.Gamma(alpha, beta).generate(100000)
    sims = distributions.Poisson(lamda).generate(100000)
    sims_mean = np.mean(sims)
    sims_std = np.std(sims)
    assert np.isclose(sims_mean, alpha * beta, 1e-2)
    assert np.isclose(sims_std**2, alpha * beta + alpha * beta**2, 1e-2)
    assert sims.coupled_variable_group == lamda.coupled_variable_group


def test_gamma_exp() -> None:
    """Tests the Exponential distribution with a gamma distributed lambda."""
    set_random_seed(12345678910)
    alpha = 1.5
    beta = 3
    lamda = distributions.Gamma(alpha, beta).generate(1000000)
    sims = distributions.Exponential(lamda).generate(1000000)
    sims_mean = np.mean(sims)
    sims_std = np.std(sims)
    assert np.isclose(sims_mean, alpha * beta, 1e-2)
    assert np.isclose(sims_std**2, (2 * alpha + alpha**2) * beta**2, 1e-2)
    assert sims.coupled_variable_group == lamda.coupled_variable_group


def test_beta() -> None:
    set_random_seed(12345678910)
    alpha = 2.0
    beta = 3.0
    scale = 10000000.0
    loc = 1000000.0
    dist = distributions.Beta(alpha, beta, scale, loc)
    assert dist.cdf(1000000) == 0.0
    assert dist.invcdf(0) == 1000000
    assert np.allclose(
        dist.invcdf(dist.cdf(np.array([1234560.1, 2345670, 3456780]))),
        np.array([1234560.1, 2345670, 3456780]),
        1e-8,
    )

    sims = dist.generate(1000000)
    assert np.allclose(np.mean(sims), alpha / (alpha + beta) * scale + loc, 1e-3)
    assert np.allclose(
        np.std(sims),
        math.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))) * scale,
        1e-3,
    )


def test_gpd() -> None:
    set_random_seed(12345678910)
    shape = 0.25
    scale = 100000
    threshold = 1000000
    dist = distributions.GPD(shape, scale, threshold)
    assert dist.cdf(1000000) == 0.0
    assert dist.cdf(1500000) == pytest.approx(0.960981557689, 1e-4)
    assert dist.invcdf(0) == 1000000
    assert dist.invcdf(0.960981557689) == pytest.approx(1500000, 1e-4)

    sims = dist.generate(100000000)
    assert np.mean(sims) == pytest.approx(scale / (1 - shape) + threshold, 1e-3)
    assert np.std(sims) == pytest.approx(scale / (1 - shape) / math.sqrt(1 - 2 * shape), 1e-3)


def test_burr() -> None:
    set_random_seed(12345678910)
    power = 2
    shape = 3
    scale = 100000
    loc = 1000000
    dist = distributions.Burr(power, shape, scale, loc)
    assert dist.cdf(1000000) == 0.0
    assert dist.cdf(1500000) == pytest.approx(0.9999431042330451, 1e-8)
    assert dist.invcdf(0) == 1000000
    assert dist.invcdf(0.9999431042330451) == pytest.approx(1500000, 1e-8)

    sims = dist.generate(10000000)
    assert np.mean(sims) == pytest.approx(
        shape * scipy.special.beta(shape - 1 / power, 1 + 1 / power) * scale + loc, 1e-3
    )
    assert np.std(sims) == pytest.approx(
        math.sqrt(
            shape * scipy.special.beta(shape - 2 / power, 1 + 2 / power)
            - shape**2 * scipy.special.beta(shape - 1 / power, 1 + 1 / power) ** 2
        )
        * scale,
        1e-3,
    )


def test_inverse_burr() -> None:
    set_random_seed(12345678910)
    power = 4
    shape = 5
    scale = 100000
    loc = 1000000
    dist = distributions.InverseBurr(power, shape, scale, loc)
    assert dist.cdf(1000000) == 0.0
    assert dist.invcdf(0) == 1000000

    assert dist.invcdf(0.5) == scale * (1 / (2 ** (1 / shape) - 1)) ** (1 / power) + loc

    sims = dist.generate(10000000)

    assert np.mean(sims) == pytest.approx(
        gamma(1 - 1 / power) * gamma(shape + 1 / power) / gamma(shape) * scale + loc,
        1e-3,
    )
    assert np.std(sims) == pytest.approx(
        math.sqrt(
            gamma(1 - 2 / power) * gamma(shape + 2 / power) / gamma(shape)
            - (gamma(1 - 1 / power) * gamma(shape + 1 / power) / gamma(shape)) ** 2
        )
        * scale,
        1e-3,
    )


def test_logistic() -> None:
    set_random_seed(12345678910)
    mu = 2.5
    sigma = 2
    dist = distributions.Logistic(mu, sigma)
    assert dist.cdf(2.5) == 0.5
    assert dist.invcdf(0.5) == 2.5
    assert np.allclose(
        dist.invcdf(dist.cdf(np.array([1.1, 2, 3]))),
        np.array([1.1, 2, 3]),
    )

    sims = dist.generate(10000000)
    sims_mean = np.mean(sims)
    sims_std = np.std(sims)
    assert np.isclose(sims_mean, mu, 1e-3)
    assert np.isclose(sims_std, np.pi * sigma / np.sqrt(3), 1e-3)


def test_log_logistic() -> None:
    set_random_seed(12345678910)
    shape = 4
    scale = 100000
    loc = 1000000
    dist = distributions.LogLogistic(shape, scale, loc)
    assert dist.cdf(1000000) == 0.0
    assert dist.cdf(1500000) == pytest.approx(0.9984025559105432, 1e-8)
    assert dist.invcdf(0) == 1000000
    assert dist.invcdf(0.5) == scale + loc
    assert dist.invcdf(0.9984025559105432) == pytest.approx(1500000, 1e-8)

    sims = dist.generate(100000000)
    assert np.mean(sims) == pytest.approx(scipy.special.beta(1 - 1 / shape, 1 + 1 / shape) * scale + loc, 1e-3)
    assert np.std(sims) == pytest.approx(
        math.sqrt(
            scipy.special.beta(1 - 2 / shape, 1 + 2 / shape) - scipy.special.beta(1 - 1 / shape, 1 + 1 / shape) ** 2
        )
        * scale,
        1e-3,
    )


def test_para_logistic() -> None:
    set_random_seed(12345678910)
    shape = 2.5
    scale = 100000
    loc = 1000000
    dist = distributions.Paralogistic(shape, scale, loc)

    assert dist.cdf(1000000) == 0.0
    assert dist.invcdf(0) == 1000000
    assert dist.invcdf(dist.cdf(np.array([1234560.1, 2345670, 3456780]))) == pytest.approx(
        np.array([1234560.1, 2345670, 3456780]), 1e-8
    )

    sims = dist.generate(100000000)

    assert np.mean(sims) == pytest.approx(
        scale * gamma(1 + 1 / shape) * gamma(shape - 1 / shape) / gamma(shape) + loc,
        1e-5,
    )


def test_inverse_para_logistic() -> None:
    set_random_seed(12345678910)
    shape = 5
    scale = 100000
    loc = 1000000
    dist = distributions.InverseParalogistic(shape, scale, loc)

    assert dist.cdf(1000000) == 0.0
    assert dist.invcdf(0) == 1000000
    assert dist.invcdf(dist.cdf(np.array([1234560.1, 2345670, 3456780]))) == pytest.approx(
        np.array([1234560.1, 2345670, 3456780]), 1e-8
    )

    sims = dist.generate(100000000)

    assert np.mean(sims) == pytest.approx(
        scale * gamma(shape + 1 / shape) * gamma(1 - 1 / shape) / gamma(shape) + loc,
        1e-3,
    )
    assert np.std(sims) == pytest.approx(
        scale
        * np.sqrt(
            (gamma(shape + 2 / shape) * gamma(1 - 2 / shape) / gamma(shape))
            - (gamma(shape + 1 / shape) * gamma(1 - 1 / shape) / gamma(shape)) ** 2
        ),
        1e-3,
    )


def test_weibull() -> None:
    set_random_seed(12345678910)
    shape = 2
    scale = 1000000
    loc = 1000000
    dist = distributions.Weibull(shape, scale, loc)

    assert dist.cdf(1000000) == 0.0
    assert dist.invcdf(0) == 1000000
    assert dist.invcdf(dist.cdf(np.array([1234560.1, 2345670, 3456780]))) == pytest.approx(
        np.array([1234560.1, 2345670, 3456780]), 1e-8
    )

    sims = dist.generate(100000000)

    assert np.mean(sims) == pytest.approx(scale * gamma(1 + 1 / shape) + loc, 1e-3)
    assert np.std(sims) == pytest.approx(scale * np.sqrt(gamma(1 + 2 / shape) - (gamma(1 + 1 / shape)) ** 2), 1e-3)


def test_inverse_weibull() -> None:
    set_random_seed(12345678910)
    shape = 4
    scale = 1000000
    loc = 1000000
    dist = distributions.InverseWeibull(shape, scale, loc)

    assert dist.cdf(1000000) == 0.0
    assert dist.invcdf(0) == 1000000
    assert dist.invcdf(dist.cdf(np.array([1234560.1, 2345670, 3456780]))) == pytest.approx(
        np.array([1234560.1, 2345670, 3456780]), 1e-8
    )

    sims = dist.generate(100000000)

    assert np.mean(sims) == pytest.approx(scale * gamma(1 - 1 / shape) + loc, 1e-3)
    assert np.std(sims) == pytest.approx(scale * np.sqrt(gamma(1 - 2 / shape) - (gamma(1 - 1 / shape)) ** 2), 1e-3)


def test_exponential() -> None:
    set_random_seed(12345678910)
    scale = 1000000
    loc = 1000000
    dist = distributions.Exponential(scale, loc)

    assert dist.cdf(1000000) == 0.0
    assert dist.invcdf(0) == 1000000
    assert dist.invcdf(dist.cdf(np.array([1234560.1, 2345670, 3456780]))) == pytest.approx(
        np.array([1234560.1, 2345670, 3456780]), 1e-8
    )

    sims = dist.generate(100000000)

    assert np.mean(sims) == pytest.approx(scale + loc, 1e-3)
    assert np.std(sims) == pytest.approx(scale, 1e-3)


def test_inverse_exponential() -> None:
    set_random_seed(12345678910)
    scale = 1000000
    loc = 1000000
    dist = distributions.InverseExponential(scale, loc)

    assert dist.cdf(1000000) == 0.0
    assert dist.invcdf(0) == 1000000
    assert dist.invcdf(dist.cdf(np.array([1234560.1, 2345670, 3456780]))) == pytest.approx(
        np.array([1234560.1, 2345670, 3456780]), 1e-8
    )


def test_gamma() -> None:
    set_random_seed(12345678910)
    scale = 1000000
    shape = 4.5
    loc = 1000000
    dist = distributions.Gamma(shape, scale, loc)

    assert dist.cdf(1000000) == 0.0
    assert dist.invcdf(0) == 1000000
    assert np.allclose(
        dist.invcdf(dist.cdf(np.array([1234560.1, 2345670, 3456780]))),
        np.array([1234560.1, 2345670, 3456780]),
        1e-8,
    )

    sims = dist.generate(10000000)

    assert np.allclose(np.mean(sims), scale * shape + loc, 1e-3)
    assert np.allclose(np.std(sims), scale * np.sqrt(shape), 1e-3)


def test_log_normal() -> None:
    set_random_seed(12345678910)
    mu = 8
    sigma = 1.25
    dist = distributions.LogNormal(mu, sigma)

    assert dist.cdf(0) == 0.0
    assert dist.invcdf(0) == 0
    assert np.allclose(
        dist.invcdf(dist.cdf(np.array([1234560.1, 2345670, 3456780]))),
        np.array([1234560.1, 2345670, 3456780]),
        1e-8,
    )

    sims = dist.generate(100000000)

    mean = np.exp(mu + 0.5 * sigma**2)
    sd = np.sqrt((np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2))

    assert np.allclose(np.mean(sims), mean, 1e-3)
    assert np.allclose(np.std(sims), sd, 1e-3)


def test_inverse_gamma() -> None:
    set_random_seed(12345678910)
    scale = 1000000
    shape = 3.5
    loc = 1000000
    dist = distributions.InverseGamma(shape, scale, loc)

    assert dist.cdf(1000000) == 0.0
    assert dist.invcdf(0) == 1000000
    assert np.allclose(
        dist.invcdf(dist.cdf(np.array([1234560.1, 2345670, 3456780]))),
        np.array([1234560.1, 2345670, 3456780]),
        1e-8,
    )

    sims = dist.generate(10000000)

    assert np.allclose(np.mean(sims), scale * gamma(shape - 1) / gamma(shape) + loc, 1e-3)
    assert np.allclose(
        np.std(sims),
        scale * np.sqrt(gamma(shape - 2) / gamma(shape) - (gamma(shape - 1) / gamma(shape)) ** 2),
        1e-3,
    )


def test_gev_gumbel() -> None:
    """Test GEV distribution with shape=0 (Gumbel case)."""
    set_random_seed(12345678910)
    shape = 0.0  # Gumbel
    scale = 100000.0
    loc = 1000000.0
    dist = distributions.GEV(shape, scale, loc)

    # Test known CDF values for Gumbel: F(x) = exp(-exp(-(x-μ)/σ))
    assert dist.cdf(loc) == pytest.approx(np.exp(-1), 1e-8)
    assert dist.cdf(loc + scale) == pytest.approx(np.exp(-np.exp(-1)), 1e-8)
    assert dist.cdf(loc - scale) == pytest.approx(np.exp(-np.exp(1)), 1e-8)

    # Test inverse CDF
    assert dist.invcdf(0.5) == pytest.approx(loc - scale * np.log(-np.log(0.5)), 1e-6)
    assert dist.invcdf(0.9) == pytest.approx(loc - scale * np.log(-np.log(0.9)), 1e-6)

    # Test round-trip
    assert np.allclose(
        dist.invcdf(dist.cdf(np.array([loc - 50000, loc, loc + 50000, loc + 150000]))),
        np.array([loc - 50000, loc, loc + 50000, loc + 150000]),
        1e-6,
    )

    # Test statistical moments for Gumbel
    # Mean: μ + σ * γ (where γ ≈ 0.5772 is Euler-Mascheroni constant)
    # Std: σ * π / sqrt(6)
    sims = dist.generate(10000000)
    euler_gamma = 0.5772156649015329
    expected_mean = loc + scale * euler_gamma
    expected_std = scale * np.pi / np.sqrt(6)

    assert np.mean(sims) == pytest.approx(expected_mean, rel=1e-3)
    assert np.std(sims) == pytest.approx(expected_std, rel=1e-3)


def test_gev_frechet() -> None:
    """Test GEV distribution with shape>0 (Fréchet case)."""
    set_random_seed(12345678910)
    shape = 0.2  # Fréchet
    scale = 100000.0
    loc = 1000000.0
    dist = distributions.GEV(shape, scale, loc)

    # Test CDF at specific points
    x = loc + scale
    expected_cdf = np.exp(-np.power(1 + shape, -1 / shape))
    assert dist.cdf(x) == pytest.approx(expected_cdf, 1e-8)

    # Test inverse CDF
    u = 0.5
    expected_x = loc + scale * (np.power(-np.log(u), -shape) - 1) / shape
    assert dist.invcdf(u) == pytest.approx(expected_x, 1e-6)

    # Test round-trip
    test_points = np.array([loc + 10000, loc + 50000, loc + 100000, loc + 200000])
    assert np.allclose(dist.invcdf(dist.cdf(test_points)), test_points, 1e-6)

    # Test statistical moments
    # Mean: μ + σ * (Γ(1-ξ) - 1) / ξ for ξ < 1
    # Variance: σ² * (Γ(1-2ξ) - Γ(1-ξ)²) / ξ² for ξ < 0.5
    sims = dist.generate(10000000)
    expected_mean = loc + scale * (gamma(1 - shape) - 1) / shape
    expected_var = scale**2 * (gamma(1 - 2 * shape) - gamma(1 - shape) ** 2) / shape**2

    assert np.mean(sims) == pytest.approx(expected_mean, rel=1e-3)
    assert np.var(sims) == pytest.approx(expected_var, rel=1e-2)


def test_gev_weibull() -> None:
    """Test GEV distribution with shape<0 (Weibull case)."""
    set_random_seed(12345678910)
    shape = -0.15  # Weibull
    scale = 100000.0
    loc = 1000000.0
    dist = distributions.GEV(shape, scale, loc)

    # Test CDF at specific points
    x = loc + scale / 2
    z = (x - loc) / scale
    expected_cdf = np.exp(-np.power(1 + shape * z, -1 / shape))
    assert dist.cdf(x) == pytest.approx(expected_cdf, 1e-8)

    # Test inverse CDF
    u = 0.7
    expected_x = loc + scale * (np.power(-np.log(u), -shape) - 1) / shape
    assert dist.invcdf(u) == pytest.approx(expected_x, 1e-6)

    # Test round-trip
    test_points = np.array([loc + 10000, loc + 50000, loc + 100000, loc + 200000])
    assert np.allclose(dist.invcdf(dist.cdf(test_points)), test_points, 1e-6)

    # Test statistical moments (same formulas as Fréchet when ξ < 0)
    sims = dist.generate(10000000)
    expected_mean = loc + scale * (gamma(1 - shape) - 1) / shape
    expected_var = scale**2 * (gamma(1 - 2 * shape) - gamma(1 - shape) ** 2) / shape**2

    assert np.mean(sims) == pytest.approx(expected_mean, rel=1e-3)
    assert np.var(sims) == pytest.approx(expected_var, rel=1e-2)


def test_studentst_standard() -> None:
    """Test Student's t distribution (standard, centered at 0)."""
    set_random_seed(12345678910)
    nu = 5.0  # degrees of freedom
    mu = 0.0
    sigma = 1.0
    dist = distributions.StudentsT(nu, mu, sigma)

    # Test CDF at 0 should be 0.5 for centered distribution
    assert dist.cdf(0.0) == pytest.approx(0.5, 1e-8)

    # Test symmetry: CDF(-x) = 1 - CDF(x)
    x = 1.5
    assert dist.cdf(-x) == pytest.approx(1 - dist.cdf(x), 1e-8)

    # Test inverse CDF
    assert dist.invcdf(0.5) == pytest.approx(0.0, 1e-8)

    # Test round-trip
    test_points = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert np.allclose(dist.invcdf(dist.cdf(test_points)), test_points, 1e-8)

    # Test statistical moments
    # Mean: μ for ν > 1
    # Variance: ν/(ν-2) * σ² for ν > 2
    sims = dist.generate(10000000)
    expected_mean = mu
    expected_var = nu / (nu - 2) * sigma**2

    assert np.mean(sims) == pytest.approx(expected_mean, abs=1e-3)
    assert np.var(sims) == pytest.approx(expected_var, rel=1e-2)


def test_studentst_general() -> None:
    """Test Student's t distribution with non-standard location and scale."""
    set_random_seed(12345678910)
    nu = 10.0
    mu = 5000.0
    sigma = 100.0
    dist = distributions.StudentsT(nu, mu, sigma)

    # Test CDF at location should be 0.5
    assert dist.cdf(mu) == pytest.approx(0.5, 1e-8)

    # Test inverse CDF
    assert dist.invcdf(0.5) == pytest.approx(mu, 1e-6)

    # Test round-trip
    test_points = np.array([mu - 200, mu - 100, mu, mu + 100, mu + 200])
    assert np.allclose(dist.invcdf(dist.cdf(test_points)), test_points, 1e-6)

    # Test statistical moments
    sims = dist.generate(10000000)
    expected_mean = mu
    expected_var = nu / (nu - 2) * sigma**2

    assert np.mean(sims) == pytest.approx(expected_mean, rel=1e-3)
    assert np.var(sims) == pytest.approx(expected_var, rel=1e-2)


def test_studentst_heavy_tails() -> None:
    """Test that Student's t has heavier tails than normal."""
    set_random_seed(12345678910)
    nu = 3.0
    mu = 0.0
    sigma = 1.0
    t_dist = distributions.StudentsT(nu, mu, sigma)
    normal_dist = distributions.Normal(mu, sigma)

    # At extreme values, t distribution should have higher probability
    x_extreme = 3.0
    assert t_dist.cdf(x_extreme) < normal_dist.cdf(x_extreme)
    assert t_dist.cdf(-x_extreme) > normal_dist.cdf(-x_extreme)

    # Generate samples and check for more extreme values in t distribution
    t_sims = t_dist.generate(1000000)
    normal_sims = normal_dist.generate(1000000)

    # Count proportion beyond 3 standard deviations
    t_extreme = np.mean(np.abs(t_sims.values) > 3)
    normal_extreme = np.mean(np.abs(normal_sims.values) > 3)

    # t distribution should have more extreme values
    assert t_extreme > normal_extreme


def test_inversegaussian() -> None:
    """Test Inverse Gaussian distribution."""
    set_random_seed(12345678910)
    mu = 1000.0
    lambda_ = 500.0
    dist = distributions.InverseGaussian(mu, lambda_)

    # Test CDF at specific points
    x = mu
    # At x = μ, the CDF has a specific form but it's complex
    # Just verify it's between 0 and 1
    cdf_at_mu = dist.cdf(x)
    assert 0 < cdf_at_mu < 1

    # Test CDF is monotonically increasing
    x_values = np.array([mu / 2, mu, 2 * mu, 3 * mu])
    cdf_values = dist.cdf(x_values)
    assert np.all(np.diff(cdf_values) > 0)

    # Test that invcdf raises NotImplementedError
    with pytest.raises(NotImplementedError):
        dist.invcdf(0.5)

    # Test statistical moments
    # Mean: μ
    # Variance: μ³/λ
    sims = dist.generate(10000000)
    expected_mean = mu
    expected_var = mu**3 / lambda_

    assert np.mean(sims) == pytest.approx(expected_mean, rel=1e-3)
    assert np.var(sims) == pytest.approx(expected_var, rel=1e-2)


def test_inversegaussian_cdf_properties() -> None:
    """Test mathematical properties of Inverse Gaussian CDF."""
    set_random_seed(12345678910)
    mu = 500.0
    lambda_ = 100.0
    dist = distributions.InverseGaussian(mu, lambda_)

    # Test CDF approaches 0 as x approaches 0 from the right
    small_x = 0.01
    assert dist.cdf(small_x) < 0.01

    # Test CDF approaches 1 as x approaches infinity
    large_x = mu * 1000
    assert dist.cdf(large_x) > 0.9999

    # Test CDF at mean is greater than 0.5 (distribution is right-skewed)
    # For inverse Gaussian, CDF(μ) ≈ 0.668 (depends on λ/μ ratio)
    cdf_at_mean = dist.cdf(mu)
    assert 0.5 < cdf_at_mean < 0.8

def test_hypergeometric() -> None:
    """Test HyperGeometric distribution implementation."""
    set_random_seed(12345)
    
    # Parameters
    ngood = 50
    nbad = 60
    n_draws = 30
    
    dist = distributions.HyperGeometric(ngood, nbad, n_draws)
    
    # Test generation consistency
    sims = dist.generate(n_sims=10000)
    assert sims.n_sims == 10000
    
    # Check mean/variance
    # Theoretical mean: n * (K / N) where n=draws, K=good, N=total
    population = ngood + nbad
    expected_mean = n_draws * (ngood / population)
    
    # Variance: n * (K/N) * ((N-K)/N) * ((N-n)/(N-1))
    p = ngood / population
    expected_var = n_draws * p * (1 - p) * ((population - n_draws) / (population - 1))
    
    print(f"HyperMean: {np.mean(sims)}, Expected: {expected_mean}")
    assert np.mean(sims) == pytest.approx(expected_mean, rel=0.02)
    assert np.var(sims) == pytest.approx(expected_var, rel=0.05)
    
    # Check CDF against Scipy
    # Scipy hypergeom(M, n, N) -> M=population, n=ngood, N=draws
    from scipy.stats import hypergeom
    rv = hypergeom(population, ngood, n_draws)
    
    # Check a few points
    for k in [10, 15, 20]:
        assert dist.cdf(k) == pytest.approx(rv.cdf(k))
        # Inverse CDF check
        p_val = rv.cdf(k)
        # Verify round trip or direct PPF match
        assert dist.invcdf(p_val) == pytest.approx(rv.ppf(p_val))


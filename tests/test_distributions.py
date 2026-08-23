"""Tests for probability distribution implementations.

Comprehensive tests for actuarial distributions including parametric tests,
CDF/inverse CDF validation, and simulation accuracy checks.
"""
# pyright: reportUnknownMemberType=false

import math

import pytest  # pyright: ignore[reportUnknownMemberType] - pytest.approx not fully typed
import scipy.integrate
import scipy.special
from scipy.special import gamma

from pal import distributions
from pal._maths import xp as np
from pal.config import set_random_seed
from pal.stochastic_scalar import StochasticScalar
from tests._assertions import allclose


def test_poisson() -> None:
    set_random_seed(12345678910)
    lamda = 3.5
    dist = distributions.Poisson(lamda)
    assert dist.cdf(0) == np.exp(-lamda)
    assert dist.invcdf(0) == 0
    assert allclose(
        dist.invcdf(dist.cdf(StochasticScalar([0, 2, 5, 10]))),
        StochasticScalar([0, 2, 5, 10]),
        1e-8,
    )
    sims = dist.generate(100000)
    assert np.isclose(np.mean(sims), lamda, 1e-3)
    assert np.isclose(np.std(sims) ** 2, lamda, 1e-2)


def test_poisson_gamma() -> None:
    """Tests the Poisson distribution with a gamma distributed lambda."""
    set_random_seed(12345678910)
    n_sims = 100000
    alpha = 0.5
    beta = 3
    lamda = distributions.Gamma(alpha, beta).generate(n_sims)
    sims = distributions.Poisson(lamda).generate(n_sims)
    sims_mean = np.mean(sims)
    sims_variance = np.var(sims)

    expected_mean = alpha * beta
    expected_variance = alpha * beta * (1 + beta)
    p = 1 / (1 + beta)
    excess_kurtosis = 6 / alpha + p**2 / (alpha * (1 - p))
    mean_se = math.sqrt(expected_variance / n_sims)
    variance_se = expected_variance * math.sqrt((excess_kurtosis + 2) / n_sims)

    assert sims_mean == pytest.approx(expected_mean, abs=5 * mean_se)
    assert sims_variance == pytest.approx(expected_variance, abs=5 * variance_se)
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
    assert allclose(
        dist.invcdf(dist.cdf(StochasticScalar([1234560.1, 2345670, 3456780]))),
        StochasticScalar([1234560.1, 2345670, 3456780]),
        1e-8,
    )

    sims = dist.generate(1000000)
    assert allclose(np.mean(sims), alpha / (alpha + beta) * scale + loc, 1e-3)
    assert allclose(
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
    assert allclose(
        dist.invcdf(dist.cdf(StochasticScalar([1.1, 2, 3]))),
        StochasticScalar([1.1, 2, 3]),
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
    assert dist.invcdf(dist.cdf(StochasticScalar([1234560.1, 2345670, 3456780]))) == pytest.approx(
        StochasticScalar([1234560.1, 2345670, 3456780]), 1e-8
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
    assert dist.invcdf(dist.cdf(StochasticScalar([1234560.1, 2345670, 3456780]))) == pytest.approx(
        StochasticScalar([1234560.1, 2345670, 3456780]), 1e-8
    )

    sims = dist.generate(100000000)

    assert np.mean(sims) == pytest.approx(
        scale * gamma(shape + 1 / shape) * gamma(1 - 1 / shape) / gamma(shape) + loc,
        1e-3,
    )
    assert np.std(sims) == pytest.approx(
        float(
            scale
            * np.sqrt(
                (gamma(shape + 2 / shape) * gamma(1 - 2 / shape) / gamma(shape))
                - (gamma(shape + 1 / shape) * gamma(1 - 1 / shape) / gamma(shape)) ** 2
            )
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
    assert dist.invcdf(dist.cdf(StochasticScalar([1234560.1, 2345670, 3456780]))) == pytest.approx(
        StochasticScalar([1234560.1, 2345670, 3456780]), 1e-8
    )

    sims = dist.generate(100000000)

    assert np.mean(sims) == pytest.approx(scale * gamma(1 + 1 / shape) + loc, 1e-3)
    assert np.std(sims) == pytest.approx(
        float(scale * np.sqrt(gamma(1 + 2 / shape) - (gamma(1 + 1 / shape)) ** 2)), 1e-3
    )


def test_inverse_weibull() -> None:
    set_random_seed(12345678910)
    shape = 4
    scale = 1000000
    loc = 1000000
    dist = distributions.InverseWeibull(shape, scale, loc)

    assert dist.cdf(1000000) == 0.0
    assert dist.invcdf(0) == 1000000
    assert dist.invcdf(dist.cdf(StochasticScalar([1234560.1, 2345670, 3456780]))) == pytest.approx(
        StochasticScalar([1234560.1, 2345670, 3456780]), 1e-8
    )

    sims = dist.generate(100000000)

    assert np.mean(sims) == pytest.approx(scale * gamma(1 - 1 / shape) + loc, 1e-3)
    assert np.std(sims) == pytest.approx(
        float(scale * np.sqrt(gamma(1 - 2 / shape) - (gamma(1 - 1 / shape)) ** 2)), 1e-3
    )


def test_exponential() -> None:
    set_random_seed(12345678910)
    scale = 1000000
    loc = 1000000
    dist = distributions.Exponential(scale, loc)

    assert dist.cdf(1000000) == 0.0
    assert dist.invcdf(0) == 1000000
    assert dist.invcdf(dist.cdf(StochasticScalar([1234560.1, 2345670, 3456780]))) == pytest.approx(
        StochasticScalar([1234560.1, 2345670, 3456780]), 1e-8
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
    assert dist.invcdf(dist.cdf(StochasticScalar([1234560.1, 2345670, 3456780]))) == pytest.approx(
        StochasticScalar([1234560.1, 2345670, 3456780]), 1e-8
    )


def test_gamma() -> None:
    set_random_seed(12345678910)
    scale = 1000000
    shape = 4.5
    loc = 1000000
    dist = distributions.Gamma(shape, scale, loc)

    assert dist.cdf(1000000) == 0.0
    assert dist.invcdf(0) == 1000000
    assert allclose(
        dist.invcdf(dist.cdf(StochasticScalar([1234560.1, 2345670, 3456780]))),
        StochasticScalar([1234560.1, 2345670, 3456780]),
        1e-8,
    )

    sims = dist.generate(10000000)

    assert allclose(np.mean(sims), scale * shape + loc, 1e-3)
    assert allclose(np.std(sims), scale * np.sqrt(shape), 1e-3)


@pytest.mark.skipif(np.__name__ == "cupy", reason="noncentral chi-squared CDF is CPU-only")
def test_noncentral_chi_squared_numerical_values() -> None:
    """Test noncentral chi-squared CDF and quantiles against fixed values."""
    dist = distributions.NonCentralChiSquared(df=5.0, nonc=2.5)
    points = StochasticScalar([1.0, 5.0, 10.0, 20.0])
    probabilities = StochasticScalar([0.01, 0.1, 0.5, 0.9, 0.99])

    assert allclose(
        dist.cdf(points),
        StochasticScalar(
            [
                0.012707864888969142,
                0.33342145852973737,
                0.7551199172247817,
                0.9848319619365804,
            ]
        ),
        rtol=1e-12,
    )
    assert allclose(
        dist.invcdf(probabilities),
        StochasticScalar(
            [
                0.9020641591262842,
                2.5664768724950924,
                6.666110074584612,
                13.525482664406804,
                21.33367317975303,
            ]
        ),
        rtol=1e-12,
    )
    assert allclose(dist.cdf(dist.invcdf(probabilities)), probabilities, rtol=1e-11)
    assert dist.cdf(0.0) == 0.0
    assert dist.invcdf(0.0) == 0.0


def test_noncentral_chi_squared_theoretical_moments() -> None:
    """Test simulated noncentral chi-squared moments against theory."""
    set_random_seed(12345678910)
    df = 5.0
    nonc = 2.5
    dist = distributions.NonCentralChiSquared(df=df, nonc=nonc)
    sims = dist.generate(200000)

    assert np.mean(sims) == pytest.approx(df + nonc, rel=5e-3)
    assert np.var(sims) == pytest.approx(2 * (df + 2 * nonc), rel=1e-2)


def test_noncentral_chi_squared_stochastic_parameters() -> None:
    """Test generation with scenario-varying distribution parameters."""
    set_random_seed(12345678910)
    df = StochasticScalar([2.0, 4.0, 6.0])
    nonc = StochasticScalar([0.0, 1.0, 2.0])
    sims = distributions.NonCentralChiSquared(df=df, nonc=nonc).generate(3)

    assert sims.n_sims == 3
    assert bool(np.all(sims.values >= 0))
    assert sims.coupled_variable_group == df.coupled_variable_group
    assert sims.coupled_variable_group == nonc.coupled_variable_group


@pytest.mark.skipif(np.__name__ == "cupy", reason="noncentral chi-squared CDF is CPU-only")
@pytest.mark.parametrize("method, argument", [("cdf", 5.0), ("invcdf", 0.5)])
def test_noncentral_chi_squared_stochastic_parameter_coupling(method: str, argument: float) -> None:
    """Test results are coupled to every stochastic input and parameter."""
    df = StochasticScalar([2.0, 4.0, 6.0])
    nonc = StochasticScalar([0.0, 1.0, 2.0])
    stochastic_argument = StochasticScalar([argument] * 3)
    dist = distributions.NonCentralChiSquared(df=df, nonc=nonc)

    result = getattr(dist, method)(stochastic_argument)

    assert isinstance(result, StochasticScalar)
    assert result.coupled_variable_group is stochastic_argument.coupled_variable_group
    assert result.coupled_variable_group is df.coupled_variable_group
    assert result.coupled_variable_group is nonc.coupled_variable_group


@pytest.mark.skipif(np.__name__ == "cupy", reason="noncentral chi-squared CDF is CPU-only")
def test_noncentral_chi_squared_stochastic_parameter_with_scalar_argument() -> None:
    """Test a stochastic parameter produces a coupled stochastic CDF result."""
    df = StochasticScalar([2.0, 4.0, 6.0])
    result = distributions.NonCentralChiSquared(df=df, nonc=1.0).cdf(5.0)

    assert isinstance(result, StochasticScalar)
    assert result.coupled_variable_group is df.coupled_variable_group


def test_noncentral_chi_squared_generator() -> None:
    """Test construction through the named continuous generator."""
    generator = distributions.ContinuousDistributionGenerator("noncentralchisquared", [5.0, 2.5])

    assert generator.generate(10).n_sims == 10


@pytest.mark.parametrize("df, nonc", [(0.0, 1.0), (-1.0, 1.0), (2.0, -0.1)])
def test_noncentral_chi_squared_rejects_invalid_parameters(df: float, nonc: float) -> None:
    """Test validation of degrees of freedom and noncentrality."""
    with pytest.raises(ValueError):
        distributions.NonCentralChiSquared(df=df, nonc=nonc).generate(1)


def test_log_normal() -> None:
    set_random_seed(12345678910)
    mu = 8
    sigma = 1.25
    dist = distributions.LogNormal(mu, sigma)

    assert dist.cdf(0) == 0.0
    assert dist.invcdf(0) == 0
    assert allclose(
        dist.invcdf(dist.cdf(StochasticScalar([1234560.1, 2345670, 3456780]))),
        StochasticScalar([1234560.1, 2345670, 3456780]),
        1e-8,
    )

    sims = dist.generate(100000000)

    mean = np.exp(mu + 0.5 * sigma**2)
    sd = np.sqrt((np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2))

    assert allclose(np.mean(sims), mean, 1e-3)
    assert allclose(np.std(sims), sd, 1e-3)


def test_inverse_gamma() -> None:
    set_random_seed(12345678910)
    scale = 1000000
    shape = 3.5
    loc = 1000000
    dist = distributions.InverseGamma(shape, scale, loc)

    assert dist.cdf(1000000) == 0.0
    assert dist.invcdf(0) == 1000000
    assert allclose(
        dist.invcdf(dist.cdf(StochasticScalar([1234560.1, 2345670, 3456780]))),
        StochasticScalar([1234560.1, 2345670, 3456780]),
        1e-8,
    )

    sims = dist.generate(10000000)

    assert allclose(np.mean(sims), scale * gamma(shape - 1) / gamma(shape) + loc, 1e-3)
    assert allclose(
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
    assert dist.cdf(loc) == pytest.approx(float(np.exp(-1)), 1e-8)
    assert dist.cdf(loc + scale) == pytest.approx(float(np.exp(-np.exp(-1))), 1e-8)
    assert dist.cdf(loc - scale) == pytest.approx(float(np.exp(-np.exp(1))), 1e-8)

    # Test inverse CDF
    assert dist.invcdf(0.5) == pytest.approx(float(loc - scale * np.log(-np.log(0.5))), 1e-6)
    assert dist.invcdf(0.9) == pytest.approx(float(loc - scale * np.log(-np.log(0.9))), 1e-6)

    # Test round-trip
    assert allclose(
        dist.invcdf(dist.cdf(StochasticScalar([loc - 50000, loc, loc + 50000, loc + 150000]))),
        StochasticScalar([loc - 50000, loc, loc + 50000, loc + 150000]),
        1e-6,
    )

    # Test statistical moments for Gumbel
    # Mean: μ + σ * γ (where γ ≈ 0.5772 is Euler-Mascheroni constant)
    # Std: σ * π / sqrt(6)
    sims = dist.generate(10000000)
    euler_gamma = 0.5772156649015329
    expected_mean = loc + scale * euler_gamma
    expected_std = float(scale * np.pi / np.sqrt(6))

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
    assert dist.cdf(x) == pytest.approx(float(expected_cdf), 1e-8)

    # Test inverse CDF
    u = 0.5
    expected_x = loc + scale * (np.power(-np.log(u), -shape) - 1) / shape
    assert dist.invcdf(u) == pytest.approx(float(expected_x), 1e-6)

    # Test round-trip
    test_points = StochasticScalar([loc + 10000, loc + 50000, loc + 100000, loc + 200000])
    assert allclose(dist.invcdf(dist.cdf(test_points)), test_points, 1e-6)

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
    assert dist.cdf(x) == pytest.approx(float(expected_cdf), 1e-8)

    # Test inverse CDF
    u = 0.7
    expected_x = loc + scale * (np.power(-np.log(u), -shape) - 1) / shape
    assert dist.invcdf(u) == pytest.approx(float(expected_x), 1e-6)

    # Test round-trip
    test_points = StochasticScalar([loc + 10000, loc + 50000, loc + 100000, loc + 200000])
    assert allclose(dist.invcdf(dist.cdf(test_points)), test_points, 1e-6)

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
    test_points = StochasticScalar([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert allclose(dist.invcdf(dist.cdf(test_points)), test_points, 1e-8)

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
    test_points = StochasticScalar([mu - 200, mu - 100, mu, mu + 100, mu + 200])
    assert allclose(dist.invcdf(dist.cdf(test_points)), test_points, 1e-6)

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


def test_studentst_generation_with_stochastic_parameters() -> None:
    """Generate directly while retaining coupling to stochastic parameters."""
    n_sims = 4
    nu = StochasticScalar([3.0, 4.0, 5.0, 6.0])
    mu = StochasticScalar([0.0, 1.0, 2.0, 3.0])
    sigma = StochasticScalar([1.0, 1.5, 2.0, 2.5])

    simulations = distributions.StudentsT(nu, mu, sigma).generate(n_sims)

    assert simulations.n_sims == n_sims
    assert simulations.coupled_variable_group is nu.coupled_variable_group
    assert simulations.coupled_variable_group is mu.coupled_variable_group
    assert simulations.coupled_variable_group is sigma.coupled_variable_group


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
    x_values = StochasticScalar([mu / 2, mu, 2 * mu, 3 * mu])
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


@pytest.mark.skipif(np.__name__ == "cupy", reason="generalized inverse Gaussian is CPU-only")
def test_generalized_inverse_gaussian() -> None:
    """Test the generalized inverse Gaussian distribution."""
    set_random_seed(12345678910)
    p = 0.75
    chi = 2.5
    psi = 1.25
    loc = 3.0
    dist = distributions.GeneralizedInverseGaussian(p, chi, psi, loc)

    probabilities = StochasticScalar([0.01, 0.25, 0.5, 0.9, 0.99])
    quantiles = dist.invcdf(probabilities)
    assert isinstance(quantiles, StochasticScalar)
    assert np.allclose(dist.cdf(quantiles), probabilities, rtol=1e-10)
    assert dist.cdf(loc) == 0.0
    assert dist.invcdf(0.0) == loc
    assert np.allclose(
        dist.cdf(StochasticScalar([3.5, 4.0, 5.0, 10.0])),
        StochasticScalar(
            [
                0.0214939224506453,
                0.157612365127509,
                0.491799593366621,
                0.977313683799349,
            ]
        ),
        rtol=1e-12,
    )
    assert np.allclose(
        dist.invcdf(StochasticScalar([0.01, 0.1, 0.5, 0.9, 0.99])),
        StochasticScalar(
            [
                3.41283685274434,
                3.82366853423611,
                5.02868285940702,
                7.66168386171783,
                11.2859643689873,
            ]
        ),
        rtol=1e-12,
    )

    sims = dist.generate(200000)
    z = math.sqrt(chi * psi)
    expected_mean = math.sqrt(chi / psi) * scipy.special.kv(p + 1, z) / scipy.special.kv(p, z)
    expected_second_moment = chi / psi * scipy.special.kv(p + 2, z) / scipy.special.kv(p, z)
    expected_variance = expected_second_moment - expected_mean**2

    assert np.mean(sims) == pytest.approx(loc + expected_mean, rel=5e-3)
    assert np.var(sims) == pytest.approx(expected_variance, rel=1e-2)


@pytest.mark.skipif(np.__name__ == "cupy", reason="generalized inverse Gaussian is CPU-only")
@pytest.mark.parametrize("method, argument", [("cdf", 5.0), ("invcdf", 0.5)])
def test_generalized_inverse_gaussian_stochastic_parameter_coupling(method: str, argument: float) -> None:
    """Test GIG results are coupled to every stochastic input and parameter."""
    p = StochasticScalar([0.5, 0.75, 1.0])
    chi = StochasticScalar([2.0, 2.5, 3.0])
    psi = StochasticScalar([1.0, 1.25, 1.5])
    loc = StochasticScalar([0.0, 0.5, 1.0])
    stochastic_argument = StochasticScalar([argument] * 3)
    dist = distributions.GeneralizedInverseGaussian(p=p, chi=chi, psi=psi, loc=loc)

    result = getattr(dist, method)(stochastic_argument)

    assert isinstance(result, StochasticScalar)
    assert result.coupled_variable_group is stochastic_argument.coupled_variable_group
    assert result.coupled_variable_group is p.coupled_variable_group
    assert result.coupled_variable_group is chi.coupled_variable_group
    assert result.coupled_variable_group is psi.coupled_variable_group
    assert result.coupled_variable_group is loc.coupled_variable_group


@pytest.mark.skipif(np.__name__ == "cupy", reason="generalized inverse Gaussian is CPU-only")
def test_generalized_inverse_gaussian_stochastic_parameter_generation_coupling() -> None:
    """Test GIG generation is coupled to every stochastic parameter."""
    set_random_seed(12345678910)
    p = StochasticScalar([0.5, 0.75, 1.0])
    chi = StochasticScalar([2.0, 2.5, 3.0])
    psi = StochasticScalar([1.0, 1.25, 1.5])
    result = distributions.GeneralizedInverseGaussian(p=p, chi=chi, psi=psi).generate(3)

    assert result.coupled_variable_group is p.coupled_variable_group
    assert result.coupled_variable_group is chi.coupled_variable_group
    assert result.coupled_variable_group is psi.coupled_variable_group


@pytest.mark.skipif(np.__name__ == "cupy", reason="generalized inverse Gaussian is CPU-only")
def test_generalized_inverse_gaussian_contains_inverse_gaussian() -> None:
    """Test the inverse Gaussian special case of the GIG distribution."""
    mu = 4.0
    lambda_ig = 7.0
    gig = distributions.GeneralizedInverseGaussian(
        p=-0.5,
        chi=lambda_ig,
        psi=lambda_ig / mu**2,
    )
    inverse_gaussian = distributions.InverseGaussian(mu, lambda_ig)
    points = StochasticScalar([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])

    assert np.allclose(gig.cdf(points), inverse_gaussian.cdf(points), rtol=1e-10)


@pytest.mark.skipif(np.__name__ == "cupy", reason="generalized inverse Gaussian is CPU-only")
def test_generalized_inverse_gaussian_generator() -> None:
    """Test construction through the named continuous generator."""
    generator = distributions.ContinuousDistributionGenerator(
        "generalizedinversegaussian",
        [-0.5, 7.0, 7.0 / 16.0],
    )

    assert generator.invcdf(0.5) > 0


@pytest.mark.parametrize("chi, psi", [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)])
@pytest.mark.skipif(np.__name__ == "cupy", reason="generalized inverse Gaussian is CPU-only")
def test_generalized_inverse_gaussian_rejects_invalid_scales(chi: float, psi: float) -> None:
    """Test that the GIG scale parameters must be strictly positive."""
    dist = distributions.GeneralizedInverseGaussian(p=0.5, chi=chi, psi=psi)

    with pytest.raises(ValueError):
        dist.cdf(1.0)


def test_mbbefd() -> None:
    """Test Bernegger's MBBEFD distribution and total-loss atom."""
    set_random_seed(12345678910)
    g = 25.0
    b = 4.0
    dist = distributions.MBBEFD(g, b)
    points = StochasticScalar([0.01, 0.2, 0.5, 0.8, 0.99])

    cdf_values = dist.cdf(points)
    assert np.all(np.diff(cdf_values) > 0)
    assert np.allclose(dist.invcdf(cdf_values), points, rtol=1e-10)
    assert dist.cdf(0.0) == 0.0
    assert dist.cdf(1.0) == 1.0
    assert dist.invcdf(1 - 1 / g) == 1.0
    assert dist.invcdf(0.98) == 1.0
    assert np.allclose(
        dist.cdf(StochasticScalar([0.2, 0.5, 0.8, 0.99])),
        StochasticScalar(
            [
                0.885695146977805,
                0.941176470588235,
                0.955444536635840,
                0.959820516901669,
            ]
        ),
        rtol=1e-12,
    )
    assert np.allclose(
        dist.invcdf(StochasticScalar([0.1, 0.25, 0.5, 0.9, 0.95])),
        StochasticScalar(
            [
                0.00250903738331221,
                0.00755344619510412,
                0.0229018448065623,
                0.238219021971494,
                0.649780140929453,
            ]
        ),
        rtol=1e-12,
    )
    assert np.allclose(
        dist.exposure_curve(StochasticScalar([0.25, 0.5, 0.75])),
        StochasticScalar(
            [
                0.583200958298682,
                0.765739458521128,
                0.893865142318307,
            ]
        ),
        rtol=1e-12,
    )

    sims = dist.generate(200000)
    expected_mean = (1 - b) * math.log(g * b) / ((1 - g * b) * math.log(b))

    def survival(x: float) -> float:
        return (1 - b) * b**x / ((g - 1) * b + (1 - g * b) * b**x)

    expected_second_moment = (
        2
        * scipy.integrate.quad(
            lambda x: x * survival(x),
            0,
            1,
            epsabs=1e-12,
            epsrel=1e-12,
        )[0]
    )
    expected_variance = expected_second_moment - expected_mean**2

    assert expected_mean == pytest.approx(0.100664487723859, rel=1e-12)
    assert expected_second_moment == pytest.approx(0.0575175543010063, rel=1e-12)
    assert np.mean(sims) == pytest.approx(expected_mean, rel=5e-3)
    assert np.mean(sims**2) == pytest.approx(expected_second_moment, rel=1e-2)
    assert np.var(sims) == pytest.approx(expected_variance, rel=1e-2)
    assert np.mean(sims == 1) == pytest.approx(1 / g, abs=1e-3)


@pytest.mark.parametrize(
    "method, argument",
    [("cdf", 0.5), ("invcdf", 0.5), ("exposure_curve", 0.5)],
)
def test_mbbefd_stochastic_parameter_coupling(method: str, argument: float) -> None:
    """Test MBBEFD results are coupled to every stochastic input and parameter."""
    g = StochasticScalar([20.0, 25.0, 30.0])
    b = StochasticScalar([2.0, 3.0, 4.0])
    stochastic_argument = StochasticScalar([argument] * 3)
    dist = distributions.MBBEFD(g=g, b=b)

    result = getattr(dist, method)(stochastic_argument)

    assert isinstance(result, StochasticScalar)
    assert result.coupled_variable_group is stochastic_argument.coupled_variable_group
    assert result.coupled_variable_group is g.coupled_variable_group
    assert result.coupled_variable_group is b.coupled_variable_group


def test_mbbefd_stochastic_parameter_generation_coupling() -> None:
    """Test MBBEFD generation is coupled to every stochastic parameter."""
    set_random_seed(12345678910)
    g = StochasticScalar([20.0, 25.0, 30.0])
    b = StochasticScalar([2.0, 3.0, 4.0])
    result = distributions.MBBEFD(g=g, b=b).generate(3)

    assert result.coupled_variable_group is g.coupled_variable_group
    assert result.coupled_variable_group is b.coupled_variable_group


@pytest.mark.parametrize("g, b", [(5.0, 1.0), (4.0, 0.25)])
def test_mbbefd_limiting_cases(g: float, b: float) -> None:
    """Test the b=1 and bg=1 limiting cases."""
    dist = distributions.MBBEFD(g, b)
    points = StochasticScalar([0.01, 0.2, 0.5, 0.8, 0.99])

    assert np.allclose(dist.invcdf(dist.cdf(points)), points, rtol=1e-10)
    assert dist.exposure_curve(0.0) == 0.0
    assert dist.exposure_curve(1.0) == 1.0


def test_mbbefd_degenerate_case() -> None:
    """Test the point-mass boundary of the MBBEFD family."""
    dist = distributions.MBBEFD(g=1.0, b=2.0)

    assert dist.cdf(0.999) == 0.0
    assert dist.cdf(1.0) == 1.0
    assert dist.invcdf(0.0) == 1.0
    assert np.all(dist.generate(1000).values == 1.0)
    assert dist.exposure_curve(0.4) == pytest.approx(0.4)


def test_mbbefd_generator() -> None:
    """Test construction through the named continuous generator."""
    generator = distributions.ContinuousDistributionGenerator("mbbefd", [25.0, 4.0])

    assert generator.invcdf(0.5) < 1


def test_mbbefd_from_swiss_re_c() -> None:
    """Test conversion from the standard Swiss Re curve parameter."""
    dist = distributions.MBBEFD.from_c(3.0)

    assert dist._params["b"] == pytest.approx(3.6692966676)
    assert dist._params["g"] == pytest.approx(30.569414012)


@pytest.mark.parametrize("g, b", [(0.99, 2.0), (2.0, -0.01), (math.inf, 2.0)])
def test_mbbefd_rejects_invalid_parameters(g: float, b: float) -> None:
    """Test that MBBEFD parameters outside the admissible region are rejected."""
    dist = distributions.MBBEFD(g, b)

    with pytest.raises(ValueError):
        dist.cdf(0.5)


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

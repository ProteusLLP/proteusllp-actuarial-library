"""Tests for copula functionality and margin validation.

Tests covering copula sampling, margin validation, and integration
with ProteusVariable for dependency modeling in actuarial applications.
"""

import numpy as np
import numpy.typing as npt
import pal.maths as pnp
import pytest
import scipy
import scipy.special
import scipy.stats  # ignore:import-untyped
from pal import config, copulas, distributions
from pal.variables import ProteusVariable, StochasticScalar


def copula_margins(
    copula_samples: list[StochasticScalar] | ProteusVariable[StochasticScalar],
):
    # check values are between 0 and 1
    if isinstance(copula_samples, ProteusVariable):
        copula_samples = list(copula_samples)
    y = ProteusVariable[StochasticScalar](
        "dim1",
        {f"margin_{i}": (x >= 0) & (x <= 1) for i, x in enumerate(copula_samples)},
    )

    assert pnp.all(y)

    # check the values are uniform by checking the moments
    for u in copula_samples:
        assert np.isclose(np.mean(u), 0.5, atol=1e-2)
        assert np.isclose(np.std(u), 1 / np.sqrt(12), atol=1e-2)
        assert np.isclose(scipy.stats.skew(u), 0, atol=1e-1)
        assert np.isclose(scipy.stats.kurtosis(u, fisher=False), 3 - 6 / 5, atol=1e-1)


@pytest.mark.parametrize("correlation", [-0.999, 0.5, 0, -0.5, 0.25, 0.75, 0.999])
def test_gaussian_copula(correlation: float):
    samples = copulas.GaussianCopula([[1, correlation], [correlation, 1]]).generate(
        100000
    )
    # test the correlations
    emp_corr = np.corrcoef((samples[0].values, samples[1].values))[0, 1]
    # convert from rank to linear
    rank_corr = 2 * np.sin(emp_corr * np.pi / 6)
    assert np.isclose(rank_corr, correlation, atol=1e-2)
    k = scipy.stats.kendalltau(samples[0].values, samples[1].values).statistic
    assert np.isclose(k, 2 / np.pi * np.asin(correlation), atol=1e-2)
    # test the margins
    copula_margins(samples)


@pytest.mark.parametrize("correlation", [-0.999, 0.5, 0, -0.5, 0.25, 0.75, 0.999])
def test_gaussian_copula_apply(correlation: float):
    n_sims = 100000
    samples = [
        distributions.Gamma(2, 50).generate(n_sims),
        distributions.LogNormal(2, 1.5).generate(n_sims),
    ]
    copulas.GaussianCopula([[1, correlation], [correlation, 1]]).apply(samples)
    # test the correlations
    emp_corr = np.corrcoef((samples[0].ranks.values, samples[1].ranks.values))[0, 1]
    # convert from rank to linear
    linear_corr = 2 * np.sin(emp_corr * np.pi / 6)
    assert np.isclose(linear_corr, correlation, atol=1e-2)
    k = scipy.stats.kendalltau(samples[0].values, samples[1].values).statistic
    assert np.isclose(k, 2 / np.pi * np.asin(correlation), atol=1e-2)


@pytest.mark.parametrize("dof", [1.5, 5, 9, 100])
@pytest.mark.parametrize("correlation", [-0.999, 0.5, -0.5, 0, 0.25, 0.75, 0.999])
def test_studentst_copula(correlation: float, dof: float):
    samples = copulas.StudentsTCopula(
        [[1, correlation], [correlation, 1]], dof
    ).generate(100000)
    k = scipy.stats.kendalltau(samples[0].values, samples[1].values).statistic
    assert np.isclose(k, 2 / np.pi * np.asin(correlation), atol=1e-2)
    # test the margins
    copula_margins(samples)


@pytest.mark.parametrize("dof", [1.5, 5, 9, 100])
@pytest.mark.parametrize("correlation", [-0.999, 0.5, 0, -0.5, 0.25, 0.75, 0.999])
def test_studentst_copula_apply(correlation: float, dof: float):
    n_sims = 100000
    samples = [
        distributions.Gamma(2, 50).generate(n_sims),
        distributions.LogNormal(2, 1.5).generate(n_sims),
    ]
    copulas.StudentsTCopula([[1, correlation], [correlation, 1]], dof).apply(samples)
    # test the correlations
    k = scipy.stats.kendalltau(samples[0].values, samples[1].values).statistic
    assert np.isclose(k, 2 / np.pi * np.asin(correlation), atol=1e-2)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.25, 2.75])
def test_clayton_copula(alpha: float):
    samples = copulas.ClaytonCopula(alpha, 2).generate(100000)
    k = scipy.stats.kendalltau(samples[0].values, samples[1].values).statistic
    assert np.isclose(k, alpha / (2 + alpha), atol=1e-2)
    # test the margins
    copula_margins(samples)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.25, 2.75])
def test_clayton_copula_apply(alpha: float):
    n_sims = 100000
    samples = [
        distributions.Gamma(2, 50).generate(n_sims),
        distributions.LogNormal(2, 1.5).generate(n_sims),
    ]
    copulas.ClaytonCopula(alpha, 2).apply(samples)
    k = scipy.stats.kendalltau(samples[0].values, samples[1].values).statistic
    assert np.isclose(k, alpha / (2 + alpha), atol=1e-2)


@pytest.mark.parametrize("theta", [1.001, 1.25, 2.2, 5])
def test_gumbel_copula(theta: float):
    samples = copulas.GumbelCopula(theta, 2).generate(1000000)
    # calculate the Kendall's tau value
    k = scipy.stats.kendalltau(samples[0].values, samples[1].values).statistic
    assert np.isclose(k, 1 - 1 / theta, atol=1e-2)
    # test the tail dependence
    expected_tail_dependence = 2 - 2 ** (1 / theta)
    threshold = 0.995
    u_exceed = (samples[0] > threshold).mean()
    both_exceed = ((samples[0] > threshold) * (samples[1] > threshold)).mean()
    estimated_tail_dependence = both_exceed / u_exceed
    assert np.isclose(estimated_tail_dependence, expected_tail_dependence, atol=1e-2)
    # test the margins
    copula_margins(samples)


@pytest.mark.parametrize("theta", [1.001, 1.25, 2.2, 5])
def test_gumbel_copula_apply(theta: float):
    n_sims = 100000
    samples = [
        distributions.Gamma(2, 50).generate(n_sims),
        distributions.LogNormal(2, 1.5).generate(n_sims),
    ]
    copulas.GumbelCopula(theta, 2).apply(samples)
    k = scipy.stats.kendalltau(samples[0].values, samples[1].values).statistic
    assert np.isclose(k, 1 - 1 / theta, atol=1e-2)


@pytest.mark.parametrize("theta", [1.001, 1.25, 2.2, 3])
def test_joe_copula(theta: float):
    samples = copulas.JoeCopula(theta, 2).generate(100000)
    # calculate the Kendall's tau value
    k = scipy.stats.kendalltau(samples[0].values, samples[1].values).statistic
    assert np.isclose(
        k,
        1
        + 2
        / (2 - theta)
        * (scipy.special.digamma(2) - scipy.special.digamma(2 / theta + 1)),
        atol=1e-2,
    )
    # test the margins
    copula_margins(samples)


def debye1(x: float) -> float:
    """The first Debye function."""
    # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType] - scipy.special functions not fully typed
    return (  # pyright: ignore[reportUnknownVariableType]
        np.log(1 - np.exp(-x)) * x
        + scipy.special.zeta(2)
        - scipy.special.spence(1 - np.exp(-x))
    ) / x


@pytest.mark.parametrize("theta", [0.001, 0.5, 2, 4])
def test_frank_copula(theta: float):
    samples = copulas.FrankCopula(theta, 2).generate(100000)
    # calculate the Kendall's tau value
    k = scipy.stats.kendalltau(samples[0].values, samples[1].values).statistic
    assert np.isclose(
        k,
        1 + 4 / theta * (debye1(theta) - 1),
        atol=1e-2,
    )
    # test the margins
    copula_margins(samples)


@pytest.mark.parametrize("theta", [0.00001, 0.1, 0.5, 2, 4])
def test_galambos_copula(theta: float):
    config.rng = np.random.default_rng(42)
    samples = copulas.GalambosCopula(theta, 2).generate(100000)
    # test the tail dependence
    expected_tail_dependence = 2 ** (-1 / theta)
    threshold = 0.995
    u_exceed = (samples[0] > threshold).mean()
    both_exceed = ((samples[0] > threshold) * (samples[1] > threshold)).mean()
    estimated_tail_dependence = both_exceed / u_exceed
    assert np.isclose(estimated_tail_dependence, expected_tail_dependence, atol=4e-2)

    # calculate the Blomqvist's beta value
    beta = 4 * ((samples[0] <= 0.5) * (samples[1] <= 0.5)).mean() - 1
    assert np.isclose(
        beta,
        2 ** (2 ** (-1 / theta)) - 1,
        atol=1e-2,
    )
    # test the margins
    copula_margins(samples)


@pytest.mark.parametrize("delta", [0.1, 0.5, 2, 5, 10])
def test_plackett_copula(delta: float):
    samples = copulas.PlackettCopula(delta).generate(100000)
    # calculate the Spearman's rho value
    r = scipy.stats.spearmanr(samples[0].values, samples[1].values).statistic
    # Theoretical Spearman's rho for Plackett copula
    expected_result = (delta + 1) / (delta - 1) - (2 * delta * np.log(delta)) / (
        (delta - 1) ** 2
    )
    assert np.isclose(
        r,
        expected_result,
        atol=1e-2,
    )
    # test the margins
    copula_margins(samples)


@pytest.mark.parametrize(
    "matrix",
    (
        [[0, 1.25], [1.25, 0]],
        [[0, 2.5], [2.5, 0]],
        [[0, 0.5], [0.5, 0]],
        [[0, 0.5, 0.25], [0.5, 0, 0.4], [0.25, 0.4, 0]],
        [[0, 1.5, 1], [1.5, 0, 0.5], [1, 0.5, 0]],
        [
            [0, 1.5, 1, np.inf],
            [1.5, 0, 0.5, np.inf],
            [1, 0.5, 0, np.inf],
            [np.inf, np.inf, np.inf, 0],
        ],
    ),
)
def test_huslerreiss_copula(matrix: list[list[float]]):
    d = len(matrix)
    config.rng = np.random.default_rng(42)
    samples = copulas.HuslerReissCopula(np.array(matrix)).generate(100000)
    # test the tail dependence

    expected_tail_dependence: npt.NDArray[np.floating] = 2 * (  # type: ignore[reportUnknownVariableType]
        1 - scipy.stats.norm.cdf(matrix)
    )
    threshold = 0.995
    estimated_tail_dependence = [
        [
            ((samples[i] > threshold) & (samples[j] > threshold)).mean()
            / (samples[i] > threshold).mean()
            for j in range(d)
        ]
        for i in range(d)
    ]
    assert np.allclose(estimated_tail_dependence, expected_tail_dependence, atol=5e-2)  # type: ignore[reportUnknownVariableType]
    # test the margins
    copula_margins(samples)


def test_huslerreiss_copula_parameter_errors():
    """Test that invalid parameters raise errors."""
    with pytest.raises(ValueError, match="Parameter matrix must be square"):
        copulas.HuslerReissCopula(np.array([[0, 1], [1, 0], [0, 1]]))
    with pytest.raises(ValueError, match="Matrix diagonal must be zero"):
        copulas.HuslerReissCopula(np.array([[0, 1], [1, 2]]))
    with pytest.raises(ValueError, match="Matrix must be symmetric"):
        copulas.HuslerReissCopula(np.array([[0, 1], [2, 0]]))


def test_hulerreiss_copula_methods():
    """Test Husler-Reiss copula methods."""
    lambda_matrix = np.array([[0, 1.25], [1.25, 0]])
    tail_dependence_matrix = copulas.HuslerReissCopula(
        lambda_matrix
    ).tail_dependence_matrix
    expected_tail_dependency_matrix = 2 * (1 - scipy.stats.norm.cdf(lambda_matrix))
    assert np.allclose(tail_dependence_matrix, expected_tail_dependency_matrix)
    lambda_matrix = copulas.HuslerReissCopula.calculate_lambda_from_tail_dependence(
        tail_dependence_matrix
    )
    assert np.allclose(lambda_matrix, lambda_matrix)
    copula = copulas.HuslerReissCopula.from_tail_dependence_matrix(
        tail_dependence_matrix
    )
    assert np.allclose(copula.adjusted_lambda_matrix, lambda_matrix)


@pytest.mark.parametrize("theta", [1.01, 1.25, 2])
@pytest.mark.parametrize(
    "delta_matrix",
    [[[1, None], [2.2, 1]], [[None, None, None], [1.2, None, None], [1.5, 2.5, None]]],
)
def test_mm1_copula(delta_matrix: list[list[float]], theta: float):
    config.rng = np.random.default_rng(12345678)

    samples = copulas.MM1Copula(delta_matrix=delta_matrix, theta=theta).generate(
        1_000_000
    )
    # calculate the tail dependency coefficient of each bivariate margin
    threshold = 0.99
    upper_tail_coefficient = [
        [((u > threshold) * (v > threshold)).mean() / (1 - threshold) for u in samples]
        for v in samples
    ]

    def mm1_tail_coeff(delta_ij: float, theta: float, d: int):
        """Calculate the upper tail dependence coefficient for MM1 copula."""
        return 2 - (
            ((2 ** (1 / delta_ij)) / (d - 1) + 2 * (d - 2) / (d - 1)) ** (1 / theta)
        )

    expected_tail_coefficients = [
        [mm1_tail_coeff(delta_matrix[i][j], theta, len(delta_matrix)) for j in range(i)]
        for i in range(0, len(delta_matrix))
    ]
    for i in range(1, len(delta_matrix)):
        for j in range(i):
            assert np.isclose(
                upper_tail_coefficient[i][j],
                expected_tail_coefficients[i][j],
                atol=5e-2,
            )

    copula_margins(samples)

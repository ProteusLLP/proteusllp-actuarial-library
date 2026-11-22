"""Tests for copula functionality and margin validation.

Tests covering copula sampling, margin validation, and integration
with ProteusVariable for dependency modeling in actuarial applications.
"""

import re

import numpy as np
import numpy.typing as npt
import pytest
import scipy
import scipy.special
import scipy.stats  # ignore:import-untyped

import pal.maths as pnp
from pal import config, copulas, distributions
from pal.variables import ProteusVariable, StochasticScalar
from tests._assertions import allclose, host_values


@pytest.fixture(autouse=True)
def _reset_config_rng() -> None:
    """Reset PAL's global RNG before each test.

    Several tests in this module rely on Monte Carlo estimates with tight tolerances.
    Other tests (in this file and elsewhere) may mutate `config.rng`, so we reset it
    here to avoid order-dependent flakiness.
    """
    config.rng = np.random.default_rng(config.seed)


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
    samples = copulas.GaussianCopula([[1, correlation], [correlation, 1]]).generate(100000)
    # test the correlations
    emp_corr = np.corrcoef((host_values(samples[0]), host_values(samples[1])))[0, 1]
    # convert from rank to linear
    rank_corr = 2 * np.sin(emp_corr * np.pi / 6)
    assert np.isclose(rank_corr, correlation, atol=1e-2)
    k = scipy.stats.kendalltau(host_values(samples[0]), host_values(samples[1])).statistic
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
    emp_corr = np.corrcoef((host_values(samples[0].ranks), host_values(samples[1].ranks)))[0, 1]
    # convert from rank to linear
    linear_corr = 2 * np.sin(emp_corr * np.pi / 6)
    assert np.isclose(linear_corr, correlation, atol=1e-2)
    k = scipy.stats.kendalltau(host_values(samples[0]), host_values(samples[1])).statistic
    assert np.isclose(k, 2 / np.pi * np.asin(correlation), atol=1e-2)


@pytest.mark.parametrize("dof", [1.5, 5, 9, 100])
@pytest.mark.parametrize("correlation", [-0.999, 0.5, -0.5, 0, 0.25, 0.75, 0.999])
def test_studentst_copula(correlation: float, dof: float):
    samples = copulas.StudentsTCopula([[1, correlation], [correlation, 1]], dof).generate(100000)
    k = scipy.stats.kendalltau(host_values(samples[0]), host_values(samples[1])).statistic
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
    k = scipy.stats.kendalltau(host_values(samples[0]), host_values(samples[1])).statistic
    assert np.isclose(k, 2 / np.pi * np.asin(correlation), atol=1e-2)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.25, 2.75])
def test_clayton_copula(alpha: float):
    samples = copulas.ClaytonCopula(alpha, 2).generate(100000)
    k = scipy.stats.kendalltau(host_values(samples[0]), host_values(samples[1])).statistic
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
    k = scipy.stats.kendalltau(host_values(samples[0]), host_values(samples[1])).statistic
    assert np.isclose(k, alpha / (2 + alpha), atol=1e-2)


@pytest.mark.parametrize("theta", [1.001, 1.25, 2.2, 5])
def test_gumbel_copula(theta: float):
    samples = copulas.GumbelCopula(theta, 2).generate(1000000)
    # calculate the Kendall's tau value
    k = scipy.stats.kendalltau(host_values(samples[0]), host_values(samples[1])).statistic
    assert np.isclose(k, 1 - 1 / theta, atol=1.5e-2)
    # test the tail dependence
    expected_tail_dependence = 2 - 2 ** (1 / theta)
    threshold = 0.995
    u_exceed = (samples[0] > threshold).mean()
    both_exceed = ((samples[0] > threshold) * (samples[1] > threshold)).mean()
    estimated_tail_dependence = both_exceed / u_exceed
    assert np.isclose(estimated_tail_dependence, expected_tail_dependence, atol=1.5e-2)
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
    k = scipy.stats.kendalltau(host_values(samples[0]), host_values(samples[1])).statistic
    assert np.isclose(k, 1 - 1 / theta, atol=1e-2)


@pytest.mark.parametrize("theta", [1.001, 1.25, 2.2, 3])
def test_joe_copula(theta: float):
    samples = copulas.JoeCopula(theta, 2).generate(100000)
    # calculate the Kendall's tau value
    k = scipy.stats.kendalltau(host_values(samples[0]), host_values(samples[1])).statistic
    assert np.isclose(
        k,
        1 + 2 / (2 - theta) * (scipy.special.digamma(2) - scipy.special.digamma(2 / theta + 1)),
        atol=1e-2,
    )
    # test the margins
    copula_margins(samples)


def debye1(x: float) -> float:
    """The first Debye function."""
    # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType] - scipy.special functions not fully typed
    return (  # pyright: ignore[reportUnknownVariableType]
        np.log(1 - np.exp(-x)) * x + scipy.special.zeta(2) - scipy.special.spence(1 - np.exp(-x))
    ) / x


@pytest.mark.parametrize("theta", [0.001, 0.5, 2, 4])
def test_frank_copula(theta: float):
    samples = copulas.FrankCopula(theta, 2).generate(100000)
    # calculate the Kendall's tau value
    k = scipy.stats.kendalltau(host_values(samples[0]), host_values(samples[1])).statistic
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
    r = scipy.stats.spearmanr(host_values(samples[0]), host_values(samples[1])).statistic
    # Theoretical Spearman's rho for Plackett copula
    expected_result = (delta + 1) / (delta - 1) - (2 * delta * np.log(delta)) / ((delta - 1) ** 2)
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
            ((samples[i] > threshold) & (samples[j] > threshold)).mean() / (samples[i] > threshold).mean()
            for j in range(d)
        ]
        for i in range(d)
    ]
    assert allclose(estimated_tail_dependence, expected_tail_dependence, atol=5e-2)  # type: ignore[reportUnknownVariableType]
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
    tail_dependence_matrix = copulas.HuslerReissCopula(lambda_matrix).tail_dependence_matrix
    expected_tail_dependency_matrix = 2 * (1 - scipy.stats.norm.cdf(lambda_matrix))
    assert allclose(tail_dependence_matrix, expected_tail_dependency_matrix)
    lambda_matrix = copulas.HuslerReissCopula.calculate_lambda_from_tail_dependence(tail_dependence_matrix)
    assert allclose(lambda_matrix, lambda_matrix)
    copula = copulas.HuslerReissCopula.from_tail_dependence_matrix(tail_dependence_matrix)
    assert allclose(copula.adjusted_lambda_matrix, lambda_matrix)


@pytest.mark.parametrize(
    "matrix",
    (
        [[1, 0.5], [0.5, 1]],
        [[1.0, 0.5, 0.0], [0.5, 1.0, 0.7], [0.0, 0.7, 1.0]],
        [
            [1.0, 0.5, 0.0, -0.2],
            [0.5, 1.0, 0.7, -0.3],
            [0.0, 0.7, 1.0, -0.4],
            [-0.2, -0.3, -0.4, 1.0],
        ],
    ),
)
@pytest.mark.parametrize("nu", [1.5, 5, 10])
def test_extremalt_copula(nu: float, matrix: list[list[float]]):
    config.rng = np.random.default_rng(123456)
    rho = np.array(matrix)
    samples = copulas.ExtremalTCopula(rho, nu).generate(200000)
    # test the tail dependence
    coeff = -np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
    lambda_theo = 2 * scipy.stats.t.cdf(coeff, df=nu + 1)
    thresh = 0.995
    lambda_emp = np.zeros(rho.shape)
    for i in range(rho.shape[0]):
        for j in range(rho.shape[0]):
            both = np.sum((samples[i] > thresh) & (samples[j] > thresh))
            one = np.sum(samples[j] > thresh)
            lambda_emp[i, j] = both / one
    assert allclose(lambda_emp, lambda_theo, atol=5e-2)
    # test the blomqvist's beta
    beta_emp = np.zeros(rho.shape)
    for i in range(rho.shape[0]):
        for j in range(rho.shape[0]):
            in_square = np.sum((samples[i] <= 0.5) & (samples[j] <= 0.5))
            beta_emp[i, j] = 4 * in_square / samples.n_sims - 1
    beta_theo = (2**lambda_theo) - 1
    assert allclose(beta_emp, beta_theo, atol=1e-2)
    # test the margins
    copula_margins(samples)


def test_extremalt_copula_parameter_errors():
    """Test that invalid parameters raise errors."""
    with pytest.raises(ValueError, match="Correlation matrix must be square"):
        copulas.ExtremalTCopula(np.array([[0, 1], [1, 0], [0, 1]]), 1)
    with pytest.raises(ValueError, match="Correlation matrix diagonal must be all ones"):
        copulas.ExtremalTCopula(np.array([[0, 1], [1, 0]]), 1)
    with pytest.raises(
        ValueError,
        match=re.escape("Correlation matrix values must be in the range [-1, 1]"),
    ):  # noqa: E501
        copulas.ExtremalTCopula(
            np.array([[1, 1.5], [1.5, 1]]),
            1,
        )
    with pytest.raises(
        ValueError,
        match=re.escape("Degrees of freedom nu must be in the range (0, inf)"),
    ):
        copulas.ExtremalTCopula(
            np.array([[1]]),
            -1,
        )


def test_extremalt_copula_methods():
    """Test the Extremal-t copula from_tail_dependence method."""
    # test that we can recover rho from a given tail dependence matrix
    rho = np.array([[1, 0.0, 0.5], [0.0, 1, -0.5], [0.5, -0.5, 1]])
    nu = 5
    coeff = -np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
    lambda_theo = 2 * scipy.stats.t.cdf(coeff, df=nu + 1)
    copula = copulas.ExtremalTCopula.from_tail_dependence_matrix(lambda_theo, nu)
    found_rho = copula.correlation_matrix
    assert allclose(found_rho, rho)


@pytest.mark.parametrize("theta", [1.01, 1.25, 2])
@pytest.mark.parametrize(
    "delta_matrix",
    [[[1, None], [2.2, 1]], [[None, None, None], [1.2, None, None], [1.5, 2.5, None]]],
)
def test_mm1_copula(delta_matrix: list[list[float]], theta: float):
    config.rng = np.random.default_rng(12345678)

    samples = copulas.MM1Copula(delta_matrix=delta_matrix, theta=theta).generate(1_000_000)
    # calculate the tail dependency coefficient of each bivariate margin
    threshold = 0.99
    upper_tail_coefficient = [
        [((u > threshold) * (v > threshold)).mean() / (1 - threshold) for u in samples] for v in samples
    ]

    def mm1_tail_coeff(delta_ij: float, theta: float, d: int):
        """Calculate the upper tail dependence coefficient for MM1 copula."""
        return 2 - (((2 ** (1 / delta_ij)) / (d - 1) + 2 * (d - 2) / (d - 1)) ** (1 / theta))

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


@pytest.mark.parametrize("n_margins", [2, 5])
def test_apply_copula_matches_independent_rank_reference(n_margins: int):
    """Match the original rank transformation, including ties, on both backends."""
    n_sims = 13
    raw_variables = [
        np.array([(7 * i + j * j + 3 * j) % (5 + i) for j in range(n_sims)], dtype=float) for i in range(n_margins)
    ]
    raw_samples = [
        np.array([(3 * j + i * i + (j // 2)) % (7 + i) for j in range(n_sims)], dtype=float) for i in range(n_margins)
    ]

    # Independent NumPy reference for the pre-optimisation implementation.
    copula_sorts = np.argsort(np.array(raw_samples), axis=1, kind="stable")
    copula_ranks = np.empty_like(copula_sorts)
    copula_ranks[np.arange(n_margins)[:, None], copula_sorts] = np.arange(n_sims)
    variable_sorts = np.argsort(np.array(raw_variables), axis=1, kind="stable")
    first_ranks = np.empty(n_sims, dtype=int)
    first_ranks[variable_sorts[0]] = np.arange(n_sims)
    sample_positions = copula_sorts[0, first_ranks]
    expected = [raw_variables[0]] + [
        raw_variables[i][variable_sorts[i, copula_ranks[i, sample_positions]]] for i in range(1, n_margins)
    ]

    variables = [StochasticScalar(v) for v in raw_variables]
    samples = [StochasticScalar(v) for v in raw_samples]
    derived = variables[1] * 3
    copulas.apply_copula(variables, samples)

    for actual, reference in zip(variables, expected):
        np.testing.assert_array_equal(host_values(actual), reference)
    np.testing.assert_array_equal(host_values(derived), 3 * expected[1])
    assert all(var.coupled_variable_group is variables[0].coupled_variable_group for var in variables)
    assert derived.coupled_variable_group is variables[0].coupled_variable_group


@pytest.mark.parametrize("sample_position", [0, 2])
@pytest.mark.parametrize("derived_sample", [False, True])
def test_apply_copula_rejects_samples_coupled_to_marginals_without_mutation(sample_position: int, derived_sample: bool):
    """A sample cannot be a marginal or any variable coupled to a marginal."""
    a = StochasticScalar([70.0, 10.0, 50.0, 90.0, 30.0, 60.0, 20.0])
    b = StochasticScalar([40.0, 70.0, 10.0, 60.0, 20.0, 50.0, 30.0])
    c = StochasticScalar([12.0, 18.0, 10.0, 15.0, 17.0, 11.0, 16.0])
    variables = [a, b, c]
    samples = [
        StochasticScalar([0.7, 0.1, 0.4, 0.9, 0.3, 0.6, 0.2]),
        StochasticScalar([0.4, 0.2, 0.7, 0.5, 0.1, 0.6, 0.3]),
        StochasticScalar([0.6, 0.1, 0.3, 0.2, 0.7, 0.4, 0.5]),
    ]
    marginal = a if sample_position == 0 else b
    samples[sample_position] = marginal * 3 if derived_sample else marginal
    all_scalars = variables + samples
    original_values = [host_values(x).copy() for x in all_scalars]
    original_groups = [x.coupled_variable_group for x in all_scalars]

    with pytest.raises(ValueError, match=f"Copula sample at position {sample_position} is coupled"):
        copulas.apply_copula(variables, samples)

    # The validation must happen before reordering or merging any groups.
    for scalar, expected, group in zip(all_scalars, original_values, original_groups):
        np.testing.assert_array_equal(host_values(scalar), expected)
        assert scalar.coupled_variable_group is group


def test_apply_copula_allows_coupled_copula_samples():
    """Copula dimensions are normally dependent on one another."""
    a = StochasticScalar([10.0, 20.0, 30.0, 40.0])
    b = StochasticScalar([400.0, 100.0, 300.0, 200.0])
    first_sample = StochasticScalar([0.3, 0.1, 0.4, 0.2])
    second_sample = first_sample * 3
    assert first_sample.coupled_variable_group is second_sample.coupled_variable_group

    copulas.apply_copula([a, b], [first_sample, second_sample])

    np.testing.assert_array_equal(host_values(b), [100.0, 200.0, 300.0, 400.0])
    assert a.coupled_variable_group is b.coupled_variable_group
    assert first_sample.coupled_variable_group is second_sample.coupled_variable_group


def test_apply_copula_reorders_coupled_frequency_severity_indices():
    """Event-to-simulation indices follow the same permutation as aggregates."""
    from pal.frequency_severity import FreqSevSims

    events = FreqSevSims([0, 0, 2, 3, 3, 4], [10.0, 20.0, 30.0, 2.0, 5.0, 40.0], 5)
    aggregate = events.aggregate()
    samples = [
        StochasticScalar([0.9, 0.1, 0.7, 0.4, 0.3]),
        StochasticScalar([0.6, 0.3, 0.1, 0.9, 0.5]),
    ]

    # Attach events to the *reordered* margin, as in a compound-loss copula.
    events2 = FreqSevSims([0, 0, 1, 3, 4], [1.0, 4.0, 5.0, 7.0, 9.0], 5)
    other = events2.aggregate()
    other_before = host_values(other).copy()
    left_ranks = np.empty(5, dtype=int)
    left_ranks[np.argsort(host_values(aggregate), kind="stable")] = np.arange(5)
    sample_positions = np.argsort(host_values(samples[0]), kind="stable")[left_ranks]
    right_rank = np.empty(5, dtype=int)
    right_rank[np.argsort(host_values(samples[1]), kind="stable")] = np.arange(5)
    permutation = np.argsort(other_before, kind="stable")[right_rank[sample_positions]]
    previous_indices = host_values(events2.sim_index).copy()
    previous_values = host_values(events2.values).copy()

    copulas.apply_copula([aggregate, other], samples)

    expected_inverse = np.empty(5, dtype=int)
    expected_inverse[permutation] = np.arange(5)
    np.testing.assert_array_equal(host_values(events2.sim_index), expected_inverse[previous_indices])
    np.testing.assert_array_equal(host_values(events2.values), previous_values)
    np.testing.assert_array_equal(host_values(other), other_before[permutation])
    assert events2.coupled_variable_group is aggregate.coupled_variable_group


def test_apply_copula_rejects_mismatched_scenario_counts_without_mutation():
    first = StochasticScalar([4.0, 2.0, 3.0])
    second = StochasticScalar([1.0, 5.0, 2.0])
    before = host_values(second).copy()
    with pytest.raises(ValueError, match="same number"):
        copulas.apply_copula(
            [first, second],
            [StochasticScalar([0.2, 0.4, 0.6]), StochasticScalar([0.1, 0.3])],
        )
    np.testing.assert_array_equal(host_values(second), before)

"""Tests for multivariate probability distributions."""

import numpy as np
import pytest
import scipy.stats

from pal import set_random_seed
from pal._maths import asnumpy, xp
from pal.multivariate_distributions import (
    Dirichlet,
    GeneralizedDirichlet,
    InverseWishart,
    InvertedDirichlet,
    InvertedGeneralizedDirichlet,
    Multinomial,
    MultivariateNormal,
    MultivariateStudentsT,
    Wishart,
)
from pal.stochastic_scalar import StochasticScalar
from pal.variables import ProteusVariable


def _sample_matrix(samples: ProteusVariable[StochasticScalar]) -> np.ndarray:
    """Return simulations in rows and components in columns."""
    return asnumpy(xp.stack([component.values for component in samples], axis=1))


def _matrix_sample_array(
    samples: ProteusVariable[ProteusVariable[StochasticScalar]],
) -> np.ndarray:
    """Return matrix simulations with shape ``(n_sims, rows, columns)``."""
    return asnumpy(
        xp.stack(
            [xp.stack([entry.values for entry in row], axis=1) for row in samples],
            axis=1,
        )
    )


def test_multivariate_normal_density_and_moments() -> None:
    """Match SciPy's density and the theoretical first two moments."""
    set_random_seed(48291)
    mean = [1.0, -2.0, 0.5]
    covariance = np.array(
        [
            [2.0, 0.6, -0.2],
            [0.6, 1.5, 0.4],
            [-0.2, 0.4, 1.0],
        ]
    )
    distribution = MultivariateNormal(
        mean,
        covariance,
        component_names=["property", "casualty", "market"],
    )

    point = np.array([0.2, -1.3, 0.8])
    assert distribution.logpdf(point) == pytest.approx(
        scipy.stats.multivariate_normal.logpdf(point, mean=mean, cov=covariance)
    )
    assert distribution.pdf(point) == pytest.approx(
        scipy.stats.multivariate_normal.pdf(point, mean=mean, cov=covariance)
    )

    samples = distribution.generate(120_000)
    values = _sample_matrix(samples)
    assert list(samples.values) == ["property", "casualty", "market"]
    assert np.mean(values, axis=0) == pytest.approx(mean, abs=0.02)
    assert np.cov(values, rowvar=False) == pytest.approx(covariance, abs=0.025)
    assert all(component.coupled_variable_group is samples[0].coupled_variable_group for component in samples)


def test_multivariate_students_t_density_and_moments() -> None:
    """Match SciPy's density and the finite theoretical covariance."""
    set_random_seed(19763)
    nu = 8.0
    mean = np.array([0.5, -1.0])
    scale = np.array([[1.2, 0.45], [0.45, 0.8]])
    distribution = MultivariateStudentsT(nu, mean, scale)

    point = np.array([1.1, -0.4])
    assert distribution.logpdf(point) == pytest.approx(
        scipy.stats.multivariate_t.logpdf(point, loc=mean, shape=scale, df=nu)
    )

    samples = distribution.generate(150_000)
    values = _sample_matrix(samples)
    assert np.mean(values, axis=0) == pytest.approx(mean, abs=0.02)
    assert np.cov(values, rowvar=False) == pytest.approx(nu / (nu - 2) * scale, abs=0.035)


def test_multinomial_probability_and_moments() -> None:
    """Match SciPy's probability mass and the theoretical moments."""
    set_random_seed(34781)
    n = 20
    probabilities = np.array([0.2, 0.3, 0.5])
    distribution = Multinomial(
        n,
        probabilities,
        component_names=["property", "casualty", "specialty"],
    )
    point = np.array([4, 6, 10])

    assert distribution.logpmf(point) == pytest.approx(scipy.stats.multinomial.logpmf(point, n, probabilities))
    assert distribution.pmf(point) == pytest.approx(scipy.stats.multinomial.pmf(point, n, probabilities))
    assert distribution.pmf([4, 6, 9]) == 0.0
    assert distribution.pmf([4, 5.5, 10.5]) == 0.0

    samples = distribution.generate(100_000)
    values = _sample_matrix(samples)
    expected_covariance = n * (np.diag(probabilities) - np.outer(probabilities, probabilities))
    assert list(samples.values) == ["property", "casualty", "specialty"]
    assert np.sum(values, axis=1) == pytest.approx(np.full(len(values), n))
    assert np.mean(values, axis=0) == pytest.approx(n * probabilities, abs=0.025)
    assert np.cov(values, rowvar=False) == pytest.approx(expected_covariance, abs=0.04)


def test_wishart_density_and_moments() -> None:
    """Match SciPy's density and the theoretical mean matrix."""
    set_random_seed(74219)
    df = 8.0
    scale = np.array([[1.0, 0.2], [0.2, 0.7]])
    distribution = Wishart(
        df,
        scale,
        component_names=["property", "casualty"],
    )
    point = np.array([[5.0, 0.4], [0.4, 3.0]])

    assert distribution.logpdf(point) == pytest.approx(scipy.stats.wishart.logpdf(point, df=df, scale=scale))
    assert distribution.pdf(point) == pytest.approx(scipy.stats.wishart.pdf(point, df=df, scale=scale))
    assert distribution.pdf([[1.0, 2.0], [2.0, 1.0]]) == 0.0

    samples = distribution.generate(60_000)
    values = _matrix_sample_array(samples)
    assert list(samples.values) == ["property", "casualty"]
    assert list(samples[0].values) == ["property", "casualty"]
    assert samples.dimensions == ["row", "column"]
    assert np.mean(values, axis=0) == pytest.approx(df * scale, abs=0.04)
    assert values == pytest.approx(np.swapaxes(values, 1, 2), abs=1e-12)
    assert np.all(np.linalg.eigvalsh(values[:1_000]) > 0)


def test_inverse_wishart_density_and_moments() -> None:
    """Match SciPy's density and the finite theoretical mean matrix."""
    set_random_seed(96541)
    df = 12.0
    scale = np.array([[1.0, 0.2], [0.2, 0.7]])
    distribution = InverseWishart(df, scale)
    point = np.array([[0.2, 0.03], [0.03, 0.15]])

    assert distribution.logpdf(point) == pytest.approx(scipy.stats.invwishart.logpdf(point, df=df, scale=scale))
    assert distribution.pdf(point) == pytest.approx(scipy.stats.invwishart.pdf(point, df=df, scale=scale))
    assert distribution.pdf([[1.0, 2.0], [2.0, 1.0]]) == 0.0

    samples = distribution.generate(80_000)
    values = _matrix_sample_array(samples)
    expected_mean = scale / (df - len(scale) - 1)
    assert np.mean(values, axis=0) == pytest.approx(expected_mean, abs=0.0015)
    assert values == pytest.approx(np.swapaxes(values, 1, 2), abs=1e-12)
    assert np.all(np.linalg.eigvalsh(values[:1_000]) > 0)


def test_dirichlet_density_simplex_and_moments() -> None:
    """Match SciPy's density and Dirichlet moments."""
    set_random_seed(58123)
    alpha = np.array([2.0, 3.0, 5.0])
    distribution = Dirichlet(alpha)
    point = np.array([0.2, 0.3, 0.5])

    assert distribution.logpdf(point) == pytest.approx(scipy.stats.dirichlet.logpdf(point, alpha))
    assert distribution.pdf(point) == pytest.approx(scipy.stats.dirichlet.pdf(point, alpha))
    assert distribution.pdf([0.2, 0.3, 0.6]) == 0.0

    samples = distribution.generate(120_000)
    values = _sample_matrix(samples)
    total = alpha.sum()
    expected_covariance = (np.diag(alpha * (total - alpha)) - (np.outer(alpha, alpha) - np.diag(alpha**2))) / (
        total**2 * (total + 1)
    )
    assert np.sum(values, axis=1) == pytest.approx(np.ones(len(values)), abs=1e-12)
    assert np.mean(values, axis=0) == pytest.approx(alpha / total, abs=0.003)
    assert np.cov(values, rowvar=False) == pytest.approx(expected_covariance, abs=0.001)


def test_inverted_dirichlet_density_and_moments() -> None:
    """Match the theoretical density, mean, variance, and covariance."""
    set_random_seed(66142)
    alpha = np.array([2.0, 3.0, 10.0])
    distribution = InvertedDirichlet(alpha)
    point = np.array([0.4, 1.2])
    log_normalizer = scipy.special.gammaln(alpha.sum()) - scipy.special.gammaln(alpha).sum()
    expected_logpdf = log_normalizer + np.sum((alpha[:-1] - 1) * np.log(point)) - alpha.sum() * np.log1p(point.sum())
    assert distribution.logpdf(point) == pytest.approx(expected_logpdf)

    samples = distribution.generate(160_000)
    values = _sample_matrix(samples)
    tail = alpha[-1]
    expected_mean = alpha[:-1] / (tail - 1)
    expected_variance = alpha[:-1] * (alpha[:-1] + tail - 1) / ((tail - 1) ** 2 * (tail - 2))
    expected_covariance = alpha[0] * alpha[1] / ((tail - 1) ** 2 * (tail - 2))
    assert np.mean(values, axis=0) == pytest.approx(expected_mean, abs=0.006)
    assert np.var(values, axis=0) == pytest.approx(expected_variance, abs=0.004)
    assert np.cov(values, rowvar=False)[0, 1] == pytest.approx(expected_covariance, abs=0.0015)


def test_generalized_dirichlet_moments_and_dirichlet_reduction() -> None:
    """Check stick-breaking means and the exact Dirichlet special case."""
    set_random_seed(72913)
    alpha = np.array([2.0, 4.0, 3.0])
    beta = np.array([5.0, 2.0, 6.0])
    distribution = GeneralizedDirichlet(alpha, beta)
    samples = distribution.generate(120_000)
    values = _sample_matrix(samples)

    expected_mean = []
    expected_remainder = 1.0
    for alpha_i, beta_i in zip(alpha, beta, strict=True):
        expected_mean.append(expected_remainder * alpha_i / (alpha_i + beta_i))
        expected_remainder *= beta_i / (alpha_i + beta_i)
    expected_mean.append(expected_remainder)
    assert np.sum(values, axis=1) == pytest.approx(np.ones(len(values)), abs=1e-12)
    assert np.mean(values, axis=0) == pytest.approx(expected_mean, abs=0.003)

    dirichlet_alpha = np.array([2.0, 3.0, 4.0])
    reduced = GeneralizedDirichlet(
        alpha=dirichlet_alpha[:-1],
        beta=[dirichlet_alpha[1:].sum(), dirichlet_alpha[2]],
    )
    point = np.array([0.2, 0.3, 0.5])
    assert reduced.logpdf(point) == pytest.approx(scipy.stats.dirichlet.logpdf(point, dirichlet_alpha))


def test_inverted_generalized_dirichlet_reduces_to_inverted_dirichlet() -> None:
    """Check the inverted Dirichlet special case in density and moments."""
    set_random_seed(91357)
    alpha = np.array([2.0, 3.0])
    beta = np.array([10.0, 12.0])
    generalized = InvertedGeneralizedDirichlet(alpha, beta)
    inverted = InvertedDirichlet([2.0, 3.0, 10.0])
    point = np.array([0.4, 1.2])
    assert generalized.logpdf(point) == pytest.approx(inverted.logpdf(point))

    samples = generalized.generate(160_000)
    values = _sample_matrix(samples)
    expected_mean = np.array([2.0 / 9.0, 3.0 / 9.0])
    expected_covariance = 2.0 * 3.0 / (9.0**2 * 8.0)
    assert np.mean(values, axis=0) == pytest.approx(expected_mean, abs=0.006)
    assert np.cov(values, rowvar=False)[0, 1] == pytest.approx(expected_covariance, abs=0.0015)


def test_multivariate_distribution_stochastic_parameter_coupling() -> None:
    """Couple every output component to every stochastic parameter."""
    n_sims = 2_000
    stochastic_alpha = StochasticScalar(xp.full(n_sims, 2.0))
    distribution = Dirichlet([stochastic_alpha, 3.0, 4.0])
    samples = distribution.generate(n_sims)

    group = samples[0].coupled_variable_group
    assert stochastic_alpha.coupled_variable_group is group
    assert all(component.coupled_variable_group is group for component in samples)

    density = distribution.logpdf(samples)
    assert isinstance(density, StochasticScalar)
    assert density.coupled_variable_group is samples[0].coupled_variable_group
    assert density.coupled_variable_group is stochastic_alpha.coupled_variable_group


def test_multinomial_stochastic_parameter_coupling() -> None:
    """Couple multinomial counts and probability mass to stochastic parameters."""
    n_sims = 2_000
    stochastic_n = StochasticScalar(xp.full(n_sims, 20))
    stochastic_probability = StochasticScalar(xp.full(n_sims, 0.2))
    distribution = Multinomial(stochastic_n, [stochastic_probability, 0.3, 0.5])
    samples = distribution.generate(n_sims)

    group = samples[0].coupled_variable_group
    assert stochastic_n.coupled_variable_group is group
    assert stochastic_probability.coupled_variable_group is group
    assert all(component.coupled_variable_group is group for component in samples)
    probability = distribution.logpmf(samples)
    assert isinstance(probability, StochasticScalar)
    assert probability.coupled_variable_group is samples[0].coupled_variable_group
    assert probability.coupled_variable_group is stochastic_n.coupled_variable_group
    assert probability.coupled_variable_group is stochastic_probability.coupled_variable_group


def test_matrix_distribution_stochastic_parameter_coupling() -> None:
    """Couple every matrix entry and its density to stochastic degrees of freedom."""
    n_sims = 1_000
    stochastic_df = StochasticScalar(xp.full(n_sims, 8.0))
    distribution = Wishart(stochastic_df, [[1.0, 0.2], [0.2, 0.7]])
    samples = distribution.generate(n_sims)

    group = samples[0][0].coupled_variable_group
    assert stochastic_df.coupled_variable_group is group
    assert all(entry.coupled_variable_group is group for row in samples for entry in row)
    density = distribution.logpdf(samples)
    assert isinstance(density, StochasticScalar)
    assert density.coupled_variable_group is samples[0][0].coupled_variable_group
    assert density.coupled_variable_group is stochastic_df.coupled_variable_group


@pytest.mark.parametrize(
    ("constructor", "match"),
    [
        (lambda: MultivariateNormal([0.0, 0.0], [[1.0, 2.0], [2.0, 1.0]]), "positive definite"),
        (lambda: MultivariateStudentsT(0.0, [0.0], [[1.0]]), "nu"),
        (lambda: Dirichlet([1.0, -1.0]), "positive"),
        (lambda: InvertedDirichlet([1.0]), "at least 2"),
        (lambda: GeneralizedDirichlet([1.0, 2.0], [1.0]), "same length"),
        (lambda: InvertedGeneralizedDirichlet([1.0], [0.0]), "positive"),
        (lambda: Multinomial(-1, [0.5, 0.5]), "non-negative integers"),
        (lambda: Multinomial(10, [0.4, 0.5]), "sum to one"),
        (lambda: Wishart(1, [[1.0, 0.0], [0.0, 1.0]]), "df"),
        (lambda: InverseWishart(3, [[1.0, 2.0], [2.0, 1.0]]), "positive definite"),
    ],
)
def test_multivariate_distribution_validation(constructor: object, match: str) -> None:
    """Reject invalid dimensions, matrices, degrees of freedom, and shape parameters."""
    with pytest.raises(ValueError, match=match):
        constructor()  # type: ignore[operator]

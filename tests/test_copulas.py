"""Tests for copula functionality and margin validation.

Tests covering copula sampling, margin validation, and integration
with ProteusVariable for dependency modeling in actuarial applications.
"""

import numpy as np
import pal.maths as pnp
import pytest
import scipy
import scipy.special
import scipy.stats  # ignore:import-untyped
from pal import copulas, distributions
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
    samples = copulas.GumbelCopula(theta, 2).generate(100000)
    # calculate the Kendall's tau value
    k = scipy.stats.kendalltau(samples[0].values, samples[1].values).statistic
    assert np.isclose(k, 1 - 1 / theta, atol=1e-2)
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

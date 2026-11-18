"""Statistical and scenario-alignment tests for every PAL copula under MC/RQMC.

These tests use a known bivariate copula diagonal as an independent reference.
The median joint CDF tests dependence without relying on an unreliable
extreme-quantile estimate from a small random sample. Separate RQMC tests check
that sequential draws retain the identity of active paths in iterative samplers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from pal import config, set_random_seed
from pal._maths import asnumpy, xp
from pal.config import SamplingMethod, set_sampling_method
from pal.copulas import (
    ClaytonCopula,
    ExtremalTCopula,
    FrankCopula,
    GalambosCopula,
    GaussianCopula,
    GumbelCopula,
    HuslerReissCopula,
    JoeCopula,
    MM1Copula,
    PlackettCopula,
    StudentsTCopula,
)
from pal.distributions import Gamma, LogNormal
from pal.qmc import RandomizedQMCGenerator, _draw_active_scenarios


def _extreme_value_diagonal_exponent(copula) -> float | None:
    """Return C(u,u)=u**theta on the diagonal for a max-stable copula."""
    if isinstance(copula, GumbelCopula):
        return 2 ** (1 / copula.theta)
    if isinstance(copula, MM1Copula) and copula.dimension == 2:
        return 2 ** (1 / (copula.theta * copula.delta_matrix[1][0]))
    if isinstance(copula, GalambosCopula):
        return 2 - 2 ** (-1 / copula.theta)
    if isinstance(copula, HuslerReissCopula):
        lam = float(asnumpy(copula.adjusted_lambda_matrix)[0, 1])
        return 2 * stats.norm.cdf(lam)
    if isinstance(copula, ExtremalTCopula):
        rho = float(asnumpy(copula.correlation_matrix)[0, 1])
        nu = copula.nu
        threshold = -math.sqrt((nu + 1) * (1 - rho) / (1 + rho))
        return 2 - 2 * stats.t.cdf(threshold, df=nu + 1)
    return None


def _copula_examples():
    """Bivariate examples paired with their theoretical C(0.5, 0.5)."""
    rho = 0.6
    theta_frank = 3.0
    theta_joe = 2.0
    delta = 2.0
    lambda_hr = 0.75
    nu = 3.0
    ext_t = 2 * stats.t.cdf(-math.sqrt((nu + 1) * (1 - rho) / (1 + rho)), df=nu + 1)
    ratio = (math.exp(-theta_frank / 2) - 1) ** 2 / (math.exp(-theta_frank) - 1)
    return [
        pytest.param(
            GaussianCopula([[1, rho], [rho, 1]]),
            0.25 + math.asin(rho) / (2 * math.pi),
            id="gaussian",
        ),
        pytest.param(
            StudentsTCopula([[1, rho], [rho, 1]], dof=nu),
            0.25 + math.asin(rho) / (2 * math.pi),
            id="student-t",
        ),
        pytest.param(ClaytonCopula(1.5, 2), (2**2.5 - 1) ** (-1 / 1.5), id="clayton"),
        pytest.param(GumbelCopula(2.0, 2), 2 ** (-math.sqrt(2)), id="gumbel"),
        pytest.param(FrankCopula(theta_frank, 2), -math.log1p(ratio) / theta_frank, id="frank"),
        pytest.param(
            JoeCopula(theta_joe, 2),
            1 - (1 - (1 - 0.5**theta_joe) ** 2) ** (1 / theta_joe),
            id="joe",
        ),
        pytest.param(PlackettCopula(delta), math.sqrt(delta) / (2 * (math.sqrt(delta) + 1)), id="plackett"),
        pytest.param(
            MM1Copula([[1.0, 1.7], [1.7, 1.0]], theta=1.3),
            2 ** (-(2 ** (1 / (1.7 * 1.3)))),
            id="mm1",
        ),
        pytest.param(GalambosCopula(2.0, 2), 0.25 * 2 ** (2 ** (-0.5)), id="galambos"),
        pytest.param(
            HuslerReissCopula(xp.asarray([[0.0, lambda_hr], [lambda_hr, 0.0]])),
            2 ** (-2 * stats.norm.cdf(lambda_hr)),
            id="husler-reiss",
        ),
        pytest.param(
            ExtremalTCopula(xp.asarray([[1.0, rho], [rho, 1.0]]), nu),
            2 ** (-(2 - ext_t)),
            id="extremal-t",
        ),
    ]


@pytest.mark.parametrize("copula,median_probability", _copula_examples())
@pytest.mark.parametrize("method", [SamplingMethod.MC, SamplingMethod.RQMC], ids=["mc", "rqmc"])
def test_every_copula_has_correct_margins_and_median_dependence(copula, median_probability: float, method) -> None:
    """Run all copulas through both generators and check the target joint law."""
    original = (config.seed, config.rng, config.sampling_method, config.qmc_dimensions)
    try:
        set_sampling_method(method, dimensions=16)
        set_random_seed(4223)
        n_sims = 4096
        samples = list(copula.generate(n_sims))
        assert len(samples) == 2
        u, v = (asnumpy(sample.values) for sample in samples)
        assert u.shape == (n_sims,) and v.shape == (n_sims,)
        assert np.all(np.isfinite(u)) and np.all(np.isfinite(v))
        assert np.all((u >= 0) & (u <= 1))
        assert np.all((v >= 0) & (v <= 1))

        # The one-dimensional CDF and centre-of-unit-square probability have
        # known analytic references for every family in the matrix.
        for margin in (u, v):
            for quantile in (0.25, 0.5, 0.75):
                assert np.mean(margin <= quantile) == pytest.approx(quantile, abs=0.047)
        assert np.mean((u <= 0.5) & (v <= 0.5)) == pytest.approx(median_probability, abs=0.05)
        exponent = _extreme_value_diagonal_exponent(copula)
        if exponent is not None:
            # Unlike an asymptotic tail-dependence estimate, the finite-q
            # joint exceedance probability 1 - 2q + C(q,q) is exact here.
            q = 0.9
            expected_joint_tail = 1 - 2 * q + q**exponent
            assert np.mean((u > q) & (v > q)) == pytest.approx(expected_joint_tail, abs=0.027)
        assert samples[0].coupled_variable_group is samples[1].coupled_variable_group
    finally:
        config.seed, config.rng, config.sampling_method, config.qmc_dimensions = original


def test_active_draw_preserves_legacy_randint_monte_carlo() -> None:
    """CuPy RandomState, like NumPy's legacy RNG, provides randint, not integers."""
    legacy = np.random.RandomState(823)
    reference = np.random.RandomState(823)
    chosen = xp.asarray([0, 3, 9, 12])
    actual = _draw_active_scenarios(legacy, "integers", 32, chosen, low=0, high=7, size=4)
    np.testing.assert_array_equal(asnumpy(actual), reference.randint(low=0, high=7, size=4))


@pytest.mark.parametrize("dimensions", [2, 4, 16])
def test_active_sobol_coordinates_equal_complete_design(dimensions: int) -> None:
    """Shrinking path sets cannot create independent Sobol designs."""
    n = 128
    selected = [xp.asarray([0, 5, 37, 88, 127]), xp.asarray([5, 88]), xp.asarray([37])]
    full = RandomizedQMCGenerator(seed=581, dimensions=dimensions)
    active = RandomizedQMCGenerator(seed=581, dimensions=dimensions)
    for indices in selected:
        expected = asnumpy(full.random(n))[asnumpy(indices)]
        actual = asnumpy(active._active_uniforms(n, indices, (len(indices),), scenario_axis=0))
        np.testing.assert_array_equal(actual, expected)

    indices = selected[0]
    expected = asnumpy(full.random(size=(3, n), scenario_axis=1))[:, asnumpy(indices)]
    actual = asnumpy(active._active_uniforms(n, indices, (3, len(indices)), scenario_axis=1))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(asnumpy(active.random(n)), asnumpy(full.random(n)))
    assert set(active._streams) == {n}


@pytest.mark.parametrize("method", ["integers", "normal", "exponential", "chisquare", "standard_normal"])
def test_active_sobol_inverse_transforms_match_full_path_draws(method: str) -> None:
    n = 128
    indices = xp.asarray([0, 3, 23, 119])
    params = {
        "integers": {"low": 0, "high": 5},
        "normal": {"loc": 1.0, "scale": 2.5},
        "exponential": {"scale": 1.5},
        "chisquare": {"df": 4.5},
        "standard_normal": {},
    }[method]
    full = RandomizedQMCGenerator(seed=701, dimensions=8)
    active = RandomizedQMCGenerator(seed=701, dimensions=8)
    expected = asnumpy(getattr(full, method)(size=n, **params))[asnumpy(indices)]
    got = asnumpy(_draw_active_scenarios(active, method, n, indices, size=len(indices), **params))
    np.testing.assert_allclose(got, expected, rtol=3e-12, atol=3e-12)
    np.testing.assert_array_equal(asnumpy(active.random(n)), asnumpy(full.random(n)))


@pytest.mark.parametrize(
    "copula",
    [
        GalambosCopula(1.0, 2),
        HuslerReissCopula(xp.asarray([[0.0, 0.7], [0.7, 0.0]])),
        ExtremalTCopula(xp.asarray([[1.0, 0.6], [0.6, 1.0]]), 3.0),
    ],
    ids=["galambos", "husler-reiss", "extremal-t"],
)
def test_iterative_copulas_keep_one_sobol_path_stream(copula) -> None:
    """All active-set sizes consume the original path-indexed Sobol stream."""
    n = 512
    first_rng = RandomizedQMCGenerator(seed=1211, dimensions=8)
    second_rng = RandomizedQMCGenerator(seed=1211, dimensions=8)
    first = list(copula.generate(n_sims=n, rng=first_rng))
    second = list(copula.generate(n_sims=n, rng=second_rng))
    for a, b in zip(first, second):
        np.testing.assert_array_equal(asnumpy(a.values), asnumpy(b.values))
    assert set(first_rng._streams) == {n}
    assert set(second_rng._streams) == {n}


@pytest.mark.parametrize(
    "copula",
    [
        GalambosCopula(1.5, 3),
        HuslerReissCopula(xp.asarray([[0.0, 0.8, 0.9], [0.8, 0.0, 1.1], [0.9, 1.1, 0.0]])),
        ExtremalTCopula(xp.asarray([[1.0, 0.6, 0.3], [0.6, 1.0, 0.5], [0.3, 0.5, 1.0]]), 3.0),
    ],
    ids=["galambos-3d", "husler-reiss-3d", "extremal-t-3d"],
)
def test_iterative_copulas_preserve_multivariate_qmc_structure(copula) -> None:
    """Exercise changing active sets and multiple coordinates per proposal."""
    n = 768
    rng = RandomizedQMCGenerator(seed=311, dimensions=4)
    variables = list(copula.generate(n_sims=n, rng=rng))
    assert len(variables) == 3
    for variable in variables:
        values = asnumpy(variable.values)
        assert values.shape == (n,)
        assert np.all(np.isfinite(values))
        assert np.all((values >= 0) & (values <= 1))
        assert np.mean(values <= 0.5) == pytest.approx(0.5, abs=0.075)
        assert variable.coupled_variable_group is variables[0].coupled_variable_group
    assert set(rng._streams) == {n}


@pytest.mark.parametrize("copula,expected_median", _copula_examples())
def test_every_copula_can_reorder_coupled_stochastic_variables_under_rqmc(copula, expected_median: float) -> None:
    """Applying a copula must retain both exact marginal samples and chained values."""
    original = (config.seed, config.rng, config.sampling_method, config.qmc_dimensions)
    try:
        set_sampling_method(SamplingMethod.RQMC, dimensions=8)
        set_random_seed(912)
        n = 512
        first = Gamma(alpha=2.0, theta=5.0).generate(n)
        derived = first * 1.5 + 3.0
        second = LogNormal(mu=1.5, sigma=0.6).generate(n)
        marginal_first = np.sort(asnumpy(first.values).copy())
        marginal_second = np.sort(asnumpy(second.values).copy())

        copula.apply([first, second])

        np.testing.assert_array_equal(np.sort(asnumpy(first.values)), marginal_first)
        np.testing.assert_array_equal(np.sort(asnumpy(second.values)), marginal_second)
        np.testing.assert_allclose(asnumpy(derived.values), asnumpy(first.values) * 1.5 + 3.0)
        assert first.coupled_variable_group is second.coupled_variable_group
        assert first.coupled_variable_group is derived.coupled_variable_group
    finally:
        config.seed, config.rng, config.sampling_method, config.qmc_dimensions = original

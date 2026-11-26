"""Independent numerical and coupling checks for simplified vine copulas."""

import math

import numpy as np
import pytest
from scipy.special import ndtri

from pal import StochasticScalar, config, set_random_seed
from pal._maths import asnumpy, xp
from pal.config import set_sampling_method
from pal.copulas import PairCopula, VineCopula


def _cdf(family, theta, u, v):
    """Evaluate the unrotated bivariate CDF at an interior point."""
    if family == "independence":
        return u * v
    if family == "gaussian":
        return 0.25 + math.asin(theta) / (2 * math.pi)
    if family == "clayton":
        return (u ** (-theta) + v ** (-theta) - 1) ** (-1 / theta)
    if family == "gumbel":
        return math.exp(-(((-math.log(u)) ** theta + (-math.log(v)) ** theta) ** (1 / theta)))
    return -math.log1p(math.expm1(-theta * u) * math.expm1(-theta * v) / math.expm1(-theta)) / theta


@pytest.mark.parametrize("method", ["mc", "rqmc"])
@pytest.mark.parametrize(
    "family,theta",
    [("independence", 0.0), ("gaussian", -0.6), ("clayton", 2.0), ("gumbel", 2.0), ("frank", 4.0)],
)
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_pair_margins_and_joint_probability(method, family, theta, rotation):
    """Check an analytic joint probability, uniformity and scenario coupling."""
    original = (config.seed, config.rng, config.sampling_method, config.qmc_dimensions)
    try:
        set_sampling_method(method, dimensions=16)
        set_random_seed(831)
        pair = PairCopula(family, theta, rotation)
        result = list(VineCopula([[pair]]).generate(8192))
        u, v = (asnumpy(sample.values) for sample in result)
        assert result[0].coupled_variable_group is result[1].coupled_variable_group
        for margin in (u, v):
            assert np.all((margin > 0) & (margin < 1))
            for q in (0.1, 0.5, 0.9):
                assert np.mean(margin < q) == pytest.approx(q, abs=0.025)
        a, b = (0.5, 0.5) if family == "gaussian" else (0.3, 0.7)
        if rotation == 90:
            expected = b - _cdf(family, theta, 1 - a, b)
        elif rotation == 180:
            expected = a + b - 1 + _cdf(family, theta, 1 - a, 1 - b)
        elif rotation == 270:
            expected = a - _cdf(family, theta, a, 1 - b)
        else:
            expected = _cdf(family, theta, a, b)
        assert np.mean((u < a) & (v < b)) == pytest.approx(expected, abs=0.025)
    finally:
        config.seed, config.rng, config.sampling_method, config.qmc_dimensions = original


@pytest.mark.parametrize("structure", ["c", "d"])
@pytest.mark.parametrize("method", ["mc", "rqmc"])
def test_gaussian_vine_reproduces_partial_correlation_matrix(structure, method):
    """An all-Gaussian vine is an ordinary Gaussian copula in any dimension."""
    rows = [[0.45, -0.3, 0.65], [0.2, -0.4], [0.35]]
    corr = np.eye(4)
    for k, row in enumerate(rows):
        for i, rho in enumerate(row):
            if structure == "c":
                a, b, cond = k, k + i + 1, list(range(k))
            else:
                a, b, cond = i, i + k + 1, list(range(i + 1, i + k + 1))
            if cond:
                inv = np.linalg.inv(corr[np.ix_(cond, cond)])
                ca, cb = corr[a, cond], corr[b, cond]
                val = ca @ inv @ cb + rho * np.sqrt((1 - ca @ inv @ ca) * (1 - cb @ inv @ cb))
            else:
                val = rho
            corr[a, b] = corr[b, a] = val
    original = (config.seed, config.rng, config.sampling_method, config.qmc_dimensions)
    try:
        set_sampling_method(method, dimensions=16)
        set_random_seed(438)
        vine = VineCopula([[PairCopula("gaussian", p) for p in row] for row in rows], structure=structure)
        samples = list(vine.generate(16384))
        values = np.stack([asnumpy(sample.values) for sample in samples])
        np.testing.assert_allclose(np.corrcoef(ndtri(values)), corr, atol=0.035)
        for sample in samples:
            assert sample.coupled_variable_group is samples[0].coupled_variable_group
    finally:
        config.seed, config.rng, config.sampling_method, config.qmc_dimensions = original


@pytest.mark.parametrize("structure", ["c", "d"])
def test_mixed_vine_preserves_margins_and_apply_coupling(structure):
    """Mixed conditional families must preserve all uniform margins and ranks."""
    vine = VineCopula(
        [
            [PairCopula("clayton", 2, 180), PairCopula("gumbel", 1.7, 90), PairCopula("frank", 3)],
            [PairCopula("gaussian", 0.4), PairCopula("clayton", 1.5, 270)],
            [PairCopula("gumbel", 1.5)],
        ],
        structure=structure,
    )
    rng = np.random.default_rng(51)
    samples = list(vine.generate(8192, rng=rng))
    for sample in samples:
        values = asnumpy(sample.values)
        assert np.all(np.isfinite(values))
        assert np.mean(values <= 0.5) == pytest.approx(0.5, abs=0.025)
    variables = [StochasticScalar(xp.asarray(rng.normal(size=2048))) for _ in range(4)]
    derived = 2 * variables[-1]
    old = [np.sort(asnumpy(var.values)) for var in variables]
    vine.apply(variables)
    for var, expected in zip(variables, old, strict=True):
        np.testing.assert_array_equal(np.sort(asnumpy(var.values)), expected)
        assert var.coupled_variable_group is variables[0].coupled_variable_group
    np.testing.assert_array_equal(asnumpy(derived.values), 2 * asnumpy(variables[-1].values))
    with pytest.raises(ValueError, match="Number of variables"):
        vine.apply(variables[:3])
    assert vine.dimension == 4


@pytest.mark.parametrize("structure", ["c", "d"])
def test_independent_vine_uses_one_coordinate_per_margin(structure):
    """At independence the inverse transform is exactly the identity."""
    vine = VineCopula([[PairCopula(), PairCopula()], [PairCopula()]], structure=structure)
    expected = np.random.default_rng(77).uniform(size=(3, 32))
    samples = list(vine.generate(32, rng=np.random.default_rng(77)))
    np.testing.assert_array_equal(np.stack([asnumpy(s.values) for s in samples]), expected)


@pytest.mark.parametrize(
    "family,parameter",
    [("gaussian", 1.0), ("clayton", -1.0), ("gumbel", 0.9), ("frank", -2.0), ("independence", 0.2)],
)
def test_invalid_pair_parameters(family, parameter):
    with pytest.raises(ValueError, match="Invalid parameter"):
        PairCopula(family, parameter)


def test_invalid_structure_and_rows():
    with pytest.raises(ValueError, match="structure"):
        VineCopula([[PairCopula()]], structure="r")
    with pytest.raises(ValueError, match="rows"):
        VineCopula([[PairCopula(), PairCopula()]])
    with pytest.raises(ValueError, match="rows"):
        VineCopula([])
    with pytest.raises(ValueError, match="rotation"):
        PairCopula(rotation=45)
    with pytest.raises(ValueError, match="finite"):
        PairCopula("gaussian", float("nan"))
    with pytest.raises(TypeError, match="PairCopula"):
        VineCopula([[None]])
    with pytest.raises(ValueError, match="positive"):
        VineCopula([[PairCopula()]]).generate(0)

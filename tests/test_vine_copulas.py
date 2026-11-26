"""Independent numerical and coupling checks for simplified vine copulas."""

import math

import numpy as np
import pytest
from scipy.special import ndtri
from scipy.stats import norm, t

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


@pytest.mark.parametrize(
    "pair,expected",
    [
        (PairCopula.Independence(), PairCopula()),
        (PairCopula.Gaussian(-0.4), PairCopula("gaussian", -0.4)),
        (PairCopula.StudentsT(0.4, 5), PairCopula("students_t", 0.4, parameter2=5)),
        (PairCopula.Clayton(2, rotation=180), PairCopula("clayton", 2, 180)),
        (PairCopula.Gumbel(1.2), PairCopula("gumbel", 1.2)),
        (PairCopula.Frank(3), PairCopula("frank", 3)),
        (PairCopula.Joe(2), PairCopula("joe", 2)),
        (PairCopula.Galambos(2), PairCopula("galambos", 2)),
        (PairCopula.Plackett(0.4), PairCopula("plackett", 0.4)),
        (PairCopula.HuslerReiss(0.75), PairCopula("husler_reiss", 0.75)),
        (PairCopula.ExtremalT(0.6, 3), PairCopula("extremal_t", 0.6, parameter2=3)),
        (PairCopula.MM1(1.2, 1.5), PairCopula("mm1", 1.2, parameter2=1.5)),
    ],
)
def test_named_constructors(pair, expected):
    assert pair == expected


def _extended_cdf(pair, u, v):
    """Independent CDF reference for the additional pair families."""
    theta = pair.parameter
    a, b = -math.log(u), -math.log(v)
    if pair.family == "joe":
        return 1 - ((1 - u) ** theta + (1 - v) ** theta - ((1 - u) * (1 - v)) ** theta) ** (1 / theta)
    if pair.family == "galambos":
        return u * v * math.exp((a ** (-theta) + b ** (-theta)) ** (-1 / theta))
    if pair.family == "plackett":
        z = 1 + (theta - 1) * (u + v)
        return 2 * theta * u * v / (z + math.sqrt(z * z - 4 * theta * (theta - 1) * u * v))
    if pair.family == "husler_reiss":
        return math.exp(
            -a * norm.cdf(theta + math.log(a / b) / (2 * theta))
            - b * norm.cdf(theta + math.log(b / a) / (2 * theta))
        )
    if pair.family == "extremal_t":
        nu = pair.parameter2
        scale = math.sqrt((nu + 1) / (1 - theta**2))
        return math.exp(
            -a * t.cdf(scale * ((a / b) ** (1 / nu) - theta), nu + 1)
            - b * t.cdf(scale * ((b / a) ** (1 / nu) - theta), nu + 1)
        )
    return _cdf("gumbel", theta * pair.parameter2, u, v)


def _extra_pairs():
    """Additional families with positive and negative association."""
    return [
        PairCopula.Joe(2),
        PairCopula.Galambos(1.5),
        PairCopula.Plackett(0.3),
        PairCopula.Plackett(4),
        PairCopula.HuslerReiss(0.75),
        PairCopula.ExtremalT(0.6, 3),
        PairCopula.ExtremalT(-0.3, 2),
        PairCopula.MM1(1.2, 1.5),
    ]


@pytest.mark.parametrize("pair", _extra_pairs())
def test_extended_pair_conditional_against_cdf_derivative(pair):
    """The h-function must equal the derivative of an independent CDF."""
    u = np.array([0.05, 0.2, 0.5, 0.75, 0.95])
    v = np.array([0.8, 0.4, 0.1, 0.65, 0.3])
    eps = 1e-6
    expected = np.array(
        [
            (_extended_cdf(pair, x, y + eps) - _extended_cdf(pair, x, y - eps)) / (2 * eps)
            for x, y in zip(u, v, strict=True)
        ]
    )
    h = pair._conditional(xp.asarray(u), xp.asarray(v), first=True)
    np.testing.assert_allclose(asnumpy(h), expected, atol=2e-8)
    np.testing.assert_allclose(asnumpy(pair._inverse(h, xp.asarray(v), first=True)), u, atol=3e-9)


@pytest.mark.parametrize("method", ["mc", "rqmc"])
@pytest.mark.parametrize("pair", [*_extra_pairs(), PairCopula.StudentsT(0.6, 3), PairCopula.StudentsT(0.0, 4)])
def test_extended_pair_simulation(method, pair):
    """Check a joint probability and uniformity independently of the inverse."""
    original = (config.seed, config.rng, config.sampling_method, config.qmc_dimensions)
    try:
        set_sampling_method(method, dimensions=16)
        set_random_seed(714)
        samples = list(VineCopula([[pair]]).generate(8192))
        u, v = (asnumpy(sample.values) for sample in samples)
        assert np.all((u > 0) & (u < 1))
        assert np.all((v > 0) & (v < 1))
        for margin in (u, v):
            for q in (0.1, 0.5, 0.9):
                assert np.mean(margin < q) == pytest.approx(q, abs=0.025)
        if pair.family == "students_t":
            expected = 0.25 + math.asin(pair.parameter) / (2 * math.pi)
        else:
            expected = _extended_cdf(pair, 0.5, 0.5)
        assert np.mean((u < 0.5) & (v < 0.5)) == pytest.approx(expected, abs=0.025)
    finally:
        config.seed, config.rng, config.sampling_method, config.qmc_dimensions = original


@pytest.mark.parametrize("rho,dof", [(0.0, 3), (0.6, 5), (-0.5, 2)])
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_students_t_conditional_reference(rho, dof, rotation):
    """The conditional t has one additional degree of freedom, including at rho=0."""
    pair = PairCopula.StudentsT(rho, dof, rotation=rotation)
    u = np.array([0.02, 0.2, 0.6, 0.98])
    v = np.array([0.91, 0.05, 0.75, 0.3])
    reflected_u = 1 - u if rotation in (90, 180) else u
    reflected_v = 1 - v if rotation in (180, 270) else v
    x, y = t.ppf(reflected_u, dof), t.ppf(reflected_v, dof)
    expected = t.cdf((x - rho * y) / np.sqrt((dof + y**2) * (1 - rho**2) / (dof + 1)), dof + 1)
    if rotation in (90, 180):
        expected = 1 - expected
    h = pair._conditional(xp.asarray(u), xp.asarray(v), first=True)
    np.testing.assert_allclose(asnumpy(h), expected, atol=2e-8)
    np.testing.assert_allclose(asnumpy(pair._inverse(h, xp.asarray(v), first=True)), u, atol=2e-8)


@pytest.mark.parametrize("structure", ["c", "d"])
def test_extended_families_in_higher_trees(structure):
    vine = VineCopula(
        [
            [PairCopula.StudentsT(0.4, 5), PairCopula.Joe(2, rotation=180), PairCopula.HuslerReiss(0.8)],
            [PairCopula.Galambos(1.5, rotation=90), PairCopula.ExtremalT(0.4, 3)],
            [PairCopula.Plackett(0.5)],
        ],
        structure=structure,
    )
    samples = vine.generate(8192, rng=np.random.default_rng(58))
    for sample in samples:
        values = asnumpy(sample.values)
        assert np.all((values > 0) & (values < 1))
        for q in (0.1, 0.5, 0.9):
            assert np.mean(values < q) == pytest.approx(q, abs=0.025)


def test_extended_pair_validation_and_independence_limits():
    for pair in (PairCopula.Joe(1), PairCopula.Plackett(1), PairCopula.MM1(1, 1)):
        u = xp.asarray([0.1, 0.5, 0.9])
        v = xp.asarray([0.8, 0.3, 0.4])
        np.testing.assert_array_equal(asnumpy(pair._inverse(u, v, first=False)), asnumpy(u))
    with pytest.raises(ValueError, match="second parameter"):
        PairCopula("students_t", 0.4)
    with pytest.raises(ValueError, match="second parameter"):
        PairCopula.ExtremalT(0.5, 0)
    with pytest.raises(ValueError, match="second parameter"):
        PairCopula("gumbel", 2, parameter2=3)
    with pytest.raises(ValueError, match="Invalid parameter"):
        PairCopula.MM1(2, 0.5)
    with pytest.raises(ValueError, match="Invalid parameter"):
        PairCopula.HuslerReiss(0)

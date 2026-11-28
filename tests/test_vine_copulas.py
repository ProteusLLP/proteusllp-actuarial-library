"""Independent numerical and coupling checks for simplified vine copulas."""

import math

import numpy as np
import pytest
from scipy.special import ndtri
from scipy.stats import norm, t

from pal import StochasticScalar, config, set_random_seed
from pal._maths import asnumpy, xp
from pal.config import set_sampling_method
from pal.copulas import (
    _INVERSION_MAX_ITERATIONS,
    PairCopula,
    VineCopula,
    VineEdge,
    _student_t_tail_from_beta,
)


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
        VineCopula([[PairCopula(), PairCopula()], [PairCopula(), PairCopula()]])
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
            -a * norm.cdf(theta + math.log(a / b) / (2 * theta)) - b * norm.cdf(theta + math.log(b / a) / (2 * theta))
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


@pytest.mark.parametrize("dof", [1.5, 4.0, 31.0, 1001.0])
def test_student_t_tail_from_beta_keeps_relative_precision(dof):
    """The incomplete-beta Student's t tail matches SciPy near the centre and far into the tails.

    This formulation is used on backends without a native Student's t CDF.
    Near the centre the naive argument dof/(dof + x^2) rounds towards one, so
    the comparison includes very small |x| as well as extreme tails.
    """
    x = np.concatenate([[0.0], np.logspace(-12, 0, 40), np.linspace(1, 60, 200), np.logspace(2, 150, 40)])
    # CuPy's incomplete-beta normalisation subtracts log-gamma terms whose
    # rounding error grows with the degrees of freedom. Retain a relative
    # check in the tails, allowing that floating-point error on the GPU.
    rtol = max(2e-13, 8 * np.finfo(float).eps * dof) if xp.__name__ == "cupy" else 2e-13
    for values in (x, -x):
        tail = asnumpy(_student_t_tail_from_beta(xp.asarray(values), dof))
        np.testing.assert_allclose(tail, t.cdf(-np.abs(values), dof), rtol=rtol, atol=1e-300)


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


def _strong_pairs():
    """Strongly dependent pairs whose conditional quantiles lack a simple closed form."""
    return [
        PairCopula.Gumbel(30),
        PairCopula.MM1(3, 4),
        PairCopula.Joe(20),
        PairCopula.Galambos(10),
        PairCopula.HuslerReiss(0.1),
        PairCopula.HuslerReiss(4),
        PairCopula.ExtremalT(0.9, 2),
        PairCopula.ExtremalT(-0.8, 10),
        PairCopula.Plackett(0.002),
        PairCopula.Plackett(500),
    ]


@pytest.mark.parametrize(
    "pair",
    [
        *_strong_pairs(),
        PairCopula.Gumbel(1.0001),
        PairCopula.Gumbel(1.5),
        PairCopula.Joe(1.0001),
        PairCopula.Joe(1.5),
        PairCopula.Joe(1e4),
    ],
)
def test_conditional_quantiles_are_representable_roots_in_both_tails(pair):
    """Each conditional quantile is within two units in the last place of the exact root.

    The conditional CDF is increasing, so the target probability must lie
    between its values at the neighbouring representable arguments. Near one,
    adjacent representable arguments can change the conditional CDF
    substantially under strong dependence, so a residual tolerance alone would
    be either too loose in the body or unattainable in the corners. Weak and
    extreme Gumbel and Joe parameters exercise both regimes of their
    dedicated Newton starting bounds.
    """
    grid = np.array([1e-12, 1e-6, 0.01, 0.3, 0.7, 0.99, 1 - 1e-6])
    given = np.array([1e-10, 1e-4, 0.05, 0.5, 0.95, 1 - 1e-4, 1 - 1e-10])
    p, v = (values.ravel() for values in np.meshgrid(grid, given))
    slack = 1e-12 * p
    for first in (True, False):
        u = asnumpy(pair._inverse(xp.asarray(p), xp.asarray(v), first=first))
        lower = np.nextafter(np.nextafter(u, 0), 0)
        upper = np.nextafter(np.nextafter(u, 1), 1)
        h_lower = asnumpy(pair._conditional(xp.asarray(lower), xp.asarray(v), first=first))
        h_upper = asnumpy(pair._conditional(xp.asarray(upper), xp.asarray(v), first=first))
        assert np.all(h_lower <= p + slack)
        assert np.all(h_upper >= p - slack)


@pytest.mark.parametrize(
    "pair",
    [PairCopula.Galambos(1.5), PairCopula.HuslerReiss(0.05), PairCopula.ExtremalT(0.6, 3)],
)
def test_newton_inversion_converges_well_within_iteration_limit(pair, monkeypatch):
    """Every point must converge, not merely most points, so the safeguard limit is never reached."""
    evaluations = 0
    original = PairCopula._conditional_and_density

    def counted(self, u, v):
        nonlocal evaluations
        evaluations += 1
        return original(self, u, v)

    monkeypatch.setattr(PairCopula, "_conditional_and_density", counted)
    rng = np.random.default_rng(12)
    pair._inverse(xp.asarray(rng.uniform(size=20000)), xp.asarray(rng.uniform(size=20000)), first=True)
    assert evaluations < _INVERSION_MAX_ITERATIONS / 2


def _regular_trees():
    """A six-margin vine whose first tree is neither a star nor a chain."""
    return [
        [(0, 1, (), 0.5), (0, 2, (), -0.3), (0, 3, (), 0.35), (3, 4, (), 0.6), (3, 5, (), 0.2)],
        [(1, 2, (0,), 0.3), (2, 3, (0,), -0.2), (0, 4, (3,), 0.1), (4, 5, (3,), 0.25)],
        [(1, 3, (0, 2), 0.2), (2, 4, (0, 3), 0.3), (0, 5, (3, 4), -0.2)],
        [(1, 4, (0, 2, 3), -0.3), (2, 5, (0, 3, 4), 0.15)],
        [(1, 5, (0, 2, 3, 4), 0.4)],
    ]


def _reference_correlation(trees, truncation):
    """Recover unconditional correlations from Gaussian partial correlations."""
    corr = np.eye(6)
    for level, tree in enumerate(trees):
        for a, b, conditioning, rho in tree:
            if level >= truncation:
                rho = 0.0
            cond = list(conditioning)
            if cond:
                inv = np.linalg.inv(corr[np.ix_(cond, cond)])
                ca, cb = corr[a, cond], corr[b, cond]
                rho = ca @ inv @ cb + rho * np.sqrt((1 - ca @ inv @ ca) * (1 - cb @ inv @ cb))
            corr[a, b] = corr[b, a] = rho
    return corr


@pytest.mark.parametrize("truncation", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("method", ["mc", "rqmc"])
def test_regular_vine_gaussian_reference(truncation, method):
    """General vines, including truncated ones, must reproduce Gaussian partial correlations."""
    trees = _regular_trees()
    # Permute labels and reverse the edge order to check that no special
    # variable ordering or input edge ordering is needed.
    labels = [4, 2, 5, 0, 1, 3]
    edges = [
        [
            VineEdge(labels[b], labels[a], PairCopula.Gaussian(rho), tuple(labels[i] for i in cond))
            for a, b, cond, rho in reversed(tree)
        ]
        for tree in trees[:truncation]
    ]
    vine = VineCopula.from_edges(edges)
    assert vine.dimension == 6
    assert vine.truncation_level == truncation
    assert vine.structure == "r"
    original = (config.seed, config.rng, config.sampling_method, config.qmc_dimensions)
    try:
        set_sampling_method(method, dimensions=16)
        set_random_seed(915)
        samples = list(vine.generate(16384))
        z = np.stack([ndtri(asnumpy(samples[i].values)) for i in labels])
        np.testing.assert_allclose(np.corrcoef(z), _reference_correlation(trees, truncation), atol=0.04)
        for sample in samples:
            assert sample.coupled_variable_group is samples[0].coupled_variable_group
    finally:
        config.seed, config.rng, config.sampling_method, config.qmc_dimensions = original


@pytest.mark.parametrize("rotation", [0, 180])
@pytest.mark.parametrize("method", ["mc", "rqmc"])
def test_regular_clayton_vine_reproduces_multivariate_clayton(rotation, method):
    """Clayton pairs with parameter theta/(1 + k theta) in tree k+1 form the Clayton copula.

    The multivariate Clayton copula is a simplified vine for every regular-vine
    structure (Stöber, Joe and Czado, 2013, J. Multivariate Anal. 119, 101-118),
    so its closed-form CDF checks conditioning and edge orientation in the
    higher trees independently of the Gaussian case. Rotating every pair by 180
    degrees gives the distribution of one minus a Clayton vector.
    """
    theta = 2.0
    labels = [4, 2, 5, 0, 1, 3]
    edges = [
        [
            VineEdge(
                labels[a],
                labels[b],
                PairCopula.Clayton(theta / (1 + level * theta), rotation=rotation),
                tuple(labels[i] for i in cond),
            )
            for a, b, cond, _ in tree
        ]
        for level, tree in enumerate(_regular_trees())
    ]
    n_sims = 2**15
    original = (config.seed, config.rng, config.sampling_method, config.qmc_dimensions)
    try:
        set_sampling_method(method, dimensions=16)
        set_random_seed(2013)
        samples = np.stack([asnumpy(sample.values) for sample in VineCopula.from_edges(edges).generate(n_sims)])
    finally:
        config.seed, config.rng, config.sampling_method, config.qmc_dimensions = original
    if rotation == 180:
        samples = 1 - samples
    for point in ([0.5] * 6, [0.2, 0.9, 0.5, 0.7, 0.3, 0.8], [0.1] * 6):
        expected = (sum(u ** (-theta) for u in point) - len(point) + 1) ** (-1 / theta)
        empirical = np.mean(np.all(samples < np.array(point)[:, None], axis=0))
        standard_error = math.sqrt(expected * (1 - expected) / n_sims)
        assert empirical == pytest.approx(expected, abs=4 * standard_error)


@pytest.mark.parametrize("structure", ["c", "d"])
def test_compact_and_regular_vine_equivalence(structure):
    """The compact syntax and its expanded edge representation are identical."""
    compact = VineCopula(
        [
            [PairCopula.Gumbel(1.5), PairCopula.Clayton(2, rotation=90), PairCopula.Frank(2)],
            [PairCopula.StudentsT(0.3, 4), PairCopula.Joe(2, rotation=270)],
            [PairCopula.Galambos(1.5)],
        ],
        structure=structure,
    )
    regular = VineCopula.from_edges(compact.trees)
    a = compact.generate(1024, rng=np.random.default_rng(22))
    b = regular.generate(1024, rng=np.random.default_rng(22))
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(asnumpy(x.values), asnumpy(y.values))


@pytest.mark.parametrize("structure", ["c", "d"])
def test_compact_truncated_vine(structure):
    """Omitted higher trees and their explicit independent pairs are equivalent in law."""
    pairs = [[PairCopula.Gaussian(0.5), PairCopula.Gaussian(0.4), PairCopula.Gaussian(-0.3)]]
    truncated = VineCopula(pairs, structure=structure)
    full = VineCopula([*pairs, [PairCopula(), PairCopula()], [PairCopula()]], structure=structure)
    assert truncated.truncation_level == 1
    for vine in (truncated, full):
        samples = vine.generate(16384, rng=np.random.default_rng(678))
        corr = np.corrcoef(np.stack([ndtri(asnumpy(sample.values)) for sample in samples]))
        expected = 0.5 * 0.4 if structure == "c" else 0.4
        assert corr[1, 2] == pytest.approx(expected, abs=0.035)


def test_regular_vine_rotations_keep_edge_orientation():
    """An edge's first and second coordinates are significant for rotations."""
    pair = PairCopula.Clayton(2, rotation=90)
    vine = VineCopula.from_edges(
        [[VineEdge(2, 0, pair), VineEdge(1, 2, PairCopula.Joe(2)), VineEdge(2, 3, PairCopula.Gumbel(1.5))]]
    )
    samples = list(vine.generate(16384, rng=np.random.default_rng(849)))
    u, v = (asnumpy(samples[i].values) for i in (2, 0))
    a, b = 0.2, 0.6
    expected = b - _cdf("clayton", 2, 1 - a, b)
    assert np.mean((u < a) & (v < b)) == pytest.approx(expected, abs=0.02)
    for sample in samples:
        assert np.mean(asnumpy(sample.values) < 0.5) == pytest.approx(0.5, abs=0.025)


def test_regular_vine_validation():
    pair = PairCopula.Independence()
    chain = [VineEdge(0, 1, pair), VineEdge(1, 2, pair), VineEdge(2, 3, pair)]
    with pytest.raises(ValueError, match="connected and acyclic"):
        VineCopula.from_edges([[VineEdge(0, 1, pair), VineEdge(1, 2, pair), VineEdge(0, 2, pair)]])
    with pytest.raises(ValueError, match="Duplicate"):
        VineCopula.from_edges([[chain[0], chain[0], chain[2]]])
    with pytest.raises(ValueError, match="preceding tree"):
        VineCopula.from_edges([[VineEdge(0, 4, pair), chain[1], chain[2]]])
    with pytest.raises(ValueError, match="preceding tree"):
        VineCopula.from_edges([chain, [VineEdge(0, 3, pair, (1,)), VineEdge(1, 3, pair, (2,))]])
    with pytest.raises(ValueError, match="conditioning variable"):
        VineCopula.from_edges([chain, [VineEdge(0, 2, pair), VineEdge(1, 3, pair, (2,))]])
    with pytest.raises(ValueError, match="distinct"):
        VineCopula.from_edges([chain, [VineEdge(1, 2, pair, (1,)), VineEdge(1, 3, pair, (2,))]])
    with pytest.raises(TypeError, match="PairCopula"):
        VineCopula.from_edges([[VineEdge(0, 1, None)]])

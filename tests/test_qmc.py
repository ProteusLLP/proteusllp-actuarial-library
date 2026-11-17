"""Statistical and integration tests for PAL's randomized QMC mode."""

import numpy as np
import pytest
from scipy import stats

from pal import StochasticScalar, config, set_random_seed
from pal._maths import asnumpy, create_random_generator, xp
from pal.config import SamplingMethod, set_sampling_method, start_new_sampling_block
from pal.copulas import GaussianCopula
from pal.distributions import Binomial, Exponential, Gamma, GeneralizedInverseGaussian, Normal, Poisson, Uniform
from pal.frequency_severity import FrequencySeverityModel
from pal.qmc import RandomizedQMCGenerator


@pytest.fixture(autouse=True)
def restore_rng():
    """Do not change the RNG or sampling settings for other tests."""
    previous = (config.seed, config.sampling_method, config.qmc_dimensions, config.rng)
    yield
    config.seed, config.sampling_method, config.qmc_dimensions, config.rng = previous


def _assert_sobol_stratification(values):
    """A scrambled 2**m Sobol coordinate visits each 1/n stratum once."""
    values = asnumpy(values)
    n = len(values)
    assert np.array_equal(np.sort(np.floor(values * n).astype(int)), np.arange(n))


def test_independent_scrambled_blocks_and_dimension_recycling():
    rng = RandomizedQMCGenerator(seed=143, dimensions=2)
    x = rng.random(1024)
    y = rng.random(1024)
    z = rng.random(1024)  # First coordinate of the next block.
    for values in (x, y, z):
        _assert_sobol_stratification(values)
    assert not np.array_equal(asnumpy(x), asnumpy(z))
    assert abs(np.corrcoef(asnumpy(x), asnumpy(y))[0, 1]) < 0.08
    assert abs(np.corrcoef(asnumpy(x), asnumpy(z))[0, 1]) < 0.08

    rng2 = RandomizedQMCGenerator(seed=143, dimensions=2)
    assert np.array_equal(asnumpy(x), asnumpy(rng2.random(1024)))
    assert np.array_equal(asnumpy(y), asnumpy(rng2.random(1024)))
    assert np.array_equal(asnumpy(z), asnumpy(rng2.random(1024)))


def test_manual_block_reset_and_non_power_of_two():
    rng = RandomizedQMCGenerator(seed=7, dimensions=4)
    first = asnumpy(rng.uniform(size=1000))
    rng.new_block()
    second = asnumpy(rng.uniform(size=1000))
    assert first.shape == (1000,)
    assert np.all((first >= 0) & (first < 1))
    assert not np.array_equal(first, second)
    assert np.unique(first).size == 1000


def test_matrix_draw_uses_one_coordinate_per_component():
    rng = RandomizedQMCGenerator(seed=11, dimensions=4)
    draws = asnumpy(rng.standard_normal(size=(3, 1024)))
    assert draws.shape == (3, 1024)
    for row in draws:
        _assert_sobol_stratification(stats.norm.cdf(row))
    assert abs(np.corrcoef(draws[0], draws[1])[0, 1]) < 0.08
    matrices = asnumpy(rng.random(size=(1024, 2, 2)))
    assert matrices.shape == (1024, 2, 2)
    for i in range(2):
        for j in range(2):
            _assert_sobol_stratification(matrices[:, i, j])


def test_scipy_quantile_marginals_with_stochastic_parameters():
    rng = RandomizedQMCGenerator(seed=9, dimensions=3)
    n = 4096
    shape = np.linspace(1.0, 3.0, n)
    gamma_draws = asnumpy(rng.gamma(shape, 2.0, size=n))
    assert gamma_draws.shape == (n,)
    assert np.all(gamma_draws > 0)
    # PIT with varying shape parameters tests the inverse transform elementwise.
    probabilities = stats.gamma.cdf(gamma_draws, a=shape, scale=2.0)
    assert np.max(np.abs(probabilities - asnumpy(rng2_uniform_reference(9, n)))) < 1e-12


def rng2_uniform_reference(seed, n):
    """Recreate the first coordinate from the same scrambled block."""
    return RandomizedQMCGenerator(seed=seed, dimensions=3).uniform(size=n)


def test_distribution_moments_and_seed_api():
    set_sampling_method(SamplingMethod.RQMC, dimensions=3)
    set_random_seed(451)
    n = 4096
    normal = Normal(mu=2, sigma=3).generate(n)
    gamma = Gamma(alpha=3, theta=4).generate(n)
    poisson = Poisson(mean=2.5).generate(n)
    assert abs(float(np.mean(asnumpy(normal.values))) - 2.0) < 0.03
    assert abs(float(np.mean(asnumpy(gamma.values))) - 12.0) < 0.10
    assert abs(float(np.mean(asnumpy(poisson.values))) - 2.5) < 0.05

    first = asnumpy(normal.values).copy()
    set_random_seed(451)
    assert np.array_equal(first, asnumpy(Normal(mu=2, sigma=3).generate(n).values))
    start_new_sampling_block()
    assert not np.array_equal(first, asnumpy(Normal(mu=2, sigma=3).generate(n).values))


def test_stochastic_parameter_coupling_and_explicit_rng_override():
    set_sampling_method(SamplingMethod.RQMC, dimensions=2)
    set_random_seed(53)
    n = 1024
    base = Normal(mu=0, sigma=1).generate(n)
    parameter = np.exp(base)
    result = Gamma(alpha=2.0, theta=parameter).generate(n)
    assert result.n_sims == n
    assert base in result.coupled_variable_group
    assert parameter in result.coupled_variable_group
    assert np.all(asnumpy(result.values) > 0)

    ordinary = np.random.default_rng(7)
    expected = np.random.default_rng(7).normal(size=n)
    actual = Normal(mu=0, sigma=1).generate(n, rng=ordinary)
    # Exact equality of raw normal draws is not assumed because PAL samples by inverse CDF.
    assert np.isfinite(asnumpy(actual.values)).all()
    assert abs(np.mean(asnumpy(actual.values)) - np.mean(expected)) < 0.2


def test_qmc_frequency_severity_and_copula():
    set_sampling_method(SamplingMethod.RQMC, dimensions=4)
    set_random_seed(24)
    n = 2048
    sims = FrequencySeverityModel(Poisson(mean=1.5), Exponential(scale=100)).generate(n)
    counts = asnumpy(xp.bincount(sims.sim_index, minlength=n))
    assert counts.shape == (n,)
    assert abs(np.mean(counts) - 1.5) < 0.05
    assert np.all(asnumpy(sims.values) >= 0)
    assert abs(np.mean(asnumpy(sims.aggregate().values)) - 150) < 15

    copula = GaussianCopula([[1.0, 0.6], [0.6, 1.0]])
    generated = list(copula.generate(n))
    left = asnumpy(generated[0].values)
    right = asnumpy(generated[1].values)
    assert np.all((left > 0) & (left < 1))
    assert np.all((right > 0) & (right < 1))
    assert abs(np.corrcoef(stats.norm.ppf(left), stats.norm.ppf(right))[0, 1] - 0.6) < 0.05


def test_frequency_severity_qmc_uses_a_dimension_per_claim_ordinal():
    """Claims in a simulation must not use unrelated Sobol path rows."""
    rng = RandomizedQMCGenerator(seed=352, dimensions=3)
    n = 1024
    model = FrequencySeverityModel(Binomial(n=2, p=1), Uniform(a=0.0, b=1.0))
    samples = model.generate(n_sims=n, rng=rng)

    assert np.array_equal(asnumpy(samples.sim_index), np.repeat(np.arange(n), 2))
    first_claim = asnumpy(samples.values)[0::2]
    second_claim = asnumpy(samples.values)[1::2]
    _assert_sobol_stratification(first_claim)
    _assert_sobol_stratification(second_claim)
    assert abs(np.corrcoef(first_claim, second_claim)[0, 1]) < 0.08


def test_qmc_frequency_severity_preserves_scenario_parameter_coupling():
    rng = RandomizedQMCGenerator(seed=514, dimensions=3)
    n = 256
    scale = StochasticScalar(np.linspace(10.0, 20.0, n))
    model = FrequencySeverityModel(Binomial(n=2, p=1), Exponential(scale=scale))
    result = model.generate(n_sims=n, rng=rng)

    assert result.n_sims == n
    assert scale in result.coupled_variable_group
    assert np.all(asnumpy(result.values) >= 0)
    assert np.isclose(np.mean(asnumpy(result.values)), 15.0, rtol=0.1)


def test_generalized_inverse_gaussian_uses_sobol():
    if xp.__name__ != "numpy":
        pytest.skip("The GIG distribution itself is CPU-only.")
    set_sampling_method(SamplingMethod.RQMC, dimensions=2)
    draw = GeneralizedInverseGaussian(p=-0.5, chi=2, psi=3).generate(128)
    assert np.all(np.isfinite(asnumpy(draw.values)))
    assert np.all(asnumpy(draw.values) > 0)


def test_invalid_dimensions_and_mode():
    with pytest.raises(ValueError, match="dimensions"):
        RandomizedQMCGenerator(dimensions=0)
    with pytest.raises(ValueError, match="method"):
        set_sampling_method("halton")


def test_sampling_method_enum_and_string_compatibility():
    """The readable enum is canonical; earlier string inputs still work."""
    assert SamplingMethod.MC.value == "pseudo"
    assert SamplingMethod.RQMC.value == "rqmc"
    set_sampling_method(SamplingMethod.RQMC, dimensions=5)
    assert config.sampling_method is SamplingMethod.RQMC
    assert config.qmc_dimensions == 5
    assert isinstance(config.rng, RandomizedQMCGenerator)
    set_random_seed(42)
    assert isinstance(config.rng, RandomizedQMCGenerator)

    set_sampling_method(SamplingMethod.MC)
    assert config.sampling_method is SamplingMethod.MC
    assert isinstance(config.rng, type(create_random_generator(42)))
    set_sampling_method("rqmc")
    assert config.sampling_method is SamplingMethod.RQMC
    set_sampling_method("pseudo")
    assert config.sampling_method is SamplingMethod.MC


def test_beta_rqmc_quantiles_match_scipy_on_cpu_and_gpu():
    """The inverse incomplete beta function takes (alpha, beta, probability)."""
    rng = RandomizedQMCGenerator(seed=91, dimensions=3)
    reference = RandomizedQMCGenerator(seed=91, dimensions=3)
    alpha, beta, n_sims = 2.0, 5.0, 4096
    uniforms = asnumpy(reference.random(n_sims))
    expected = stats.beta.ppf(uniforms, alpha, beta)
    actual = asnumpy(rng.beta(alpha, beta, size=n_sims))
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-12)
    assert np.all(np.isfinite(actual))
    assert np.mean(actual) == pytest.approx(alpha / (alpha + beta), abs=0.004)


def test_qmc_memory_cache_eviction_and_reproducibility():
    """Memory-based eviction keeps only the active oversized Sobol block."""
    rng = RandomizedQMCGenerator(seed=981, dimensions=8)
    reference = RandomizedQMCGenerator(seed=981, dimensions=8)
    rng.max_cache_bytes = 1000
    reference.max_cache_bytes = 1000
    for n in (1024, 1025, 1024):
        result = rng.random(n)
        expected = reference.random(n)
        np.testing.assert_array_equal(asnumpy(result), asnumpy(expected))
        assert len(rng._streams) == 1
        block = rng._streams[n]
        assert block.gray.size == n
        assert block.tables.shape == (8, 4, 256)
        assert block.nbytes == block.gray.nbytes + block.tables.nbytes + block.shifts.nbytes
        if n == 1024:
            _assert_sobol_stratification(result)


def test_explicit_scenario_axis_handles_more_components_than_paths():
    """Each component receives a Sobol coordinate even when d exceeds n_sims."""
    n_sims, components = 2, 7
    rng = RandomizedQMCGenerator(seed=1371, dimensions=16)
    draws = asnumpy(rng.random(size=(components, n_sims), scenario_axis=1))
    assert draws.shape == (components, n_sims)
    for component in draws:
        _assert_sobol_stratification(component)

    matrices = asnumpy(rng.random(size=(n_sims, 3, 3), scenario_axis=0))
    assert matrices.shape == (n_sims, 3, 3)
    for i in range(3):
        for j in range(3):
            _assert_sobol_stratification(matrices[:, i, j])


def test_multivariate_gaussian_rqmc_with_more_components_than_sims():
    """The Gaussian copula must preserve the scenario axis for tiny path counts."""
    rng = RandomizedQMCGenerator(seed=113, dimensions=16)
    copula = GaussianCopula(np.eye(5))
    generated = list(copula.generate(2, rng=rng))
    assert len(generated) == 5
    for component in generated:
        _assert_sobol_stratification(component.values)


def test_invalid_scenario_axis():
    """Reject out-of-range scenario-axis declarations."""
    rng = RandomizedQMCGenerator(seed=12, dimensions=2)
    with pytest.raises(ValueError, match="scenario_axis"):
        rng.random(size=(2, 4), scenario_axis=2)


def test_lazy_sobol_coordinates_match_scipy_full_design():
    """Lazy CPU coordinates must match SciPy LMS+shift Sobol exactly."""
    if xp is not np:
        pytest.skip("NVIDIA cuRAND uses different scrambled direction vectors from SciPy")
    from scipy.stats import qmc

    seed, dimensions, n = 814, 4, 1024
    generator = RandomizedQMCGenerator(seed=seed, dimensions=dimensions)
    actual = np.column_stack([asnumpy(generator.random(n)) for _ in range(dimensions)])
    random = np.random.default_rng(seed)
    sobol_seed = int(random.integers(0, 2**32, dtype=np.uint64))
    expected = qmc.Sobol(d=dimensions, scramble=True, bits=32, seed=sobol_seed).random_base2(10)
    expected = expected[random.permutation(n)]
    np.testing.assert_array_equal(actual, expected)


def test_sobol_memory_is_linear_in_paths_not_dimensions():
    """Retained memory is O(n + dimensions), not O(n * dimensions)."""
    n = 4096
    rng = RandomizedQMCGenerator(seed=51, dimensions=32)
    rng.random(n)
    block = rng._streams[n]
    assert block.nbytes < 6 * n + 32 * 4096
    assert not hasattr(block, "points")

"""GPU-native probability-kernel regressions and reference checks."""

from unittest.mock import patch

import numpy as np
import pytest
from scipy import stats

from pal._gpu_distributions import generalized_inverse_gaussian_mc, noncentral_chisquare_cdf
from pal._maths import asnumpy, create_random_generator, xp
from pal.distributions import GeneralizedInverseGaussian, NonCentralChiSquared
from pal.qmc import RandomizedQMCGenerator


@pytest.mark.parametrize(
    ("df", "nonc"),
    [(0.5, 0.0), (1.0, 0.1), (2.0, 1.0), (5.0, 10.0), (20.0, 500.0), (30.0, 10000.0)],
)
def test_poisson_mixture_matches_scipy_tail_probabilities(df: float, nonc: float) -> None:
    """The mode-centred mixture must reproduce SciPy CDFs in both tails."""
    probabilities = np.asarray([1e-10, 1e-6, 0.001, 0.1, 0.5, 0.9, 0.999, 1 - 1e-6])
    quantiles = stats.ncx2.ppf(probabilities, df, nonc)
    actual = noncentral_chisquare_cdf(xp.asarray(quantiles), df, nonc)
    np.testing.assert_allclose(asnumpy(actual), probabilities, atol=4e-11, rtol=2e-9)


@pytest.mark.parametrize(("p", "b"), [(-0.5, 2.0), (0.5, 1.0), (0.0, 0.0001), (2.0, 0.01), (-2.0, 0.01), (1.0, 100.0)])
def test_vectorized_gig_sampler_matches_reference(p: float, b: float) -> None:
    """Check the full sampling distribution, including broad and skewed cases."""
    if xp is np:
        rng = np.random.default_rng(1001)
    else:
        rng = create_random_generator(1001)
    samples = generalized_inverse_gaussian_mc(rng, 10000, p, b, 1.0, 0.0)
    assert isinstance(samples, xp.ndarray)
    quantiles = stats.geninvgauss(p, b).ppf([0.01, 0.1, 0.5, 0.9, 0.99])
    empirical = np.mean(asnumpy(samples)[:, None] <= np.asarray(quantiles)[None, :], axis=0)
    np.testing.assert_allclose(empirical, [0.01, 0.1, 0.5, 0.9, 0.99], atol=0.025)


@pytest.mark.skipif(xp is np, reason="Requires a CuPy backend")
def test_gpu_native_hot_paths_do_not_call_host_scipy() -> None:
    """Disable host SciPy bridges while exercising GPU CDF, PPF, RNG, and RQMC."""
    import pal.distributions as distributions
    import pal.qmc as qmc

    with (
        patch.object(distributions, "_scipy_on_backend", side_effect=AssertionError("CPU probability fallback")),
        patch.object(qmc, "_scipy_on_backend", side_effect=AssertionError("CPU sampling fallback")),
        patch.object(qmc.scipy_stats.geninvgauss, "rvs", side_effect=AssertionError("CPU sampler")),
        patch.object(qmc, "asnumpy", side_effect=AssertionError("CPU data transfer")),
    ):
        distribution = NonCentralChiSquared(5.0, 3.0)
        cdf = distribution.cdf(8.0)
        quantile = distribution.invcdf(0.7)
        assert 0 < cdf < 1
        assert quantile > 0
        sobol = RandomizedQMCGenerator(seed=146)
        qmc_samples = sobol.noncentral_chisquare(5.0, 3.0, size=16)
        assert isinstance(qmc_samples, xp.ndarray)
        assert bool(xp.all(qmc_samples > 0))
        gig = GeneralizedInverseGaussian(-0.5, 2.0, 3.0)
        draws = gig.generate(256, rng=create_random_generator(146))
        assert isinstance(draws.values, xp.ndarray)
        assert bool(xp.all(draws.values > 0))


@pytest.mark.skipif(xp is np, reason="Requires a CuPy backend")
@pytest.mark.parametrize(("df", "nonc"), [(1.0, 0.0), (3.0, 0.5), (5.0, 3.0), (10.0, 30.0)])
def test_gpu_rqmc_poisson_mixture_has_correct_noncentral_chisquare_law(df: float, nonc: float) -> None:
    """Two-coordinate Sobol Poisson/gamma mixture must preserve the target distribution."""
    sample = RandomizedQMCGenerator(seed=146, dimensions=32).noncentral_chisquare(df, nonc, size=8192)
    assert isinstance(sample, xp.ndarray)
    checkpoints = stats.ncx2.ppf([0.05, 0.25, 0.5, 0.75, 0.95], df, nonc)
    observed = np.mean(asnumpy(sample)[:, None] < checkpoints[None, :], axis=0)
    np.testing.assert_allclose(observed, [0.05, 0.25, 0.5, 0.75, 0.95], atol=0.015)

"""Regression tests for the accelerated sampling paths and CPU fallbacks."""

from unittest.mock import patch

import numpy as np
import pytest
from scipy import stats

from pal._gpu_distributions import generalized_inverse_gaussian_mc
from pal._maths import asnumpy, create_random_generator, xp
from pal.distributions import GeneralizedInverseGaussian, NonCentralChiSquared
from pal.qmc import RandomizedQMCGenerator, _sample_generalized_inverse_gaussian


@pytest.mark.parametrize(("p", "b"), [(-0.5, 2.0), (0.5, 1.0), (-0.5, 0.01), (0.5, 100.0)])
def test_half_order_gig_exact_sampler_matches_reference(p: float, b: float) -> None:
    """GPU-native and NumPy reference paths must have the correct GIG law."""
    samples = generalized_inverse_gaussian_mc(create_random_generator(1001), 10000, p, b, 1.0, 0.0)
    assert isinstance(samples, xp.ndarray)
    quantiles = stats.geninvgauss(p, b).ppf([0.01, 0.1, 0.5, 0.9, 0.99])
    empirical = np.mean(asnumpy(samples)[:, None] <= np.asarray(quantiles)[None, :], axis=0)
    np.testing.assert_allclose(empirical, [0.01, 0.1, 0.5, 0.9, 0.99], atol=0.025)


@pytest.mark.skipif(xp is np, reason="Requires CuPy")
@pytest.mark.parametrize(("df", "nonc"), [(1.0, 0.0), (3.0, 0.5), (5.0, 3.0), (10.0, 30.0)])
def test_gpu_rqmc_poisson_mixture_noncentral_chisquare(df: float, nonc: float) -> None:
    """Two Sobol coordinates give the target noncentral chi-square law."""
    sample = RandomizedQMCGenerator(seed=146, dimensions=32).noncentral_chisquare(df, nonc, size=8192)
    assert isinstance(sample, xp.ndarray)
    checkpoints = stats.ncx2.ppf([0.05, 0.25, 0.5, 0.75, 0.95], df, nonc)
    observed = np.mean(asnumpy(sample)[:, None] < checkpoints[None, :], axis=0)
    np.testing.assert_allclose(observed, [0.05, 0.25, 0.5, 0.75, 0.95], atol=0.015)


@pytest.mark.skipif(xp is np, reason="Requires CuPy")
@pytest.mark.parametrize("p", [-0.5, 0.5])
def test_half_order_gig_gpu_rqmc_matches_distribution(p: float) -> None:
    """Device-resident GIG uses two Sobol coordinates with the right law."""
    samples = _sample_generalized_inverse_gaussian(
        RandomizedQMCGenerator(seed=146, dimensions=4), 8192, p, 2.5, 1.0, 0.0
    )
    assert isinstance(samples, xp.ndarray)
    quantiles = stats.geninvgauss(p, 2.5).ppf([0.05, 0.25, 0.5, 0.75, 0.95])
    observed = np.mean(asnumpy(samples)[:, None] < np.asarray(quantiles)[None, :], axis=0)
    np.testing.assert_allclose(observed, [0.05, 0.25, 0.5, 0.75, 0.95], atol=0.015)


@pytest.mark.skipif(xp is np, reason="Requires CuPy")
def test_native_gpu_sampling_avoids_scipy_and_array_host_transfers() -> None:
    """The accelerated MC/RQMC paths must not invoke host SciPy bridges."""
    import pal.distributions as distributions
    import pal.qmc as qmc

    with (
        patch.object(distributions, "_scipy_on_backend", side_effect=AssertionError("CPU probability bridge")),
        patch.object(qmc, "_scipy_on_backend", side_effect=AssertionError("CPU sampling bridge")),
        patch.object(qmc.scipy_stats.geninvgauss, "rvs", side_effect=AssertionError("CPU sampler")),
        patch.object(qmc, "asnumpy", side_effect=AssertionError("Host-array transfer")),
    ):
        rng = create_random_generator(146)
        assert isinstance(NonCentralChiSquared(5.0, 3.0).generate(256, rng=rng).values, xp.ndarray)
        samples = RandomizedQMCGenerator(seed=146).noncentral_chisquare(5.0, 3.0, size=256)
        assert isinstance(samples, xp.ndarray)
        assert bool(xp.all(samples > 0))
        dist = GeneralizedInverseGaussian(-0.5, 2.0, 3.0)
        assert isinstance(dist.generate(256, rng=rng).values, xp.ndarray)
        qmc_gig = dist.generate(256, rng=RandomizedQMCGenerator(seed=146))
        assert isinstance(qmc_gig.values, xp.ndarray)


@pytest.mark.parametrize("p", [0.0, 2.0, -2.0])
def test_general_index_gig_falls_back_and_preserves_distribution(p: float) -> None:
    """Slow general-index GPU sampling is replaced by the host SciPy path."""
    dist = GeneralizedInverseGaussian(p, 2.0, 3.0)
    samples = dist.generate(5000, rng=create_random_generator(146)).values
    assert isinstance(samples, xp.ndarray)
    actual = float(np.mean(asnumpy(samples)))
    mean = float(stats.geninvgauss.mean(p, np.sqrt(6.0), scale=np.sqrt(2 / 3)))
    assert np.isclose(actual, mean, rtol=0.07)


@pytest.mark.skipif(xp is np, reason="Requires CuPy")
def test_general_index_gig_cpu_bridge_uses_backend_arrays() -> None:
    """Specialised GIG host fallback transfers results back to CuPy."""
    import pal.qmc as qmc

    with patch.object(qmc.scipy_stats.geninvgauss, "rvs", wraps=stats.geninvgauss.rvs) as host:
        result = GeneralizedInverseGaussian(2.0, 2.0, 3.0).generate(
            256, rng=create_random_generator(146)
        )
    assert host.called
    assert isinstance(result.values, xp.ndarray)

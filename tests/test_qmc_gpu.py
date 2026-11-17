"""GPU-only correctness checks for PAL's native cuRAND scrambled Sobol backend."""

import numpy as np
import pytest
from scipy import stats

from pal._maths import asnumpy, xp
from pal.qmc import RandomizedQMCGenerator


@pytest.mark.skipif(xp is np, reason="Native cuRAND Sobol tests require a GPU")
def test_gpu_sobol_coordinates_stay_on_device_and_are_stratified():
    """Each scrambled Sobol coordinate must have one point in every 1/n bin."""
    n = 4096
    generator = RandomizedQMCGenerator(seed=137, dimensions=2)
    first, second, next_block = (generator.random(n) for _ in range(3))

    for values in (first, second, next_block):
        assert isinstance(values, xp.ndarray)
        strata = np.sort(np.floor(asnumpy(values) * n).astype(np.int64))
        np.testing.assert_array_equal(strata, np.arange(n))
        assert np.all(np.isfinite(asnumpy(values)))
        assert np.all((asnumpy(values) > 0) & (asnumpy(values) < 1))

    assert not np.array_equal(asnumpy(first), asnumpy(next_block))
    assert abs(np.corrcoef(asnumpy(first), asnumpy(next_block))[0, 1]) < 0.08


@pytest.mark.skipif(xp is np, reason="Native cuRAND Sobol tests require a GPU")
def test_gpu_sobol_seed_reproducibility_and_block_reset():
    """Digital shifts and GPU row permutation must both follow the PAL seed."""
    a = RandomizedQMCGenerator(seed=52, dimensions=1)
    b = RandomizedQMCGenerator(seed=52, dimensions=1)
    for _ in range(3):
        np.testing.assert_array_equal(asnumpy(a.random(1024)), asnumpy(b.random(1024)))

    a.new_block()
    b.new_block()
    np.testing.assert_array_equal(asnumpy(a.random(1024)), asnumpy(b.random(1024)))
    other = RandomizedQMCGenerator(seed=53, dimensions=1)
    assert not np.array_equal(asnumpy(a.random(1024)), asnumpy(other.random(1024)))


@pytest.mark.skipif(xp is np, reason="Native cuRAND Sobol tests require a GPU")
def test_gpu_common_quantiles_do_not_use_host_scipy(monkeypatch):
    """Keep common continuous PPFs device-resident, including stochastic parameters."""
    n = 4096
    rng = RandomizedQMCGenerator(seed=95, dimensions=8)

    def fail_host_transform(_value):
        raise AssertionError("Unexpected transfer to host for native GPU quantile")

    monkeypatch.setattr(RandomizedQMCGenerator, "_host", staticmethod(fail_host_transform))
    shape = xp.linspace(1.0, 4.0, n)
    normals = rng.normal(loc=2.0, scale=3.0, size=n)
    gamma = rng.gamma(shape, 2.0, size=n)
    beta = rng.beta(2.0, 5.0, size=n)
    exponential = rng.exponential(2.0, size=n)
    chi = rng.chisquare(4.0, size=n)

    for values in (normals, gamma, beta, exponential, chi):
        assert isinstance(values, xp.ndarray)
        assert values.shape == (n,)
        assert bool(xp.all(xp.isfinite(values)))

    np.testing.assert_allclose(asnumpy(xp.mean(normals)), 2.0, atol=0.04)
    np.testing.assert_allclose(asnumpy(xp.mean(beta)), 2 / 7, atol=0.01)
    np.testing.assert_allclose(
        asnumpy(stats.gamma.cdf(asnumpy(gamma), a=np.linspace(1.0, 4.0, n), scale=2.0)).mean(),
        0.5,
        atol=0.02,
    )

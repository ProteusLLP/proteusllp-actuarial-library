"""Regression tests for scenario-aligned FRA1 sampling."""

import numpy as np
import pytest

from pal import _fra1
from pal._maths import asnumpy, xp
from pal.copulas import FRA1Copula
from pal.qmc import RandomizedQMCGenerator


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize(("eta", "theta"), [(-1.0, -1.0), (0.2, 0.6), (0.6, 0.2)])
@pytest.mark.parametrize("dimensions", [2, 16])
def test_fra1_rqmc_joint_law(dimension, eta, theta, dimensions):
    n = 4096
    rng = RandomizedQMCGenerator(seed=714, dimensions=dimensions)
    replay = RandomizedQMCGenerator(seed=714, dimensions=dimensions)
    copula = FRA1Copula(eta, theta, dimension)
    samples = list(copula.generate(n, rng=rng))
    repeated = list(copula.generate(n, rng=replay))
    assert len(samples) == len(repeated) == dimension
    values = np.stack([asnumpy(s.values) for s in samples])
    assert values.shape == (dimension, n)
    assert np.all(np.isfinite(values))
    assert np.all((values >= 0) & (values <= 1))
    for sample, other in zip(samples, repeated):
        assert isinstance(sample.values, xp.ndarray)
        assert sample.coupled_variable_group is samples[0].coupled_variable_group
        np.testing.assert_array_equal(asnumpy(sample.values), asnumpy(other.values))
    assert set(rng._streams) == {n}
    for q in (0.1, 0.5, 0.9):
        np.testing.assert_allclose(np.mean(values <= q, axis=1), q, atol=0.04)
        log_phi = _fra1.log_inverse_generator(xp.asarray(q), eta, theta)
        expected = float(asnumpy(_fra1.generator_from_log_argument(log_phi + np.log(2), eta, theta)))
        for i in range(dimension):
            for j in range(i):
                actual = np.mean((values[i] <= q) & (values[j] <= q))
                assert actual == pytest.approx(expected, abs=0.04)

"""Exact CPU quantiles after threaded RQMC inverse-CDF evaluation."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import special

from pal._maths import xp
from pal.qmc import RandomizedQMCGenerator, _cpu_parallel_inverse


@pytest.mark.skipif(xp is not np, reason="CPU-specific quantile threading")
@pytest.mark.parametrize("n", [512, 32768, 65536])
def test_parallel_gamma_matches_original_pointwise(n):
    rng = RandomizedQMCGenerator(seed=221, dimensions=5)
    reference = RandomizedQMCGenerator(seed=221, dimensions=5)
    alpha = np.geomspace(0.05, 70.0, n)
    uniform = reference.random(n)
    actual = rng.gamma(alpha, 1.0, size=n)
    expected = special.gammaincinv(alpha, uniform)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(xp is not np, reason="CPU-specific quantile threading")
@pytest.mark.parametrize("n", [512, 32768, 65536])
def test_parallel_beta_matches_original_pointwise(n):
    rng = RandomizedQMCGenerator(seed=439, dimensions=5)
    reference = RandomizedQMCGenerator(seed=439, dimensions=5)
    a = np.geomspace(0.1, 50, n)
    b = np.geomspace(70, 0.2, n)
    actual = rng.beta(a, b, size=n)
    expected = special.betaincinv(a, b, reference.random(n))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(xp is not np, reason="CPU-specific quantile threading")
def test_parallel_beta_with_scalar_parameters_is_bitwise_identical():
    n = 131072
    rng = RandomizedQMCGenerator(seed=517, dimensions=2)
    reference = RandomizedQMCGenerator(seed=517, dimensions=2)
    result = rng.beta(2.75, 13.0, size=n)
    expected = special.betaincinv(2.75, 13.0, reference.random(n))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.skipif(xp is not np, reason="CPU-specific quantile threading")
def test_multidimensional_quantile_falls_back_without_axis_changes():
    n = 64
    u = np.linspace(0.05, 0.95, n).reshape(8, 8)
    alpha = np.linspace(0.5, 6.0, n).reshape(8, 8)
    output = _cpu_parallel_inverse(special.gammaincinv, u, alpha)
    np.testing.assert_array_equal(output, special.gammaincinv(alpha, u))

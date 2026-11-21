"""Regression coverage for distribution methods with SciPy GPU fallbacks."""

import numpy as np
import pytest
from scipy import stats

from pal import StochasticScalar
from pal._maths import asnumpy, create_random_generator, xp
from pal.distributions import GeneralizedInverseGaussian, NonCentralChiSquared


@pytest.mark.parametrize("family", ["noncentral_chi_squared", "generalized_inverse_gaussian"])
def test_probability_methods_match_scipy_and_preserve_backend_and_coupling(family: str) -> None:
    """SciPy fallbacks must preserve numerical values, device arrays and ancestry."""
    observations = StochasticScalar(xp.asarray([0.6, 1.5, 2.0, 3.0, 6.0]))
    probabilities = StochasticScalar(xp.asarray([0.02, 0.1, 0.5, 0.9, 0.98]))

    if family == "noncentral_chi_squared":
        df = StochasticScalar(xp.asarray([2.0, 3.0, 4.0, 5.0, 6.0]))
        nonc = StochasticScalar(xp.asarray([0.0, 0.5, 1.0, 2.0, 3.0]))
        parameters = (df, nonc)
        dist = NonCentralChiSquared(df, nonc)
        reference = stats.ncx2
        reference_args = (asnumpy(df.values), asnumpy(nonc.values))
    else:
        p = StochasticScalar(xp.asarray([-0.5, 0.25, 0.5, 1.0, 1.5]))
        chi = StochasticScalar(xp.asarray([1.0, 2.0, 3.0, 4.0, 5.0]))
        psi = StochasticScalar(xp.asarray([0.5, 0.75, 1.0, 1.25, 1.5]))
        parameters = (p, chi, psi)
        dist = GeneralizedInverseGaussian(p, chi, psi)
        reference = stats.geninvgauss
        reference_args = (
            asnumpy(p.values),
            np.sqrt(asnumpy(chi.values) * asnumpy(psi.values)),
        )

    result_cdf = dist.cdf(observations)
    result_ppf = dist.invcdf(probabilities)

    assert isinstance(result_cdf, StochasticScalar)
    assert isinstance(result_ppf, StochasticScalar)
    assert isinstance(result_cdf.values, xp.ndarray)
    assert isinstance(result_ppf.values, xp.ndarray)
    for result, argument in ((result_cdf, observations), (result_ppf, probabilities)):
        assert result.coupled_variable_group is argument.coupled_variable_group
        for parameter in parameters:
            assert result.coupled_variable_group is parameter.coupled_variable_group

    reference_options: dict[str, np.ndarray] = {}
    if family == "generalized_inverse_gaussian":
        reference_options["scale"] = np.sqrt(asnumpy(chi.values) / asnumpy(psi.values))
    np.testing.assert_allclose(
        asnumpy(result_cdf.values),
        reference.cdf(asnumpy(observations.values), *reference_args, **reference_options),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        asnumpy(result_ppf.values),
        reference.ppf(asnumpy(probabilities.values), *reference_args, **reference_options),
        rtol=1e-9,
        atol=1e-11,
    )


@pytest.mark.parametrize("family", ["noncentral_chi_squared", "generalized_inverse_gaussian"])
def test_monte_carlo_sampling_reproducible_and_on_backend(family: str) -> None:
    """Sampling remains reproducible even if GIG uses an external host sampler."""
    if family == "noncentral_chi_squared":
        dist = NonCentralChiSquared(5.0, 2.5)
    else:
        dist = GeneralizedInverseGaussian(-0.5, 2.0, 3.0)
    first = dist.generate(128, rng=create_random_generator(146))
    second = dist.generate(128, rng=create_random_generator(146))

    assert isinstance(first.values, xp.ndarray)
    assert first.values.shape == (128,)
    assert bool(xp.all(xp.isfinite(first.values)))
    assert bool(xp.all(first.values > 0))
    np.testing.assert_array_equal(asnumpy(first.values), asnumpy(second.values))

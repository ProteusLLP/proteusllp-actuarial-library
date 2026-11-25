"""Extreme-tail regression checks for severity probability and fitting APIs."""

import numpy as np
import pytest

from pal.distributions import Gamma, InverseBurr, InverseExponential, InverseParalogistic, InverseWeibull, LogLogistic
from pal.fitting import _frozen_distribution


@pytest.mark.parametrize(
    ("model", "x", "logsf", "logpdf"),
    [
        (InverseBurr(power=2, shape=1e30, scale=1e-15, loc=0), 1.0, -0.45867514538708176, -0.3068528194400547),
        (InverseBurr(power=1, shape=1e300, scale=1e-310, loc=0), 1.0, -23.025850929990458, -23.02585093004046),
        (InverseBurr(power=1, shape=1e-10, scale=1e-300, loc=0), 1e10, -736.8272297580946, -759.8530806880351),
        (LogLogistic(shape=2, scale=1, loc=0), 1e100, -460.51701859880916, np.log(2) - 300 * np.log(10)),
        (InverseExponential(scale=6754, loc=1), 2.0, 0.0, np.log(6754) - 6754),
    ],
)
def test_extreme_likelihood_probabilities(model, x, logsf, logpdf):
    # Constants from independent high-precision evaluation of the density and survival.
    rv = _frozen_distribution(type(model), model._params)
    assert float(rv.logsf(x)) == pytest.approx(logsf, rel=1e-12, abs=1e-12)
    assert float(rv.logpdf(x)) == pytest.approx(logpdf, rel=1e-12, abs=1e-12)
    assert float(model.logsf(x)) == pytest.approx(logsf, rel=1e-12, abs=1e-12)
    assert float(model.logpdf(x)) == pytest.approx(logpdf, rel=1e-12, abs=1e-12)
    assert 0 <= model.cdf(x) <= 1


def test_inverse_paralogistic_cdf_no_overflow():
    model = InverseParalogistic(shape=100, scale=1e-20, loc=3)
    assert model.cdf(1e100) == 1.0
    assert model.cdf(2) == 0.0
    assert np.isfinite(model.logsf(1e100))


def test_inverse_weibull_finite_log_density():
    model = InverseWeibull(shape=2, scale=100, loc=3)
    assert model.logpdf(4) == pytest.approx(np.log(2) + 2 * np.log(100) - 10000)


def test_severity_optimizer_and_parameter_count():
    x = np.linspace(1, 5, 100)
    fit = Gamma.fit(x, optimizer="L-BFGS-B", optimizer_options={"maxiter": 2000})
    baseline = Gamma.fit(x)
    assert fit.n_parameters == 2
    assert fit.log_likelihood == pytest.approx(baseline.log_likelihood, abs=1e-6)

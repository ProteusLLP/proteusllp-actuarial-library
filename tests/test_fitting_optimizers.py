"""Fit optimizer comparisons on deterministic, weighted and censored samples."""

import numpy as np
import pytest
from scipy import stats

from pal import distributions as d
from pal.fitting import _minimize_multistart


@pytest.mark.parametrize(
    ("family", "rv", "threshold"),
    [
        (d.Gamma, stats.gamma(2.5, scale=3), 2),
        (d.LogNormal, stats.lognorm(0.8, scale=10), 5),
        (d.Weibull, stats.weibull_min(1.7, scale=10), 4),
        (d.InverseBurr, stats.burr(2.5, 3, scale=10), 3),
        (d.InverseWeibull, stats.invweibull(2.5, scale=10), 3),
        (d.LogLogistic, stats.fisk(2.3, scale=10), 3),
        (d.GPD, stats.genpareto(-0.25, loc=3, scale=5), 3),
        (d.GPD, stats.genpareto(0.3, loc=3, scale=5), 3),
        (d.Beta, stats.beta(2.5, 4), 0.2),
        (d.Normal, stats.norm(4, 2), 3),
        (d.Logistic, stats.logistic(4, 2), 3),
        (d.StudentsT, stats.t(5, loc=4, scale=2), 3),
    ],
)
def test_default_optimizer_matches_simplex_on_weighted_censored_samples(family, rv, threshold):
    q = (np.arange(300) + 0.5) / 300
    uncapped = rv.isf(rv.sf(threshold) * (1 - q))
    cap = np.quantile(uncapped, 0.85)
    x = np.minimum(uncapped, cap)
    kwargs = {"threshold": threshold, "censored": uncapped >= cap, "weights": 1 + np.arange(x.size) % 3}
    default = family.fit(x, **kwargs)
    simplex = family.fit(x, optimizer="Nelder-Mead", **kwargs)
    assert default.optimizer in ("L-BFGS-B", "Nelder-Mead")
    assert default.converged
    assert default.log_likelihood >= simplex.log_likelihood - 1e-4
    assert np.isfinite(default.aic)
    assert default.n_parameters == simplex.n_parameters


def test_default_optimizer_is_lbfgsb_on_regular_gamma():
    x = stats.gamma(2.5, scale=3).ppf((np.arange(500) + 0.5) / 500)
    result = d.Gamma.fit(x)
    assert result.optimizer == "L-BFGS-B"
    assert not result.optimizer_fallback
    explicit = d.Gamma.fit(x, optimizer="L-BFGS-B")
    assert explicit.log_likelihood == pytest.approx(result.log_likelihood, abs=1e-10)


def test_analytic_estimators_are_not_replaced():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    result = d.Exponential.fit(x)
    assert result.optimizer is None
    assert result.parameters["scale"] == pytest.approx(3)


def test_multistart_retains_better_finite_solution():
    def objective(x):
        # A narrow feasible region tests the invalid-likelihood penalty path.
        if x[0] < 0 or x[0] > 1:
            return np.inf
        return (x[0] - 0.7) ** 2

    result = _minimize_multistart(objective, [np.array([0.1]), np.array([0.9])], "L-BFGS-B", None, 100)
    assert result.fun < 1e-10
    assert result.x[0] == pytest.approx(0.7, abs=1e-5)


def test_large_loss_scaling_preserves_gamma_likelihood():
    x = stats.gamma(2.5, scale=3).ppf((np.arange(500) + 0.5) / 500)
    original = d.Gamma.fit(x)
    scaled = d.Gamma.fit(x * 1e7)
    assert scaled.parameters["alpha"] == pytest.approx(original.parameters["alpha"], rel=1e-5)
    assert scaled.parameters["theta"] / 1e7 == pytest.approx(original.parameters["theta"], rel=1e-5)
    assert scaled.log_likelihood + len(x) * np.log(1e7) == pytest.approx(original.log_likelihood, abs=1e-6)

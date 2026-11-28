"""Fitting at samples where earlier estimators reported success at a wrong or nonexistent optimum."""

from __future__ import annotations

import numpy as np
import pytest

from pal._fitting_initial import _shape_coordinate, _support_shape
from pal.distributions import GPD, Exponential, Gamma, LogNormal, Pareto, Poisson


def _pareto_log_likelihood(shape: float, scale: float, x: np.ndarray, censored: np.ndarray) -> float:
    """Right-censored Pareto log-likelihood evaluated directly from its density and survival function."""
    if np.any((x < scale) & ~censored):
        return -np.inf
    density = np.log(shape) + shape * np.log(scale) - (shape + 1) * np.log(x[~censored])
    survival = np.minimum(0.0, shape * (np.log(scale) - np.log(x[censored])))
    return float(density.sum() + survival.sum())


def test_censored_pareto_scale_is_the_smallest_uncensored_observation():
    """Censoring levels below the scale have survival probability one, so they do not bound it.

    The scale was the smallest observation of any kind, which gave a
    log-likelihood of -44.9 against -41.5 at the maximum.
    """
    x = np.array([50.0, 120.0, 150.0, 200.0, 300.0, 500.0, 900.0, 2000.0])
    censored = np.array([True, False, False, False, False, False, False, True])
    fitted = Pareto.fit(x, censored=censored)
    assert fitted.parameters["scale"] == 120.0
    # Closed-form shape: uncensored count over the total exposure above the scale.
    shape = 6 / np.sum(np.log(np.maximum(x, 120.0) / 120.0))
    assert fitted.parameters["shape"] == pytest.approx(shape, rel=1e-12)
    assert fitted.log_likelihood == pytest.approx(_pareto_log_likelihood(shape, 120.0, x, censored), rel=1e-12)
    # Profile over the scale with the shape maximised for each: no scale does better.
    best = -np.inf
    for scale in np.linspace(5.0, 120.0, 2000):
        exposure = np.sum(np.log(np.maximum(x, scale) / scale))
        best = max(best, _pareto_log_likelihood(6 / exposure, scale, x, censored))
    assert fitted.log_likelihood >= best - 1e-9


def test_uncensored_pareto_fit_is_unchanged():
    x = np.array([50.0, 120.0, 150.0, 200.0, 300.0, 500.0, 900.0, 2000.0])
    fitted = Pareto.fit(x)
    assert fitted.parameters["scale"] == 50.0
    assert fitted.parameters["shape"] == pytest.approx(8 / np.sum(np.log(x / 50.0)), rel=1e-12)


def test_gpd_shape_is_limited_to_minus_one_and_warns_for_capped_losses():
    """Losses capped at a limit but not flagged as censored have no maximum-likelihood GPD.

    The likelihood is unbounded for a shape below -1, and the fit once ran
    along that ridge, returning a shape of -3.5, converged and with no warning.
    """
    losses = np.minimum(np.random.default_rng(1).exponential(1000.0, 2000), 1500.0)
    fitted = GPD.fit(losses)
    assert fitted.parameters["shape"] == pytest.approx(-1.0, abs=1e-6)
    assert any("lower limit of -1" in message for message in fitted.warnings)
    # At a shape of -1 the GPD is uniform on (0, scale), so the best scale is the largest loss.
    assert fitted.parameters["scale"] == pytest.approx(losses.max(), rel=1e-6)
    assert fitted.log_likelihood == pytest.approx(-len(losses) * np.log(losses.max()), rel=1e-9)


def test_gpd_flagged_as_censored_at_the_cap_fits_without_warning():
    losses = np.minimum(np.random.default_rng(1).exponential(1000.0, 2000), 1500.0)
    fitted = GPD.fit(losses, censored=losses >= 1500.0)
    assert fitted.warnings == ()
    # Exponential data have a GPD shape of zero.
    assert fitted.parameters["shape"] == pytest.approx(0.0, abs=0.1)
    assert fitted.parameters["scale"] == pytest.approx(1000.0, rel=0.1)


def test_gpd_shape_support_transformation_is_bounded_below():
    assert _support_shape(-5.0, scale=1.0, left=0.0, right=0.01) == -1.0
    # Shapes above -1 are unchanged by the bound.
    assert _support_shape(-0.5, scale=10.0, left=0.0, right=1.0) == pytest.approx(-0.5 / 1.05)
    with pytest.raises(ValueError, match="must be at least -1"):
        _shape_coordinate(-2.0, 10.0, 0.0, 1.0)


@pytest.mark.parametrize("family", [LogNormal, Gamma, GPD])
def test_constant_samples_are_rejected(family):
    """A constant sample has no maximum: the scale tends to zero, and sigma was returned as 1e-260."""
    with pytest.raises(ValueError, match="All observations are equal"):
        family.fit([7.0] * 5)


def test_constant_samples_remain_valid_where_the_likelihood_has_a_maximum():
    assert Exponential.fit([7.0] * 5).parameters["scale"] == pytest.approx(7.0)
    assert Poisson.fit([3, 3, 3, 3]).parameters["mean"] == pytest.approx(3.0)


@pytest.mark.parametrize("censored_fraction", [0.25, 0.75])
def test_discrete_uncertainty_rejects_censoring_indicators(censored_fraction):
    """The discrete likelihood has no censored term, so the information would silently ignore them."""
    observations = np.random.default_rng(3).poisson(3.0, 200).astype(float)
    censored = np.arange(200) < censored_fraction * 200
    fitted = Poisson.fit(observations)
    with pytest.raises(ValueError, match="does not support censored observations"):
        fitted.asymptotic_uncertainty(observations, censored=censored)
    # Indicators that are all false are consistent with the uncensored fit.
    assert fitted.asymptotic_uncertainty(observations, censored=np.zeros(200, dtype=bool)).covariance.shape == (1, 1)


def test_model_averaging_rejects_nonregular_candidates():
    """AICc's parameter penalty has no justification for an estimated support endpoint."""
    from pal.model_averaging import WeightedDistribution

    observations = np.random.default_rng(4).lognormal(2.0, 0.8, 300)
    with pytest.raises(ValueError, match="Pareto without a threshold"):
        WeightedDistribution.fit(observations, [LogNormal, Pareto])
    with pytest.raises(ValueError, match="free location"):
        WeightedDistribution.fit(observations, [LogNormal, GPD], per_family_options={GPD: {"free_location": True}})
    # With a threshold, Pareto's lower endpoint is known and averaging proceeds.
    tail = observations[observations > 10.0]
    averaged = WeightedDistribution.fit(tail, [LogNormal, Pareto], threshold=10.0)
    assert averaged.weights.sum() == pytest.approx(1.0)

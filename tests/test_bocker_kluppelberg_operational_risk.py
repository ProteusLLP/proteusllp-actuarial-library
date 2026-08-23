"""Tests for the Böcker-Klüppelberg (2005) operational-risk example."""

import pytest

from pal.config import set_random_seed
from pal.distributions import Pareto, Poisson
from pal.frequency_severity import FrequencySeverityModel


def _bocker_kluppelberg_var(
    alpha: float,
    theta: float,
    mean_frequency: float,
    confidence_level: float,
    time_horizon: float = 1.0,
) -> float:
    expected_events = mean_frequency * time_horizon
    return theta * ((expected_events / (1.0 - confidence_level)) ** (1.0 / alpha) - 1.0)


def test_shifted_pal_pareto_matches_paper_pareto_type_ii():
    alpha = 1.5
    theta = 2.0
    loss = 7.0

    pal_cdf = Pareto(shape=alpha, scale=theta).cdf(loss + theta)
    paper_cdf = 1.0 - (1.0 + loss / theta) ** (-alpha)

    assert pal_cdf == pytest.approx(paper_cdf)


def test_closed_form_var_matches_adjusted_severity_quantile():
    alpha = 1.1
    theta = 1.0
    mean_frequency = 10.0
    confidence_level = 0.999
    adjusted_probability = 1.0 - (1.0 - confidence_level) / mean_frequency

    severity_quantile = Pareto(shape=alpha, scale=theta).invcdf(adjusted_probability) - theta
    approximation = _bocker_kluppelberg_var(alpha, theta, mean_frequency, confidence_level)

    assert severity_quantile == pytest.approx(approximation)
    assert approximation == pytest.approx(4327.761281083053)


def test_simulated_pareto_poisson_high_quantile_is_close_to_single_loss_approximation():
    alpha = 1.1
    theta = 1.0
    mean_frequency = 10.0
    confidence_level = 0.998

    set_random_seed(42)
    model = FrequencySeverityModel(
        freq_dist=Poisson(mean_frequency),
        sev_dist=Pareto(shape=alpha, scale=theta),
    )
    annual_loss = (model.generate(n_sims=200_000) - theta).aggregate()

    simulated_var = annual_loss.percentile(100.0 * confidence_level)
    approximation = _bocker_kluppelberg_var(alpha, theta, mean_frequency, confidence_level)

    # The paper's result is first-order as confidence approaches one; the Monte Carlo
    # check is intentionally looser than the exact parameterisation tests above.
    assert simulated_var == pytest.approx(approximation, rel=0.20)

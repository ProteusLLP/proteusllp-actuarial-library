"""Operational-risk LDA following Böcker and Klüppelberg (2005).

Builds a Pareto-Poisson loss distribution model, simulates annual operational
losses, and compares high aggregate-loss quantiles with the paper's closed-form
single-loss approximation.
"""

import pandas as pd

from pal.config import set_random_seed
from pal.distributions import Pareto, Poisson
from pal.frequency_severity import FrequencySeverityModel

MEAN_FREQUENCY = 10.0
ALPHA = 1.1
THETA = 1.0
N_SIMS = 250_000
CONFIDENCE_LEVELS = [0.990, 0.995, 0.998, 0.999]


def bocker_kluppelberg_var(
    alpha: float,
    theta: float,
    mean_frequency: float,
    confidence_level: float,
    time_horizon: float = 1.0,
) -> float:
    """Return the Pareto-Poisson first-order operational VaR approximation."""
    expected_events = mean_frequency * time_horizon
    return theta * ((expected_events / (1.0 - confidence_level)) ** (1.0 / alpha) - 1.0)


set_random_seed(42)

model = FrequencySeverityModel(
    freq_dist=Poisson(MEAN_FREQUENCY),
    sev_dist=Pareto(shape=ALPHA, scale=THETA),
)

# PAL's Pareto is Type I. Subtracting theta gives the paper's Pareto Type II severity.
events = model.generate(n_sims=N_SIMS) - THETA
annual_loss = events.aggregate()

simulated_var = annual_loss.percentile([100 * level for level in CONFIDENCE_LEVELS])
rows = []
for confidence_level, simulated in zip(CONFIDENCE_LEVELS, simulated_var):
    approximation = bocker_kluppelberg_var(
        alpha=ALPHA,
        theta=THETA,
        mean_frequency=MEAN_FREQUENCY,
        confidence_level=confidence_level,
    )
    rows.append(
        {
            "confidence": confidence_level,
            "simulated VaR": simulated,
            "single-loss approximation": approximation,
            "relative difference": (approximation / simulated) - 1.0,
        }
    )

comparison = pd.DataFrame(rows).set_index("confidence")
print("Böcker-Klüppelberg Pareto-Poisson operational VaR")
print("=" * 58)
print(comparison)

print("\nTail-index sensitivity at 99.9%")
print("=" * 35)
shape_comparison = pd.DataFrame(
    {
        "alpha": [1.5, 1.1],
        "99.9% approximate VaR": [
            bocker_kluppelberg_var(1.5, THETA, MEAN_FREQUENCY, 0.999),
            bocker_kluppelberg_var(1.1, THETA, MEAN_FREQUENCY, 0.999),
        ],
    }
).set_index("alpha")
print(shape_comparison)

print("\nPareto alpha-root-of-time scaling at 99.9%")
print("=" * 46)
one_year = bocker_kluppelberg_var(ALPHA, THETA, MEAN_FREQUENCY, 0.999)
scaling_rows = []
for years in [1, 2, 5]:
    scaled_var = bocker_kluppelberg_var(
        ALPHA,
        THETA,
        MEAN_FREQUENCY,
        0.999,
        time_horizon=float(years),
    )
    scaling_rows.append(
        {
            "years": years,
            "closed-form VaR ratio": scaled_var / one_year,
            "alpha-root rule": years ** (1.0 / ALPHA),
        }
    )

print(pd.DataFrame(scaling_rows).set_index("years"))

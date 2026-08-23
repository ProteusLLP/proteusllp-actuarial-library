"""Property exposure rating and simulation with the MBBEFD distribution."""

from __future__ import annotations

import typing as t
from pathlib import Path

import numpy as np
import pandas as pd

from pal import Empirical, StochasticScalar, XoL, XoLTower, distributions, set_random_seed
from pal._maths import xp
from pal.frequency_severity import FreqSevSims, FrequencySeverityModel

N_SIMS = 100_000
N_SEVERITY_POINTS = 5_000
DATA_PATH = Path(__file__).parent / "data" / "property_exposures.csv"


def make_tower() -> XoLTower:
    """Create the illustrative per-risk reinsurance tower."""
    return XoLTower(
        name=["5m xs 5m", "10m xs 10m", "20m xs 20m"],
        limit=[5_000_000, 10_000_000, 20_000_000],
        excess=[5_000_000, 10_000_000, 20_000_000],
        premium=[0.0, 0.0, 0.0],
    )


def curve_increment(
    distribution: distributions.MBBEFD,
    lower: float,
    upper: float,
    maximum_loss: float,
) -> float:
    """Return the exposure-curve share between two ground-up thresholds."""
    lower_ratio = min(max(lower / maximum_loss, 0.0), 1.0)
    upper_ratio = min(max(upper / maximum_loss, 0.0), 1.0)
    lower_share = t.cast(float, distribution.exposure_curve(lower_ratio))
    upper_share = t.cast(float, distribution.exposure_curve(upper_ratio))
    return upper_share - lower_share


def policy_loss_samples(row: pd.Series, n_points: int) -> StochasticScalar:
    """Discretize one risk's conditional policy-loss severity distribution."""
    probabilities = StochasticScalar((xp.arange(n_points, dtype=float) + 0.5) / n_points)
    damage_ratio = t.cast(
        StochasticScalar,
        distributions.MBBEFD.from_c(float(row["mbbefd_c"])).invcdf(probabilities),
    )
    ground_up_loss = damage_ratio * float(row["maximum_loss"])
    return t.cast(
        StochasticScalar,
        np.minimum(
            np.maximum(ground_up_loss - float(row["policy_deductible"]), 0.0),
            float(row["policy_limit"]),
        ),
    )


def make_empirical_policy_severity(
    exposures: pd.DataFrame,
    n_points: int = N_SEVERITY_POINTS,
) -> tuple[np.ndarray, Empirical]:
    """Infer risk frequencies and combine policy severities empirically."""
    frequencies: list[float] = []
    samples: list[t.Any] = []
    weights: list[t.Any] = []

    for _, row in exposures.iterrows():
        policy_losses = policy_loss_samples(row, n_points)
        expected_annual_loss = float(row["subject_premium"]) * float(row["expected_loss_ratio"])
        frequency = expected_annual_loss / float(policy_losses.mean())

        frequencies.append(frequency)
        samples.append(policy_losses.values)
        weights.append(xp.full(n_points, frequency / n_points, dtype=float))

    return (
        np.asarray(frequencies, dtype=float),
        Empirical(samples=xp.concatenate(samples), weights=xp.concatenate(weights)),
    )


def simulate_policy_losses(
    exposures: pd.DataFrame,
    n_sims: int,
    n_points: int = N_SEVERITY_POINTS,
) -> tuple[np.ndarray, FreqSevSims]:
    """Simulate occurrence-level policy losses for the exposure schedule."""
    frequencies, severity = make_empirical_policy_severity(exposures, n_points)
    model = FrequencySeverityModel(
        distributions.Poisson(float(frequencies.sum())),
        severity,
    )
    return frequencies, model.generate(n_sims)


def expected_layer_loss(row: pd.Series, layer: XoL) -> float:
    """Calculate the exposure-rated annual expected loss for one risk and layer."""
    maximum_loss = float(row["maximum_loss"])
    policy_limit = float(row["policy_limit"])
    deductible = float(row["policy_deductible"])
    distribution = distributions.MBBEFD.from_c(float(row["mbbefd_c"]))

    policy_share = curve_increment(
        distribution,
        deductible,
        deductible + policy_limit,
        maximum_loss,
    )
    layer_lower = deductible + min(layer.excess, policy_limit)
    layer_upper = deductible + min(layer.excess + layer.limit, policy_limit)
    layer_share = curve_increment(
        distribution,
        layer_lower,
        layer_upper,
        maximum_loss,
    )

    expected_policy_loss = float(row["subject_premium"]) * float(row["expected_loss_ratio"])
    return expected_policy_loss * layer_share / policy_share


def main() -> None:
    """Run the property exposure-rating example."""
    exposures = pd.read_csv(DATA_PATH)
    set_random_seed(42)

    frequencies, policy_losses = simulate_policy_losses(exposures, N_SIMS)
    tower = make_tower()
    tower.apply(policy_losses)

    frequency_table = exposures[["risk_id"]].copy()
    frequency_table["annual_claim_frequency"] = frequencies
    print("Inferred annual claim frequencies")
    print(frequency_table.to_string(index=False, float_format=lambda x: f"{x:,.6f}"))
    print(f"\nPortfolio Poisson mean: {frequencies.sum():,.6f}\n")

    total_subject_premium = float(exposures["subject_premium"].sum())
    rows = []
    for layer in tower.layers:
        exposure_expected_loss = sum(expected_layer_loss(row, layer) for _, row in exposures.iterrows())
        simulated_expected_loss = layer.summary["mean"]
        rows.append(
            {
                "layer": layer.name,
                "exposure_expected_loss": exposure_expected_loss,
                "simulated_expected_loss": simulated_expected_loss,
                "exposure_rate": exposure_expected_loss / total_subject_premium,
                "simulated_rate": simulated_expected_loss / total_subject_premium,
            }
        )

    comparison = pd.DataFrame(rows)
    comparison["difference"] = comparison["simulated_expected_loss"] - comparison["exposure_expected_loss"]
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))


if __name__ == "__main__":
    main()

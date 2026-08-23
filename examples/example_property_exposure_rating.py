"""Property exposure rating and simulation with the MBBEFD distribution."""

from __future__ import annotations

import typing as t
from pathlib import Path

import numpy as np
import pandas as pd

from pal import Empirical, MBBEFD, StochasticScalar, XoL, XoLTower, distributions, set_random_seed
from pal._maths import xp
from pal.frequency_severity import FreqSevSims, FrequencySeverityModel

N_SIMS = 100_000
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
    distribution: MBBEFD,
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


def expected_policy_loss_given_claim(row: pd.Series) -> float:
    """Return the expected policy loss conditional on a ground-up claim."""
    maximum_loss = float(row["maximum_loss"])
    deductible = float(row["policy_deductible"])
    policy_limit = float(row["policy_limit"])
    distribution = MBBEFD.from_c(float(row["mbbefd_c"]))
    policy_share = curve_increment(
        distribution,
        deductible,
        deductible + policy_limit,
        maximum_loss,
    )
    return maximum_loss * float(distribution.mean()) * policy_share


def claim_frequencies(exposures: pd.DataFrame) -> np.ndarray:
    """Infer annual ground-up claim frequencies from premium and loss ratio."""
    frequencies = []
    for _, row in exposures.iterrows():
        expected_annual_loss = float(row["subject_premium"]) * float(row["expected_loss_ratio"])
        frequencies.append(expected_annual_loss / expected_policy_loss_given_claim(row))
    return np.asarray(frequencies, dtype=float)


def simulate_claim_rows(
    exposures: pd.DataFrame,
    frequencies: np.ndarray,
    n_sims: int,
) -> FreqSevSims:
    """Simulate which exposure row gives rise to each ground-up claim."""
    row_distribution = Empirical(
        samples=xp.arange(len(exposures), dtype=float),
        weights=xp.asarray(frequencies),
    )
    model = FrequencySeverityModel(
        distributions.Poisson(float(frequencies.sum())),
        row_distribution,
    )
    return model.generate(n_sims)


def simulate_policy_losses(
    exposures: pd.DataFrame,
    claim_rows: FreqSevSims,
) -> FreqSevSims:
    """Draw continuous MBBEFD severities for the simulated exposure rows."""
    row_index = claim_rows.values.astype(np.int64)
    if len(row_index) == 0:
        result = FreqSevSims(claim_rows.sim_index, xp.asarray([], dtype=float), claim_rows.n_sims)
        result.coupled_variable_group.merge(claim_rows.coupled_variable_group)
        return result

    maximum_loss = xp.asarray(exposures["maximum_loss"].to_numpy(dtype=float))[row_index]
    policy_limit = xp.asarray(exposures["policy_limit"].to_numpy(dtype=float))[row_index]
    deductible = xp.asarray(exposures["policy_deductible"].to_numpy(dtype=float))[row_index]
    c = xp.asarray(exposures["mbbefd_c"].to_numpy(dtype=float))[row_index]

    event_c = StochasticScalar(c)
    event_c.coupled_variable_group.merge(claim_rows.coupled_variable_group)
    damage_ratio = MBBEFD.from_c(event_c).generate(len(row_index))
    ground_up_loss = damage_ratio * maximum_loss
    policy_loss = t.cast(
        StochasticScalar,
        np.minimum(np.maximum(ground_up_loss - deductible, 0.0), policy_limit),
    )

    result = FreqSevSims(claim_rows.sim_index, policy_loss.values, claim_rows.n_sims)
    result.coupled_variable_group.merge(claim_rows.coupled_variable_group)
    result.coupled_variable_group.merge(policy_loss.coupled_variable_group)
    return result


def expected_layer_loss(row: pd.Series, layer: XoL) -> float:
    """Calculate the exposure-rated annual expected loss for one risk and layer."""
    maximum_loss = float(row["maximum_loss"])
    policy_limit = float(row["policy_limit"])
    deductible = float(row["policy_deductible"])
    distribution = MBBEFD.from_c(float(row["mbbefd_c"]))

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
    frequencies = claim_frequencies(exposures)
    set_random_seed(42)

    claim_rows = simulate_claim_rows(exposures, frequencies, N_SIMS)
    policy_losses = simulate_policy_losses(exposures, claim_rows)
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

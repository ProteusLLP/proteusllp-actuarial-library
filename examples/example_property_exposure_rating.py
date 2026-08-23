"""Property exposure rating and simulation with the MBBEFD distribution."""

from pathlib import Path

import numpy as np
import pandas as pd

from pal import MBBEFD, Empirical, StochasticScalar, XoLTower, distributions, set_random_seed
from pal.frequency_severity import FreqSevSims, FrequencySeverityModel

N_SIMS = 100_000
DATA_PATH = Path(__file__).parent / "data" / "property_exposures.csv"


def make_tower():
    return XoLTower(
        name=["5m xs 5m", "10m xs 10m", "20m xs 20m"],
        limit=[5_000_000, 10_000_000, 20_000_000],
        excess=[5_000_000, 10_000_000, 20_000_000],
        premium=[0.0, 0.0, 0.0],
    )


def curve_increment(distribution, lower, upper, maximum_loss):
    lower = min(max(lower / maximum_loss, 0.0), 1.0)
    upper = min(max(upper / maximum_loss, 0.0), 1.0)
    return distribution.exposure_curve(upper) - distribution.exposure_curve(lower)


def expected_policy_loss_given_claim(row):
    maximum_loss = row["maximum_loss"]
    deductible = row["policy_deductible"]
    policy_limit = row["policy_limit"]
    distribution = MBBEFD.from_c(row["mbbefd_c"])
    policy_share = curve_increment(distribution, deductible, deductible + policy_limit, maximum_loss)
    return maximum_loss * distribution.mean() * policy_share


def claim_frequencies(exposures):
    frequencies = []
    for _, row in exposures.iterrows():
        expected_annual_loss = row["subject_premium"] * row["expected_loss_ratio"]
        frequencies.append(expected_annual_loss / expected_policy_loss_given_claim(row))
    return np.asarray(frequencies)


def simulate_claim_rows(exposures, frequencies, n_sims):
    row_distribution = Empirical(
        samples=np.arange(len(exposures)),
        weights=frequencies,
    )
    return FrequencySeverityModel(
        distributions.Poisson(frequencies.sum()),
        row_distribution,
    ).generate(n_sims)


def simulate_policy_losses(exposures, claim_rows):
    row_index = StochasticScalar(claim_rows.values)
    if len(row_index) == 0:
        return claim_rows

    maximum_loss = StochasticScalar(exposures["maximum_loss"].values)[row_index]
    policy_limit = StochasticScalar(exposures["policy_limit"].values)[row_index]
    deductible = StochasticScalar(exposures["policy_deductible"].values)[row_index]
    c = StochasticScalar(exposures["mbbefd_c"].values)[row_index]

    damage_ratio = MBBEFD.from_c(c).generate(len(row_index))
    policy_loss = np.minimum(
        np.maximum(damage_ratio * maximum_loss - deductible, 0.0),
        policy_limit,
    )
    return FreqSevSims(claim_rows.sim_index, policy_loss.values, claim_rows.n_sims)


def expected_layer_loss(row, layer):
    maximum_loss = row["maximum_loss"]
    policy_limit = row["policy_limit"]
    deductible = row["policy_deductible"]
    distribution = MBBEFD.from_c(row["mbbefd_c"])

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

    expected_policy_loss = row["subject_premium"] * row["expected_loss_ratio"]
    return expected_policy_loss * layer_share / policy_share


def main():
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

    total_subject_premium = exposures["subject_premium"].sum()
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

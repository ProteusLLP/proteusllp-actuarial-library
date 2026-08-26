"""Property exposure rating and simulation with the MBBEFD distribution."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import pal.maths as pnp
from pal import contracts, distributions, frequency_severity, set_random_seed, stochastic_scalar, variables

N_SIMS = 100_000
DATA_PATH = Path(__file__).parent / "data" / "property_exposures.csv"


def main() -> None:
    exposure_df = pd.read_csv(DATA_PATH)
    exposure = variables.ProteusVariable(
        dim_name="field",
        values={column: stochastic_scalar.StochasticScalar(exposure_df[column]) for column in exposure_df.columns},
    )

    maximum_loss = exposure["maximum_loss"]
    policy_limit = exposure["policy_limit"]
    deductible = exposure["policy_deductible"]
    mbbefd = distributions.MBBEFD.from_c(exposure["mbbefd_c"])

    lower = pnp.minimum(1.0, deductible / maximum_loss)
    upper = pnp.minimum(1.0, (deductible + policy_limit) / maximum_loss)
    policy_share = mbbefd.exposure_curve(upper) - mbbefd.exposure_curve(lower)

    expected_policy_severity = maximum_loss * mbbefd.mean() * policy_share
    expected_policy_loss = exposure["subject_premium"] * exposure["expected_loss_ratio"]
    frequencies = expected_policy_loss / expected_policy_severity
    total_subject_premium = exposure["subject_premium"].sum()

    tower = contracts.XoLTower(
        name=["5m xs 5m", "10m xs 10m", "20m xs 20m"],
        limit=[5_000_000, 10_000_000, 20_000_000],
        excess=[5_000_000, 10_000_000, 20_000_000],
        premium=[0.0, 0.0, 0.0],
    )

    analytical_rates: list[float] = []
    layer_names: list[str] = []
    for layer in tower.layers:
        layer_lower = pnp.minimum(
            1.0,
            (deductible + pnp.minimum(layer.excess, policy_limit)) / maximum_loss,
        )
        layer_upper = pnp.minimum(
            1.0,
            (deductible + pnp.minimum(layer.excess + layer.limit, policy_limit)) / maximum_loss,
        )
        layer_share = mbbefd.exposure_curve(layer_upper) - mbbefd.exposure_curve(layer_lower)
        analytical_expected_loss = (expected_policy_loss * layer_share / policy_share).sum()
        layer_names.append(layer.name)
        analytical_rates.append(analytical_expected_loss / total_subject_premium)

    set_random_seed(42)
    row_distribution = distributions.Empirical(  # pyright: ignore[reportAttributeAccessIssue]
        samples=np.arange(len(exposure_df)),
        weights=frequencies,
    )
    claim_rows = frequency_severity.FrequencySeverityModel(
        distributions.Poisson(frequencies.sum()),
        row_distribution,
    ).generate(N_SIMS)

    row_index = stochastic_scalar.StochasticScalar(claim_rows.values)
    selected_maximum_loss = exposure["maximum_loss"][row_index]
    selected_policy_limit = exposure["policy_limit"][row_index]
    selected_deductible = exposure["policy_deductible"][row_index]
    selected_c = exposure["mbbefd_c"][row_index]

    damage_ratio = distributions.MBBEFD.from_c(selected_c).generate(len(row_index))
    policy_loss = pnp.minimum(
        selected_policy_limit,
        pnp.maximum(damage_ratio * selected_maximum_loss - selected_deductible, 0.0),
    )
    policy_losses = frequency_severity.FreqSevSims(
        claim_rows.sim_index,
        policy_loss.values,
        claim_rows.n_sims,
    )

    tower.apply(policy_losses)

    target_expected_loss = expected_policy_loss.sum()
    simulated_expected_loss = policy_losses.aggregate().mean()
    calibration = pd.DataFrame(
        {
            "Target": [
                frequencies.sum(),
                target_expected_loss,
                target_expected_loss / total_subject_premium,
            ],
            "Simulation": [
                claim_rows.count().mean(),
                simulated_expected_loss,
                simulated_expected_loss / total_subject_premium,
            ],
        },
        index=[
            "Annual ground-up claim count",
            "Annual policy loss",
            "Portfolio loss ratio",
        ],
    )
    print("Portfolio calibration check")
    print(calibration)
    print()

    severity = stochastic_scalar.StochasticScalar(policy_losses.values)
    paid_severity = severity[severity > 0]
    severity_figure = paid_severity.histogram_plot(title="Portfolio Paid-Claim Severity")

    aggregate_tower = contracts.XoLTower(
        name=layer_names,
        limit=[5_000_000, 10_000_000, 20_000_000],
        excess=[5_000_000, 10_000_000, 20_000_000],
        premium=[0.0, 0.0, 0.0],
        aggregate_deductible=[1_000_000, 1_000_000, 1_000_000],
    )
    aggregate_tower_result = aggregate_tower.apply(policy_losses)

    occurrence_only_rates = [layer.summary["mean"] / total_subject_premium for layer in tower.layers]
    aggregate_rates = [layer.summary["mean"] / total_subject_premium for layer in aggregate_tower.layers]
    comparison = pd.DataFrame(
        {
            "Analytical exposure rate": analytical_rates,
            "Simulated occurrence-only burn": occurrence_only_rates,
            "Simulated with £1m aggregate deductible": aggregate_rates,
        },
        index=layer_names,
    )
    print("Layer burn rates")
    print(comparison)
    print()

    comparison_figure = go.Figure(
        [
            go.Bar(
                x=layer_names,
                y=analytical_rates,
                name="Analytical exposure rate",
            ),
            go.Bar(
                x=layer_names,
                y=occurrence_only_rates,
                name="Simulated occurrence-only burn",
            ),
            go.Bar(
                x=layer_names,
                y=aggregate_rates,
                name="With £1m aggregate deductible",
            ),
        ]
    )
    comparison_figure.update_layout(
        title="Impact of Aggregate Terms on Layer Burn Rate",
        xaxis_title="Layer",
        yaxis_title="Rate on subject premium",
        yaxis_tickformat=".1%",
        barmode="group",
    )

    annual_gross_loss = policy_losses.aggregate()
    annual_reinsurance_recovery = aggregate_tower_result.recoveries.aggregate()
    annual_net_loss = annual_gross_loss - annual_reinsurance_recovery
    print(f"Mean annual net property loss: {annual_net_loss.mean():,.0f}")

    if os.getenv("PAL_SUPPRESS_PLOTS", "").lower() != "true":
        severity_figure.show()
        comparison_figure.show()


if __name__ == "__main__":
    main()

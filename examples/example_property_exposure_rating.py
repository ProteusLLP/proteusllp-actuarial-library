"""Property exposure rating and simulation with the MBBEFD distribution."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pal import (
    MBBEFD,
    Empirical,
    StochasticScalar,
    XoLTower,
    distributions,
    set_random_seed,
)
from pal.frequency_severity import FreqSevSims, FrequencySeverityModel

N_SIMS = 100_000
DATA_PATH = Path(__file__).parent / "data" / "property_exposures.csv"

PROTEUS_NAVY = "#001a64"
PROTEUS_BLUE = "#1d4ed8"
PROTEUS_MID_BLUE = "#5b8def"
PROTEUS_LIGHT_BLUE = "#8fb8ff"


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
    policy_share = curve_increment(
        distribution,
        deductible,
        deductible + policy_limit,
        maximum_loss,
    )
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

    maximum_loss = StochasticScalar(exposures["maximum_loss"])[row_index]
    policy_limit = StochasticScalar(exposures["policy_limit"])[row_index]
    deductible = StochasticScalar(exposures["policy_deductible"])[row_index]
    c = StochasticScalar(exposures["mbbefd_c"])[row_index]

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


def make_calibration_table(exposures, frequencies, claim_rows, policy_losses):
    total_subject_premium = float(exposures["subject_premium"].sum())
    target_expected_loss = float((exposures["subject_premium"] * exposures["expected_loss_ratio"]).sum())
    simulated_expected_loss = float(policy_losses.aggregate().mean())

    return pd.DataFrame(
        [
            {
                "metric": "Total subject premium",
                "target": total_subject_premium,
                "simulated": total_subject_premium,
            },
            {
                "metric": "Annual ground-up claim count",
                "target": float(frequencies.sum()),
                "simulated": float(claim_rows.count().mean()),
            },
            {
                "metric": "Annual policy loss",
                "target": target_expected_loss,
                "simulated": simulated_expected_loss,
            },
            {
                "metric": "Portfolio loss ratio",
                "target": target_expected_loss / total_subject_premium,
                "simulated": simulated_expected_loss / total_subject_premium,
            },
        ]
    )


def make_layer_comparison(exposures, tower, total_subject_premium):
    rows = []
    for layer in tower.layers:
        exposure_expected_loss = sum(expected_layer_loss(row, layer) for _, row in exposures.iterrows())
        simulated_expected_loss = layer.summary["mean"]
        exposure_rate = exposure_expected_loss / total_subject_premium
        simulated_rate = simulated_expected_loss / total_subject_premium
        rows.append(
            {
                "layer": layer.name,
                "exposure_expected_loss": exposure_expected_loss,
                "simulated_expected_loss": simulated_expected_loss,
                "exposure_rate": exposure_rate,
                "simulated_rate": simulated_rate,
                "rate_difference": simulated_rate - exposure_rate,
            }
        )
    return pd.DataFrame(rows)


def make_layer_burn_rates(tower, policy_losses, total_subject_premium):
    return {
        layer.name: layer.apply(policy_losses).recoveries.aggregate() / total_subject_premium
        for layer in tower.layers
    }


def _style_figure(fig):
    fig.update_layout(
        template="plotly_white",
        colorway=[
            PROTEUS_NAVY,
            PROTEUS_BLUE,
            PROTEUS_MID_BLUE,
            PROTEUS_LIGHT_BLUE,
        ],
        margin={"l": 60, "r": 30, "t": 80, "b": 80},
        legend_title_text="",
    )
    fig.add_annotation(
        text="PROTEUS",
        x=1,
        y=-0.18,
        xref="paper",
        yref="paper",
        showarrow=False,
        xanchor="right",
        font={"color": PROTEUS_NAVY, "size": 11},
    )
    return fig


def make_distribution_figure(values, title, xaxis_title, xaxis_tickformat=None):
    percentiles = np.linspace(0.0, 99.9, 300)
    quantiles = values.percentile(percentiles.tolist())
    exceedance_probabilities = 1 - percentiles / 100
    histogram_limit = values.percentile(99.5)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Histogram (to 99.5th percentile)", "Exceedance probability"),
        horizontal_spacing=0.12,
    )
    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=60,
            name="Simulation",
            marker_color=PROTEUS_BLUE,
            hovertemplate=(f"{xaxis_title}: %{{x:,.4g}}<br>Count: %{{y:,}}<extra></extra>"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=quantiles,
            y=exceedance_probabilities,
            mode="lines",
            name="Exceedance",
            line={"color": PROTEUS_NAVY, "width": 2.5},
            hovertemplate=(f"{xaxis_title}: %{{x:,.4g}}<br>Exceedance: %{{y:.2%}}<extra></extra>"),
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(
        title_text=xaxis_title,
        tickformat=xaxis_tickformat,
        range=[0, histogram_limit],
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text=xaxis_title,
        tickformat=xaxis_tickformat,
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="Simulation count", row=1, col=1)
    fig.update_yaxes(
        title_text="Probability of exceedance",
        tickformat=".0%",
        range=[0, 1],
        row=1,
        col=2,
    )
    fig.update_layout(title=title, showlegend=False)
    return _style_figure(fig)


def make_claim_severity_figure(policy_losses):
    severity = StochasticScalar(policy_losses.values)
    paid_severity = severity[severity > 0]
    return make_distribution_figure(
        paid_severity,
        "Portfolio Paid-Claim Severity",
        "Policy loss",
        ",.0f",
    )


def make_gross_burn_rate_figure(policy_losses, total_subject_premium):
    gross_burn_rate = policy_losses.aggregate() / total_subject_premium
    return make_distribution_figure(
        gross_burn_rate,
        "Annual Gross Portfolio Burn Rate",
        "Annual policy loss / subject premium",
        ".0%",
    )


def make_layer_burn_rate_figure(layer_burn_rates):
    percentiles = np.linspace(0.0, 99.9, 300)
    exceedance_probabilities = 1 - percentiles / 100

    fig = go.Figure()
    for layer_name, burn_rate in layer_burn_rates.items():
        fig.add_trace(
            go.Scatter(
                x=burn_rate.percentile(percentiles.tolist()),
                y=exceedance_probabilities,
                mode="lines",
                name=layer_name,
                hovertemplate=(
                    "Burn rate: %{x:.2%}<br>Probability of exceedance: %{y:.2%}<extra>%{fullData.name}</extra>"
                ),
            )
        )

    fig.update_layout(
        title="Annual Layer Burn-Rate Distributions",
        xaxis_title="Annual ceded loss / total subject premium",
        yaxis_title="Probability of exceedance",
    )
    fig.update_xaxes(tickformat=".1%")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return _style_figure(fig)


def make_layer_rate_comparison_figure(comparison):
    fig = go.Figure(
        [
            go.Bar(
                x=comparison["layer"],
                y=comparison["exposure_rate"],
                name="Analytical exposure rate",
                marker_color=PROTEUS_NAVY,
                hovertemplate="%{x}<br>Analytical: %{y:.3%}<extra></extra>",
            ),
            go.Bar(
                x=comparison["layer"],
                y=comparison["simulated_rate"],
                name="Simulated mean burn rate",
                marker_color=PROTEUS_BLUE,
                hovertemplate="%{x}<br>Simulated: %{y:.3%}<extra></extra>",
            ),
        ]
    )
    fig.update_layout(
        title="Analytical Exposure Rate vs Simulated Mean Burn Rate",
        xaxis_title="Layer",
        yaxis_title="Rate on subject premium",
        barmode="group",
    )
    fig.update_yaxes(tickformat=".1%")
    return _style_figure(fig)


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
    print(
        frequency_table.to_string(
            index=False,
            float_format=lambda x: f"{x:,.6f}",
        )
    )
    print(f"\nPortfolio Poisson mean: {frequencies.sum():,.6f}\n")

    calibration = make_calibration_table(
        exposures,
        frequencies,
        claim_rows,
        policy_losses,
    )
    print("Portfolio calibration check")
    print(calibration.to_string(index=False, float_format=lambda x: f"{x:,.6f}"))
    print()

    total_subject_premium = float(exposures["subject_premium"].sum())
    comparison = make_layer_comparison(
        exposures,
        tower,
        total_subject_premium,
    )
    print("Analytical exposure rates versus simulated mean burn rates")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:,.6f}"))

    layer_burn_rates = make_layer_burn_rates(
        tower,
        policy_losses,
        total_subject_premium,
    )
    figures = [
        make_claim_severity_figure(policy_losses),
        make_gross_burn_rate_figure(policy_losses, total_subject_premium),
        make_layer_burn_rate_figure(layer_burn_rates),
        make_layer_rate_comparison_figure(comparison),
    ]

    if os.getenv("PAL_SUPPRESS_PLOTS", "").lower() != "true":
        for figure in figures:
            figure.show()


if __name__ == "__main__":
    main()

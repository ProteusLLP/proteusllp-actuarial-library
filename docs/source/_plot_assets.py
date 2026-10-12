"""Generate static Plotly SVGs embedded in the HTML documentation."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pal import ProteusVariable, XoLTower, config, copulas, distributions, set_random_seed
from pal.frequency_severity import FrequencySeverityModel
from pal.risk_measures import (
    proportional_hazards_transform,
    standard_deviation_principle,
    tvar,
    wang_transform,
)

OUTPUT_DIR = Path(__file__).parent / "_static" / "generated"


def _write_svg(fig: go.Figure, filename: str) -> None:
    """Write a Plotly figure to the generated documentation directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_image(OUTPUT_DIR / filename, format="svg")


def _generate_getting_started() -> None:
    config.n_sims = 10_000
    set_random_seed(42)
    model = FrequencySeverityModel(
        freq_dist=distributions.Poisson(mean=100),
        sev_dist=distributions.LogNormal(mu=10, sigma=1.5),
    )
    aggregate_loss = model.generate().aggregate()
    _write_svg(
        aggregate_loss.cdf_plot("Aggregate Loss"),
        "getting_started_aggregate_cdf.svg",
    )


def _generate_distributions_guide() -> None:
    config.n_sims = 10_000
    set_random_seed(42)
    loss = distributions.LogNormal(mu=14, sigma=0.5).generate()
    _write_svg(loss.cdf_plot("Loss Distribution"), "distributions_guide_cdf.svg")
    _write_svg(
        loss.histogram_plot("Loss Distribution"),
        "distributions_guide_histogram.svg",
    )


def _copula_pair(family: str):
    set_random_seed(42)
    x = distributions.LogNormal(mu=10, sigma=1.0).generate()
    y = distributions.LogNormal(mu=10, sigma=1.0).generate()
    if family == "gaussian":
        copulas.GaussianCopula([[1.0, 0.8], [0.8, 1.0]]).apply([x, y])
    elif family == "gumbel":
        copulas.GumbelCopula(theta=3.0).apply([x, y])
    elif family == "clayton":
        copulas.ClaytonCopula(theta=4.0).apply([x, y])
    elif family == "student_t":
        copulas.StudentsTCopula([[1.0, 0.7], [0.7, 1.0]], dof=3).apply([x, y])
    return x, y


def _generate_copula_plots() -> None:
    config.n_sims = 10_000
    families = [
        ("Independent", "independent"),
        ("Gaussian (ρ=0.8)", "gaussian"),
        ("Gumbel (θ=3.0)", "gumbel"),
        ("Clayton (θ=4.0)", "clayton"),
        ("Student's T (ρ=0.7, ν=3)", "student_t"),
    ]
    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[title for title, _ in families],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )
    for (title, family), (row, col) in zip(families, positions):
        x, y = _copula_pair(family)
        fig.add_trace(
            go.Scattergl(
                x=x.ranks,
                y=y.ranks,
                mode="markers",
                marker={"size": 2, "opacity": 0.3},
                name=title,
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="X Rank", row=row, col=col)
        fig.update_yaxes(title_text="Y Rank", row=row, col=col)
    fig.update_layout(
        title_text="Copula Dependency Structures (Rank Space)",
        height=700,
        width=1000,
        showlegend=False,
    )
    _write_svg(fig, "copula_scatter_plots.svg")

    x, y = _copula_pair("student_t")
    dependency = ProteusVariable("variable", {"X": x, "Y": y})
    _write_svg(
        dependency.rank_scatter_plot(title="Dependency in rank space"),
        "copula_rank_scatter.svg",
    )
    _write_svg(
        dependency.value_scatter_plot(title="Dependency in value space"),
        "copula_value_scatter.svg",
    )


def _generate_risk_measure_weights() -> None:
    config.n_sims = 20_000
    set_random_seed(42)
    portfolio = ProteusVariable(
        dim_name="lob",
        values={
            "property": distributions.LogNormal(mu=14, sigma=0.8).generate(),
            "casualty": distributions.LogNormal(mu=13, sigma=0.5).generate(),
            "marine": distributions.LogNormal(mu=12, sigma=0.6).generate(),
        },
    )
    copulas.GalambosCopula(2).apply(portfolio)
    total = portfolio.sum()
    measures = [
        ("TVaR 99%", tvar(total, percentile=99)),
        ("PH (α=0.5)", proportional_hazards_transform(total, alpha=0.5)),
        ("Wang (α=1)", wang_transform(total, alpha=1.0)),
        ("Std dev (k=2)", standard_deviation_principle(total, k=2.0)),
    ]
    weights = ProteusVariable(
        dim_name="measure",
        values={name: measure.weights for name, measure in measures},
    )
    fig = weights.cdf_plot()
    for trace in fig.data:
        trace.x, trace.y = trace.y, trace.x  # type: ignore
    fig.update_layout(
        title="Risk Measure Weights by Quantile",
        xaxis_title="Quantile of Total Loss",
        yaxis_title="Weight",
    )
    fig.update_layout()
    _write_svg(fig, "risk_measure_weights.svg")


def _generate_xol_pricing_curve() -> None:
    config.n_sims = 20_000
    set_random_seed(42)
    losses = FrequencySeverityModel(
        freq_dist=distributions.Poisson(mean=2),
        sev_dist=distributions.LogNormal(mu=12, sigma=1.5),
    ).generate()
    tower = XoLTower(
        name=["2m xs 1m", "2m xs 3m", "2m xs 5m", "2m xs 7m", "2m xs 9m"],
        limit=[2_000_000] * 5,
        excess=[1_000_000, 3_000_000, 5_000_000, 7_000_000, 9_000_000],
        premium=[0] * 5,
    )

    excesses = [layer.excess / 1e6 for layer in tower.layers]
    loss_on_line: list[float] = []
    rate_on_line: list[float] = []
    loadings: list[float] = []
    for layer in tower.layers:
        recoveries = layer.apply(losses).recoveries.aggregate()
        measure = proportional_hazards_transform(recoveries, alpha=0.5)
        expected_loss = recoveries.mean()
        loss_on_line.append(expected_loss / layer.limit * 100)
        rate_on_line.append(measure.value / layer.limit * 100)
        loadings.append((measure.value / expected_loss - 1) * 100)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=excesses, y=loss_on_line, name="Loss on Line %"))
    fig.add_trace(go.Bar(x=excesses, y=rate_on_line, name="Rate on Line %"))
    fig.add_trace(
        go.Scatter(
            x=excesses,
            y=loadings,
            name="Loading %",
            mode="lines+markers",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="XoL Pricing by Attachment Point (PH α=0.5)",
        xaxis_title="Attachment Point (£m)",
        barmode="group",
    )
    fig.update_yaxes(title_text="% of Limit", secondary_y=False)
    fig.update_yaxes(title_text="Loading %", secondary_y=True)
    _write_svg(fig, "xol_pricing_curve.svg")


def generate_all_plot_assets() -> None:
    """Generate every static SVG used by the HTML tutorials."""
    _generate_getting_started()
    print("Generated getting started plots.")
    _generate_distributions_guide()
    print("Generated distributions guide plots.")
    _generate_copula_plots()
    print("Generated copula plots.")
    _generate_risk_measure_weights()
    print("Generated risk measure weights plots.")
    _generate_xol_pricing_curve()
    print("Generated XoL pricing curve plots.")


if __name__ == "__main__":
    generate_all_plot_assets()

"""Generate the figures embedded in the risk-measures tutorial.

Run from the repository root with the project dependencies installed::

    python docs/tutorials/generate_risk_measure_charts.py

The script uses Plotly for all figures and Kaleido for static PNG export
(``pip install kaleido``).

The random seed and simulation count intentionally match the tutorial so
that the figures can be regenerated when the examples change.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pal import (
    PROTEUS_BLUE,
    PROTEUS_NAVY,
    PROTEUS_ORANGE,
    PROTEUS_SKY,
    ProteusVariable,
    XoLTower,
    add_proteus_branding,
    config,
    copulas,
    distributions,
    set_random_seed,
)
from pal.frequency_severity import FrequencySeverityModel
from pal.risk_measures import (
    percentile_layer,
    proportional_hazards_transform,
    standard_deviation_principle,
    tvar,
    wang_transform,
)


OUT = Path(__file__).parent
# Proteus documentation palette (also defined in docs/source/_static/css/proteus.css).
COLORS = {"property": PROTEUS_NAVY, "casualty": PROTEUS_BLUE, "marine": PROTEUS_SKY}


def portfolio_results():
    config.n_sims = 100_000
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
    measures = {
        "TVaR 99%": tvar(total, percentile=99),
        "PH (α=0.5)": proportional_hazards_transform(total, alpha=0.5),
        "Wang (α=1)": wang_transform(total, alpha=1),
        "Std dev (k=2)": standard_deviation_principle(total, k=2),
        "CAPL (VaR 99.5%)": percentile_layer(total, float(np.percentile(total.values, 99.5))),
    }
    return portfolio, total, measures


def plot_allocations(portfolio, measures):
    names = list(measures)
    values = np.array(
        [[float(measures[name].allocate(portfolio)[lob]) / measures[name].value for lob in COLORS] for name in names]
    )
    fig = go.Figure()
    bottom = np.zeros(len(names))
    for i, lob in enumerate(COLORS):
        fig.add_bar(x=names, y=values[:, i] * 100, name=lob.title(), marker_color=COLORS[lob])
        bottom += values[:, i]
    fig.update_layout(
        template="proteus",
        title="Capital Allocation by Risk Measure",
        barmode="stack",
        yaxis={"title": "Allocated capital (%)", "range": [0, 100]},
        xaxis={"tickangle": 25},
        legend={"orientation": "h", "y": 1.12, "x": 0.5, "xanchor": "center"},
        width=900,
        height=480,
        margin={"t": 100, "b": 100},
    )
    add_proteus_branding(fig)
    fig.write_image(OUT / "capital_allocations.png", scale=2)


def plot_weights(total, measures):
    order = np.argsort(total.values)
    percentiles = np.arange(len(order)) / len(order) * 100
    fig = go.Figure()
    for name in ("TVaR 99%", "PH (α=0.5)", "Wang (α=1)", "Std dev (k=2)"):
        weights = measures[name].weights.values[order]
        # Plotly has no symmetric-log axis. This signed transform keeps the
        # negative standard-deviation weights and the large tail weights visible.
        weights = np.sign(weights) * np.log10(1 + np.abs(weights))
        fig.add_scatter(
            x=percentiles,
            y=weights,
            mode="lines",
            name=name,
        )
    fig.update_layout(
        template="proteus",
        title="Risk Measure Weights by Percentile",
        xaxis={"title": "Percentile of total loss"},
        yaxis={"title": "Signed log₁₀(1 + |simulation weight|)"},
        width=900,
        height=480,
    )
    add_proteus_branding(fig)
    fig.write_image(OUT / "risk_measure_weights.png", scale=2)


def plot_price_curve():
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
    excesses, els, prices, loadings = [], [], [], []
    for layer in tower.layers:
        rec = layer.apply(losses).recoveries.aggregate()
        rm = proportional_hazards_transform(rec, alpha=0.5)
        excesses.append(layer.excess / 1e6)
        els.append(rec.mean() / layer.limit * 100)
        prices.append(rm.value / layer.limit * 100)
        loadings.append((rm.value / rec.mean() - 1) * 100)
    labels = [f"£{e:.0f}m" for e in excesses]
    order = np.argsort(els)
    fig = go.Figure()
    fig.add_scatter(
        x=np.array(els)[order],
        y=np.array(prices)[order],
        text=np.array(labels)[order],
        customdata=np.array(loadings)[order],
        name="XoL layers",
        mode="lines+markers+text",
        textposition="top center",
        line={"color": PROTEUS_NAVY, "width": 2},
        marker={
            "size": 11,
            "color": np.array(loadings)[order],
            "colorscale": [[0, PROTEUS_SKY], [1, PROTEUS_NAVY]],
            "colorbar": {"title": "Loading (%)"},
        },
        hovertemplate="Attachment: %{text}<br>Loss on line: %{x:.2f}%<br>Rate on line: %{y:.2f}%<br>Loading: %{customdata:.1f}%<extra></extra>",
    )
    fig.update_layout(
        template="proteus",
        title="XoL Rate on Line versus Loss on Line (PH α=0.5)",
        xaxis={"title": "Loss on line (%)"},
        yaxis={"title": "Rate on line (%)"},
        width=900,
        height=480,
    )
    add_proteus_branding(fig)
    fig.write_image(OUT / "xol_price_curve.png", scale=2)


if __name__ == "__main__":
    portfolio, total, measures = portfolio_results()
    plot_allocations(portfolio, measures)
    plot_weights(total, measures)
    plot_price_curve()

"""Generate the figures embedded in the risk-measures tutorial.

Run from the repository root with the project dependencies installed::

    python docs/tutorials/generate_risk_measure_charts.py

The random seed and simulation count intentionally match the tutorial so
that the figures can be regenerated when the examples change.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pal import ProteusVariable, XoLTower, config, copulas, distributions, set_random_seed
from pal.frequency_severity import FrequencySeverityModel
from pal.risk_measures import (
    percentile_layer,
    proportional_hazards_transform,
    standard_deviation_principle,
    tvar,
    wang_transform,
)


OUT = Path(__file__).parent
COLORS = {"property": "#4c78a8", "casualty": "#f58518", "marine": "#54a24b"}


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
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    bottom = np.zeros(len(names))
    for i, lob in enumerate(COLORS):
        ax.bar(names, values[:, i] * 100, bottom=bottom * 100, label=lob.title(), color=COLORS[lob])
        bottom += values[:, i]
    ax.set_ylabel("Allocated capital (%)")
    ax.set_title("Capital Allocation by Risk Measure")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.savefig(OUT / "capital_allocations.png", dpi=150, facecolor="white")
    plt.close(fig)


def plot_weights(total, measures):
    order = np.argsort(total.values)
    percentiles = np.arange(len(order)) / len(order) * 100
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    for name in ("TVaR 99%", "PH (α=0.5)", "Wang (α=1)", "Std dev (k=2)"):
        ax.plot(percentiles, measures[name].weights.values[order], label=name, linewidth=1.2)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("Percentile of total loss")
    ax.set_ylabel("Simulation weight (symmetric log scale)")
    ax.set_title("Risk Measure Weights by Percentile")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(OUT / "risk_measure_weights.png", dpi=150, facecolor="white")
    plt.close(fig)


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
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    x = np.arange(len(excesses))
    width = 0.36
    ax.bar(x - width / 2, els, width, label="Loss on line", color="#4c78a8")
    ax.bar(x + width / 2, prices, width, label="Rate on line", color="#e45756")
    ax.set_xticks(x, [f"£{e:.0f}m" for e in excesses])
    ax.set_xlabel("Attachment point")
    ax.set_ylabel("% of limit")
    ax.set_title("XoL Pricing by Attachment Point (PH α=0.5)")
    ax2 = ax.twinx()
    ax2.plot(x, loadings, color="#222222", marker="o", linewidth=2, label="Loading")
    ax2.set_ylabel("Loading (%)")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, frameon=False, loc="upper right")
    fig.savefig(OUT / "xol_price_curve.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    portfolio, total, measures = portfolio_results()
    plot_allocations(portfolio, measures)
    plot_weights(total, measures)
    plot_price_curve()

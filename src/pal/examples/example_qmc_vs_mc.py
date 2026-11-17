"""Compare MC and RQMC copula samples with Plotly density heatmaps.

Run with:\n    python -m pal.examples.example_qmc_vs_mc --output qmc-plots

Creates an interactive HTML heatmap and a convergence comparison chart.
The figures use the same copulas, parameters and simulation counts for both
sampling methods. Histograms illustrate the sampled density; they are not
an estimate of the analytic density's pointwise error.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.integrate import quad
from scipy.stats import norm
from plotly.subplots import make_subplots

from pal import config, set_random_seed
from pal._maths import asnumpy
from pal.config import SamplingMethod, set_sampling_method
from pal.copulas import GaussianCopula, StudentsTCopula

N_SIMS = 8192
RHO = 0.7
GRID_BINS = 32


def copula_samples(family: str, method: SamplingMethod, n_sims: int, seed: int) -> np.ndarray:
    """Return a pair of uniform copula margins on the host."""
    previous = (config.seed, config.sampling_method, config.qmc_dimensions, config.rng)
    try:
        set_sampling_method(method, dimensions=32)
        set_random_seed(seed)
        correlation = [[1.0, RHO], [RHO, 1.0]]
        if family == "Gaussian":
            copula = GaussianCopula(correlation)
        elif family == "Student t (4 df)":
            copula = StudentsTCopula(correlation, dof=4)
        else:
            raise ValueError(f"Unknown copula: {family}")
        values = list(copula.generate(n_sims=n_sims))
        return np.stack([asnumpy(item.values) for item in values], axis=1)
    finally:
        config.seed, config.sampling_method, config.qmc_dimensions, config.rng = previous


def copula_density_chart(n_sims: int = N_SIMS) -> go.Figure:
    """Return matched 2x2 empirical density heatmaps with common scales."""
    methods = [SamplingMethod.MC, SamplingMethod.RQMC]
    families = ["Gaussian", "Student t (4 df)"]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"{family} · {method.name}" for family in families for method in methods],
        horizontal_spacing=0.08,
        vertical_spacing=0.13,
    )
    density_panels: list[tuple[int, int, np.ndarray]] = []
    for row, family in enumerate(families, start=1):
        for column, method in enumerate(methods, start=1):
            points = copula_samples(family, method, n_sims, 20261117)
            counts, _, _ = np.histogram2d(
                points[:, 0],
                points[:, 1],
                bins=GRID_BINS,
                range=[[0, 1], [0, 1]],
            )
            density = counts * (GRID_BINS * GRID_BINS / n_sims)
            density_panels.append((row, column, density))

    # Use the same colour mapping for all methods and copula families.
    maximum = float(np.percentile(np.concatenate([x.ravel() for _, _, x in density_panels]), 99.5))
    centres = (np.arange(GRID_BINS) + 0.5) / GRID_BINS
    for row, column, density in density_panels:
        fig.add_trace(
            go.Heatmap(
                x=centres,
                y=centres,
                z=density.T,
                coloraxis="coloraxis",
                hovertemplate="u=%{x:.2f}<br>v=%{y:.2f}<br>Density=%{z:.2f}<extra></extra>",
            ),
            row=row,
            col=column,
        )
        fig.update_xaxes(title_text="u", range=[0, 1], row=row, col=column)
        fig.update_yaxes(title_text="v", range=[0, 1], row=row, col=column)
    fig.update_layout(
        title=f"Empirical copula densities: {n_sims:,} paths per panel",
        coloraxis={"colorscale": "Viridis", "cmin": 0, "cmax": maximum, "colorbar": {"title": "Density"}},
        template="plotly_white",
        width=1050,
        height=920,
        margin={"l": 80, "r": 105, "t": 115, "b": 65},
    )
    return fig


def convergence_chart(sizes: list[int] | None = None, replicates: int = 12) -> go.Figure:
    """Compare MC and RQMC Gaussian copula joint exceedance errors."""
    if sizes is None:
        sizes = [512, 1024, 2048, 4096, 8192]
    if replicates < 2 or any(n < 1 for n in sizes):
        raise ValueError("Choose at least two replicates and positive simulation sizes")

    probability = 0.9
    z = norm.ppf(probability)
    reference, _ = quad(
        lambda x: norm.pdf(x) * norm.cdf((RHO * x - z) / np.sqrt(1 - RHO**2)),
        z,
        np.inf,
        epsabs=1e-12,
        epsrel=1e-12,
    )
    fig = go.Figure()
    for method in SamplingMethod:
        rmses = []
        for n in sizes:
            errors = []
            for replicate in range(replicates):
                points = copula_samples("Gaussian", method, n, 20261117 + 65537 * replicate)
                estimate = np.mean((points[:, 0] > probability) & (points[:, 1] > probability))
                errors.append(estimate - reference)
            rmses.append(float(np.sqrt(np.mean(np.square(errors)))))
        fig.add_trace(go.Scatter(x=sizes, y=rmses, name=method.name, mode="lines+markers"))
    fig.update_layout(
        title=f"Gaussian copula joint 90% tail: RMSE over {replicates} independent runs",
        xaxis={"title": "Simulation paths", "type": "log"},
        yaxis={"title": "RMSE of joint exceedance estimate", "type": "log"},
        template="plotly_white",
        width=900,
        height=530,
    )
    return fig


def main() -> None:
    """Write the comparison figures as interactive Plotly HTML."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("qmc-plots"))
    parser.add_argument("--n-sims", type=int, default=N_SIMS)
    parser.add_argument("--replicates", type=int, default=12)
    parser.add_argument("--svg", action="store_true", help="Also render SVGs; needs Kaleido and Chrome")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plots = {
        "qmc_copula_densities": copula_density_chart(args.n_sims),
        "qmc_vs_mc_convergence": convergence_chart(replicates=args.replicates),
    }
    for name, figure in plots.items():
        figure.write_html(args.output / f"{name}.html", include_plotlyjs=True)
        if args.svg:
            figure.write_image(args.output / f"{name}.svg")
        print(args.output / f"{name}.html")


if __name__ == "__main__":
    main()

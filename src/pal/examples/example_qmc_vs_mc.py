"""Static Monte Carlo and randomized QMC CDF and copula-density comparisons.

The figures use actual PAL copula simulations. With the optional documentation
dependencies installed (Kaleido and Chrome), run:

    python -m pal.examples.example_qmc_vs_mc --output qmc-plots

This writes static SVG surfaces and a static convergence chart, not HTML.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import quad
from scipy.stats import norm

from pal import config, set_random_seed
from pal._maths import asnumpy
from pal.config import SamplingMethod, set_sampling_method
from pal.copulas import GaussianCopula, StudentsTCopula
from pal.distributions import Normal

RHO = 0.7
GRID_BINS = 28


def copula_samples(family: str, method: SamplingMethod, n_sims: int | None = None, seed: int = 42) -> np.ndarray:
    """Generate copula pairs from PAL using its default simulation count."""
    previous = (config.seed, config.sampling_method, config.qmc_dimensions, config.rng)
    try:
        set_sampling_method(method)
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


def univariate_cdf_chart(n_sims: int | None = None, seed: int = 42) -> go.Figure:
    """Show the empirical normal CDF and its sampling error under MC/RQMC."""
    if n_sims is None:
        n_sims = config.n_sims
    if n_sims < 1:
        raise ValueError("n_sims must be positive")
    previous = (config.seed, config.sampling_method, config.qmc_dimensions, config.rng)
    samples: dict[SamplingMethod, np.ndarray] = {}
    try:
        for method in SamplingMethod:
            set_sampling_method(method)
            set_random_seed(seed)
            samples[method] = np.sort(asnumpy(Normal(mu=0, sigma=1).generate(n_sims=n_sims).values))
    finally:
        config.seed, config.sampling_method, config.qmc_dimensions, config.rng = previous

    grid = np.linspace(-3.5, 3.5, 281)
    reference = norm.cdf(grid)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.12)
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=reference,
            name="Exact normal CDF",
            mode="lines",
            line={"color": "#8798ae", "dash": "dash"},
        ),
        row=1,
        col=1,
    )
    for method, color in ((SamplingMethod.MC, "#c96c40"), (SamplingMethod.RQMC, "#2163bc")):
        empirical = np.searchsorted(samples[method], grid, side="right") / n_sims
        fig.add_trace(
            go.Scatter(x=grid, y=empirical, name=method.name, mode="lines", line={"color": color}),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=100 * (empirical - reference),
                name=f"{method.name} deviation",
                mode="lines",
                line={"color": color},
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    fig.add_shape(
        type="line",
        x0=float(grid[0]),
        x1=float(grid[-1]),
        y0=0,
        y1=0,
        xref="x2",
        yref="y2",
        line={"color": "#9cabbc", "width": 1},
    )
    fig.update_yaxes(title_text="Cumulative probability", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="CDF error (pp)", row=2, col=1)
    fig.update_xaxes(title_text="Standard normal quantile", row=2, col=1)
    fig.update_layout(
        title=f"Normal distribution: Monte Carlo versus randomized Sobol ({n_sims:,} draws)",
        template="plotly_white",
        width=1050,
        height=650,
        margin={"l": 80, "r": 50, "t": 90, "b": 65},
    )
    return fig


def copula_density_chart(n_sims: int | None = None) -> go.Figure:
    """Return matched 3D empirical-density surfaces for MC and RQMC."""
    if n_sims is None:
        n_sims = config.n_sims
    if n_sims < 1:
        raise ValueError("n_sims must be positive")

    methods = (SamplingMethod.MC, SamplingMethod.RQMC)
    densities = []
    for method in methods:
        points = copula_samples("Gaussian", method, n_sims=n_sims)
        counts, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=GRID_BINS, range=((0, 1), (0, 1)))
        # A small local smoothing kernel makes the 3D sampling surface legible.
        # This is an empirical density, not the analytical copula density.
        density = counts * GRID_BINS**2 / n_sims
        padded = np.pad(density, 1, mode="edge")
        smoothed = (
            padded[:-2, :-2]
            + 2 * padded[1:-1, :-2]
            + padded[2:, :-2]
            + 2 * padded[:-2, 1:-1]
            + 4 * padded[1:-1, 1:-1]
            + 2 * padded[2:, 1:-1]
            + padded[:-2, 2:]
            + 2 * padded[1:-1, 2:]
            + padded[2:, 2:]
        ) / 16
        densities.append(smoothed)

    centres = (np.arange(GRID_BINS) + 0.5) / GRID_BINS
    max_density = float(max(density.max() for density in densities))
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Monte Carlo", "Randomized QMC"),
        horizontal_spacing=0.04,
    )
    for col, density in enumerate(densities, start=1):
        fig.add_trace(
            go.Surface(
                x=centres,
                y=centres,
                z=density.T,
                cmin=0,
                cmax=max_density,
                colorscale="Viridis",
                showscale=col == 2,
                colorbar={"title": "Density"} if col == 2 else None,
            ),
            row=1,
            col=col,
        )
        fig.update_scenes(
            xaxis={"title": "u", "range": [0, 1]},
            yaxis={"title": "v", "range": [0, 1]},
            zaxis={"title": "Density", "range": [0, max_density]},
            camera={"eye": {"x": 1.7, "y": -1.1, "z": 1.8}},
            aspectratio={"x": 1, "y": 1, "z": 0.7},
            row=1,
            col=col,
        )
    fig.update_layout(
        title=f"Empirical Gaussian copula densities ({n_sims:,} simulated pairs per panel)",
        template="plotly_white",
        width=1280,
        height=650,
        margin={"l": 10, "r": 75, "t": 90, "b": 10},
    )
    return fig


def convergence_chart(sizes: list[int] | None = None, replicates: int = 12) -> go.Figure:
    """Compare Gaussian copula joint-tail RMSE under MC and RQMC."""
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
    """Write static SVG figures with the optional docs rendering dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("qmc-plots"))
    parser.add_argument("--n-sims", type=int, default=None, help="Defaults to PAL's configured simulation count")
    parser.add_argument("--replicates", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    charts = {
        "qmc_univariate_cdf": univariate_cdf_chart(n_sims=args.n_sims),
        "qmc_copula_surface": copula_density_chart(n_sims=args.n_sims),
        "qmc_vs_mc_convergence": convergence_chart(replicates=args.replicates),
    }
    for name, chart in charts.items():
        target = args.output / f"{name}.svg"
        chart.write_image(target)
        print(target)


if __name__ == "__main__":
    main()

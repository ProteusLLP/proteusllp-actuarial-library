"""Smoke tests for the frequency-severity scaling benchmark and static surfaces."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from benchmarks.benchmark_frequency_severity import run_benchmarks

from pal.config import SamplingMethod, config
from pal.examples.example_qmc_vs_mc import copula_density_chart, copula_samples, univariate_cdf_chart


def test_frequency_severity_benchmark_covers_cpu_and_rqmc_paths():
    """Performance measurements must include both generators and event aggregation."""
    original = (config.seed, config.rng, config.sampling_method, config.qmc_dimensions)
    result = run_benchmarks([128], replicates=1, cases=("low_frequency", "stochastic_scale"))
    assert result["backend"] in ("cpu", "gpu")
    assert len(result["results"]) == 4
    for row in result["results"]:
        assert row["n_sims"] == 128
        assert row["mean_claims"] >= 0
        assert row["median_generate_ms"] >= 0
        assert row["median_aggregate_ms"] >= 0
        assert row["claims_per_second"] > 0
        assert np.isfinite(row["aggregate_mean"])
    assert {row["method"] for row in result["results"]} == {"MC", "RQMC"}
    assert original == (config.seed, config.rng, config.sampling_method, config.qmc_dimensions)


def test_static_copula_example_uses_surface_plots():
    """Density plots use paired 3D surfaces with common limits, not heatmaps."""
    figure = copula_density_chart(n_sims=256)
    assert len(figure.data) == 2
    assert all(isinstance(trace, go.Surface) for trace in figure.data)
    assert figure.data[0].cmax == figure.data[1].cmax
    assert figure.data[0].cmin == figure.data[1].cmin
    assert figure.layout.scene.camera.eye.z >= 1.5
    assert figure.layout.scene2.camera.eye.z >= 1.5


def test_univariate_cdf_example_uses_common_x_grid():
    """The univariate display must compare MC, RQMC and the same exact CDF."""
    chart = univariate_cdf_chart(n_sims=128)
    assert len(chart.data) == 5
    assert np.array_equal(chart.data[0].x, chart.data[1].x)
    assert np.array_equal(chart.data[1].x, chart.data[2].x)


def test_copula_samples_restore_sampling_configuration():
    """Generating demonstration samples must not change the caller's setup."""
    original = (config.seed, config.rng, config.sampling_method, config.qmc_dimensions)
    points = copula_samples("Gaussian", SamplingMethod.RQMC, n_sims=128, seed=2026)
    assert points.shape == (128, 2)
    assert np.all(np.isfinite(points))
    assert original == (config.seed, config.rng, config.sampling_method, config.qmc_dimensions)

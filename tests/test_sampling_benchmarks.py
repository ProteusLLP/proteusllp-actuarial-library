"""Regression tests for executable MC/RQMC benchmark and Plotly example."""

import numpy as np
import pytest
from benchmarks.benchmark_sampling import (
    analytic_joint_tail,
    markdown_summary,
    run_comparison,
)
from pal import config
from pal.examples.example_qmc_vs_mc import convergence_chart, copula_density_chart


def test_joint_tail_reference_matches_independence():
    assert analytic_joint_tail(rho=0, probability=0.9) == pytest.approx(0.01, abs=1e-12)
    assert 0.01 < analytic_joint_tail(rho=0.65, probability=0.9) < 0.1


def test_accuracy_comparison_both_sampling_modes():
    report = run_comparison(sizes=[512], replicates=3, cases=("normal_mean", "gaussian_joint_tail"))
    assert report["schema_version"] == 1
    rows = report["results"]
    assert len(rows) == 4
    assert {row["sampling_method"] for row in rows} == {"MC", "RQMC"}
    assert {row["case"] for row in rows} == {"normal_mean", "gaussian_joint_tail"}
    for row in rows:
        assert np.isfinite(row["rmse"])
        assert np.isfinite(row["estimate_sd"])
        assert row["median_elapsed_ms"] >= 0.0
        assert row["replicates"] == 3
    assert "MC vs RQMC" in markdown_summary(report)


def test_comparison_input_validation():
    with pytest.raises(ValueError, match="replicates"):
        run_comparison([16], replicates=1)
    with pytest.raises(ValueError, match="Unknown"):
        run_comparison([16], replicates=2, cases=("unknown",))


def test_copula_density_chart_has_matching_axes_and_panels():
    before = (config.seed, config.sampling_method, config.qmc_dimensions, config.rng)
    fig = copula_density_chart(n_sims=256)
    after = (config.seed, config.sampling_method, config.qmc_dimensions, config.rng)
    assert before == after
    assert len(fig.data) == 4
    assert all(trace.type == "heatmap" for trace in fig.data)
    assert all(trace.coloraxis == "coloraxis" for trace in fig.data)
    assert all(np.asarray(trace.z).shape == (32, 32) for trace in fig.data)


def test_convergence_example_returns_two_error_curves():
    fig = convergence_chart(sizes=[64, 128], replicates=2)
    assert len(fig.data) == 2
    assert {trace.name for trace in fig.data} == {"MC", "RQMC"}
    assert all(len(trace.x) == 2 for trace in fig.data)

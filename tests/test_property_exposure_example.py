"""Integration tests for the property exposure-rating example."""

import pandas as pd
import pytest
from examples.example_property_exposure_rating import (
    DATA_PATH,
    claim_frequencies,
    make_calibration_table,
    make_claim_severity_figure,
    make_gross_burn_rate_figure,
    make_layer_burn_rate_figure,
    make_layer_burn_rates,
    make_layer_comparison,
    make_layer_rate_comparison_figure,
    make_tower,
    simulate_claim_rows,
    simulate_policy_losses,
)

from pal._maths import xp
from pal.config import set_random_seed


def test_property_exposure_row_selection_and_mbbefd_severity_workflow() -> None:
    """Select claim rows empirically, then draw continuous MBBEFD severities."""
    exposures = pd.read_csv(DATA_PATH)
    frequencies = claim_frequencies(exposures)

    assert frequencies == pytest.approx(
        [
            0.0250560446,
            0.0345541978,
            0.0568114195,
            0.1146095843,
            0.1876025068,
            0.3779235410,
            0.6720690387,
            0.4140222253,
        ],
        rel=1e-8,
    )

    set_random_seed(42)
    claim_rows = simulate_claim_rows(exposures, frequencies, n_sims=5_000)
    row_index = claim_rows.values.astype(int)

    assert claim_rows.n_sims == 5_000
    assert len(row_index) > 0
    assert bool(xp.all(row_index >= 0))
    assert bool(xp.all(row_index < len(exposures)))
    assert float(claim_rows.count().mean()) == pytest.approx(
        float(frequencies.sum()),
        abs=0.08,
    )

    policy_losses = simulate_policy_losses(exposures, claim_rows)
    selected_limits = xp.asarray(exposures["policy_limit"].to_numpy(dtype=float))[row_index]

    assert policy_losses.n_sims == claim_rows.n_sims
    assert bool(xp.array_equal(policy_losses.sim_index, claim_rows.sim_index))
    assert len(policy_losses.values) == len(claim_rows.values)
    assert bool(xp.all(policy_losses.values >= 0))
    assert bool(xp.all(policy_losses.values <= selected_limits))
    assert xp.unique(policy_losses.values).size > 100

    tower = make_tower()
    tower.apply(policy_losses)
    assert all(layer.summary["mean"] >= 0 for layer in tower.layers)

    calibration = make_calibration_table(
        exposures,
        frequencies,
        claim_rows,
        policy_losses,
    )
    assert (
        calibration.loc[
            calibration["metric"] == "Total subject premium",
            "target",
        ].item()
        == 2_150_000
    )
    assert (
        calibration.loc[
            calibration["metric"] == "Annual policy loss",
            "target",
        ].item()
        == 1_422_100
    )
    assert calibration.loc[
        calibration["metric"] == "Portfolio loss ratio",
        "target",
    ].item() == pytest.approx(1_422_100 / 2_150_000)

    total_subject_premium = float(exposures["subject_premium"].sum())
    comparison = make_layer_comparison(exposures, tower, total_subject_premium)
    assert comparison["exposure_rate"].tolist() == pytest.approx(
        [0.1163531952, 0.1229171863, 0.1015502012],
        rel=1e-8,
    )

    layer_burn_rates = make_layer_burn_rates(
        tower,
        policy_losses,
        total_subject_premium,
    )
    assert set(layer_burn_rates) == {"5m xs 5m", "10m xs 10m", "20m xs 20m"}
    assert all(len(burn_rate) == claim_rows.n_sims for burn_rate in layer_burn_rates.values())

    severity_figure = make_claim_severity_figure(policy_losses)
    gross_burn_figure = make_gross_burn_rate_figure(
        policy_losses,
        total_subject_premium,
    )
    layer_burn_figure = make_layer_burn_rate_figure(layer_burn_rates)
    comparison_figure = make_layer_rate_comparison_figure(comparison)

    assert len(severity_figure.data) == 2
    assert len(gross_burn_figure.data) == 2
    assert len(layer_burn_figure.data) == 3
    assert len(comparison_figure.data) == 2

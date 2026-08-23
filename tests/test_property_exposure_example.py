"""Integration tests for the property exposure-rating example."""

import pandas as pd
import pytest
from examples.example_property_exposure_rating import (
    DATA_PATH,
    claim_frequencies,
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
    assert float(claim_rows.count().mean()) == pytest.approx(float(frequencies.sum()), abs=0.08)

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

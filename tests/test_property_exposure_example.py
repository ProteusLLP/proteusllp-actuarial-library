"""Integration tests for the property exposure-rating example."""

import pandas as pd
import pytest
from examples.example_property_exposure_rating import (
    DATA_PATH,
    make_empirical_policy_severity,
    make_tower,
    simulate_policy_losses,
)

from pal import Empirical
from pal._maths import xp
from pal.config import set_random_seed


def test_property_exposure_empirical_frequency_severity_workflow() -> None:
    """Infer frequencies, build empirical severity and apply the reinsurance tower."""
    exposures = pd.read_csv(DATA_PATH)
    frequencies, severity = make_empirical_policy_severity(exposures)

    assert isinstance(severity, Empirical)
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
        rel=5e-4,
    )

    set_random_seed(42)
    simulated_frequencies, policy_losses = simulate_policy_losses(
        exposures,
        n_sims=2_000,
        n_points=1_000,
    )

    assert policy_losses.n_sims == 2_000
    assert len(policy_losses.values) > 0
    assert bool(xp.all(policy_losses.values >= 0))
    assert float(policy_losses.count().mean()) == pytest.approx(float(simulated_frequencies.sum()), abs=0.1)

    tower = make_tower()
    tower.apply(policy_losses)
    assert all(layer.summary["mean"] >= 0 for layer in tower.layers)

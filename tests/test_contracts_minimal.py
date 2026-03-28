"""Tests for contracts module to improve coverage."""

import numpy as np

from pal import FreqSevSims
from pal.contracts import XoL, XoLTower


def test_xol_print_summary(capsys):
    """Test XoL print_summary method (lines 248-260)."""
    sim_idx = np.array([0, 0, 1, 1, 2])
    losses = np.array([100000, 200000, 300000, 400000, 500000])
    claims = FreqSevSims(sim_idx, losses, n_sims=3)

    # Create simple XoL contract and apply (which calls calc_summary)
    layer = XoL("Test Layer", 100000, 50000, 10000)
    _ = layer.apply(claims)

    # Call print_summary to cover lines 248-260
    layer.print_summary()

    # Verify something was printed
    captured = capsys.readouterr()
    assert "Test Layer" in captured.out
    assert "Mean Recoveries" in captured.out


def test_xoltower_with_optional_params():
    """Test XoLTower with optional parameters (lines 298-305, 329-340)."""
    # Test with all optional parameters
    tower = XoLTower(
        limit=[100000, 200000],
        excess=[50000, 150000],
        premium=[10000, 20000],
        name=["Layer 1", "Layer 2"],
        reinstatement_cost=[[5000], [10000]],
        aggregate_deductible=[25000, 50000],
        aggregate_limit=[500000, 1000000],
        franchise=[1000, 2000],
        reverse_franchise=[None, None],
    )

    assert tower.n_layers == 2
    assert len(tower.layers) == 2
    assert tower.layers[0].name == "Layer 1"
    assert tower.layers[1].name == "Layer 2"


def test_xoltower_print_summary(capsys):
    """Test XoLTower print_summary method (lines 344-345)."""
    sim_idx = np.array([0, 0, 1, 1, 2])
    losses = np.array([100000, 200000, 300000, 400000, 500000])
    claims = FreqSevSims(sim_idx, losses, n_sims=3)

    # Create tower
    tower = XoLTower(
        limit=[100000, 200000],
        excess=[50000, 150000],
        premium=[10000, 20000],
    )

    # Apply tower - calc_summary is called within apply()
    _ = tower.apply(claims)  # noqa: F841
    for layer in tower.layers:
        _ = layer.apply(claims)  # Applies and calls calc_summary

    # Call print_summary to cover lines 344-345
    tower.print_summary()

    # Verify output
    captured = capsys.readouterr()
    assert "Layer 1" in captured.out or "Layer 2" in captured.out

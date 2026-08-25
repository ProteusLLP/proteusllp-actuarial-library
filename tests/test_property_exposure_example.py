"""Integration tests for the property exposure-rating example."""

import pandas as pd
import pytest

from examples import example_property_exposure_rating as example


def test_property_exposure_example_runs_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run the vectorised exposure-rating workflow, including aggregate terms."""
    exposures = pd.read_csv(example.DATA_PATH)

    assert list(exposures.columns) == [
        "maximum_loss",
        "policy_limit",
        "policy_deductible",
        "subject_premium",
        "expected_loss_ratio",
        "mbbefd_c",
    ]
    assert float(exposures["subject_premium"].sum()) == 2_150_000
    assert float(
        (exposures["subject_premium"] * exposures["expected_loss_ratio"]).sum()
    ) == 1_422_100

    monkeypatch.setattr(example, "N_SIMS", 5_000)
    monkeypatch.setenv("PAL_SUPPRESS_PLOTS", "true")

    example.main()

    output = capsys.readouterr().out
    assert "Portfolio calibration check" in output
    assert "Layer burn rates" in output
    assert "Analytical exposure rate" in output
    assert "Simulated with £1m aggregate deductible" in output
    assert "Mean annual net property loss" in output

"""Regression tests for DataFrame/CSV/Parquet PAL variable round trips."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pal import FreqSevSims, ProteusVariable, StochasticScalar
from pal._maths import asnumpy
from pal.couplings import couple_stochastic_results
from pal.serialization import load, save
from pal.variable_io import from_dataframe as decode_snapshot
from pal.variable_io import to_dataframe as encode_snapshot


@pytest.mark.parametrize("storage", ["frame", "csv", "parquet"])
def test_nested_stochastic_roundtrip_with_coupling(storage, tmp_path):
    """Value types, dimensions, labels and within-file coupling survive I/O."""
    motor = StochasticScalar(np.array([1, 2, 3], dtype=np.int64))
    property_loss = StochasticScalar(np.array([0.5, np.nan, 2.5]))
    isolated = StochasticScalar(["low", "medium", "high"])
    couple_stochastic_results((motor, property_loss))
    original = ProteusVariable(
        "region",
        {
            "UK": ProteusVariable("class", {"Motor": motor, "Property": property_loss}),
            "US": ProteusVariable("class", {"Motor": isolated}),
        },
    )

    if storage == "frame":
        restored = decode_snapshot(encode_snapshot(original))
    elif storage == "csv":
        path = tmp_path / "losses.csv"
        save(original, path)
        restored = load(path)
    else:
        pytest.importorskip("pyarrow")
        path = tmp_path / "losses.parquet"
        save(original, path)
        restored = load(path)

    assert restored.dim_name == "region"
    assert list(restored.values) == ["UK", "US"]
    assert restored["UK"].dim_name == "class"
    assert list(restored["UK"].values) == ["Motor", "Property"]
    np.testing.assert_array_equal(asnumpy(restored["UK"]["Motor"].values), np.array([1, 2, 3]))
    np.testing.assert_array_equal(asnumpy(restored["UK"]["Property"].values), np.array([0.5, np.nan, 2.5]))
    np.testing.assert_array_equal(asnumpy(restored["US"]["Motor"].values), ["low", "medium", "high"])
    assert restored["UK"]["Motor"].values.dtype.kind in "iu"
    assert restored["UK"]["Motor"].coupled_variable_group is restored["UK"]["Property"].coupled_variable_group
    assert restored["US"]["Motor"].coupled_variable_group is not restored["UK"]["Motor"].coupled_variable_group
    assert restored["UK"]["Motor"].coupled_variable_group is not motor.coupled_variable_group


@pytest.mark.parametrize("storage", ["frame", "csv", "parquet"])
def test_freqsev_claims_and_zero_claim_scenarios(storage, tmp_path):
    """Preserve event order even when event simulation indices are interleaved."""
    frequency_severity = FreqSevSims(
        np.array([2, 0, 2, 3, 0], dtype=np.int64),
        np.array([100.0, 10.0, 200.0, 40.0, 20.0]),
        n_sims=5,
    )
    variable = ProteusVariable("LoB", {"Large losses": frequency_severity})
    if storage == "frame":
        restored = decode_snapshot(encode_snapshot(variable))
    elif storage == "csv":
        path = tmp_path / "claims.csv"
        save(variable, path)
        restored = load(path)
    else:
        pytest.importorskip("pyarrow")
        path = tmp_path / "claims.parquet"
        save(variable, path)
        restored = load(path)
    result = restored["Large losses"]
    assert isinstance(result, FreqSevSims)
    assert result.n_sims == 5
    np.testing.assert_array_equal(asnumpy(result.sim_index), asnumpy(frequency_severity.sim_index))
    np.testing.assert_array_equal(asnumpy(result.values), asnumpy(frequency_severity.values))


@pytest.mark.parametrize("n_sims", [0, 3])
def test_freqsev_no_claims(n_sims):
    variable = ProteusVariable("risk", {"Cat": FreqSevSims([], np.array([], dtype=np.float64), n_sims=n_sims)})
    result = decode_snapshot(encode_snapshot(variable))["Cat"]
    assert result.n_sims == n_sims
    assert len(result.sim_index) == 0
    assert result.values.dtype == np.float64


def test_zero_simulations_stochastic():
    variable = ProteusVariable("risk", {"A": StochasticScalar(np.array([], dtype=np.int32))})
    restored = decode_snapshot(encode_snapshot(variable))
    assert restored["A"].n_sims == 0
    assert restored["A"].values.dtype == np.int32


def test_deterministic_numeric_text_and_dates(tmp_path):
    original = ProteusVariable(
        "parameter",
        {
            "Retention": 1_000_000,
            "Enabled": True,
            "Currency": "GBP",
            "Valuation date": np.datetime64("2026-11-21", "D"),
        },
    )
    path = tmp_path / "params.csv"
    save(original, path)
    restored = load(path)
    assert restored["Retention"] == 1_000_000
    assert isinstance(restored["Retention"], int)
    assert restored["Enabled"] is True
    assert restored["Currency"] == "GBP"
    assert restored["Valuation date"] == np.datetime64("2026-11-21", "D")


def test_numeric_only_csv_does_not_infer_or_round_payloads(tmp_path):
    """CSV payloads are JSON literals and must be read as text."""
    original = ProteusVariable(
        "risk",
        {"Extreme": StochasticScalar(np.array([2**63 + 1, 2**64 - 1], dtype=np.uint64))},
    )
    path = tmp_path / "numbers.csv"
    save(original, path)
    restored = load(path)
    np.testing.assert_array_equal(asnumpy(restored["Extreme"].values), asnumpy(original["Extreme"].values))
    assert restored["Extreme"].values.dtype == np.dtype("uint64")


def test_string_and_date_stochastic_values():
    original = ProteusVariable(
        "risk",
        {
            "rating": StochasticScalar(["AA", "BBB", "CCC"]),
            "date": StochasticScalar(np.array(["2025-01-01", "2026-01-01", "2027-01-01"], dtype="datetime64[D]")),
        },
    )
    restored = decode_snapshot(encode_snapshot(original))
    np.testing.assert_array_equal(asnumpy(restored["rating"].values), asnumpy(original["rating"].values))
    np.testing.assert_array_equal(asnumpy(restored["date"].values), asnumpy(original["date"].values))


def test_import_conventional_simulation_table_and_legacy_csv(tmp_path):
    frame = pd.DataFrame(
        {
            "Simulation": [11, 10, 10, 11],
            "class": ["Motor", "Motor", "Property", "Property"],
            "claim": [5.0, 2.0, 8.0, 7.0],
        }
    )
    imported = ProteusVariable.from_dataframe(frame, "class", "claim")
    np.testing.assert_array_equal(asnumpy(imported["Motor"].values), [2.0, 5.0])
    np.testing.assert_array_equal(asnumpy(imported["Property"].values), [8.0, 7.0])
    path = tmp_path / "legacy.csv"
    frame.to_csv(path, index=False)
    legacy = ProteusVariable.from_csv(path, "class", "claim")
    np.testing.assert_array_equal(asnumpy(legacy["Motor"].values), asnumpy(imported["Motor"].values))


def test_import_conventional_deterministic_table():
    frame = pd.DataFrame({"risk": ["Motor", "Property"], "premium": [10, 20]})
    restored = ProteusVariable.from_dataframe(frame, "risk", "premium", simulation_column=None)
    assert restored.values == {"Motor": 10, "Property": 20}


@pytest.mark.parametrize(
    "bad_frame",
    [
        pd.DataFrame({"risk": ["A", "A"], "value": [1, 2]}),
        pd.DataFrame({"risk": ["A", "B"], "Simulation": [0, 1], "value": [1, 2]}),
    ],
)
def test_invalid_conventional_tables(bad_frame):
    with pytest.raises(ValueError):
        ProteusVariable.from_dataframe(
            bad_frame,
            "risk",
            "value",
            simulation_column=None if "Simulation" not in bad_frame.columns else "Simulation",
        )


def test_missing_and_duplicated_simulation_ids():
    frame = pd.DataFrame({"risk": ["A", "A", "B"], "Simulation": [0, 0, 0], "value": [1, 2, 3]})
    with pytest.raises(ValueError, match="Duplicate simulation"):
        ProteusVariable.from_dataframe(frame, "risk")
    frame.loc[1, "Simulation"] = 1
    with pytest.raises(ValueError, match="same simulation IDs"):
        ProteusVariable.from_dataframe(frame, "risk")


def test_invalid_schema_and_events_are_rejected():
    variable = ProteusVariable("risk", {"A": FreqSevSims([0, 1], [1.0, 2.0], 2)})
    frame = encode_snapshot(variable)
    frame.loc[0, "_pal_schema"] = "unsupported"
    with pytest.raises(ValueError, match="schema"):
        decode_snapshot(frame)
    frame = encode_snapshot(variable)
    frame.loc[0, "Event"] = 99
    with pytest.raises(ValueError, match="event IDs"):
        decode_snapshot(frame)


def test_unsupported_leaf_is_rejected():
    variable = ProteusVariable("risk", {"A": {"unexpected": "mapping"}})
    with pytest.raises(TypeError, match="dtype"):
        encode_snapshot(variable)


def test_empty_dimension_is_rejected():
    with pytest.raises(ValueError, match="Empty ProteusVariable"):
        encode_snapshot(ProteusVariable("empty", {}))

"""Ordinary DataFrame, CSV and Parquet interchange is independent of snapshots."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pal import FreqSevSims, ProteusVariable, StochasticScalar
from pal.serialization import load, save


@pytest.mark.parametrize("format", ["dataframe", "csv", "parquet"])
def test_wide_data_exchange_is_readable_without_pal(tmp_path, format):
    """An external program sees one simulation row and one column per LoB."""
    model = ProteusVariable(
        "class",
        {
            "Motor": StochasticScalar([1.5, 2.0, 3.5]),
            "Property": StochasticScalar([10.0, 20.0, 30.0]),
        },
    )

    if format == "dataframe":
        frame = model.to_dataframe()
        imported = ProteusVariable.from_dataframe(frame, dim_name="class")
    elif format == "csv":
        path = tmp_path / "losses.csv"
        model.to_csv(path)
        frame = pd.read_csv(path)
        imported = ProteusVariable.from_csv(path, dim_name="class")
    else:
        pytest.importorskip("pyarrow")
        path = tmp_path / "losses.parquet"
        model.to_parquet(path)
        frame = pd.read_parquet(path)
        imported = ProteusVariable.from_parquet(path, dim_name="class")

    assert list(frame.columns) == ["Simulation", "Motor", "Property"]
    assert not any(str(column).startswith("_pal_") for column in frame)
    np.testing.assert_array_equal(frame["Motor"], [1.5, 2.0, 3.5])
    assert imported.dim_name == "class"
    np.testing.assert_array_equal(imported["Property"].values, [10.0, 20.0, 30.0])


def test_long_data_exchange_has_ordinary_dimensions_and_values(tmp_path):
    model = ProteusVariable(
        "LoB",
        {"Motor": StochasticScalar([10.0, 20.0]), "Property": StochasticScalar([30.0, 40.0])},
    )
    frame = model.to_dataframe(layout="long")
    assert list(frame.columns) == ["LoB", "Simulation", "value"]
    assert frame.to_dict(orient="records")[0] == {"LoB": "Motor", "Simulation": 0, "value": 10.0}

    path = tmp_path / "long.csv"
    model.to_csv(path, layout="long")
    imported = ProteusVariable.from_csv(path, dim_name="LoB")
    np.testing.assert_array_equal(imported["Motor"].values, [10.0, 20.0])


def test_claim_events_have_simple_long_format_and_optional_sim_count(tmp_path):
    """Interchange cannot infer trailing zero-claim scenarios without n_sims."""
    model = ProteusVariable("LoB", {"Cat": FreqSevSims([2, 0, 2], [100.0, 10.0, 200.0], 5)})
    frame = model.to_dataframe()
    assert list(frame.columns) == ["LoB", "Simulation", "Event", "value"]
    assert frame["Simulation"].tolist() == [2, 0, 2]
    assert frame["Event"].tolist() == [0, 1, 2]
    path = tmp_path / "claims.csv"
    model.to_csv(path)
    restored = ProteusVariable.from_csv(path, dim_name="LoB", n_sims=5)["Cat"]
    assert isinstance(restored, FreqSevSims)
    assert restored.n_sims == 5
    np.testing.assert_array_equal(restored.sim_index, [2, 0, 2])
    np.testing.assert_array_equal(restored.values, [100.0, 10.0, 200.0])


def test_deterministic_exchange_table():
    model = ProteusVariable("risk", {"Retention": 100, "Deductible": 25})
    frame = model.to_dataframe()
    assert list(frame.columns) == ["risk", "Simulation", "value"]
    restored = ProteusVariable.from_dataframe(frame, dim_name="risk", simulation_column=None)
    assert restored.values == {"Retention": 100, "Deductible": 25}


def test_object_snapshot_is_not_an_exchange_table(tmp_path):
    """The two APIs intentionally produce and consume different file formats."""
    model = ProteusVariable("LoB", {"Motor": StochasticScalar([4.0, 5.0])})
    pytest.importorskip("pyarrow")
    snapshot = tmp_path / "model.parquet"
    exchange = tmp_path / "data.parquet"
    save(model, snapshot)
    model.to_parquet(exchange)
    assert list(pd.read_parquet(exchange).columns) == ["Simulation", "Motor"]
    with pytest.raises(ValueError):
        ProteusVariable.from_parquet(snapshot)
    with pytest.raises(ValueError):
        load(exchange)
    restored = load(snapshot)
    np.testing.assert_array_equal(restored["Motor"].values, [4.0, 5.0])


def test_wide_import_rejects_duplicate_or_missing_simulations():
    frame = pd.DataFrame({"Simulation": [0, 0], "Motor": [1.0, 2.0]})
    with pytest.raises(ValueError, match="unique"):
        ProteusVariable.from_dataframe(frame)
    frame["Simulation"] = [0, np.nan]
    with pytest.raises(ValueError, match="unique"):
        ProteusVariable.from_dataframe(frame)


def test_wide_exchange_10000_simulations_avoids_per_value_json():
    rng = np.random.default_rng(123)
    model = ProteusVariable("LoB", {f"risk{i}": StochasticScalar(rng.standard_normal(10000)) for i in range(10)})
    frame = model.to_dataframe()
    assert frame.shape == (10000, 11)
    for i in range(10):
        np.testing.assert_array_equal(frame[f"risk{i}"], model[f"risk{i}"].values)

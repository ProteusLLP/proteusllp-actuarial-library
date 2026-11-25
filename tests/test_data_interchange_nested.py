"""Interchange should import all tables PAL exports, including nested headers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pal import FreqSevSims, ProteusVariable, StochasticScalar
from pal._maths import asnumpy


@pytest.fixture
def nested() -> ProteusVariable:
    return ProteusVariable(
        "Region",
        {
            "UK": ProteusVariable(
                "LoB",
                {
                    "Motor": StochasticScalar([10, 20, 30]),
                    "Property": StochasticScalar([2.5, 3.5, 4.5]),
                },
            ),
            "US": ProteusVariable(
                "LoB",
                {
                    "Motor": StochasticScalar([40, 50, 60]),
                    "Property": StochasticScalar([6.5, 7.5, 8.5]),
                },
            ),
        },
    )


@pytest.mark.parametrize("format", ["dataframe", "csv", "parquet"])
@pytest.mark.parametrize("layout", ["multiindex", "long"])
def test_nested_export_import(nested, tmp_path, format, layout):
    """Nested values survive every exported CSV, Parquet and DataFrame layout."""
    if format == "dataframe":
        frame = nested.to_dataframe(layout=layout)
        result = ProteusVariable.from_dataframe(frame)
    elif format == "csv":
        path = tmp_path / "nested.csv"
        nested.to_csv(path, layout=layout)
        result = ProteusVariable.from_csv(path)
    else:
        pytest.importorskip("pyarrow")
        path = tmp_path / "nested.parquet"
        nested.to_parquet(path, layout=layout)
        result = ProteusVariable.from_parquet(path)

    assert result.dim_name == "Region"
    assert result["UK"].dim_name == "LoB"
    np.testing.assert_array_equal(asnumpy(result["UK"]["Motor"].values), [10, 20, 30])
    np.testing.assert_array_equal(asnumpy(result["UK"]["Property"].values), [2.5, 3.5, 4.5])
    np.testing.assert_array_equal(asnumpy(result["US"]["Motor"].values), [40, 50, 60])


def test_auto_layout_prefers_multiindex_for_nested_stochastic(nested):
    frame = nested.to_dataframe()
    assert isinstance(frame.columns, pd.MultiIndex)
    assert frame.columns.names == ["Region", "LoB"]
    assert frame.index.name == "Simulation"
    assert frame["UK", "Motor"].tolist() == [10, 20, 30]


@pytest.mark.parametrize("format", ["csv", "parquet"])
def test_custom_column_names_for_nested_long(nested, tmp_path, format):
    kwargs = {
        "layout": "long",
        "dimension_columns": ["Market", "Class"],
        "simulation_column": "Run",
        "values_column": "Loss",
    }
    path = tmp_path / f"custom.{format}"
    if format == "csv":
        nested.to_csv(path, **kwargs)
        result = ProteusVariable.from_csv(
            path, dimension_columns=["Market", "Class"], simulation_column="Run", values_column="Loss"
        )
    else:
        pytest.importorskip("pyarrow")
        nested.to_parquet(path, **kwargs)
        result = ProteusVariable.from_parquet(
            path, dimension_columns=["Market", "Class"], simulation_column="Run", values_column="Loss"
        )
    assert result.dim_name == "Market"
    assert result["UK"].dim_name == "Class"
    np.testing.assert_array_equal(asnumpy(result["US"]["Property"].values), [6.5, 7.5, 8.5])


@pytest.mark.parametrize("format", ["csv", "parquet"])
def test_custom_multiindex_headers(nested, tmp_path, format):
    path = tmp_path / f"headers.{format}"
    kwargs = {
        "layout": "multiindex",
        "dimension_columns": ["Market", "Business"],
        "simulation_column": "Run",
    }
    if format == "csv":
        nested.to_csv(path, **kwargs)
        result = ProteusVariable.from_csv(path, simulation_column="Run")
    else:
        pytest.importorskip("pyarrow")
        nested.to_parquet(path, **kwargs)
        result = ProteusVariable.from_parquet(path, simulation_column="Run")
    assert result.dim_name == "Market"
    assert result["UK"].dim_name == "Business"
    np.testing.assert_array_equal(asnumpy(result["US"]["Property"].values), [6.5, 7.5, 8.5])


@pytest.mark.parametrize("format", ["csv", "parquet"])
def test_all_zero_claim_scenarios_roundtrip_without_n_sims(tmp_path, format):
    """Even if no events occur, the file has a row retaining n_sims."""
    variable = ProteusVariable("LoB", {"Cat": FreqSevSims([], np.array([], dtype=np.float64), 9)})
    path = tmp_path / f"empty_claims.{format}"
    if format == "csv":
        variable.to_csv(path)
        result = ProteusVariable.from_csv(path)
    else:
        pytest.importorskip("pyarrow")
        variable.to_parquet(path)
        result = ProteusVariable.from_parquet(path)
    assert result["Cat"].n_sims == 9
    assert len(result["Cat"].sim_index) == 0


@pytest.mark.parametrize("format", ["csv", "parquet"])
def test_trailing_zero_claim_simulations_roundtrip_without_n_sims(tmp_path, format):
    variable = ProteusVariable("LoB", {"Cat": FreqSevSims([0, 2, 0], [10.0, 20.0, 30.0], 12)})
    path = tmp_path / f"claims.{format}"
    if format == "csv":
        variable.to_csv(path)
        result = ProteusVariable.from_csv(path)["Cat"]
    else:
        pytest.importorskip("pyarrow")
        variable.to_parquet(path)
        result = ProteusVariable.from_parquet(path)["Cat"]
    assert result.n_sims == 12
    np.testing.assert_array_equal(asnumpy(result.sim_index), [0, 2, 0])
    np.testing.assert_array_equal(asnumpy(result.values), [10.0, 20.0, 30.0])


def test_configurable_external_claim_columns():
    frame = pd.DataFrame(
        {
            "Portfolio": ["Property", "Property", "Property"],
            "Run": [2, 0, 2],
            "Claim": [1, 2, 3],
            "Severity": [50.0, 25.0, 75.0],
        }
    )
    result = ProteusVariable.from_dataframe(
        frame,
        dimension_columns=["Portfolio"],
        simulation_column="Run",
        event_column="Claim",
        values_column="Severity",
        n_sims=8,
    )["Property"]
    assert isinstance(result, FreqSevSims)
    assert result.n_sims == 8
    np.testing.assert_array_equal(asnumpy(result.sim_index), [2, 0, 2])


def test_custom_external_wide_simulation_column():
    frame = pd.DataFrame({"Run": [11, 10], "A": [5, 4], "B": [7, 6]})
    result = ProteusVariable.from_dataframe(frame, simulation_column="Run", dim_name="LoB")
    assert result.dim_name == "LoB"
    np.testing.assert_array_equal(asnumpy(result["A"].values), [4, 5])


def test_multiindex_external_dataframe(nested):
    frame = nested.to_dataframe(layout="multiindex")
    renamed = frame.rename_axis(index="Run")
    result = ProteusVariable.from_dataframe(renamed, simulation_column="Run")
    np.testing.assert_array_equal(asnumpy(result["UK"]["Motor"].values), [10, 20, 30])


def test_three_levels_nested_roundtrip_csv(tmp_path):
    model = ProteusVariable(
        "Country",
        {
            "GB": ProteusVariable(
                "LoB",
                {"Motor": ProteusVariable("Coverage", {"Comp": StochasticScalar([7.0, 8.0])})},
            )
        },
    )
    path = tmp_path / "three.csv"
    model.to_csv(path)
    restored = ProteusVariable.from_csv(path)
    assert restored["GB"]["Motor"].dim_name == "Coverage"
    np.testing.assert_array_equal(asnumpy(restored["GB"]["Motor"]["Comp"].values), [7, 8])


def test_irregular_nested_variable_auto_uses_long(tmp_path):
    model = ProteusVariable(
        "Area",
        {
            "UK": ProteusVariable("LoB", {"Motor": StochasticScalar([1.0, 2.0])}),
            "France": ProteusVariable("LoB", {"Property": StochasticScalar([3.0, 4.0])}),
        },
    )
    # Uneven branches can still use MultiIndex as long as leaf depths match.
    assert isinstance(model.to_dataframe().columns, pd.MultiIndex)
    path = tmp_path / "irregular.csv"
    model.to_csv(path, layout="long")
    restored = ProteusVariable.from_csv(path)
    np.testing.assert_array_equal(asnumpy(restored["France"]["Property"].values), [3.0, 4.0])


def test_exported_zero_sim_stochastic_can_be_imported(tmp_path):
    model = ProteusVariable("LoB", {"Empty": StochasticScalar(np.array([], dtype=float))})
    path = tmp_path / "empty.csv"
    model.to_csv(path, layout="long")
    result = ProteusVariable.from_csv(path)["Empty"]
    assert result.n_sims == 0
    assert len(result.values) == 0


def test_reject_conflicting_simulation_counts():
    frame = pd.DataFrame(
        {
            "LoB": ["Cat", "Cat"],
            "Simulation": [0, 0],
            "Event": [0, 1],
            "value": [100.0, 200.0],
            "Kind": ["frequency_severity", "frequency_severity"],
            "N Simulations": [10, 11],
        }
    )
    with pytest.raises(ValueError, match="Conflicting simulation counts"):
        ProteusVariable.from_dataframe(frame)


def test_external_claim_ordinals_can_restart_for_each_simulation():
    frame = pd.DataFrame(
        {
            "LoB": ["Cat", "Cat", "Cat", "Cat"],
            "Simulation": [1, 0, 1, 0],
            "Event": [1, 0, 0, 1],
            "value": [40.0, 10.0, 30.0, 20.0],
        }
    )
    result = ProteusVariable.from_dataframe(frame, n_sims=4)["Cat"]
    assert result.n_sims == 4
    np.testing.assert_array_equal(asnumpy(result.sim_index), [0, 0, 1, 1])
    np.testing.assert_array_equal(asnumpy(result.values), [10.0, 20.0, 30.0, 40.0])

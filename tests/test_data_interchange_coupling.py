"""Imported joint simulation tables should preserve scenario alignment.

A table's scenario IDs identify a single joint sample; coupling keeps all
related outcomes together under subsequent scenario reordering. Independent
marginal samples can be imported with couple=False.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from pal import FreqSevSims, ProteusVariable, StochasticScalar
from pal._maths import asnumpy, xp
from pal.serialization import load, save


@pytest.mark.parametrize("layout", ["wide", "long", "multiindex"])
@pytest.mark.parametrize("format", ["dataframe", "csv", "parquet"])
def test_joint_table_coupled_by_default(tmp_path, layout, format):
    if format == "parquet":
        pytest.importorskip("pyarrow")
    if layout == "multiindex":
        original = ProteusVariable(
            "Country",
            {
                "UK": ProteusVariable(
                    "LoB",
                    {
                        "Motor": StochasticScalar([2.0, 6.0, 4.0]),
                        "Property": StochasticScalar([10.0, 30.0, 20.0]),
                    },
                )
            },
        )
    else:
        original = ProteusVariable(
            "LoB",
            {"Motor": StochasticScalar([2.0, 6.0, 4.0]), "Property": StochasticScalar([10.0, 30.0, 20.0])},
        )

    if format == "dataframe":
        result = ProteusVariable.from_dataframe(original.to_dataframe(layout=layout))
    elif format == "csv":
        path = tmp_path / "losses.csv"
        original.to_csv(path, layout=layout)
        result = ProteusVariable.from_csv(path)
    else:
        path = tmp_path / "losses.parquet"
        original.to_parquet(path, layout=layout)
        result = ProteusVariable.from_parquet(path)

    if layout == "multiindex":
        motor, property_loss = result["UK"]["Motor"], result["UK"]["Property"]
    else:
        motor, property_loss = result["Motor"], result["Property"]
    assert motor.coupled_variable_group is property_loss.coupled_variable_group
    assert isinstance(motor.values, xp.ndarray)
    np.testing.assert_array_equal(asnumpy(motor.values), [2.0, 6.0, 4.0])
    np.testing.assert_array_equal(asnumpy(property_loss.values), [10.0, 30.0, 20.0])


@pytest.mark.parametrize("format", ["dataframe", "csv", "parquet"])
def test_coupling_opt_out_for_independent_marginals(tmp_path, format):
    if format == "parquet":
        pytest.importorskip("pyarrow")
    original = ProteusVariable(
        "LoB",
        {"Motor": StochasticScalar([2, 3, 4]), "Property": StochasticScalar([5, 6, 7])},
    )
    if format == "dataframe":
        result = ProteusVariable.from_dataframe(original.to_dataframe(), couple=False)
    elif format == "csv":
        path = tmp_path / "marginals.csv"
        original.to_csv(path)
        result = ProteusVariable.from_csv(path, couple=False)
    else:
        path = tmp_path / "marginals.parquet"
        original.to_parquet(path)
        result = ProteusVariable.from_parquet(path, couple=False)
    assert result["Motor"].coupled_variable_group is not result["Property"].coupled_variable_group


def test_long_mixed_claims_and_annual_losses_share_joint_scenarios():
    model = ProteusVariable(
        "LoB",
        {
            "Annual": StochasticScalar([1.0, 3.0, 2.0, 4.0]),
            "Claims": FreqSevSims([2, 0, 2], [10.0, 20.0, 30.0], 4),
        },
    )
    restored = ProteusVariable.from_dataframe(model.to_dataframe(layout="long"))
    a, b = restored["Annual"], restored["Claims"]
    assert a.coupled_variable_group is b.coupled_variable_group
    assert b.n_sims == 4
    np.testing.assert_array_equal(asnumpy(b.sim_index), [2, 0, 2])
    assert isinstance(b.sim_index, xp.ndarray)

    separate = ProteusVariable.from_dataframe(model.to_dataframe(layout="long"), couple=False)
    assert separate["Annual"].coupled_variable_group is not separate["Claims"].coupled_variable_group


def test_separate_imports_do_not_couple_across_data_sources():
    frame = pd.DataFrame({"Scenario": [0, 1, 2], "Motor": [1.0, 2.0, 3.0]})
    first = ProteusVariable.from_dataframe(frame, simulation_column="Scenario")
    second = ProteusVariable.from_dataframe(frame, simulation_column="Scenario")
    assert first["Motor"].coupled_variable_group is not second["Motor"].coupled_variable_group


def test_snapshot_restores_original_group_boundaries(tmp_path):
    source = StochasticScalar([1.0, 2.0])
    linked = source * 3
    independent = StochasticScalar([4.0, 5.0])
    variable = ProteusVariable("Measure", {"Source": source, "Linked": linked, "Other": independent})
    path = tmp_path / "separate-groups.csv"
    save(variable, path)
    restored = load(path)
    assert restored["Source"].coupled_variable_group is restored["Linked"].coupled_variable_group
    assert restored["Source"].coupled_variable_group is not restored["Other"].coupled_variable_group


def test_database_roundtrip_through_pandas_retains_joint_scenario_axis():
    original = ProteusVariable(
        "Market",
        {
            "UK": ProteusVariable(
                "LoB",
                {
                    "Motor": StochasticScalar([10.0, 20.0, 30.0]),
                    "Property": StochasticScalar([5.0, 7.0, 9.0]),
                },
            )
        },
    )
    connection = sqlite3.connect(":memory:")
    try:
        original.to_dataframe(layout="long").to_sql("annual_losses", connection, if_exists="replace", index=False)
        frame = pd.read_sql_query(
            "SELECT * FROM annual_losses WHERE Simulation >= ?",
            connection,
            params=(0,),
        )
    finally:
        connection.close()

    restored = ProteusVariable.from_dataframe(
        frame,
        dimension_columns=["Market", "LoB"],
        simulation_column="Simulation",
        values_column="value",
    )
    motor, property_loss = restored["UK"]["Motor"], restored["UK"]["Property"]
    assert motor.coupled_variable_group is property_loss.coupled_variable_group
    np.testing.assert_array_equal(asnumpy(property_loss.values), [5.0, 7.0, 9.0])

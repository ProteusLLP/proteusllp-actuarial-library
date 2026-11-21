"""Parquet columnar codec regression tests and large-array invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pal import FreqSevSims, ProteusVariable, StochasticScalar


pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def test_columnar_parquet_avoids_long_dataframe(tmp_path, monkeypatch):
    """Writing numeric vectors must not generate one Python row per scenario."""
    rng = np.random.default_rng(123)
    samples = {f"LoB {i}": StochasticScalar(rng.normal(size=25000)) for i in range(12)}
    variable = ProteusVariable("class", samples)
    path = tmp_path / "large.parquet"

    def fail_dataframe():
        raise AssertionError("Compact Parquet must not create a long-form DataFrame")

    monkeypatch.setattr(variable, "to_dataframe", fail_dataframe)
    variable.to_parquet(path)

    metadata = pq.read_metadata(path)
    assert metadata.num_rows == 1
    assert b"pal.variable.columnar" in metadata.metadata
    restored = ProteusVariable.from_parquet(path)
    for label, value in samples.items():
        np.testing.assert_array_equal(restored[label].values, value.values)


def test_columnar_parquet_mixed_types_nested_and_empty_claims(tmp_path):
    """Data types and zero-claim scenario count must survive typed storage."""
    a = StochasticScalar(np.array([2**63 + 1, 2**64 - 1], dtype=np.uint64))
    b = StochasticScalar(np.array(["AA", "BBB"], dtype="U3"))
    c = StochasticScalar(np.array(["2026-01-01", "2026-12-31"], dtype="datetime64[D]"))
    event = FreqSevSims([1, 0, 1], [10.0, 20.0, 30.0], 4)
    original = ProteusVariable(
        "region",
        {
            "UK": ProteusVariable("type", {"capital": a, "rating": b, "event": event}),
            "US": ProteusVariable("type", {"date": c, "premium": 1_000_000}),
        },
    )
    path = tmp_path / "mixed.parquet"
    original.to_parquet(path)
    restored = ProteusVariable.from_parquet(path)
    assert restored.dim_name == "region"
    np.testing.assert_array_equal(restored["UK"]["capital"].values, a.values)
    assert restored["UK"]["capital"].values.dtype == np.dtype("uint64")
    np.testing.assert_array_equal(restored["UK"]["rating"].values, b.values)
    np.testing.assert_array_equal(restored["US"]["date"].values, c.values)
    assert restored["US"]["date"].values.dtype == c.values.dtype
    assert restored["US"]["premium"] == 1_000_000
    np.testing.assert_array_equal(restored["UK"]["event"].sim_index, [1, 0, 1])
    assert restored["UK"]["event"].n_sims == 4


def test_compact_parquet_empty_vectors(tmp_path):
    """Zero simulations and zero events have unambiguous typed columns."""
    original = ProteusVariable(
        "risk",
        {
            "zero draws": StochasticScalar(np.array([], dtype=np.float64)),
            "zero claims": FreqSevSims([], np.array([], dtype=np.int32), n_sims=3),
        },
    )
    path = tmp_path / "empty.parquet"
    original.to_parquet(path)
    restored = ProteusVariable.from_parquet(path)
    assert restored["zero draws"].n_sims == 0
    assert restored["zero draws"].values.dtype == np.float64
    assert restored["zero claims"].n_sims == 3
    assert restored["zero claims"].values.dtype == np.int32
    assert len(restored["zero claims"].sim_index) == 0


def test_existing_parquet_file_remains_readable(tmp_path):
    """Older PAL row-form files and ordinary Pandas tables are still accepted."""
    original = ProteusVariable("risk", {"A": StochasticScalar([10.0, 20.0])})
    old = tmp_path / "old.parquet"
    original.to_dataframe().to_parquet(old, index=False)
    restored = ProteusVariable.from_parquet(old)
    np.testing.assert_array_equal(restored["A"].values, [10.0, 20.0])
    conventional = tmp_path / "ordinary.parquet"
    pd.DataFrame({"Simulation": [0, 1], "risk": ["A", "A"], "loss": [5.0, 6.0]}).to_parquet(
        conventional, index=False
    )
    imported = ProteusVariable.from_parquet(conventional, "risk", "loss")
    np.testing.assert_array_equal(imported["A"].values, [5.0, 6.0])


def test_parquet_coupling_is_recreated_independent_of_original(tmp_path):
    a = StochasticScalar([1.0, 2.0])
    b = a * 2
    third = StochasticScalar([100.0, 200.0])
    original = ProteusVariable("line", {"A": a, "B": b, "C": third})
    path = tmp_path / "coupled.parquet"
    original.to_parquet(path)
    restored = ProteusVariable.from_parquet(path)
    assert restored["A"].coupled_variable_group is restored["B"].coupled_variable_group
    assert restored["C"].coupled_variable_group is not restored["A"].coupled_variable_group
    assert restored["A"].coupled_variable_group is not a.coupled_variable_group


def test_parquet_dtype_rejected_for_unsupported_objects(tmp_path):
    variable = ProteusVariable("line", {"A": {"invalid": 1}})
    with pytest.raises(TypeError, match="dtype"):
        variable.to_parquet(tmp_path / "invalid.parquet")

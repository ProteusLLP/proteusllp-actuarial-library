"""CPU and GPU residence checks for tabular interchange and PAL snapshots.

Pandas, CSV and PyArrow are host formats, but imported numeric PAL leaves
must use the configured NumPy or CuPy backend without changing semantics.
"""

from __future__ import annotations

import numpy as np
import pytest

from pal import FreqSevSims, ProteusVariable, StochasticScalar
from pal._maths import asnumpy, xp
from pal.serialization import load, save


@pytest.mark.parametrize("storage", ["dataframe", "csv", "parquet"])
@pytest.mark.parametrize("layout", ["multiindex", "long"])
def test_nested_numeric_exchange_uses_active_backend(tmp_path, storage, layout):
    """Exporting must stage on the host; importing restores backend arrays."""
    model = ProteusVariable(
        "Region",
        {
            "UK": ProteusVariable(
                "LoB",
                {
                    "Motor": StochasticScalar([10.0, 20.0, 30.0]),
                    "Property": StochasticScalar([40.0, 50.0, 60.0]),
                },
            ),
            "US": ProteusVariable("LoB", {"Motor": StochasticScalar([1.0, 2.0, 3.0])}),
        },
    )
    assert isinstance(model["UK"]["Motor"].values, xp.ndarray)

    if storage == "dataframe":
        frame = model.to_dataframe(layout=layout)
        assert isinstance(frame.to_numpy(), np.ndarray)
        result = ProteusVariable.from_dataframe(frame)
    elif storage == "csv":
        path = tmp_path / "nested.csv"
        model.to_csv(path, layout=layout)
        result = ProteusVariable.from_csv(path)
    else:
        pytest.importorskip("pyarrow")
        path = tmp_path / "nested.parquet"
        model.to_parquet(path, layout=layout)
        result = ProteusVariable.from_parquet(path)

    assert isinstance(result["UK"]["Motor"].values, xp.ndarray)
    assert isinstance(result["US"]["Motor"].values, xp.ndarray)
    np.testing.assert_array_equal(asnumpy(result["UK"]["Motor"].values), [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(asnumpy(result["US"]["Motor"].values), [1.0, 2.0, 3.0])
    assert isinstance((result["UK"]["Motor"] + 1).values, xp.ndarray)


@pytest.mark.parametrize("storage", ["csv", "parquet"])
def test_frequency_severity_exchange_uses_active_backend(tmp_path, storage):
    """Claim severities and simulation indices must return to the GPU."""
    claims = FreqSevSims([2, 0, 2], [100.0, 10.0, 200.0], 7)
    assert isinstance(claims.values, xp.ndarray)
    assert isinstance(claims.sim_index, xp.ndarray)
    model = ProteusVariable("LoB", {"Cat": claims})
    path = tmp_path / f"claims.{storage}"

    if storage == "csv":
        model.to_csv(path)
        result = ProteusVariable.from_csv(path)["Cat"]
    else:
        pytest.importorskip("pyarrow")
        model.to_parquet(path)
        result = ProteusVariable.from_parquet(path)["Cat"]

    assert isinstance(result.values, xp.ndarray)
    assert isinstance(result.sim_index, xp.ndarray)
    assert result.n_sims == 7
    np.testing.assert_array_equal(asnumpy(result.values), [100.0, 10.0, 200.0])
    np.testing.assert_array_equal(asnumpy(result.sim_index), [2, 0, 2])
    totals = result.aggregate()
    assert isinstance(totals.values, xp.ndarray)
    np.testing.assert_array_equal(asnumpy(totals.values), [10, 0, 300, 0, 0, 0, 0])


@pytest.mark.parametrize("storage", ["csv", "parquet"])
def test_empty_claims_import_backend(tmp_path, storage):
    """Metadata-only rows must restore an empty device array on the GPU."""
    model = ProteusVariable("LoB", {"Cat": FreqSevSims([], np.array([], dtype=float), 11)})
    path = tmp_path / f"empty.{storage}"
    if storage == "csv":
        model.to_csv(path)
        result = ProteusVariable.from_csv(path)["Cat"]
    else:
        pytest.importorskip("pyarrow")
        model.to_parquet(path)
        result = ProteusVariable.from_parquet(path)["Cat"]
    assert result.n_sims == 11
    assert isinstance(result.sim_index, xp.ndarray)
    assert isinstance(result.values, xp.ndarray)
    assert len(result.sim_index) == 0
    np.testing.assert_array_equal(asnumpy(result.aggregate().values), np.zeros(11))


@pytest.mark.parametrize("storage", ["csv", "parquet"])
def test_pal_snapshot_roundtrip_preserves_backend_and_coupling(tmp_path, storage):
    """Serialized stochastic leaves must return as GPU arrays, still coupled."""
    motor = StochasticScalar([10.0, 20.0, 30.0])
    linked = motor * 2
    claim = FreqSevSims([0, 1, 1], [2.0, 3.0, 4.0], 3)
    model = ProteusVariable(
        "Region",
        {"UK": ProteusVariable("LoB", {"Motor": motor, "Linked": linked, "Cat": claim})},
    )
    path = tmp_path / f"snapshot.{storage}"
    if storage == "parquet":
        pytest.importorskip("pyarrow")

    save(model, path)
    restored = load(path)["UK"]
    assert isinstance(restored["Motor"].values, xp.ndarray)
    assert isinstance(restored["Linked"].values, xp.ndarray)
    assert isinstance(restored["Cat"].sim_index, xp.ndarray)
    assert isinstance(restored["Cat"].values, xp.ndarray)
    assert restored["Motor"].coupled_variable_group is restored["Linked"].coupled_variable_group
    np.testing.assert_array_equal(asnumpy(restored["Motor"].values), [10, 20, 30])
    np.testing.assert_array_equal(asnumpy(restored["Cat"].aggregate().values), [2, 7, 0])


@pytest.mark.skipif(xp is np, reason="Requires GPU backend and CUDA")
def test_gpu_device_residence_is_not_numpy_host():
    """Fail if CuPy was configured but numeric results are secretly host arrays."""
    import cupy

    assert xp is cupy
    model = ProteusVariable("LoB", {"Motor": StochasticScalar([1.0, 2.0, 3.0])})
    assert type(model["Motor"].values) is cupy.ndarray
    imported = ProteusVariable.from_dataframe(model.to_dataframe())
    assert type(imported["Motor"].values) is cupy.ndarray
    assert type((imported["Motor"] + 1).values) is cupy.ndarray

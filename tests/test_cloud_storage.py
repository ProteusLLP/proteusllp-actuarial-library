"""Remote filesystem and file-like storage tests for PAL interchange.

The fsspec in-memory protocol runs in CI without real Azure or AWS credentials.
Provider-specific integrations are delegated to adlfs and s3fs.
"""

from __future__ import annotations

import io

import fsspec
import numpy as np
import pytest

from pal import FreqSevSims, ProteusVariable, StochasticScalar
from pal._maths import asnumpy, xp
from pal.serialization import load, save
from pal.storage import file_extension


def _portfolio() -> ProteusVariable:
    return ProteusVariable(
        "Region",
        {
            "UK": ProteusVariable(
                "LoB",
                {
                    "Motor": StochasticScalar([1.0, 2.0, 3.0]),
                    "Property": StochasticScalar([11.0, 12.0, 13.0]),
                },
            ),
            "US": ProteusVariable("LoB", {"Motor": StochasticScalar([5.0, 6.0, 7.0])}),
        },
    )


@pytest.mark.parametrize("format", ["csv", "parquet"])
@pytest.mark.parametrize("layout", ["multiindex", "long"])
def test_memory_filesystem_nested_interchange(tmp_path, format, layout):
    """Every remote file we export must be reimportable without losing dimensions."""
    if format == "parquet":
        pytest.importorskip("pyarrow")
    path = f"memory://pal-test-{tmp_path.name}/nested.{format}"
    variable = _portfolio()
    kwargs = {"layout": layout}
    if format == "csv":
        variable.to_csv(path, **kwargs)
        restored = ProteusVariable.from_csv(path)
    else:
        variable.to_parquet(path, **kwargs)
        restored = ProteusVariable.from_parquet(path)
    assert restored.dim_name == "Region"
    assert restored["UK"].dim_name == "LoB"
    assert isinstance(restored["UK"]["Motor"].values, xp.ndarray)
    np.testing.assert_array_equal(asnumpy(restored["UK"]["Motor"].values), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(asnumpy(restored["US"]["Motor"].values), [5.0, 6.0, 7.0])


@pytest.mark.parametrize("format", ["csv", "parquet"])
def test_memory_filesystem_claims_zero_simulations(tmp_path, format):
    if format == "parquet":
        pytest.importorskip("pyarrow")
    variable = ProteusVariable("LoB", {"Cat": FreqSevSims([2, 0, 2], [10.0, 25.0, 50.0], 15)})
    path = f"memory://pal-test-{tmp_path.name}/claims.{format}"
    if format == "csv":
        variable.to_csv(path)
        restored = ProteusVariable.from_csv(path)["Cat"]
    else:
        variable.to_parquet(path)
        restored = ProteusVariable.from_parquet(path)["Cat"]
    assert restored.n_sims == 15
    assert isinstance(restored.sim_index, xp.ndarray)
    np.testing.assert_array_equal(asnumpy(restored.sim_index), [2, 0, 2])
    np.testing.assert_array_equal(asnumpy(restored.values), [10.0, 25.0, 50.0])


@pytest.mark.parametrize("format", ["csv", "parquet"])
def test_remote_snapshots_preserve_coupling_and_claims(tmp_path, format):
    if format == "parquet":
        pytest.importorskip("pyarrow")
    a = StochasticScalar([1.0, 2.0, 3.0])
    b = a * 2
    original = ProteusVariable(
        "Region",
        {"UK": ProteusVariable("LoB", {
            "Motor": a,
            "Doubled": b,
            "Claims": FreqSevSims([0, 2], [9.0, 19.0], 3),
        })},
    )
    path = f"memory://pal-test-{tmp_path.name}/snapshot.{format}"
    save(original, path)
    result = load(path)["UK"]
    assert result["Motor"].coupled_variable_group is result["Doubled"].coupled_variable_group
    assert result["Claims"].n_sims == 3
    assert isinstance(result["Motor"].values, xp.ndarray)
    assert isinstance(result["Claims"].values, xp.ndarray)
    np.testing.assert_array_equal(asnumpy(result["Claims"].sim_index), [0, 2])
    np.testing.assert_array_equal(asnumpy(result["Doubled"].values), [2.0, 4.0, 6.0])


@pytest.mark.parametrize("format", ["csv", "parquet"])
def test_filelike_interchange_and_snapshots(format):
    if format == "parquet":
        pytest.importorskip("pyarrow")
    variable = _portfolio()
    stream = io.BytesIO()
    if format == "csv":
        variable.to_csv(stream, layout="multiindex")
    else:
        variable.to_parquet(stream, layout="multiindex")
    assert not stream.closed
    stream.seek(0)
    restored = ProteusVariable.from_csv(stream) if format == "csv" else ProteusVariable.from_parquet(stream)
    assert not stream.closed
    np.testing.assert_array_equal(asnumpy(restored["UK"]["Motor"].values), [1.0, 2.0, 3.0])

    stream = io.BytesIO()
    save(variable, stream, format=format)
    assert not stream.closed
    stream.seek(0)
    restored = load(stream, format=format)
    assert not stream.closed
    np.testing.assert_array_equal(asnumpy(restored["US"]["Motor"].values), [5.0, 6.0, 7.0])


@pytest.mark.parametrize("format", ["csv", "parquet"])
def test_in_memory_remote_storage_options_forwarded(tmp_path, format, monkeypatch):
    if format == "parquet":
        pytest.importorskip("pyarrow")
    original_open = fsspec.open
    opened = []

    def open_recording(path, mode="rb", **kwargs):
        opened.append((path, mode, kwargs))
        return original_open(path, mode=mode, **kwargs)

    monkeypatch.setattr(fsspec, "open", open_recording)
    path = f"memory://pal-test-{tmp_path.name}/mapped.{format}"
    options = {"account_name": "demo-cloud-account"}
    variable = ProteusVariable("LoB", {"Motor": StochasticScalar([1.0, 2.0])})
    if format == "csv":
        variable.to_csv(path, storage_options=options)
        restored = ProteusVariable.from_csv(path, storage_options=options)
    else:
        variable.to_parquet(path, storage_options=options)
        restored = ProteusVariable.from_parquet(path, storage_options=options)
    assert len(opened) == 2
    assert all(x[2] == options for x in opened)
    assert all(x[0] == path for x in opened)
    np.testing.assert_array_equal(asnumpy(restored["Motor"].values), [1.0, 2.0])


def test_stream_rejects_storage_options_and_requires_snapshot_format():
    variable = _portfolio()
    with pytest.raises(ValueError, match="storage_options"):
        variable.to_csv(io.BytesIO(), storage_options={"token": "unused"})
    with pytest.raises(ValueError, match="format"):
        save(variable, io.BytesIO())


def test_snapshot_filename_extension_ignores_query_string():
    assert file_extension("az://container/folder/snapshot.parquet?sig=abc") == ".parquet"
    assert file_extension("s3://bucket/snapshot.csv") == ".csv"
    assert file_extension(io.BytesIO(), format="parquet") == ".parquet"

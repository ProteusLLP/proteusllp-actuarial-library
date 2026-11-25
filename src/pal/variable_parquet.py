"""Compact typed Parquet I/O for large PAL variable hierarchies.

Array values are written as Arrow list columns without per-simulation JSON
encoding. Paths, data types and coupling groups live in schema metadata.
The original tabular CSV/Parquet representation remains readable.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import typing as t

import numpy as np
import pandas as pd

from .couplings import ProteusStochasticVariable, couple_stochastic_results
from .frequency_severity import FreqSevSims
from .stochastic_scalar import StochasticScalar
from .variable_io import _array, _dtype, _parse_path, from_dataframe
from .variables import ProteusVariable

_METADATA_KEY = b"pal.variable.columnar"
_SCHEMA = "pal-columnar-parquet-v1"


def _arrow_values(values: np.ndarray, pa: t.Any) -> t.Any:
    """Create an Arrow array directly from numeric storage where possible.

    Args:
        pa: PyArrow module used to construct or read Arrow columns.
        values: Values to encode or decode in their stored order.
    """
    if values.dtype.kind == "M":
        values = values.view(np.int64)
    elif values.dtype.kind in "US":
        return pa.array(values.tolist())
    return pa.array(values)


def _list_column(values: np.ndarray, pa: t.Any) -> t.Any:
    """Store one typed vector as a LargeList rather than repeated rows.

    Args:
        pa: PyArrow module used to construct or read Arrow columns.
        values: Values to encode or decode in their stored order.
    """
    offsets = pa.array([0, len(values)], type=pa.int64())
    return pa.LargeListArray.from_arrays(offsets, _arrow_values(values, pa))


def write_parquet(variable: ProteusVariable[t.Any], path: str | os.PathLike[str] | t.BinaryIO) -> None:
    """Write a compact, self-describing Parquet file using typed Arrow columns.

    Each leaf has a typed vector; claim-level leaves additionally store their
    simulation indices. This avoids materialising a long-form DataFrame.
    Install the optional parquet extra to supply PyArrow.

    Args:
        path: Ordered dimension keys identifying the variable leaf.
        variable: PAL variable to convert or serialize.
    """
    try:
        import pyarrow as pa  # pyright: ignore[reportMissingImports]  # Optional parquet extra
        import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]  # Optional parquet extra
    except ImportError as exc:
        raise ImportError("Compact Parquet I/O requires pyarrow; install the parquet extra") from exc

    columns: list[t.Any] = []
    names: list[str] = []
    leaves: list[dict[str, t.Any]] = []
    group_ids: dict[int, int] = {}

    def visit(value: t.Any, parts: list[tuple[str, str]]) -> None:
        """Traverse the variable hierarchy and serialize its leaves.

        Args:
            parts: Dimension path components identifying the nested variable.
            value: Value to encode, decode or evaluate.
        """
        if isinstance(value, ProteusVariable):
            if not value.values:
                raise ValueError("Empty ProteusVariable dimensions cannot be serialized")
            for label, child in value.values.items():
                visit(child, [*parts, (value.dim_name, str(label))])
            return

        group = -1
        if isinstance(value, ProteusStochasticVariable):
            identity = id(value.coupled_variable_group)
            if identity not in group_ids:
                group_ids[identity] = len(group_ids)
            group = group_ids[identity]

        indices: np.ndarray | None = None
        if isinstance(value, StochasticScalar):
            kind, n_sims, samples = "stochastic", int(value.n_sims), _array(value.values)
            if n_sims != len(samples):
                raise ValueError("StochasticScalar n_sims does not match its values")
        elif isinstance(value, FreqSevSims):
            kind, n_sims, samples = "freqsev", int(value.n_sims), _array(value.values)
            indices = _array(value.sim_index)
            if not np.issubdtype(indices.dtype, np.integer):
                raise TypeError("Frequency-severity simulation indices must be integers")
            if n_sims < 0 or len(indices) != len(samples) or np.any(indices < 0) or np.any(indices >= n_sims):
                raise ValueError("Frequency-severity simulation indices are out of range")
        else:
            kind, n_sims = "scalar", 1
            samples = (
                np.asarray([np.datetime64(value)]) if isinstance(value, (dt.date, dt.datetime)) else np.asarray([value])
            )

        if not parts:
            raise ValueError("PAL variable must have a dimension")
        dtype = _dtype(samples)
        index = len(leaves)
        names.append(f"values_{index}")
        columns.append(_list_column(samples, pa))
        if kind == "freqsev":
            if indices is None:
                raise ValueError("Missing frequency-severity simulation indices")
            names.append(f"sim_index_{index}")
            columns.append(_list_column(indices.astype(np.int64, copy=False), pa))
        leaves.append({"path": parts, "kind": kind, "dtype": dtype, "n_sims": n_sims, "group": group})

    visit(variable, [])
    metadata = {
        _METADATA_KEY: json.dumps(
            {"schema": _SCHEMA, "leaves": leaves}, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    }
    table = pa.Table.from_arrays(columns, names=names).replace_schema_metadata(metadata)
    pq.write_table(table, path, compression="zstd")


def _read_column(table: t.Any, name: str, dtype: str, pa: t.Any) -> np.ndarray:
    """Read and decode the stored Arrow column.

    Args:
        dtype: Data type of the stored values.
        name: Name of the Arrow column.
        pa: PyArrow module used to construct or read Arrow columns.
        table: Arrow table containing serialized variable data.
    """
    if name not in table.column_names:
        raise ValueError(f"Missing PAL column {name!r}")
    column = table.column(name)
    if len(column) != 1 or not pa.types.is_large_list(column.type):
        raise ValueError(f"Invalid PAL column {name!r}")
    values = column[0].values.to_numpy(zero_copy_only=False)
    try:
        data_type = np.dtype(dtype)
        if data_type.kind == "M":
            return np.asarray(values, dtype=np.int64).view(data_type)
        if data_type.kind not in "biufUS":
            raise ValueError(f"Unsupported PAL dtype: {dtype}")
        return np.asarray(values, dtype=data_type)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid values in PAL column {name!r}") from exc


def _rebuild(leaves: list[tuple[tuple[tuple[str, str], ...], t.Any]], level: int) -> ProteusVariable[t.Any]:
    """Reconstruct the variable hierarchy from its serialized leaves.

    Args:
        leaves: Serialized leaf variables and their dimension paths.
        level: Current depth in the nested variable hierarchy.
    """
    dimensions = {path[level][0] for path, _ in leaves if len(path) > level}
    if len(dimensions) != 1 or any(len(path) <= level for path, _ in leaves):
        raise ValueError("Inconsistent PAL variable dimension structure")
    dim_name = dimensions.pop()
    by_label: dict[str, list[tuple[tuple[tuple[str, str], ...], t.Any]]] = {}
    for path, value in leaves:
        by_label.setdefault(path[level][1], []).append((path, value))
    values: dict[str, t.Any] = {}
    for label, entries in by_label.items():
        at_leaf = [len(path) == level + 1 for path, _ in entries]
        if all(at_leaf) and len(entries) == 1:
            values[label] = entries[0][1]
        elif any(at_leaf):
            raise ValueError("PAL variable path is both a leaf and a dimension")
        else:
            values[label] = _rebuild(entries, level + 1)
    return ProteusVariable(dim_name, values)


def read_parquet(
    path: str | os.PathLike[str] | t.BinaryIO,
    dim_name: str | None = None,
    values_column: str = "value",
    simulation_column: str | None = "Simulation",
) -> ProteusVariable[t.Any]:
    """Read compact PAL Parquet or any earlier PAL/conventional Parquet table.

    Args:
        dim_name: Name of the dimension indexing the restored entries.
        path: Ordered dimension keys identifying the variable leaf.
        simulation_column: Name of the column containing simulation indices.
        values_column: Name of the column containing variable values.
    """
    try:
        import pyarrow as pa  # pyright: ignore[reportMissingImports]  # Optional parquet extra
        import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]  # Optional parquet extra
    except ImportError:
        return from_dataframe(pd.read_parquet(path), dim_name, values_column, simulation_column)

    # One seekable handle works for local and remote fsspec files. Reuse the
    # reader for the body to avoid reopening the remote object after metadata.
    reader = pq.ParquetFile(path)
    metadata = reader.metadata.metadata or {}
    if _METADATA_KEY not in metadata:
        return from_dataframe(reader.read().to_pandas(), dim_name, values_column, simulation_column)

    try:
        spec = json.loads(metadata[_METADATA_KEY])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid PAL Parquet metadata") from exc
    if not isinstance(spec, dict) or spec.get("schema") != _SCHEMA or not isinstance(spec.get("leaves"), list):
        raise ValueError("Unsupported PAL Parquet schema")
    if not spec["leaves"]:
        raise ValueError("PAL Parquet file has no leaves")

    table = reader.read()
    leaves: list[tuple[tuple[tuple[str, str], ...], t.Any]] = []
    groups: dict[int, list[ProteusStochasticVariable]] = {}
    for index, info in enumerate(spec["leaves"]):
        if not isinstance(info, dict):
            raise ValueError("Invalid PAL Parquet leaf metadata")
        leaf_path = _parse_path(json.dumps(info["path"]))
        kind = info["kind"]
        dtype = info["dtype"]
        n_sims = info["n_sims"]
        group_id = info["group"]
        if not isinstance(n_sims, int) or n_sims < 0 or not isinstance(group_id, int):
            raise ValueError("Invalid PAL simulation or coupling metadata")
        values = _read_column(table, f"values_{index}", dtype, pa)
        if kind == "scalar":
            if len(values) != 1 or n_sims != 1 or group_id != -1:
                raise ValueError("Invalid deterministic PAL leaf")
            scalar = values[0]
            leaf: t.Any = scalar if isinstance(scalar, np.datetime64) else scalar.item()
        elif kind == "stochastic":
            if len(values) != n_sims or group_id < 0:
                raise ValueError("Invalid stochastic PAL leaf")
            leaf = StochasticScalar(values)
        elif kind == "freqsev":
            indices = _read_column(table, f"sim_index_{index}", "<i8", pa)
            if group_id < 0 or len(indices) != len(values) or np.any(indices < 0) or np.any(indices >= n_sims):
                raise ValueError("Invalid frequency-severity PAL leaf")
            leaf = FreqSevSims(indices, values, n_sims)
        else:
            raise ValueError(f"Unknown PAL leaf type: {kind!r}")
        if isinstance(leaf, ProteusStochasticVariable):
            groups.setdefault(group_id, []).append(leaf)
        leaves.append((leaf_path, leaf))

    for members in groups.values():
        if len(members) > 1:
            couple_stochastic_results(members)
    return _rebuild(leaves, 0)

"""Tabular import and export for PAL variable hierarchies.

The versioned long-form schema is shared by DataFrame, CSV and Parquet.
It records leaf types, data types, simulation identities and claim-event order.
Coupling relationships among serialized leaves are reconstructed when loading,
but no connection is retained to objects outside the serialized hierarchy.
"""

from __future__ import annotations

import datetime as dt
import json
import typing as t

import numpy as np
import pandas as pd

from ._maths import asnumpy
from .couplings import ProteusStochasticVariable, couple_stochastic_results
from .frequency_severity import FreqSevSims
from .stochastic_scalar import StochasticScalar
from .variables import ProteusVariable

_SCHEMA = "pal-variable-v1"
_PATH = "_pal_path"
_KIND = "_pal_kind"
_DTYPE = "_pal_dtype"
_N_SIMS = "_pal_n_sims"
_GROUP = "_pal_group"
_VERSION = "_pal_schema"
_SIM = "Simulation"
_EVENT = "Event"
_VALUE = "value"
_COLUMNS = [_VERSION, _PATH, _KIND, _DTYPE, _N_SIMS, _GROUP, _SIM, _EVENT, _VALUE]


def _array(values: t.Any) -> np.ndarray:
    """Copy values to a host array for disk serialization."""
    return np.asarray(asnumpy(values))


def _dtype(values: np.ndarray) -> str:
    if values.dtype.kind not in "biufUSM":
        raise TypeError(f"Cannot serialize values with dtype {values.dtype!s}")
    return values.dtype.str


def _encode(value: t.Any) -> str:
    if isinstance(value, np.datetime64):
        value = str(value)
    elif isinstance(value, np.generic):
        value = value.item()
    elif isinstance(value, (dt.date, dt.datetime)):
        value = value.isoformat()
    return json.dumps(value, ensure_ascii=False, allow_nan=True)


def _row(
    path: str, kind: str, dtype: str, n_sims: int, group: int, simulation: int, event: int, value: str
) -> tuple[str, str, str, str, int, int, int, int, str]:
    return (_SCHEMA, path, kind, dtype, n_sims, group, simulation, event, value)


def to_dataframe(variable: ProteusVariable[t.Any]) -> pd.DataFrame:
    """Convert a variable hierarchy to a self-describing, long-form DataFrame.

    The returned DataFrame can be saved as CSV or Parquet without an external
    metadata file. Events are numbered in their original order; a row with
    Event=-1 and an empty value represents a simulation with no claims.
    """
    rows: list[tuple[str, str, str, str, int, int, int, int, str]] = []
    group_ids: dict[int, int] = {}

    def visit(value: t.Any, path: list[tuple[str, str]]) -> None:
        if isinstance(value, ProteusVariable):
            if not value.values:
                raise ValueError("Empty ProteusVariable dimensions cannot be serialized")
            for label, child in value.values.items():
                visit(child, [*path, (value.dim_name, str(label))])
            return

        encoded_path = json.dumps(path, ensure_ascii=False, separators=(",", ":"))
        group = -1
        if isinstance(value, ProteusStochasticVariable):
            identity = id(value.coupled_variable_group)
            if identity not in group_ids:
                group_ids[identity] = len(group_ids)
            group = group_ids[identity]
        if isinstance(value, StochasticScalar):
            samples = _array(value.values)
            dtype = _dtype(samples)
            for i, sample in enumerate(samples):
                rows.append(_row(encoded_path, "stochastic", dtype, value.n_sims, group, i, -1, _encode(sample)))
            if value.n_sims == 0:
                rows.append(_row(encoded_path, "stochastic", dtype, 0, group, -1, -1, ""))
        elif isinstance(value, FreqSevSims):
            samples = _array(value.values)
            indices = _array(value.sim_index)
            dtype = _dtype(samples)
            n_sims = int(value.n_sims)
            if n_sims < 0 or np.any(indices < 0) or np.any(indices >= n_sims):
                raise ValueError("Frequency-severity simulation indices are out of range")
            if not np.issubdtype(indices.dtype, np.integer):
                raise TypeError("Frequency-severity simulation indices must be integers")
            for event, (simulation, sample) in enumerate(zip(indices, samples)):
                rows.append(
                    _row(encoded_path, "freqsev", dtype, n_sims, group, int(simulation), event, _encode(sample))
                )
            populated = set(indices.tolist())
            for simulation in range(n_sims):
                if simulation not in populated:
                    rows.append(_row(encoded_path, "freqsev", dtype, n_sims, group, simulation, -1, ""))
            if n_sims == 0:
                rows.append(_row(encoded_path, "freqsev", dtype, 0, group, -1, -1, ""))
        else:
            if isinstance(value, (dt.date, dt.datetime)):
                sample = np.asarray([np.datetime64(value)])
            else:
                sample = np.asarray([value])
            dtype = _dtype(sample)
            rows.append(_row(encoded_path, "scalar", dtype, 1, -1, -1, -1, _encode(sample[0])))

    visit(variable, [])
    return pd.DataFrame.from_records(rows, columns=_COLUMNS)


def _parse_path(value: t.Any) -> tuple[tuple[str, str], ...]:
    try:
        path = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid PAL variable path") from exc
    if not isinstance(path, list) or not path:
        raise ValueError("A PAL variable path must contain at least one dimension")
    pairs: list[tuple[str, str]] = []
    for pair in path:
        if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(x, str) for x in pair):
            raise ValueError("Invalid dimension name or label in PAL variable path")
        pairs.append((pair[0], pair[1]))
    if len(set(name for name, _ in pairs)) != len(pairs):
        raise ValueError("Repeated dimension name in PAL variable path")
    return tuple(pairs)


def _decode(values: t.Iterable[t.Any], dtype: str) -> np.ndarray:
    try:
        parsed = [json.loads(value) for value in values]
        result = np.asarray(parsed, dtype=np.dtype(dtype))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot decode PAL values as {dtype}") from exc
    if result.dtype.kind not in "biufUSM":
        raise ValueError(f"Unsupported PAL value dtype: {dtype}")
    return result


def _unique(frame: pd.DataFrame, column: str) -> t.Any:
    values = frame[column].unique()
    if len(values) != 1:
        raise ValueError(f"PAL leaf has inconsistent {column}")
    return values[0]


def _restore_leaf(frame: pd.DataFrame) -> t.Any:
    kind = _unique(frame, _KIND)
    dtype = str(_unique(frame, _DTYPE))
    n_sims = int(_unique(frame, _N_SIMS))
    if n_sims < 0:
        raise ValueError("Number of simulations cannot be negative")
    sim = frame[_SIM].astype(np.int64).to_numpy()
    event = frame[_EVENT].astype(np.int64).to_numpy()

    if kind == "scalar":
        if len(frame) != 1 or sim[0] != -1 or event[0] != -1 or n_sims != 1:
            raise ValueError("Invalid deterministic PAL leaf")
        value = _decode(frame[_VALUE], dtype)[0]
        return value if isinstance(value, np.datetime64) else value.item()

    if kind == "stochastic":
        present = sim >= 0
        if (not np.all(event == -1)) or (
            not np.array_equal(np.sort(sim[present]), np.arange(n_sims))
        ) or (np.any(~present) and (n_sims != 0 or len(frame) != 1)):
            raise ValueError("Stochastic simulations must appear exactly once, indexed from zero")
        ordered = frame.iloc[np.argsort(sim)] if n_sims else frame.iloc[0:0]
        return StochasticScalar(_decode(ordered[_VALUE], dtype))

    if kind == "freqsev":
        if np.any(sim < -1) or np.any(sim >= n_sims):
            raise ValueError("Frequency-severity simulation index is out of range")
        actual = event >= 0
        if not np.array_equal(np.sort(event[actual]), np.arange(np.sum(actual))):
            raise ValueError("Frequency-severity event IDs must be unique and consecutive")
        missing = frame.loc[~actual]
        if not (missing[_EVENT] == -1).all():
            raise ValueError("Invalid frequency-severity empty-event marker")
        ordered = frame.loc[actual].iloc[np.argsort(event[actual])]
        indices = ordered[_SIM].astype(np.int64).to_numpy()
        if np.any(indices < 0):
            raise ValueError("Claim events must have a valid simulation index")
        empty_sims = set(range(n_sims)) - set(indices.tolist())
        marked = missing[_SIM].astype(np.int64).tolist()
        if n_sims == 0:
            if marked != [-1]:
                raise ValueError("Empty frequency-severity variable needs one marker")
        elif set(marked) != empty_sims or len(marked) != len(empty_sims):
            raise ValueError("Empty-claim simulations must be represented exactly once")
        return FreqSevSims(indices, _decode(ordered[_VALUE], dtype), n_sims)

    raise ValueError(f"Unknown PAL leaf type: {kind!r}")


def _from_pal_frame(frame: pd.DataFrame) -> ProteusVariable[t.Any]:
    if frame.empty or not set(_COLUMNS).issubset(frame.columns):
        raise ValueError("Missing PAL serialization columns")
    if not (frame[_VERSION] == _SCHEMA).all():
        raise ValueError("Unsupported PAL variable schema version")

    leaves: list[tuple[tuple[tuple[str, str], ...], t.Any]] = []
    groups: dict[int, list[ProteusStochasticVariable]] = {}
    for path, group in frame.groupby(_PATH, sort=False):
        value = _restore_leaf(group)
        group_id = int(_unique(group, _GROUP))
        if isinstance(value, ProteusStochasticVariable):
            if group_id < 0:
                raise ValueError("Missing coupling-group marker for stochastic leaf")
            groups.setdefault(group_id, []).append(value)
        elif group_id != -1:
            raise ValueError("Deterministic leaves cannot belong to a coupling group")
        leaves.append((_parse_path(path), value))
    for members in groups.values():
        if len(members) > 1:
            couple_stochastic_results(members)
    if not leaves:
        raise ValueError("A PAL variable must have at least one leaf")

    def build(items: list[tuple[tuple[tuple[str, str], ...], t.Any]], level: int) -> ProteusVariable[t.Any]:
        dimensions = {path[level][0] for path, _ in items if len(path) > level}
        if len(dimensions) != 1 or any(len(path) <= level for path, _ in items):
            raise ValueError("Inconsistent PAL variable dimension structure")
        dim_name = dimensions.pop()
        by_label: dict[str, list[tuple[tuple[tuple[str, str], ...], t.Any]]] = {}
        for path, value in items:
            by_label.setdefault(path[level][1], []).append((path, value))
        values: dict[str, t.Any] = {}
        for label, entries in by_label.items():
            at_leaf = [len(path) == level + 1 for path, _ in entries]
            if all(at_leaf) and len(entries) == 1:
                values[label] = entries[0][1]
            elif any(at_leaf):
                raise ValueError("PAL variable path is both a leaf and a dimension")
            else:
                values[label] = build(entries, level + 1)
        return ProteusVariable(dim_name, values)

    return build(leaves, 0)


def from_dataframe(
    frame: pd.DataFrame,
    dim_name: str | None = None,
    values_column: str = "value",
    simulation_column: str | None = "Simulation",
) -> ProteusVariable[t.Any]:
    """Load a PAL export, or import a conventional one-dimensional table.

    For ordinary tables, provide the dimension and value column names. Supply
    simulation_column=None to import deterministic values. Simulation IDs may
    start at any integer, but all dimension labels must use the same IDs.
    """
    if _VERSION in frame.columns:
        return _from_pal_frame(frame)
    if dim_name is None:
        raise ValueError("Provide dim_name to read a non-PAL DataFrame")
    required = {dim_name, values_column}
    if simulation_column is not None:
        required.add(simulation_column)
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    if frame.empty:
        raise ValueError("Cannot import an empty DataFrame")

    labels = frame[dim_name].drop_duplicates().tolist()
    if frame[dim_name].isna().any() or frame[values_column].isna().any():
        raise ValueError("Missing dimension labels or values")
    if simulation_column is None:
        if frame[dim_name].duplicated().any():
            raise ValueError("Duplicate dimension labels in deterministic table")
        return ProteusVariable(dim_name, {str(label): frame.loc[frame[dim_name] == label, values_column].iloc[0]
                                          for label in labels})

    if frame[[dim_name, simulation_column]].duplicated().any():
        raise ValueError("Duplicate simulation ID for a dimension label")
    if frame[simulation_column].isna().any():
        raise ValueError("Missing simulation IDs")
    sims = sorted(frame[simulation_column].unique().tolist())
    variables: dict[str, StochasticScalar] = {}
    for label in labels:
        subset = frame.loc[frame[dim_name] == label].set_index(simulation_column)
        if set(subset.index.tolist()) != set(sims):
            raise ValueError("All dimension labels must have the same simulation IDs")
        variables[str(label)] = StochasticScalar(subset.loc[sims, values_column].to_numpy())
    return ProteusVariable(dim_name, variables)

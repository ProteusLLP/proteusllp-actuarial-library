"""Interchange PAL variables with ordinary pandas, CSV and Parquet tables.

These are human-readable data extracts, not PAL object snapshots. Object
persistence, including coupling, is provided separately by pal.serialization.
"""

from __future__ import annotations

import datetime as dt
import typing as t
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ._maths import asnumpy
from .frequency_severity import FreqSevSims
from .stochastic_scalar import StochasticScalar
from .variables import ProteusVariable

_SIM = "Simulation"
_EVENT = "Event"
_VALUE = "value"
_KIND = "Kind"
_COUNT = "N Simulations"


def _leaves(
    value: t.Any,
    path: tuple[tuple[str, str], ...] = (),
) -> t.Iterator[tuple[tuple[tuple[str, str], ...], t.Any]]:
    if isinstance(value, ProteusVariable):
        for label, child in value.values.items():
            yield from _leaves(child, (*path, (value.dim_name, str(label))))
    else:
        yield path, value


def _dimension_names(
    leaves: list[tuple[tuple[tuple[str, str], ...], t.Any]],
    dimension_columns: Sequence[str] | None,
) -> list[str]:
    depth = max(len(path) for path, _ in leaves)
    if dimension_columns is not None:
        if len(dimension_columns) != depth:
            raise ValueError(f"Expected {depth} dimension column names, got {len(dimension_columns)}")
        names = list(dimension_columns)
    else:
        names = []
        for level in range(depth):
            found = {path[level][0] for path, _ in leaves if len(path) > level}
            if len(found) != 1:
                raise ValueError(
                    "Different dimension names occur at the same level; "
                    "pass dimension_columns to select a common column name"
                )
            names.append(next(iter(found)))
    if len(names) != len(set(names)):
        raise ValueError("Dimension column names must be distinct")
    return names


def to_dataframe(
    variable: ProteusVariable[t.Any],
    layout: str = "auto",
    *,
    dimension_columns: Sequence[str] | None = None,
    simulation_column: str = _SIM,
    event_column: str = _EVENT,
    values_column: str = _VALUE,
    kind_column: str = _KIND,
    n_sims_column: str = _COUNT,
) -> pd.DataFrame:
    """Export as a wide, nested MultiIndex, or long-form pandas DataFrame.

    Auto selects wide for 1D stochastic values, MultiIndex for regular nested
    stochastic values, and long for heterogeneous, deterministic or claim data.
    Long claim tables retain their simulation count, including zero-claim
    simulations, through an ordinary 'N Simulations' column.
    """
    if layout not in ("auto", "wide", "long", "multiindex"):
        raise ValueError("layout must be 'auto', 'wide', 'multiindex' or 'long'")
    leaves = list(_leaves(variable))
    if not leaves:
        raise ValueError("Cannot export an empty variable as a table")
    dim_names = _dimension_names(leaves, dimension_columns)
    reserved = {simulation_column, event_column, values_column, kind_column, n_sims_column}
    if len(reserved) != 5 or reserved.intersection(dim_names):
        raise ValueError("Simulation, event, value, kind, count and dimension columns must be distinct")

    homogeneous = (
        all(isinstance(value, StochasticScalar) for _, value in leaves)
        and len({int(value.n_sims) for _, value in leaves}) == 1
    )
    depths = {len(path) for path, _ in leaves}
    regular = homogeneous and len(depths) == 1
    depth = next(iter(depths)) if len(depths) == 1 else 0
    paths = [tuple(label for _, label in path) for path, _ in leaves]
    unique_paths = len(set(paths)) == len(paths)
    wide = regular and depth == 1 and unique_paths and not {p[0] for p in paths}.intersection(reserved)
    multiindex = regular and depth >= 2 and unique_paths

    if layout == "wide" and not wide:
        raise ValueError("Wide layout requires 1D StochasticScalar leaves with matching n_sims")
    if layout == "multiindex" and not multiindex:
        raise ValueError("MultiIndex layout requires regular nested StochasticScalar leaves")
    if layout == "wide" or (layout == "auto" and wide):
        data: dict[str, t.Any] = {simulation_column: np.arange(int(leaves[0][1].n_sims))}
        for path, value in leaves:
            data[path[0][1]] = asnumpy(value.values)
        return pd.DataFrame(data, copy=False)
    if layout == "multiindex" or (layout == "auto" and multiindex):
        values = {labels: asnumpy(value.values) for labels, (_, value) in zip(paths, leaves)}
        frame = pd.DataFrame(
            values,
            index=pd.RangeIndex(int(leaves[0][1].n_sims), name=simulation_column),
            copy=False,
        )
        frame.columns = pd.MultiIndex.from_tuples(list(values), names=dim_names)
        return frame

    include_events = any(isinstance(value, FreqSevSims) for _, value in leaves)
    include_metadata = include_events or any(
        isinstance(value, StochasticScalar) and int(value.n_sims) == 0 for _, value in leaves
    )
    frames: list[pd.DataFrame] = []
    for path, value in leaves:
        labels = {name: label for name, (_, label) in zip(dim_names, path)}
        if isinstance(value, StochasticScalar):
            vals = np.asarray(asnumpy(value.values))
            sims: t.Any = np.arange(len(vals), dtype=np.int64)
            events: t.Any = np.full(len(vals), pd.NA, dtype=object)
            kind, count = "stochastic", int(value.n_sims)
        elif isinstance(value, FreqSevSims):
            vals = np.asarray(asnumpy(value.values))
            sims = np.asarray(asnumpy(value.sim_index), dtype=np.int64)
            events = np.arange(len(vals), dtype=np.int64)
            kind, count = "frequency_severity", int(value.n_sims)
        else:
            vals = np.asarray([np.datetime64(value) if isinstance(value, (dt.date, dt.datetime)) else value])
            sims = np.array([None], dtype=object)
            events = np.array([None], dtype=object)
            kind, count = "deterministic", None

        # An empty event list must still record the existence and number of
        # simulations, otherwise export/import silently drops the variable.
        if len(vals) == 0:
            vals = np.asarray([None], dtype=object)
            sims = np.asarray([None], dtype=object)
            events = np.asarray([None], dtype=object)
        row: dict[str, t.Any] = dict(labels)
        row.update({simulation_column: sims, values_column: vals})
        if include_events:
            row[event_column] = events
        if include_metadata:
            row[kind_column] = kind
            row[n_sims_column] = count
        frames.append(pd.DataFrame(row, copy=False))

    columns = [
        *dim_names,
        simulation_column,
        *([event_column] if include_events else []),
        values_column,
        *([kind_column, n_sims_column] if include_metadata else []),
    ]
    return pd.concat(frames, ignore_index=True).reindex(columns=columns)


def _nested_variable(values: dict[tuple[str, ...], t.Any], dimension_columns: Sequence[str]) -> ProteusVariable[t.Any]:
    """Reconstruct a heterogeneous hierarchy from its ordered leaf paths."""

    def build(prefix: tuple[str, ...], level: int) -> ProteusVariable[t.Any]:
        children: dict[str, t.Any] = {}
        for path, leaf in values.items():
            if path[:level] != prefix:
                continue
            if len(path) <= level:
                raise ValueError("Dimension hierarchy contains an invalid path")
            label = path[level]
            if label in children:
                continue
            if len(path) == level + 1:
                if any(p[: level + 1] == (*prefix, label) and len(p) > level + 1 for p in values):
                    raise ValueError("A dimension label cannot be both a leaf and a nested variable")
                children[label] = leaf
            else:
                children[label] = build((*prefix, label), level + 1)
        return ProteusVariable(str(dimension_columns[level]), children)

    return build((), 0)


def _multiindex_import(
    frame: pd.DataFrame,
    simulation_column: str | None,
    dimension_columns: Sequence[str] | None,
) -> ProteusVariable[t.Any]:
    if simulation_column is None:
        raise ValueError("MultiIndex stochastic data requires a simulation index")
    if frame.index.hasnans or not frame.index.is_unique:
        raise ValueError("Simulation IDs must be present and unique")
    level_count = frame.columns.nlevels
    names = list(dimension_columns) if dimension_columns is not None else list(frame.columns.names)
    if len(names) != level_count:
        raise ValueError("dimension_columns must contain one name per header level")
    if any(name is None for name in names):
        raise ValueError("MultiIndex column levels must be named or supplied via dimension_columns")
    ordered = frame.sort_index()
    vals: dict[tuple[str, ...], t.Any] = {}
    for column in ordered.columns:
        path = tuple(str(label) for label in column)
        if path in vals:
            raise ValueError("Duplicate MultiIndex column paths")
        vals[path] = StochasticScalar(ordered[column].to_numpy(copy=False))
    if not vals:
        raise ValueError("MultiIndex table has no value columns")
    return _nested_variable(vals, [str(name) for name in names])


def from_dataframe(
    frame: pd.DataFrame,
    dim_name: str | None = None,
    values_column: str = _VALUE,
    simulation_column: str | None = _SIM,
    *,
    dimension_columns: Sequence[str] | None = None,
    event_column: str = _EVENT,
    kind_column: str = _KIND,
    n_sims_column: str = _COUNT,
    n_sims: int | None = None,
) -> ProteusVariable[t.Any]:
    """Import wide, long or MultiIndex data, with selectable column mappings.

    Dimension columns can describe multiple nested levels in long data. Event
    rows create FreqSevSims; the optional N Simulations column keeps zero-claim
    simulations. This is data interchange; coupling is not reconstructed.
    """
    if "_pal_schema" in frame or "_pal_path" in frame:
        raise ValueError("PAL snapshots must be loaded through pal.serialization.load()")
    if isinstance(frame.columns, pd.MultiIndex):
        return _multiindex_import(frame, simulation_column, dimension_columns)

    if values_column not in frame.columns:
        # A wide table has a separate stochastic variable in each value column.
        if simulation_column is None:
            if len(frame) != 1:
                raise ValueError("Wide deterministic tables must have exactly one row")
            return ProteusVariable(dim_name or "Variable", {str(c): frame[c].iloc[0] for c in frame})
        if simulation_column not in frame:
            raise ValueError(f"Missing simulation column {simulation_column!r}")
        if frame[simulation_column].isna().any() or frame[simulation_column].duplicated().any():
            raise ValueError("Simulation IDs must be present and unique in wide tables")
        labels = [c for c in frame if c != simulation_column]
        if not labels:
            raise ValueError("Wide table has no value columns")
        ordered = frame.sort_values(simulation_column)
        return ProteusVariable(
            dim_name or (dimension_columns[0] if dimension_columns else "Variable"),
            {str(c): StochasticScalar(ordered[c].to_numpy(copy=False)) for c in labels},
        )

    control = {values_column, simulation_column, event_column, kind_column, n_sims_column}
    dimensions = (
        list(dimension_columns) if dimension_columns is not None else [str(c) for c in frame if c not in control]
    )
    if dim_name is not None:
        if dimension_columns is not None and len(dimension_columns) > 1:
            raise ValueError("Use dimension_columns alone to select nested dimension names")
        dimensions = [dim_name]
    if not dimensions or len(dimensions) != len(set(dimensions)):
        raise ValueError("Provide distinct dimension_columns for long-form data")
    if any(c not in frame.columns for c in dimensions):
        raise ValueError("Missing requested dimension column")
    if any(c in control for c in dimensions):
        raise ValueError("Dimension and control columns must be distinct")
    if simulation_column is not None and simulation_column not in frame.columns:
        raise ValueError(f"Missing simulation column {simulation_column!r}")

    leaves: dict[tuple[str, ...], t.Any] = {}
    reference_sims: np.ndarray | None = None
    for raw_path, group in frame.groupby(dimensions, sort=False, dropna=False):
        path_values = raw_path if isinstance(raw_path, tuple) else (raw_path,)
        path: list[str] = []
        encountered_missing = False
        for label in path_values:
            if pd.isna(t.cast(t.Any, label)):
                encountered_missing = True
            elif encountered_missing:
                raise ValueError("Dimension paths must not contain gaps")
            else:
                path.append(str(label))
        if not path:
            raise ValueError("Dimension labels cannot all be missing")

        declared_kind: str | None = None
        if kind_column in group:
            kinds = group[kind_column].dropna().unique()
            if len(kinds) > 1:
                raise ValueError("Conflicting kinds for the same dimension path")
            if len(kinds):
                declared_kind = str(kinds[0])
        has_event = event_column in group and group[event_column].notna().any()
        has_simulation = simulation_column is not None and group[simulation_column].notna().any()
        kind = declared_kind or (
            "frequency_severity" if has_event else "stochastic" if has_simulation else "deterministic"
        )

        stored_n_sims: int | None = None
        if n_sims_column in group:
            counts = group[n_sims_column].dropna().unique()
            if len(counts) > 1:
                raise ValueError("Conflicting simulation counts for the same dimension path")
            if len(counts):
                stored_n_sims = int(counts[0])
        if kind == "frequency_severity":
            if simulation_column is None or event_column not in group:
                raise ValueError("Frequency-severity input requires simulation and event columns")
            claims = group.loc[group[event_column].notna()]
            if claims[[simulation_column, event_column]].duplicated().any():
                raise ValueError("Event IDs must be unique within each simulation")
            if claims[simulation_column].isna().any():
                raise ValueError("Claim rows must have a simulation ID")
            # Exported PAL event IDs give a global order. External data often
            # numbers claims separately within each simulation.
            sort_columns = (
                [simulation_column, event_column] if claims[event_column].duplicated().any() else [event_column]
            )
            ordered = claims.sort_values(sort_columns, kind="stable")
            indices = ordered[simulation_column].to_numpy(dtype=np.int64)
            if (indices < 0).any():
                raise ValueError("Simulation indices must be nonnegative")
            observed = int(indices.max()) + 1 if len(indices) else 0
            total = n_sims if n_sims is not None else (stored_n_sims if stored_n_sims is not None else observed)
            if total < observed:
                raise ValueError("n_sims cannot be smaller than the largest simulation index")
            vals = ordered[values_column].to_numpy(copy=False) if len(ordered) else np.array([], dtype=float)
            leaves[tuple(path)] = FreqSevSims(indices, vals, total)
        elif kind == "stochastic":
            if simulation_column is None:
                raise ValueError("Stochastic input requires a simulation column")
            samples = group.loc[group[simulation_column].notna()]
            if samples[simulation_column].duplicated().any():
                raise ValueError("Duplicate simulation ID for a dimension label")
            ordered = samples.sort_values(simulation_column)
            ids = ordered[simulation_column].to_numpy()
            if reference_sims is None:
                reference_sims = ids
            elif not np.array_equal(reference_sims, ids):
                raise ValueError("All dimension labels must have the same simulation IDs")
            if stored_n_sims is not None and len(ordered) != stored_n_sims:
                raise ValueError("Stochastic data does not match its stored simulation count")
            leaves[tuple(path)] = StochasticScalar(
                ordered[values_column].to_numpy(copy=False) if len(ordered) else np.array([], dtype=float)
            )
        elif kind == "deterministic":
            if len(group) != 1:
                raise ValueError("Deterministic dimension labels must be unique")
            leaves[tuple(path)] = group[values_column].iloc[0]
        else:
            raise ValueError(f"Unknown value kind {kind!r}")

    return _nested_variable(leaves, dimensions)

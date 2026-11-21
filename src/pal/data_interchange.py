"""CSV, Parquet and DataFrame interchange with ordinary tabular data.

Tables contain user-facing dimension labels, simulation IDs, claim-event
indices and values. These are data extracts, not lossless PAL object snapshots.
For preserving types, nesting and coupling, use pal.serialization instead.
"""

from __future__ import annotations

import datetime as dt
import typing as t

import numpy as np
import pandas as pd

from ._maths import asnumpy
from .frequency_severity import FreqSevSims
from .stochastic_scalar import StochasticScalar
from .variables import ProteusVariable

_SIM = "Simulation"
_EVENT = "Event"
_VALUE = "value"


def _leaves(
    value: t.Any,
    path: tuple[tuple[str, str], ...] = (),
) -> t.Iterator[tuple[tuple[tuple[str, str], ...], t.Any]]:
    if isinstance(value, ProteusVariable):
        for label, child in value.values.items():
            yield from _leaves(child, (*path, (value.dim_name, str(label))))
    else:
        yield path, value


def to_dataframe(variable: ProteusVariable[t.Any], layout: str = "auto") -> pd.DataFrame:
    """Create a conventional table for exchange with pandas or other systems.

    The default uses a wide table for a one-dimensional collection of equally
    sized StochasticScalars: Simulation, Motor, Property, etc. Otherwise it uses
    long-form dimension columns with Simulation, Event (if needed), and value.
    This interchange representation does not retain coupling metadata or
    distinguish trailing zero-claim simulations.
    """
    if layout not in ("auto", "wide", "long"):
        raise ValueError("layout must be 'auto', 'wide' or 'long'")
    leaves = list(_leaves(variable))
    if not leaves:
        raise ValueError("Cannot export an empty variable as a table")
    is_wide = (
        all(len(path) == 1 and isinstance(value, StochasticScalar) for path, value in leaves)
        and len({int(value.n_sims) for _, value in leaves}) == 1
        and len({path[0][1] for path, _ in leaves}) == len(leaves)
        and all(path[0][1] != _SIM for path, _ in leaves)
    )
    if layout == "wide" and not is_wide:
        raise ValueError("Wide layout requires one-dimensional StochasticScalar leaves with matching n_sims")
    if layout == "wide" or (layout == "auto" and is_wide):
        n_sims = int(leaves[0][1].n_sims)
        data: dict[str, t.Any] = {_SIM: np.arange(n_sims, dtype=np.int64)}
        for path, value in leaves:
            data[path[0][1]] = asnumpy(value.values)
        return pd.DataFrame(data, copy=False)

    dimensions = list(dict.fromkeys(name for path, _ in leaves for name, _ in path))
    reserved = {_SIM, _EVENT, _VALUE}
    if reserved.intersection(dimensions):
        raise ValueError("Dimension names Simulation, Event and value are reserved in long-form tables")
    include_events = any(isinstance(value, FreqSevSims) for _, value in leaves)
    frames: list[pd.DataFrame] = []
    for path, value in leaves:
        if isinstance(value, StochasticScalar):
            values = np.asarray(asnumpy(value.values))
            sims = np.arange(len(values), dtype=np.int64)
            events: t.Any = np.full(len(values), pd.NA, dtype=object)
        elif isinstance(value, FreqSevSims):
            values = np.asarray(asnumpy(value.values))
            sims = np.asarray(asnumpy(value.sim_index), dtype=np.int64)
            events = np.arange(len(values), dtype=np.int64)
        else:
            values = np.asarray([np.datetime64(value)]) if isinstance(value, (dt.date, dt.datetime)) else np.asarray([value])
            sims = np.array([None], dtype=object)
            events = np.array([None], dtype=object)
        data = {name: label for name, label in path}
        data.update({_SIM: sims, _VALUE: values})
        if include_events:
            data[_EVENT] = events
        frames.append(pd.DataFrame(data, copy=False))
    columns = [*dimensions, _SIM, *([_EVENT] if include_events else []), _VALUE]
    return pd.concat(frames, ignore_index=True).reindex(columns=columns)


def from_dataframe(
    frame: pd.DataFrame,
    dim_name: str | None = None,
    values_column: str = _VALUE,
    simulation_column: str | None = _SIM,
    *,
    n_sims: int | None = None,
) -> ProteusVariable[t.Any]:
    """Read a conventional wide table or a one-dimensional long table.

    For wide data, columns other than Simulation are separate stochastic
    variables. For long data, specify the label and value column names.
    An optional Event column identifies individual claims. Supply n_sims
    for event-list data if there are trailing zero-claim simulations.
    Data exchange does not preserve coupling groups.
    """
    if frame.empty:
        raise ValueError("Cannot import an empty table")
    if "_pal_schema" in frame or "_pal_path" in frame:
        raise ValueError("PAL object snapshots must be loaded with pal.serialization.load()")
    if values_column not in frame.columns:
        # Wide table: one column per variable, one row per simulation.
        if simulation_column is None:
            if len(frame) != 1:
                raise ValueError("Wide deterministic tables must have exactly one row")
            columns = list(frame.columns)
            return ProteusVariable(dim_name or "Variable", {str(key): frame[key].iloc[0] for key in columns})
        if simulation_column not in frame:
            raise ValueError(f"Missing simulation column {simulation_column!r}")
        if frame[simulation_column].isna().any() or frame[simulation_column].duplicated().any():
            raise ValueError("Simulation IDs must be present and unique in wide tables")
        labels = [key for key in frame if key != simulation_column]
        if not labels:
            raise ValueError("Wide table has no value columns")
        ordered = frame.sort_values(simulation_column)
        return ProteusVariable(
            dim_name or "Variable",
            {str(key): StochasticScalar(ordered[key].to_numpy(copy=False)) for key in labels},
        )

    if dim_name is None:
        candidates = [key for key in frame if key not in {values_column, simulation_column, _EVENT}]
        if len(candidates) != 1:
            raise ValueError("Provide dim_name to identify the dimension label column")
        dim_name = str(candidates[0])
    if dim_name not in frame:
        raise ValueError(f"Missing dimension column {dim_name!r}")
    if frame[dim_name].isna().any():
        raise ValueError("Dimension labels cannot be missing")
    if frame[values_column].isna().any():
        raise ValueError("Values cannot be missing in long-form imports")

    if simulation_column is None:
        if frame[dim_name].duplicated().any():
            raise ValueError("Duplicate dimension labels in deterministic table")
        return ProteusVariable(dim_name, {
            str(label): subset[values_column].iloc[0] for label, subset in frame.groupby(dim_name, sort=False)
        })
    if simulation_column not in frame:
        raise ValueError(f"Missing simulation column {simulation_column!r}")

    variables: dict[str, t.Any] = {}
    has_events = _EVENT in frame.columns and frame[_EVENT].notna().any()
    if has_events:
        valid_sims = frame[simulation_column].dropna()
        observed = int(valid_sims.max()) + 1 if len(valid_sims) else 0
        total_sims = observed if n_sims is None else n_sims
        if total_sims < observed:
            raise ValueError("n_sims cannot be smaller than the largest simulation index")
        for label, group in frame.groupby(dim_name, sort=False):
            if group[_EVENT].isna().any() or group[simulation_column].isna().any():
                raise ValueError("Mixed deterministic and event-list rows are not supported for import")
            event_ids = group[_EVENT].to_numpy()
            if len(np.unique(event_ids)) != len(event_ids):
                raise ValueError("Event IDs must be unique within each variable")
            ordered = group.sort_values(_EVENT)
            indices = ordered[simulation_column].to_numpy(dtype=np.int64)
            if (indices < 0).any():
                raise ValueError("Simulation indices must be nonnegative")
            variables[str(label)] = FreqSevSims(
                indices, ordered[values_column].to_numpy(copy=False), total_sims
            )
        return ProteusVariable(dim_name, variables)

    if frame[simulation_column].isna().any():
        raise ValueError("Simulation IDs cannot be missing in stochastic tables")
    if frame[[dim_name, simulation_column]].duplicated().any():
        raise ValueError("Duplicate simulation ID for a dimension label")
    sims = np.sort(frame[simulation_column].unique())
    for label, group in frame.groupby(dim_name, sort=False):
        ordered = group.sort_values(simulation_column)
        if not np.array_equal(ordered[simulation_column].to_numpy(), sims):
            raise ValueError("All dimension labels must have the same simulation IDs")
        variables[str(label)] = StochasticScalar(ordered[values_column].to_numpy(copy=False))
    return ProteusVariable(dim_name, variables)

"""Array handling for simulated values, including dates and category labels."""

from __future__ import annotations

import datetime as dt
import numbers
import typing as t

import numpy as np
import numpy.typing as npt

from ._maths import xp


def stochastic_array(values: t.Any) -> npt.NDArray[t.Any]:
    """Create a one-dimensional array without converting dates or labels to numbers."""
    if isinstance(values, np.ndarray):
        candidate = values
    elif isinstance(values, (list, tuple)) and values and isinstance(values[0], (str, bytes, dt.date, np.datetime64)):
        candidate = np.asarray(values)
    else:
        candidate = values

    dtype = getattr(candidate, "dtype", None)
    if dtype is not None and dtype.kind == "O":
        items = np.asarray(candidate).ravel().tolist()
        if all(isinstance(item, str) for item in items):
            return np.asarray(candidate, dtype=str)
        if all(isinstance(item, (dt.date, np.datetime64)) for item in items):
            units = [
                item.dtype
                if isinstance(item, np.datetime64)
                else np.dtype("datetime64[us]" if isinstance(item, dt.datetime) else "datetime64[D]")
                for item in items
            ]
            dtype = np.result_type(*units) if units else np.dtype("datetime64[D]")
            return np.asarray(candidate, dtype=dtype)
        if all(isinstance(item, numbers.Number) for item in items):
            numeric = np.asarray(items).reshape(np.asarray(candidate).shape)
            if numeric.dtype.kind in "biufc":
                return t.cast(npt.NDArray[t.Any], xp.asarray(numeric))
        raise TypeError("Simulated values must have a consistent numeric, string or date type")

    if dtype is not None and dtype.kind in "USMm":
        return np.asarray(candidate)
    return t.cast(npt.NDArray[t.Any], xp.asarray(values))


def host_values(values: t.Any) -> bool:
    """Whether an array, scalar or sequence needs the NumPy host backend."""
    if isinstance(values, np.ndarray):
        return values.dtype.kind in "USMmO"
    if isinstance(values, (str, bytes, dt.date, np.datetime64, np.timedelta64)):
        return True
    if isinstance(values, (list, tuple)):
        if any(host_values(item) for item in values):
            return True
        try:
            return np.asarray(values).dtype.kind in "USMmO"
        except (TypeError, ValueError):
            return True
    return False

"""Array handling for simulated values, including dates and category labels."""

from __future__ import annotations

import datetime as dt
import typing as t

import numpy as np
import numpy.typing as npt

from ._maths import xp


def stochastic_array(values: t.Any) -> npt.NDArray[t.Any]:
    """Create a one-dimensional array without converting dates or labels to numbers."""
    if isinstance(values, np.ndarray):
        candidate = values
    elif isinstance(values, (list, tuple)) and values and isinstance(
        values[0], (str, bytes, dt.date, np.datetime64)
    ):
        candidate = np.asarray(values)
    else:
        candidate = values

    dtype = getattr(candidate, "dtype", None)
    if dtype is not None and dtype.kind == "O":
        items = np.asarray(candidate).ravel().tolist()
        if all(isinstance(item, dt.datetime) for item in items):
            return np.asarray(candidate, dtype="datetime64[us]")
        if all(isinstance(item, dt.date) for item in items):
            return np.asarray(candidate, dtype="datetime64[D]")
        raise TypeError("Simulated values must have a consistent numeric, string or date type")

    if dtype is not None and dtype.kind in "USMm":
        return np.asarray(candidate)
    return t.cast(npt.NDArray[t.Any], xp.asarray(values))


def host_values(values: t.Any) -> bool:
    """Whether values are stored in NumPy because the active backend lacks this dtype."""
    return isinstance(values, np.ndarray) and values.dtype.kind in "USMm"

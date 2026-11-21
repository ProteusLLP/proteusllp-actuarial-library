"""Lossless PAL object persistence, separate from tabular data interchange.

Use save/load for a versioned snapshot of a ProteusVariable hierarchy. This
preserves leaf types, value dtypes, simulation counts and internal coupling
groups. It is intended for PAL-to-PAL storage, not third-party data exchange.
"""

from __future__ import annotations

import os
import typing as t
from pathlib import Path

import pandas as pd

from .variable_io import from_dataframe as _restore_csv
from .variable_io import to_dataframe as _snapshot_csv
from .variable_parquet import read_parquet as _restore_parquet
from .variable_parquet import write_parquet as _snapshot_parquet
from .variables import ProteusVariable


def save(variable: ProteusVariable[t.Any], path: str | os.PathLike[str]) -> None:
    """Save a PAL object in versioned Parquet or CSV format.

    Parquet is the preferred format for large models. CSV snapshots are
    supported for portability but are much slower and can be larger.
    """
    extension = Path(path).suffix.lower()
    if extension == ".parquet":
        _snapshot_parquet(variable, path)
    elif extension == ".csv":
        _snapshot_csv(variable).to_csv(path, index=False)
    else:
        raise ValueError("PAL snapshots require a .parquet or .csv file")


def load(path: str | os.PathLike[str]) -> ProteusVariable[t.Any]:
    """Restore a PAL variable hierarchy from a versioned snapshot.

    All saved leaves and their coupling relationships are reconstructed,
    without linking them to the in-memory objects from which they originated.
    """
    extension = Path(path).suffix.lower()
    if extension == ".parquet":
        return _restore_parquet(path)
    if extension == ".csv":
        frame: pd.DataFrame = pd.read_csv(path, dtype={"value": str}, keep_default_na=False)
        if "_pal_schema" not in frame:
            raise ValueError("Not a PAL snapshot; use ProteusVariable.from_csv() for tabular data")
        return _restore_csv(frame)
    raise ValueError("PAL snapshots require a .parquet or .csv file")

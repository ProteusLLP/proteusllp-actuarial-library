"""Lossless PAL object persistence, separate from tabular data interchange.

Versioned CSV/Parquet snapshots preserve PAL leaf types, simulation counts,
value dtypes and internal coupling groups. Storage may be local or remote.
"""

from __future__ import annotations

import typing as t

import pandas as pd

from .storage import StorageLocation, StorageOptions, file_extension, open_file
from .variable_io import from_dataframe as _restore_csv
from .variable_io import to_dataframe as _snapshot_csv
from .variable_parquet import read_parquet as _restore_parquet
from .variable_parquet import write_parquet as _snapshot_parquet
from .variables import ProteusVariable


def save(
    variable: ProteusVariable[t.Any],
    path: StorageLocation,
    *,
    storage_options: StorageOptions = None,
    format: str | None = None,
) -> None:
    """Save a PAL object to CSV or Parquet on local, S3, Azure or other storage.

    Parquet snapshots are preferred for large models. For unnamed streams use
    format='csv' or format='parquet'; the caller keeps ownership of the stream.

    Args:
        format: Serialization format of the variable.
        path: File path or storage URL of the serialized variable.
        storage_options: Options passed to the storage filesystem.
        variable: PAL variable to convert or serialize.
    """
    extension = file_extension(path, format)
    with open_file(path, "wb", storage_options=storage_options) as output:
        if extension == ".parquet":
            _snapshot_parquet(variable, output)
        else:
            _snapshot_csv(variable).to_csv(output, index=False, encoding="utf-8")


def load(
    path: StorageLocation,
    *,
    storage_options: StorageOptions = None,
    format: str | None = None,
) -> ProteusVariable[t.Any]:
    """Restore a versioned PAL snapshot from a local or remote filesystem.

    Args:
        format: Serialization format of the variable.
        path: File path or storage URL of the serialized variable.
        storage_options: Options passed to the storage filesystem.
    """
    extension = file_extension(path, format)
    with open_file(path, "rb", storage_options=storage_options) as input_file:
        if extension == ".parquet":
            return _restore_parquet(input_file)
        frame: pd.DataFrame = pd.read_csv(input_file, dtype={"value": str}, keep_default_na=False)
        if "_pal_schema" not in frame:
            raise ValueError("Not a PAL snapshot; use ProteusVariable.from_csv() for tabular data")
        return _restore_csv(frame)

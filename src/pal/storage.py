"""Unified path and file-like I/O for local files and fsspec cloud storage.

S3 and Azure support is supplied by the optional s3fs/adlfs filesystem
implementations. Credentials are passed only to the underlying filesystem
and are never embedded in PAL table or snapshot metadata.
"""

from __future__ import annotations

import os
import typing as t
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import fsspec

StorageLocation = str | os.PathLike[str] | t.IO[t.Any]
StorageOptions = Mapping[str, t.Any] | None


@contextmanager
def open_file(
    location: StorageLocation,
    mode: str,
    *,
    storage_options: StorageOptions = None,
) -> Iterator[t.BinaryIO]:
    """Open a local path, remote URI or caller-owned file-like object.

    Remote locations use fsspec, for example s3://bucket/key.parquet or
    az://container/key.parquet. A file-like object stays open after use;
    the caller owns its lifecycle. Storage options apply to paths only.

    Args:
        location: Local path or remote storage URL.
        mode: File access mode.
        storage_options: Options passed to the storage filesystem.
    """
    if isinstance(location, (str, os.PathLike)):
        with fsspec.open(os.fspath(location), mode=mode, **dict(storage_options or {})) as stream:
            yield t.cast(t.BinaryIO, stream)
    else:
        if storage_options:
            raise ValueError("storage_options cannot be used with an already-open file")
        if ("r" in mode and not hasattr(location, "read")) or ("w" in mode and not hasattr(location, "write")):
            raise TypeError("location must be a path, URI or readable/writable file-like object")
        yield t.cast(t.BinaryIO, location)


def file_extension(location: StorageLocation, format: str | None = None) -> str:
    """Identify CSV/Parquet snapshots from the suffix or an optional format.

    Args:
        format: Serialization format of the variable.
        location: Local path or remote storage URL.
    """
    if format is not None:
        extension = "." + format.lower().lstrip(".")
    else:
        name = os.fspath(location) if isinstance(location, (str, os.PathLike)) else getattr(location, "name", "")
        extension = Path(str(name).split("?", 1)[0]).suffix.lower()
    if extension not in (".csv", ".parquet"):
        raise ValueError("PAL snapshots require a .parquet or .csv path (or format='csv'/'parquet')")
    return extension

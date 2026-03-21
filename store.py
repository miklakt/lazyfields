import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping

import pandas as pd

from .persistence import LOADERS_BY_SUFFIX, hdf5_get


def _load_row(path: str | Path) -> Mapping[str, Any]:
    try:
        return LOADERS_BY_SUFFIX[Path(path).suffix.lower()](path)
    except KeyError as exc:
        raise ValueError(f"Could not infer a storage loader from '{Path(path).name}'.") from exc


def _path_get(value: Any, key: str) -> Any:
    for part in key.split("/"):
        value = value[part]
    return value


def _row_value(path: str | Path, key: str, strict: bool = False) -> Any:
    path = Path(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        try: return hdf5_get(path, key)
        except (KeyError, TypeError):
            if strict: raise
            return pd.NA
    try:
        return _path_get(_load_row(path), key)
    except (KeyError, TypeError):
        if strict: raise
        return pd.NA


def _stored_row(path: str | Path, keys: list[str]) -> dict[str, Any]:
    return {key: _row_value(path, key, strict=True) for key in keys}


def _reference_rows(
    directory: str | Path = "data", 
    columns: list[str] | None = None, *, 
    reference_path: str | Path | None = None, file_pattern: str = "*") -> Iterator[dict[str, Any]]:
    for storage_path in sorted(
        path for path in Path(directory).glob(file_pattern) if path.suffix.lower() in LOADERS_BY_SUFFIX
    ):
        try:
            row_data = _load_row(storage_path)
        except Exception as exc:
            print(f"Failed to load {storage_path.name}: {exc}")
            continue

        if not isinstance(row_data, Mapping):
            print(f"Skipping {storage_path.name}: expected a mapping row dictionary.")
            continue

        yield _reference_row(row_data, storage_path, columns=columns, reference_path=reference_path)

def create_reference_table(
    directory: str | Path = "data", 
    columns: list[str] | None = None, 
    *, reference_path: str | Path | None = None, 
    file_pattern: str = "*") -> pd.DataFrame:
    df = pd.DataFrame(_reference_rows(directory, columns, reference_path=reference_path, file_pattern=file_pattern))
    if reference_path is not None: df.attrs["reference_path"] = str(reference_path)
    return df


def _reference_row(
    row_data: Mapping[str, Any], 
    storage_path: Path, *, 
    columns: list[str] | None = None, 
    reference_path: str | Path | None = None) -> dict[str, Any]:
    row = {k: v for k, v in row_data.items() if pd.api.types.is_scalar(v)}
    if columns is not None:
        row = {k: v for k, v in row.items() if k in columns}
    base = Path(reference_path).parent if reference_path is not None else None
    return {
        **row,
        "storage_file": str(storage_path) if base is None else os.path.relpath(storage_path, start=base),
        "non_scalar_keys": [k for k, v in row_data.items() if not pd.api.types.is_scalar(v)],
        "creation_time": datetime.fromtimestamp(storage_path.stat().st_ctime),
    }


class StoreAccessor:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj
        self._base_dir = (
            Path(pandas_obj.attrs["reference_path"]).parent
            if "reference_path" in pandas_obj.attrs
            else None
        )

    def _path(self, stored_path: str | Path) -> Path:
        path = Path(stored_path)
        return path if path.is_absolute() or self._base_dir is None else self._base_dir / path

    def __getitem__(self, keys):
        series = isinstance(self._obj, pd.Series)
        files = self._path(self._obj.storage_file) if series else self._obj["storage_file"].map(self._path)
        if keys == slice(None):
            return _stored_row(files, self._obj.non_scalar_keys) if series else [
                _stored_row(path, row_keys) for path, row_keys in zip(files, self._obj["non_scalar_keys"])
            ]
        if isinstance(keys, str):
            if series:
                return _row_value(files, keys, strict=True)
            values = pd.Series((_row_value(path, keys) for path in files), index=self._obj.index, dtype="object")
            if values.isna().all():
                raise KeyError(keys)
            return values
        if isinstance(keys, list): return {k: self[k] for k in keys} if series else pd.DataFrame({k: self[k] for k in keys}, index=self._obj.index)
        raise TypeError("Keys must be str, list[str], or [:]")


def register_accessors() -> None:
    if "store" not in pd.DataFrame._accessors: pd.api.extensions.register_dataframe_accessor("store")(StoreAccessor)
    if "store" not in pd.Series._accessors: pd.api.extensions.register_series_accessor("store")(StoreAccessor)

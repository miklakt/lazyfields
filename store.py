import os
from copy import deepcopy
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


def _select(value: Any, selection: Any | None) -> Any:
    return value if selection is None else value[selection]


def _row_value(path: str | Path, key: str, *, strict: bool = False, selection: Any | None = None) -> Any:
    path = Path(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        try:
            return hdf5_get(path, key, selection=selection)
        except (KeyError, TypeError):
            if strict:
                raise
            return pd.NA
    try:
        return _select(_path_get(_load_row(path), key), selection)
    except (KeyError, TypeError):
        if strict:
            raise
        return pd.NA


def _apply_pipe(row_data: Mapping[str, Any], pipe: list[Any] | None, *, scoped: bool = False) -> Mapping[str, Any] | bool:
    last_was_filter = False
    for step in pipe or []:
        result = _apply_pipe(deepcopy(row_data), step, scoped=True) if isinstance(step, list) else step(row_data)
        if result is None: last_was_filter = False
        elif isinstance(result, Mapping): row_data, last_was_filter = result, False
        elif isinstance(result, bool):
            if not result: return False
            last_was_filter = True
        else: raise TypeError("Pipe callbacks must return a mapping, bool, or None.")
    if scoped and not last_was_filter: raise TypeError("Scoped pipe blocks must end with a filter.")
    return True if scoped else row_data


def _reference_rows(
    directory: str | Path = "data", 
    columns: list[str] | None = None, *, 
    reference_path: str | Path | None = None, file_pattern: str = "*",
    search_subdirectories: bool = False,
    pipe: list[Any] | None = None,
) -> Iterator[dict[str, Any]]:
    scan = Path(directory).rglob if search_subdirectories else Path(directory).glob
    for storage_path in sorted(
        path for path in scan(file_pattern) if path.suffix.lower() in LOADERS_BY_SUFFIX
    ):
        try:
            row_data = _load_row(storage_path)
        except Exception as exc:
            print(f"Failed to load {storage_path.name}: {exc}")
            continue

        if not isinstance(row_data, Mapping):
            print(f"Skipping {storage_path.name}: expected a mapping row dictionary.")
            continue

        non_scalar_keys = [k for k, v in row_data.items() if not pd.api.types.is_scalar(v)]
        try:
            processed_row_data = _apply_pipe(row_data, pipe)
        except Exception as exc:
            print(f"Failed to apply pipe to {storage_path.name}: {exc}")
            continue
        if processed_row_data is False:
            continue

        yield _reference_row(
            processed_row_data,
            storage_path,
            columns=columns,
            reference_path=reference_path,
            non_scalar_keys=non_scalar_keys,
        )


def create_reference_table(
    directory: str | Path = "data", 
    columns: list[str] | None = None, 
    *, reference_path: str | Path | None = None, 
    file_pattern: str = "*",
    search_subdirectories: bool = False,
    pipe: list[Any] | None = None,
) -> pd.DataFrame:
    df = pd.DataFrame(
        _reference_rows(
            directory,
            columns,
            reference_path=reference_path,
            file_pattern=file_pattern,
            search_subdirectories=search_subdirectories,
            pipe=pipe,
        )
    )
    if reference_path is not None:
        df.attrs["reference_path"] = str(reference_path)
    return df


def _reference_row(
    row_data: Mapping[str, Any], 
    storage_path: Path, *, 
    columns: list[str] | None = None, 
    reference_path: str | Path | None = None,
    non_scalar_keys: list[str] | None = None,
) -> dict[str, Any]:
    row = {k: v for k, v in row_data.items() if pd.api.types.is_scalar(v)}
    if columns is not None:
        row = {k: v for k, v in row.items() if k in columns}
    base = Path(reference_path).parent if reference_path is not None else None
    return {
        **row,
        "storage_file": str(storage_path) if base is None else os.path.relpath(storage_path, start=base),
        "non_scalar_keys": non_scalar_keys if non_scalar_keys is not None else [k for k, v in row_data.items() if not pd.api.types.is_scalar(v)],
        "creation_time": datetime.fromtimestamp(storage_path.stat().st_ctime),
    }


class StoreAccessor:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    def rebase_reference_paths(self, reference_path: str | Path) -> pd.DataFrame:
        self._obj.attrs["reference_path"] = str(reference_path)
        return self._obj

    @property
    def _base_dir(self) -> Path | None:
        if "reference_path" not in self._obj.attrs:
            return None
        return Path(self._obj.attrs["reference_path"]).parent

    def _path(self, stored_path: str | Path) -> Path:
        path = Path(stored_path)
        return path if path.is_absolute() or self._base_dir is None else self._base_dir / path

    def __getitem__(self, keys):
        series = isinstance(self._obj, pd.Series)
        files = self._path(self._obj.storage_file) if series else self._obj["storage_file"].map(self._path)
        if keys == slice(None):
            return _load_row(files) if series else [_load_row(path) for path in files]

        selection = None
        if isinstance(keys, tuple):
            if len(keys) != 2 or not isinstance(keys[0], str):
                raise TypeError("Tuple access must be ('field', selection).")
            keys, selection = keys

        if isinstance(keys, str):
            if series:
                return _row_value(files, keys, strict=True, selection=selection)
            values = pd.Series(
                (_row_value(path, keys, selection=selection) for path in files),
                index=self._obj.index,
                dtype="object",
            )
            if values.isna().all():
                raise KeyError(keys)
            return values
        if isinstance(keys, list): return {k: self[k] for k in keys} if series else pd.DataFrame({k: self[k] for k in keys}, index=self._obj.index)
        raise TypeError("Keys must be str, tuple[str, selection], list[str], or [:]")


def register_accessors() -> None:
    if "store" not in pd.DataFrame._accessors: pd.api.extensions.register_dataframe_accessor("store")(StoreAccessor)
    if "store" not in pd.Series._accessors: pd.api.extensions.register_series_accessor("store")(StoreAccessor)

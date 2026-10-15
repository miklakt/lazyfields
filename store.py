import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from .persistence import LOADERS_BY_SUFFIX, archive_suffix, hdf5_get, unpacked_archive_member


def _load_row(path: str | Path) -> Mapping[str, Any]:
    try:
        return LOADERS_BY_SUFFIX[Path(path).suffix.lower()](path)
    except KeyError as exc:
        raise ValueError(f"Could not infer a storage loader from '{Path(path).name}'.") from exc


def _path_get(value: Any, key: str) -> Any:
    for part in filter(None, key.split("/")):
        quoted = len(part) >= 2 and part[0] == part[-1] and part[0] in {"'", '"'}
        if quoted:
            part = part[1:-1]
        try:
            int_part = int(part)
        except ValueError:
            int_part = None
        if isinstance(value, Mapping):
            if quoted and int_part is not None:
                # Quoting distinguishes intentional string keys from numeric path indices.
                warnings.warn(f"Quoted path segment {part!r} is treating the segment as a string key.", stacklevel=4)
            if quoted or part in value:
                value = value[part]
            elif int_part is not None and int_part in value:
                value = value[int_part]
            else:
                value = value[part]
        elif not quoted:
            value = value[int_part if int_part is not None else part]
        else:
            value = value[part]
    return value


def _row_value(path: str | Path, key: str, *, strict: bool = False, selection: Any | None = None) -> Any:
    path = Path(path)
    if archive_suffix(path) is not None:
        try:
            with unpacked_archive_member(path) as inner_path:
                return _row_value(inner_path, key, strict=True, selection=selection)
        except (KeyError, TypeError, IndexError, ValueError):
            if strict:
                raise
            return pd.NA
    if path.suffix.lower() in {".h5", ".hdf5"}:
        try:
            return hdf5_get(path, key, selection=selection)
        except (KeyError, TypeError):
            if strict:
                raise
            return pd.NA
    try:
        value = _path_get(_load_row(path), key)
        return value if selection is None else value[selection]
    except (KeyError, TypeError, IndexError):
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


def _reference_row_from_path(
    storage_path: Path, *,
    columns: list[str] | None = None,
    relative_to: Path,
    pipe: list[Any] | None = None,
) -> dict[str, Any] | None:
    try:
        row_data = _load_row(storage_path)
    except Exception as exc:
        print(f"Failed to load {storage_path.name}: {exc}")
        return None

    if not isinstance(row_data, Mapping):
        print(f"Skipping {storage_path.name}: expected a mapping row dictionary.")
        return None

    non_scalar_keys = [k for k, v in row_data.items() if not pd.api.types.is_scalar(v)]
    try:
        processed_row_data = _apply_pipe(row_data, pipe)
    except Exception as exc:
        print(f"Failed to apply pipe to {storage_path.name}: {exc}")
        return None
    if processed_row_data is False:
        return None

    row = {k: v for k, v in processed_row_data.items() if pd.api.types.is_scalar(v)}
    if columns is not None:
        row = {k: v for k, v in row.items() if k in columns}
    return {
        **row,
        "storage_file": _storage_file_value(storage_path, relative_to),
        "non_scalar_keys": non_scalar_keys,
        "creation_time": datetime.fromtimestamp(storage_path.stat().st_ctime),
    }


def _find_storage_files(directory: str | Path, file_pattern: str, search_subdirectories: bool) -> list[Path]:
    scan = Path(directory).rglob if search_subdirectories else Path(directory).glob
    return sorted(
        path for path in scan(file_pattern)
        if path.suffix.lower() in LOADERS_BY_SUFFIX and (path.suffix.lower() not in {".gz", ".bz2", ".xz"} or archive_suffix(path))
    )


def _storage_file_value(storage_path: Path, relative_to: Path) -> str:
    return os.path.relpath(storage_path.resolve(), start=relative_to)


def _stat_fingerprint(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def create_reference_table(
    directory: str | Path = "data", 
    columns: list[str] | None = None, 
    *, relative_to: str | Path | None = None,
    file_pattern: str = "*",
    search_subdirectories: bool = False,
    pipe: list[Any] | None = None,
    workers: int | None = None,
) -> pd.DataFrame:
    if workers is not None and workers < 1:
        raise ValueError("workers must be None or a positive integer.")

    relative_to = Path.cwd().resolve() if relative_to is None else Path(relative_to).resolve().parent
    storage_paths = _find_storage_files(directory, file_pattern, search_subdirectories)

    def make_row(storage_path: Path) -> dict[str, Any] | None:
        return _reference_row_from_path(
            storage_path,
            columns=columns,
            relative_to=relative_to,
            pipe=pipe,
        )

    rows = map(make_row, storage_paths)
    if workers not in (None, 1):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = executor.map(make_row, storage_paths)
            df = pd.DataFrame(row for row in rows if row is not None)
    else:
        df = pd.DataFrame(row for row in rows if row is not None)

    df.attrs["relative_to"] = str(relative_to)
    df.attrs["stat_fingerprints"] = {
        _storage_file_value(path, relative_to): _stat_fingerprint(path)
        for path in storage_paths
    }
    return df


def update_reference_table(
    reference_df: pd.DataFrame,
    directory: str | Path = "data",
    columns: list[str] | None = None,
    *,
    relative_to: str | Path | None = None,
    file_pattern: str = "*",
    search_subdirectories: bool = False,
    pipe: list[Any] | None = None,
) -> pd.DataFrame:
    relative_to = Path(relative_to).resolve().parent if relative_to is not None else Path(reference_df.attrs["relative_to"])
    old_rows = reference_df.set_index("storage_file", drop=False) if "storage_file" in reference_df else pd.DataFrame()
    old_fingerprints = reference_df.attrs.get("stat_fingerprints", {})
    rows = []
    fingerprints = {}
    for path in _find_storage_files(directory, file_pattern, search_subdirectories):
        stored_path = _storage_file_value(path, relative_to)
        fingerprint = _stat_fingerprint(path)
        fingerprints[stored_path] = fingerprint
        if stored_path in old_rows.index and old_fingerprints.get(stored_path) == fingerprint:
            rows.append(old_rows.loc[stored_path].to_dict())
        elif (row := _reference_row_from_path(path, columns=columns, relative_to=relative_to, pipe=pipe)) is not None:
            rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["relative_to"] = str(relative_to)
    df.attrs["stat_fingerprints"] = fingerprints
    return df


@pd.api.extensions.register_dataframe_accessor("store")
@pd.api.extensions.register_series_accessor("store")
class StoreAccessor:
    def __init__(self, pandas_obj: pd.DataFrame | pd.Series):
        self._obj = pandas_obj
        self._methods: dict[str, Callable[..., Any]] = pandas_obj.attrs.get("store_methods", {})

    def register_method(self, method: Callable[..., Any]) -> Callable[..., Any]:
        """Register a row-wise method for this pandas object and its selections."""
        name = getattr(method, "__name__", None)
        if not callable(method) or not isinstance(name, str) or not name.isidentifier():
            raise TypeError("A store method must be a named callable.")
        if hasattr(type(self), name) or name in self._methods:
            raise ValueError(f"A store method named {name!r} already exists.")
        self._methods = {**self._methods, name: method}
        self._obj.attrs["store_methods"] = self._methods
        return method

    def __getattr__(self, name: str) -> Any:
        try:
            method = self._methods[name]
        except KeyError:
            raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}") from None

        @wraps(method)
        def call(*args: Any, **kwargs: Any) -> Any:
            if isinstance(self._obj, pd.Series):
                return method(self._obj, *args, **kwargs)
            values = []
            for _, row in self._obj.iterrows():
                row.attrs = self._obj.attrs.copy()
                row.store._methods = self._methods
                values.append(method(row, *args, **kwargs))
            return pd.Series(values, index=self._obj.index, dtype="object", name=name)

        return call

    def set_relative_to(self, relative_to: str | Path) -> pd.DataFrame | pd.Series:
        self._obj.attrs["relative_to"] = str(Path(relative_to).resolve().parent)
        return self._obj

    def _path(self, stored_path: str | Path) -> Path:
        path = Path(stored_path)
        try:
            return Path(self._obj.attrs["relative_to"]) / path
        except KeyError:
            raise RuntimeError("Call .store.set_relative_to(path) after loading this table.") from None

    def __getitem__(self, keys):
        series = isinstance(self._obj, pd.Series)
        files = self._path(self._obj["storage_file"]) if series else self._obj["storage_file"].map(self._path)
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
        if isinstance(keys, list):
            values = {k: self[k] for k in keys}
            return values if series else pd.DataFrame(values, index=self._obj.index)
        raise TypeError("Keys must be str, tuple[str, selection], list[str], or [:]")

import ast, inspect, os, textwrap
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from .readers import _reader_match, require_reader


def _row_value(path: str | Path, key: str, *, strict: bool = False, selection: Any | None = None) -> Any:
    reader = require_reader(path)
    try:
        return reader.get(path, key, selection)
    except reader.missing_errors:
        if strict: raise
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
    file_path: Path, *,
    columns: list[str] | None = None,
    relative_to: Path,
    pipe: list[Any] | None = None,
) -> dict[str, Any] | None:
    try:
        reader = require_reader(file_path)
        if not pipe and reader.inspect is not None:
            row, non_scalar_keys = {}, []
            for key, is_scalar, value in reader.items(file_path):
                if is_scalar:
                    row[key] = value
                else:
                    non_scalar_keys.append(key)
        else:
            row_data = reader.load(file_path)
    except Exception as exc:
        print(f"Failed to load {file_path.name}: {exc}"); return None

    if pipe or reader.inspect is None:
        if not isinstance(row_data, Mapping):
            print(f"Skipping {file_path.name}: expected a mapping row dictionary."); return None
        non_scalar_keys = [k for k, v in row_data.items() if not pd.api.types.is_scalar(v)]
        try:
            processed_row_data = _apply_pipe(row_data, pipe)
        except Exception as exc:
            print(f"Failed to apply pipe to {file_path.name}: {exc}"); return None
        if processed_row_data is False: return None
        row = {k: v for k, v in processed_row_data.items() if pd.api.types.is_scalar(v)}
    if columns is not None: row = {k: v for k, v in row.items() if k in columns}
    return {
        **row,
        "file_path": _relative_file_path(file_path, relative_to),
        "non_scalar_keys": non_scalar_keys,
        "creation_time": datetime.fromtimestamp(file_path.stat().st_ctime),
    }


def _find_files(directory: str | Path, file_pattern: str, search_subdirectories: bool) -> list[Path]:
    scan = Path(directory).rglob if search_subdirectories else Path(directory).glob
    return sorted(path for path in scan(file_pattern) if _reader_match(path)[0] is not None)


def _relative_file_path(path: Path, relative_to: Path) -> str:
    return os.path.relpath(path.resolve(), start=relative_to)


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
    file_paths = _find_files(directory, file_pattern, search_subdirectories)

    def make_row(file_path: Path) -> dict[str, Any] | None:
        return _reference_row_from_path(
            file_path,
            columns=columns,
            relative_to=relative_to,
            pipe=pipe,
        )

    rows = map(make_row, file_paths)
    if workers not in (None, 1):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = executor.map(make_row, file_paths)
            df = pd.DataFrame(row for row in rows if row is not None)
    else:
        df = pd.DataFrame(row for row in rows if row is not None)

    df.attrs["relative_to"] = str(relative_to)
    df.attrs["stat_fingerprints"] = {
        _relative_file_path(path, relative_to): _stat_fingerprint(path)
        for path in file_paths
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
    old_rows = reference_df.set_index("file_path", drop=False) if "file_path" in reference_df else pd.DataFrame()
    old_fingerprints = reference_df.attrs.get("stat_fingerprints", {})
    rows = []
    fingerprints = {}
    for path in _find_files(directory, file_pattern, search_subdirectories):
        stored_path = _relative_file_path(path, relative_to)
        fingerprint = _stat_fingerprint(path)
        fingerprints[stored_path] = fingerprint
        if stored_path in old_rows.index and old_fingerprints.get(stored_path) == fingerprint:
            rows.append(old_rows.loc[stored_path].to_dict())
        else:
            row = _reference_row_from_path(path, columns=columns, relative_to=relative_to, pipe=pipe)
            if row is not None: rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["relative_to"] = str(relative_to)
    df.attrs["stat_fingerprints"] = fingerprints
    return df


@pd.api.extensions.register_dataframe_accessor("store")
@pd.api.extensions.register_series_accessor("store")
class StoreAccessor:
    def __init__(self, pandas_obj: pd.DataFrame | pd.Series):
        self._obj = pandas_obj
        self._methods: dict[str, str] = pandas_obj.attrs.get("store_methods", {})
        self._compiled_methods: dict[str, Callable[..., Any]] = {}

    def register_method(self, method: Callable[..., Any]) -> Callable[..., Any]:
        """Register a row-wise method for this pandas object and its selections."""
        name = getattr(method, "__name__", None)
        if not callable(method) or not isinstance(name, str) or not name.isidentifier():
            raise TypeError("A store method must be a named callable.")
        if hasattr(type(self), name) or name in self._methods:
            raise ValueError(f"A store method named {name!r} already exists.")
        try:
            node = ast.parse(textwrap.dedent(inspect.getsource(method))).body[0]
        except (OSError, TypeError, SyntaxError) as exc:
            raise ValueError("A registered method must have inspectable source code.") from exc
        if not isinstance(node, ast.FunctionDef) or node.name != name:
            raise ValueError("A registered method must be a named function.")
        node.decorator_list = []
        self._methods = {**self._methods, name: ast.unparse(node)}
        self._obj.attrs["store_methods"] = self._methods
        self._compiled_methods[name] = method
        return method

    def __getattr__(self, name: str) -> Any:
        try:
            source = self._methods[name]
        except KeyError:
            raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}") from None
        if name not in self._compiled_methods:
            namespace: dict[str, Any] = {}
            exec(source, namespace)
            self._compiled_methods[name] = namespace[name]
        method = self._compiled_methods[name]

        @wraps(method)
        def call(*args: Any, **kwargs: Any) -> Any:
            if isinstance(self._obj, pd.Series):
                return method(self._obj, *args, **kwargs)
            values = []
            for _, row in self._obj.iterrows():
                row.attrs = self._obj.attrs.copy()
                row.store._methods = self._methods
                row.store._compiled_methods = self._compiled_methods
                values.append(method(row, *args, **kwargs))
            return pd.Series(values, index=self._obj.index, dtype="object", name=name)

        return call

    def get_registered_methods(self):
        return list(self._methods.keys())

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
        files = self._path(self._obj["file_path"]) if series else self._obj["file_path"].map(self._path)
        if keys == slice(None):
            return require_reader(files).load(files) if series else [require_reader(path).load(path) for path in files]

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

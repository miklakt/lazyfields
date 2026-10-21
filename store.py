import ast, inspect, os, sys, textwrap
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from .pipe import _apply_pipe
from .readers import _reader_match, require_reader


def _row_value(path: str | Path, key: str, *, strict: bool = False, selection: Any | None = None) -> Any:
    reader = require_reader(path)
    try:
        return reader.get(path, key, selection)
    except reader.missing_errors:
        if strict: raise
        return pd.NA


class _InvalidKeyError(ValueError):
    pass


def _validate_keys(value: Any) -> None:
    pending, seen = [value], set()
    while pending:
        current = pending.pop()
        if not isinstance(current, (Mapping, list, tuple)) or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, Mapping):
            for key, item in current.items():
                if isinstance(key, str) and "/" in key:
                    raise _InvalidKeyError(f"Key {key!r} contains '/', which is reserved for nested paths.")
                pending.append(item)
        else:
            pending.extend(current)


def _reference_row_from_path(
    file_path: Path, *,
    columns: list[str] | None = None,
    relative_to: Path,
    pipe: list[Any] | None = None,
    pipe_error_skip: bool = True,
) -> dict[str, Any] | None:
    try:
        reader = require_reader(file_path)
        if not pipe and reader.inspect is not None:
            row, non_scalar_keys = {}, []
            for key, is_scalar, value in reader.items(file_path):
                _validate_keys({key: value})
                if is_scalar:
                    row[key] = value
                else:
                    non_scalar_keys.append(key)
        else:
            row_data = reader.load(file_path)
            _validate_keys(row_data)
    except _InvalidKeyError:
        raise
    except Exception as exc:
        print(f"Failed to load {file_path.name}: {exc}"); return None

    if pipe or reader.inspect is None:
        if not isinstance(row_data, Mapping):
            print(f"Skipping {file_path.name}: expected a mapping row dictionary."); return None
        non_scalar_keys = [k for k, v in row_data.items() if not pd.api.types.is_scalar(v)]
        if pipe:
            try:
                processed_row_data = _apply_pipe(row_data, pipe, reader, file_path)
            except Exception as exc:
                if not pipe_error_skip: raise
                print(f"Failed to apply pipe to {file_path}: {type(exc).__name__}: {exc}", file=sys.stderr)
                return None
            if processed_row_data is False: return None
            _validate_keys(processed_row_data)
        else:
            processed_row_data = row_data
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
    pipe_error_skip: bool = True,
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
            pipe_error_skip=pipe_error_skip,
        )

    if workers not in (None, 1):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            df = pd.DataFrame(row for row in executor.map(make_row, file_paths) if row is not None)
    else:
        df = pd.DataFrame(row for row in map(make_row, file_paths) if row is not None)

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
    pipe_error_skip: bool = True,
) -> pd.DataFrame:
    original_attrs = reference_df.attrs.copy()
    relative_to = Path(relative_to).resolve().parent if relative_to is not None else Path(reference_df.attrs["relative_to"])
    old_rows = reference_df.set_index("file_path", drop=False) if "file_path" in reference_df else pd.DataFrame()
    previous_paths = {(Path(reference_df.attrs["relative_to"]) / path).resolve() for path in old_rows.index}
    scan = Path(directory).rglob if search_subdirectories else Path(directory).glob
    for path in scan(file_pattern):
        if path.resolve() in previous_paths and _reader_match(path)[0] is None:
            raise ValueError(f"No reader registered for previously indexed file: {path}")
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
            row = _reference_row_from_path(path, columns=columns, relative_to=relative_to, pipe=pipe, pipe_error_skip=pipe_error_skip)
            if row is not None: rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs = original_attrs
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

    def unregister_method(self, name: str) -> None:
        """Remove a registered method by name; raise KeyError if it is missing."""
        methods = self._methods.copy()
        del methods[name]
        self._methods = methods
        self._obj.attrs["store_methods"] = methods
        self._compiled_methods.pop(name, None)

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
            if len(keys) < 2 or not isinstance(keys[0], str):
                raise TypeError("Tuple access must start with a field name and a selection.")
            keys, selection = keys[0], keys[1] if len(keys) == 2 else keys[1:]

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

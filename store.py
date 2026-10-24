import ast
import inspect
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from .pipe import _apply_pipe
from .filesystem import FileSystem, Node


def _row_value(filesystem: FileSystem, path: str, key: str, selection: Any | None = None) -> Any:
    try:
        return filesystem.read(f"{path}/{key}", selection)
    except (KeyError, TypeError, IndexError):
        return pd.NA


class _InvalidKeyError(ValueError):
    pass


def _validate_keys(value):
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


def _creation_time(node):
    try:
        paths = {source.file_path for source in node.locate()}
        return datetime.fromtimestamp(max(path.stat().st_ctime for path in paths)) if paths else pd.NaT
    except OSError:
        return pd.NaT


def _reference_row_from_node(node: Node, *, attrs, columns=None, pipe=None, pipe_error_skip=True):
    scalars, non_scalar_keys = {}, []
    try:
        for key, kind, value in node.items():
            _validate_keys({key: value})
            if kind == "scalar": scalars[key] = value
            else: non_scalar_keys.append(key)
    except _InvalidKeyError:
        raise
    except Exception as exc:
        print(f"Failed to load {node.path}: {exc}"); return None
    if pipe:
        row = pd.Series({**scalars, "file_path": node.path, "non_scalar_keys": non_scalar_keys}, dtype="object")
        row.attrs = {**attrs, "store_path": node.path}
        row.__dict__["_store_filesystem"] = node._filesystem
        try:
            row = _apply_pipe(row, pipe)
        except Exception as exc:
            if not pipe_error_skip: raise
            print(f"Failed to apply pipe to {node.path}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return None
        if row is False: return None
        _validate_keys(row.to_dict())
        scalars = {key: value for key, value in row.items() if pd.api.types.is_scalar(value)}
    if columns is not None: scalars = {key: value for key, value in scalars.items() if key in columns}
    return {**scalars, "file_path": node.path, "non_scalar_keys": non_scalar_keys,
            "creation_time": _creation_time(node)}


def create_reference_table(
    filesystem_spec: Mapping,
    columns: list[str] | None = None,
    *,
    relative_to: str | Path | None = None,
    pipe: list[Any] | None = None,
    pipe_error_skip: bool = True,
    workers: int | None = None,
) -> pd.DataFrame:
    """Index root groups from a specification saved in the table's attributes."""
    return update_reference_table(
        pd.DataFrame(), filesystem_spec, columns, relative_to=relative_to, pipe=pipe,
        pipe_error_skip=pipe_error_skip, workers=workers,
    )


def update_reference_table(
    reference_df: pd.DataFrame,
    filesystem_spec: Mapping | None = None,
    columns: list[str] | None = None,
    *,
    relative_to: str | Path | None = None,
    pipe: list[Any] | None = None,
    pipe_error_skip: bool = True,
    workers: int | None = None,
) -> pd.DataFrame:
    """Rebuild the saved specification and reindex, retaining table attributes."""
    if workers is not None and workers < 1:
        raise ValueError("workers must be None or a positive integer.")
    if filesystem_spec is None:
        filesystem_spec = reference_df.attrs.get("filesystem_spec")
    if not isinstance(filesystem_spec, Mapping):
        raise TypeError("filesystem_spec must be a filesystem specification mapping.")
    filesystem_spec = deepcopy(dict(filesystem_spec))
    if relative_to is None:
        relative_to = reference_df.attrs.get("filesystem_relative_to")
    relative_to = str(relative_to) if relative_to is not None else None
    filesystem = FileSystem.from_spec(filesystem_spec, relative_to=relative_to)
    paths = sorted(name for name, kind, _ in filesystem.items() if kind == "group")
    attrs = {**reference_df.attrs, "filesystem_spec": filesystem_spec,
             "filesystem_relative_to": relative_to}

    def make_row(path):
        return _reference_row_from_node(filesystem.node(path), attrs=attrs, columns=columns,
                                        pipe=pipe, pipe_error_skip=pipe_error_skip)

    pool = ThreadPoolExecutor(max_workers=workers) if workers not in (None, 1) else nullcontext()
    with pool as executor:
        rows = (executor.map if executor else map)(make_row, paths)
        df = pd.DataFrame(row for row in rows if row is not None)
    df.attrs = attrs
    df.__dict__["_store_filesystem"] = filesystem
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
                row.__dict__["_store_filesystem"] = self.filesystem
                row.store._methods = self._methods
                row.store._compiled_methods = self._compiled_methods
                values.append(method(row, *args, **kwargs))
            return pd.Series(values, index=self._obj.index, dtype="object", name=name)

        return call

    def get_registered_methods(self):
        return list(self._methods.keys())

    @property
    def filesystem(self) -> FileSystem:
        """Reconstruct the saved specification once for this pandas object."""
        cache = self._obj.__dict__
        if "_store_filesystem" not in cache:
            try:
                spec = self._obj.attrs["filesystem_spec"]
            except KeyError:
                raise RuntimeError("This table has no filesystem specification.") from None
            cache["_store_filesystem"] = FileSystem.from_spec(
                spec, relative_to=self._obj.attrs.get("filesystem_relative_to"),
            )
        return cache["_store_filesystem"]

    def __getitem__(self, keys):
        series = isinstance(self._obj, pd.Series)
        filesystem = self.filesystem
        paths = self._obj.attrs["store_path"] if series and "store_path" in self._obj.attrs else self._obj["file_path"]
        if keys == slice(None):
            return filesystem.read(paths) if series else [filesystem.read(path) for path in paths]

        selection = None
        if isinstance(keys, tuple):
            if len(keys) < 2 or not isinstance(keys[0], str):
                raise TypeError("Tuple access must start with a field name and a selection.")
            keys, selection = keys[0], keys[1] if len(keys) == 2 else keys[1:]

        if isinstance(keys, str):
            if series:
                return filesystem.read(f"{paths}/{keys}", selection)
            values = pd.Series(
                (_row_value(filesystem, path, keys, selection=selection) for path in paths),
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

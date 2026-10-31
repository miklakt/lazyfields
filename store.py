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

from .pipe import _NodeRow, _apply_pipe
from .filesystem import FileSystem, Node

# Columns added to every row; indexed fields cannot use these names.
_RESERVED = ("lf_path", "lf_store_keys", "lf_mtime", "lf_digest")

def _row_value(filesystem: FileSystem, path: str, key: str, selection: Any | None = None) -> Any:
    try:
        return filesystem.read(f"{path}/{key}", selection)
    except (KeyError, TypeError, IndexError):
        return pd.NA


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
                    raise ValueError(f"Key {key!r} contains '/', which is reserved for nested paths.")
                pending.append(item)
        else:
            pending.extend(current)


def _modification_time(node):
    try:
        return datetime.fromtimestamp(max(path.stat().st_mtime for path in node.files()))
    except (OSError, ValueError):  # A missing file, or a node without files.
        return pd.NaT


def _reference_row_from_node(node: Node, *, columns=None, pipe=None, pipe_error_skip=True):
    try:
        items = list(node.items())
    except Exception as exc:
        print(f"Failed to load {node.path}: {type(exc).__name__}: {exc}", file=sys.stderr); return None
    _validate_keys({key: value for key, _, value in items})
    scalars = {key: value for key, kind, value in items if kind == "scalar"}
    store_keys = [key for key, kind, _ in items if kind != "scalar"]
    if pipe:
        try:
            row = _apply_pipe(_NodeRow(node, scalars), pipe)
        except Exception as exc:
            if not pipe_error_skip: raise
            print(f"Failed to apply pipe to {node.path}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return None
        if row is False: return None
        _validate_keys(row)
        scalars = {key: value for key, value in row.items() if pd.api.types.is_scalar(value)}
    if columns is not None: scalars = {key: value for key, value in scalars.items() if key in columns}
    if scalars.keys() & _RESERVED:
        raise ValueError(f"Keys {sorted(scalars.keys() & _RESERVED)} of {node.path} are reserved for lazyfields columns.")
    return {**scalars, "lf_path": node.path, "lf_store_keys": store_keys,
            "lf_mtime": _modification_time(node), "lf_digest": node.digest()}


def create_reference_table(
    source: str | Path | Mapping,
    columns: list[str] | None = None,
    *,
    relative_to: str | Path | None = None,
    pipe: list[Any] | None = None,
    pipe_error_skip: bool = True,
    workers: int | None = None,
) -> pd.DataFrame:
    """Index the groups at the root of a directory or filesystem specification."""
    return update_reference_table(
        pd.DataFrame(), source, columns, relative_to=relative_to, pipe=pipe,
        pipe_error_skip=pipe_error_skip, workers=workers,
    )


def update_reference_table(
    reference_df: pd.DataFrame,
    source: str | Path | Mapping | None = None,
    columns: list[str] | None = None,
    *,
    relative_to: str | Path | None = None,
    pipe: list[Any] | None = None,
    pipe_error_skip: bool = True,
    workers: int | None = None,
) -> pd.DataFrame:
    """Refresh the index, reusing rows whose sources are unchanged."""
    if workers is not None and workers < 1:
        raise ValueError("workers must be None or a positive integer.")
    if source is None:
        source = reference_df.attrs["filesystem_spec"]
        if relative_to is None:
            relative_to = reference_df.attrs["filesystem_relative_to"]
    elif isinstance(source, (str, Path)):
        if relative_to is not None:
            raise ValueError("relative_to is only supported for filesystem specifications; "
                             "a directory source defines its own root.")
        relative_to = str(Path(source).resolve())
        source = {"mount": [{"glob": "*"}]}
    elif relative_to is None:
        relative_to = reference_df.attrs.get("filesystem_relative_to")
    relative_to = str(relative_to) if relative_to is not None else None
    if not isinstance(source, Mapping):
        raise TypeError("source must be a directory or a filesystem specification mapping.")
    filesystem_spec = deepcopy(dict(source))
    filesystem = FileSystem.from_spec(filesystem_spec, relative_to=relative_to)
    relative_to = str(filesystem.relative_to)
    paths = sorted(name for name, kind, _ in filesystem.items() if kind == "group")
    old_rows = {row["lf_path"]: row for row in reference_df.to_dict("records")}
    attrs = {**reference_df.attrs, "filesystem_spec": filesystem_spec,
             "filesystem_relative_to": relative_to}

    def make_row(path):
        node = filesystem.node(path)
        if path in old_rows and old_rows[path]["lf_digest"] == node.digest():
            return old_rows[path]
        return _reference_row_from_node(node, columns=columns,
                                        pipe=pipe, pipe_error_skip=pipe_error_skip)

    pool = ThreadPoolExecutor(max_workers=workers) if workers not in (None, 1) else nullcontext()
    with pool as executor:
        rows = (executor.map if executor else map)(make_row, paths)
        df = pd.DataFrame(row for row in rows if row is not None)
    df.attrs = attrs
    df.__dict__["_filesystem"] = filesystem
    return df


@pd.api.extensions.register_dataframe_accessor("store")
@pd.api.extensions.register_series_accessor("store")
class StoreAccessor:
    def __init__(self, pandas_obj: pd.DataFrame | pd.Series):
        self._obj = pandas_obj
        self._methods: dict[str, str] = pandas_obj.attrs.get("store_methods", {})
        self._method_categories: dict[str, str] = pandas_obj.attrs.get("store_method_categories", {})
        self._compiled_methods: dict[str, Callable[..., Any]] = {}

    def _set_method_categories(self, categories: dict[str, str]) -> None:
        self._method_categories = categories
        if categories:
            self._obj.attrs["store_method_categories"] = categories
        else:
            self._obj.attrs.pop("store_method_categories", None)

    def register_method(self, method: Callable[..., Any] | None = None, *,
                        category: str | None = None) -> Callable[..., Any]:
        """Register a row-wise method, optionally below a slash-separated browser category."""
        if method is None:
            return lambda method: self.register_method(method, category=category)
        name = getattr(method, "__name__", None)
        if not callable(method) or not isinstance(name, str) or not name.isidentifier():
            raise TypeError("A store method must be a named callable.")
        if hasattr(type(self), name) or hasattr(_NodeRow(None), name) or name in self._methods:
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
        categories = self._method_categories.copy()
        if category is None:
            categories.pop(name, None)
        else:
            categories[name] = category
        self._set_method_categories(categories)
        self._compiled_methods[name] = method
        return method

    def unregister_method(self, name: str) -> None:
        """Remove a registered method by name; raise KeyError if it is missing."""
        methods = self._methods.copy()
        del methods[name]
        self._methods = methods
        self._obj.attrs["store_methods"] = methods
        categories = self._method_categories.copy()
        categories.pop(name, None)
        self._set_method_categories(categories)
        self._compiled_methods.pop(name, None)

    def _method(self, name: str) -> Callable[..., Any]:
        """Compile a registered method once; raise KeyError if it is missing."""
        if name not in self._compiled_methods:
            namespace: dict[str, Any] = {}
            exec(self._methods[name], namespace)
            self._compiled_methods[name] = namespace[name]
        return self._compiled_methods[name]

    def __getattr__(self, name: str) -> Any:
        try:
            method = self._method(name)
        except KeyError:
            raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}") from None

        @wraps(method)
        def call(*args: Any, **kwargs: Any) -> Any:
            def apply(row: pd.Series) -> Any:
                fields = {key: value for key, value in row.items()
                          if not (pd.api.types.is_scalar(value) and pd.isna(value))}
                node = self.filesystem.node(row["lf_path"])
                return method(_NodeRow(node, fields, self._method), *args, **kwargs)

            if isinstance(self._obj, pd.Series):
                return apply(self._obj)
            return pd.Series([apply(row) for _, row in self._obj.iterrows()],
                             index=self._obj.index, dtype="object", name=name)

        return call

    def get_registered_methods(self):
        return list(self._methods.keys())

    def get_registered_method_category(self, name: str) -> str | None:
        """Return menu metadata supplied when this method was registered."""
        return self._method_categories.get(name)

    @property
    def filesystem(self) -> FileSystem:
        """The table's filesystem, restored from its saved specification."""
        cache = self._obj.__dict__
        if "_filesystem" not in cache:
            try:
                spec = self._obj.attrs["filesystem_spec"]
            except KeyError:
                raise RuntimeError("This table has no filesystem specification.") from None
            cache["_filesystem"] = FileSystem.from_spec(
                spec, relative_to=self._obj.attrs["filesystem_relative_to"],
            )
        return cache["_filesystem"]

    def __getitem__(self, keys):
        series = isinstance(self._obj, pd.Series)
        filesystem = self.filesystem
        paths = self._obj["lf_path"]
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

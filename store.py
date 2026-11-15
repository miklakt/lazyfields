import __future__
import ast
import inspect
import sys
import textwrap
import typing
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime
from functools import cache, wraps
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from .pipe import _NodeRow, _apply_pipe
from .filesystem import FileSystem, Node

# Columns added to every row; indexed fields cannot use these names.
_RESERVED = ("lf_path", "lf_store_keys", "lf_mtime", "lf_digest")


def _read_only(self, *args, **kwargs):
    raise TypeError("A reference table's attrs value is read-only; assign a new one instead.")


class _FrozenDict(dict):
    """A dict that cannot change in place, so frames derived from a table share it instead of copying.

    It compares, encodes to JSON and pickles as a plain dict.
    """

    __setitem__ = __delitem__ = __ior__ = clear = pop = popitem = setdefault = update = _read_only

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return dict, (dict(self),)


class _FrozenList(list):
    __setitem__ = __delitem__ = __iadd__ = __imul__ = _read_only
    append = clear = extend = insert = pop = remove = reverse = sort = _read_only

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return list, (list(self),)


def _freeze(value: Any) -> Any:
    if isinstance(value, (_FrozenDict, _FrozenList)):
        return value
    if isinstance(value, Mapping):
        return _FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return _FrozenList(_freeze(item) for item in value)
    return value


class _TableFileSystem(_FrozenDict):
    """A table's {"spec", "relative_to"}; the FileSystem built from them is shared by its derived frames."""

    _filesystem: FileSystem | None = None

    @property
    def filesystem(self) -> FileSystem:
        if self._filesystem is None:
            self._filesystem = FileSystem.from_spec(self["spec"], relative_to=self["relative_to"])
        return self._filesystem


def _freeze_attrs(attrs: dict) -> None:
    """Freeze the attrs values lazyfields owns, which load from pickle or parquet as plain ones."""
    if "store_methods" in attrs:
        attrs["store_methods"] = _freeze(attrs["store_methods"])
    if "filesystem" in attrs and not isinstance(attrs["filesystem"], _TableFileSystem):
        attrs["filesystem"] = _TableFileSystem(_freeze(attrs["filesystem"]))


@cache
def _compile(name: str, source: str) -> Callable[..., Any]:
    """A registered method's function, compiled from its source once."""
    # Annotations stay strings, so a missing name never breaks compiling; the typing names
    # let the browser resolve common ones such as Literal for its argument suggestions.
    namespace: dict[str, Any] = {key: getattr(typing, key) for key in typing.__all__}
    exec(compile(source, f"<store method {name}>", "exec",
                 flags=__future__.annotations.compiler_flag, dont_inherit=True), namespace)
    return namespace[name]


def _row_value(node: Node, key: Any) -> Any:
    try:
        return node[key]
    except (KeyError, TypeError, IndexError):
        return pd.NA


def _validate_keys(keys):
    """Names become column names and path segments, so none may contain '/'."""
    invalid = next((key for key in keys if isinstance(key, str) and "/" in key), None)
    if invalid is not None:
        raise ValueError(f"Key {invalid!r} contains '/', which is reserved for nested paths.")


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
    _validate_keys(key for key, _, _ in items)
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
    previous = reference_df.attrs.get("filesystem", {})
    if source is None:
        source = previous["spec"]
        if relative_to is None:
            relative_to = previous["relative_to"]
    elif isinstance(source, (str, Path)):
        if relative_to is not None:
            raise ValueError("relative_to is only supported for filesystem specifications; "
                             "a directory source defines its own root.")
        relative_to = str(Path(source).resolve())
        source = {"mount": [{"glob": "*"}]}
    elif relative_to is None:
        relative_to = previous.get("relative_to")
    relative_to = str(relative_to) if relative_to is not None else None
    if not isinstance(source, Mapping):
        raise TypeError("source must be a directory or a filesystem specification mapping.")
    filesystem = FileSystem.from_spec(source, relative_to=relative_to)
    paths = sorted(name for name, kind, _ in filesystem.items() if kind == "group")
    old_rows = {row["lf_path"]: row for row in reference_df.to_dict("records")}
    handle = _TableFileSystem(_freeze({"spec": source, "relative_to": str(filesystem.relative_to)}))
    handle._filesystem = filesystem  # Built for indexing already.
    attrs = {**reference_df.attrs, "filesystem": handle}
    _freeze_attrs(attrs)

    def make_row(path):
        node = filesystem.node(path)
        if path in old_rows and old_rows[path]["lf_digest"] == node.digest():
            return old_rows[path]
        return _reference_row_from_node(node, columns=columns,
                                        pipe=pipe, pipe_error_skip=pipe_error_skip)

    pool = ThreadPoolExecutor(max_workers=workers) if workers not in (None, 1) else nullcontext()
    with pool as executor:
        df = pd.DataFrame(row for row in (executor.map if executor else map)(make_row, paths)
                          if row is not None)
    df.attrs = attrs
    return df


@pd.api.extensions.register_dataframe_accessor("store")
@pd.api.extensions.register_series_accessor("store")
class StoreAccessor:
    def __init__(self, pandas_obj: pd.DataFrame | pd.Series):
        _freeze_attrs(pandas_obj.attrs)
        self._obj = pandas_obj

    @property
    def _methods(self) -> Mapping[str, Mapping[str, str | None]]:
        """Registered methods by name, each with its "source", browser "category" and "scope"."""
        return self._obj.attrs.get("store_methods", {})

    def register_method(self, method: Callable[..., Any] | None = None, *, category: str | None = None,
                        scope: str = "row") -> Callable[..., Any]:
        """Register a method, optionally below a slash-separated browser category.

        A "row" method gets one row first, a "table" method the whole frame (in the browser, the
        current view); any further parameters are settings.
        """
        if method is None:
            return lambda method: self.register_method(method, category=category, scope=scope)
        if scope not in ("row", "table"):
            raise ValueError(f"scope must be 'row' or 'table', not {scope!r}.")
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
        self._obj.attrs["store_methods"] = _freeze(
            {**self._methods, name: {"source": ast.unparse(node), "category": category, "scope": scope}})
        return method

    def unregister_method(self, name: str) -> None:
        """Remove a registered method by name; raise KeyError if it is missing."""
        methods = dict(self._methods)
        del methods[name]
        self._obj.attrs["store_methods"] = _freeze(methods)

    def _method(self, name: str) -> Callable[..., Any]:
        """A registered method's function; raise KeyError if it is missing."""
        return _compile(name, self._methods[name]["source"])

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
                node = self._filesystem.node(row["lf_path"])
                return method(_NodeRow(node, fields, self._method), *args, **kwargs)

            if self._methods[name]["scope"] == "table":
                return method(self._obj, *args, **kwargs)
            if isinstance(self._obj, pd.Series):
                return apply(self._obj)
            return pd.Series([apply(row) for _, row in self._obj.iterrows()],
                             index=self._obj.index, dtype="object", name=name)

        return call

    @property
    def _filesystem(self) -> FileSystem:
        """The data namespace, built on first use and shared with every frame derived from this one."""
        if "filesystem" not in self._obj.attrs:
            raise RuntimeError("This table has no filesystem specification.")
        return self._obj.attrs["filesystem"].filesystem

    def __getitem__(self, key):
        """Read each row's node as Node indexing does; a list of field names gives one column each."""
        series = isinstance(self._obj, pd.Series)
        if isinstance(key, list):
            values = {k: self[k] for k in key}
            return values if series else pd.DataFrame(values, index=self._obj.index)
        if series:
            return self._filesystem.node(self._obj["lf_path"])[key]
        values = pd.Series((_row_value(self._filesystem.node(path), key) for path in self._obj["lf_path"]),
                           index=self._obj.index, dtype="object")
        if values.isna().all():
            raise KeyError(key)
        return values

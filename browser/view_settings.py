"""Table browser views remembered between runs, keyed by a hash so no path is stored."""

import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

_KEEP = 100  # Most recently saved table views to keep.
_FIELDS = {"filter": str, "sort": list, "column_order": list, "hidden": list}


def _file() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "lazyfields" / "table_views.json"


def table_key(table: pd.DataFrame) -> str:
    """Hash a table's filesystem specification and root with its column names, in any order.

    Tables with the same columns over the same data share a view, whether pickled or not, and it
    survives rebuilding the table or moving its pickle; moving the data root or changing the
    columns loses it.
    """
    text = json.dumps([table.attrs["filesystem"], sorted(map(str, table.columns))], sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _conformant(view) -> bool:
    """Whether view has the shape the table browser saves: a filter, [column, ascending] sort
    levels with column "#0" for the index, and lists of column names."""
    return (isinstance(view, dict) and set(view) <= set(_FIELDS)
            and all(isinstance(view.get(field, kind()), kind) for field, kind in _FIELDS.items())
            and all(isinstance(level, list) and len(level) == 2 and isinstance(level[0], str)
                    and isinstance(level[1], bool) for level in view.get("sort", []))
            and all(isinstance(name, str) for field in ("column_order", "hidden") for name in view.get(field, [])))


def _read() -> dict:
    """Saved views by key, without those of any other shape, which the next save purges."""
    try:
        tables = json.loads(_file().read_text())
    except (OSError, ValueError):
        return {}
    return {key: view for key, view in tables.items() if _conformant(view)} if isinstance(tables, dict) else {}


def load(key: str) -> dict:
    """The view saved under key, or an empty one."""
    return _read().get(key, {})


def save(key: str, view: dict) -> None:
    """Remember view under key, forgetting the least recently saved beyond _KEEP; failures are ignored.

    The file is replaced atomically and synced first, so a crash or power loss leaves either the
    old file or the new one, never a truncated one.
    """
    tables = _read()
    tables.pop(key, None)
    tables[key] = view  # Last in the file is the most recent.
    path, temp = _file(), None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name, suffix=".tmp",
                                         delete=False, encoding="utf-8") as file:
            temp = file.name
            json.dump(dict(list(tables.items())[-_KEEP:]), file, indent=1)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, path)
    except OSError:
        if temp is not None:
            with contextlib.suppress(OSError):
                os.unlink(temp)

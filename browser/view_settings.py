"""Table browser views remembered between runs, keyed by a hash so no path is stored."""

import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

_KEEP = 100  # Most recently saved table views to keep.


def _file() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "lazyfields" / "table_views.json"


def table_key(table: pd.DataFrame) -> str:
    """Hash a table's filesystem specification and root with its column names, in any order.

    Tables with the same columns over the same data share a view, whether pickled or not, and it
    survives rebuilding the table or moving its pickle; moving the data root or changing the
    columns loses it.
    """
    text = json.dumps([table.attrs["filesystem_spec"], table.attrs["filesystem_relative_to"],
                       sorted(map(str, table.columns))], sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _read() -> dict:
    try:
        tables = json.loads(_file().read_text())
    except (OSError, ValueError):
        return {}
    return tables if isinstance(tables, dict) else {}


def load(key: str) -> dict:
    """The view saved under key, or an empty one."""
    view = _read().get(key)
    return view if isinstance(view, dict) else {}


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

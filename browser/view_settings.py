"""Table browser views remembered between runs, keyed by a hash so no table path is stored."""

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

_KEEP = 100  # Most recently saved table views to keep.


def _file() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "lazyfields" / "table_views.json"


def table_key(source: pd.DataFrame | str | Path) -> str:
    """Hash a table pickle's resolved path, or a DataFrame's column names and lf_path values."""
    if isinstance(source, (str, Path)):
        text = str(Path(source).resolve())
    else:
        text = "\0".join([*map(str, source.columns), "", *source["lf_path"].astype(str)])
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
    """Remember view under key, forgetting the least recently saved beyond _KEEP; failures are ignored."""
    tables = _read()
    tables.pop(key, None)
    tables[key] = view  # Last in the file is the most recent.
    try:
        _file().parent.mkdir(parents=True, exist_ok=True)
        _file().write_text(json.dumps(dict(list(tables.items())[-_KEEP:]), indent=1))
    except OSError:
        pass

from collections.abc import Mapping
from copy import deepcopy

import pandas as pd


def _apply_pipe(row: pd.Series, pipe: list, *, scoped: bool = False):
    """Apply callbacks to ordinary pandas rows, isolating scoped row values."""
    last_was_filter = False
    for step in pipe:
        result = _apply_pipe(row.map(deepcopy), step, scoped=True) if isinstance(step, list) else step(row)
        if result is None: last_was_filter = False
        elif isinstance(result, (Mapping, pd.Series)):
            attrs = row.attrs
            row = result if isinstance(result, pd.Series) else pd.Series(result, dtype="object")
            row.attrs = attrs.copy()
            last_was_filter = False
        elif pd.api.types.is_bool(result):
            if not result: return False
            last_was_filter = True
        else: raise TypeError("Pipe callbacks must return a mapping, Series, bool, or None.")
    if scoped and not last_was_filter: raise TypeError("Scoped pipe blocks must end with a filter.")
    return True if scoped else row

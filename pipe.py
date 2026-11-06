from collections.abc import Mapping
from copy import deepcopy
from functools import partial

import pandas as pd

from .filesystem.filesystem import _split_key


class _NodeRow(dict):
    """Row fields over lazy reads from the row's node, for pipes and store methods."""

    def __init__(self, node, fields=(), methods=None):
        super().__init__(fields)
        self._node, self._methods = node, methods

    def __getattr__(self, name):
        methods = self.__dict__.get("_methods")
        try:
            if methods is None: raise KeyError(name)
            return partial(methods(name), self)
        except KeyError:
            raise AttributeError(f"Row has no registered method {name!r}.") from None

    def __getitem__(self, key):
        path, selection = _split_key(key)
        if path not in self:
            return self._node[key]
        value = super().__getitem__(path)
        return value if selection is None else value[selection]

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, IndexError, TypeError, NotADirectoryError):
            return default


def _apply_pipe(row: _NodeRow, pipe: list, *, scoped: bool = False):
    """Apply callbacks to a row, isolating scoped row values."""
    last_was_filter = False
    for step in pipe:
        result = (_apply_pipe(_NodeRow(row._node, deepcopy(dict(row))), step, scoped=True)
                  if isinstance(step, list) else step(row))
        if result is None: last_was_filter = False
        elif isinstance(result, Mapping): row, last_was_filter = _NodeRow(row._node, result), False
        elif pd.api.types.is_bool(result):
            if not result: return False
            last_was_filter = True
        else: raise TypeError("Pipe callbacks must return a mapping, bool, or None.")
    if scoped and not last_was_filter: raise TypeError("Scoped pipe blocks must end with a filter.")
    return True if scoped else row

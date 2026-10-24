from collections.abc import Mapping
from copy import deepcopy
from functools import partial

import pandas as pd


class _NodeRow(dict):
    """Row fields over lazy reads from the row's node, for pipes and store methods."""

    def __init__(self, node, fields=(), methods=None):
        super().__init__(fields)
        self.node, self._methods = node, methods

    def __getattr__(self, name):
        # Registered store methods, applied to this row.
        methods = self.__dict__.get("_methods")
        try:
            if methods is None: raise KeyError(name)
            return partial(methods(name), self)
        except KeyError:
            raise AttributeError(f"Row has no registered method {name!r}.") from None

    def __getitem__(self, key):
        if isinstance(key, tuple):
            name, selection = key[0], key[1] if len(key) == 2 else key[1:]
            return self[name][selection] if name in self else self.node.node(name).read(selection)
        return super().__getitem__(key) if key in self else self.node.node(key).read()

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, IndexError, TypeError, NotADirectoryError):
            return default


def _apply_pipe(row: _NodeRow, pipe: list, *, scoped: bool = False):
    """Apply callbacks to a row, isolating scoped row values."""
    last_was_filter = False
    for step in pipe:
        result = (_apply_pipe(_NodeRow(row.node, deepcopy(dict(row))), step, scoped=True)
                  if isinstance(step, list) else step(row))
        if result is None: last_was_filter = False
        elif isinstance(result, Mapping): row, last_was_filter = _NodeRow(row.node, result), False
        elif pd.api.types.is_bool(result):
            if not result: return False
            last_was_filter = True
        else: raise TypeError("Pipe callbacks must return a mapping, bool, or None.")
    if scoped and not last_was_filter: raise TypeError("Scoped pipe blocks must end with a filter.")
    return True if scoped else row

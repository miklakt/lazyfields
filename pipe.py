import warnings
from collections.abc import Mapping
from copy import deepcopy
from functools import partial

import pandas as pd

from .filesystem.filesystem import _split_key
from .readers import _names, _path_get


class _NodeRow(dict):
    """A row's fields over its read-only source, for pipes and store methods.

    row[path] reads the field named by the longest prefix of path, else the source; row.node reads the source.
    """

    def __init__(self, node, fields=(), methods=None):
        super().__init__(fields)
        self.node, self._methods = node, methods

    def __getattr__(self, name):
        methods = self.__dict__.get("_methods")
        try:
            if methods is None: raise KeyError(name)
            return partial(methods(name), self)
        except KeyError:
            raise AttributeError(f"Row has no registered method {name!r}.") from None

    def __getitem__(self, key):
        path, selection = _split_key(key)
        parts = path.split("/")
        for end in range(len(parts), 0, -1):
            if (field := "/".join(parts[:end])) in self:
                value = _path_get(super().__getitem__(field), _names("/".join(parts[end:])))
                return value if selection is None else value[selection]
        return self.node[key]

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


_shadowing: set[str] = set()  # Field names already reported as shadowing their source.


def _warn_shadowing(row: _NodeRow, source: Mapping) -> None:
    """Warn, once per name, of fields hiding source data; a source scalar kept as it was hides nothing."""
    for key, value in dict.items(row):
        if key in _shadowing or value is source.get(key, row):
            continue
        try:
            row.node.node(key).info(full=False)
        except (KeyError, IndexError, NotADirectoryError, ValueError):
            continue
        _shadowing.add(key)
        warnings.warn(f"Field {key!r} shadows the source's {key!r} and everything below it in row[...]; "
                      f"row.node[{key!r}] still reads the source. Further shadowing by {key!r} is not reported.")

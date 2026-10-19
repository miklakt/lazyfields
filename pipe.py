from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .readers import Reader, _path_get


class _PipeRow(dict):
    def __init__(self, data: Mapping[str, Any], reader: Reader, path: Path):
        super().__init__(data)
        self._reader, self._path = reader, path

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple):
            if len(key) < 2 or not isinstance(key[0], str):
                raise TypeError("Tuple access must start with a field name and a selection.")
            name, selection = key[0], key[1] if len(key) == 2 else key[1:]
            if selection is not None and self._reader.read is not None:
                return self._reader.get(self._path, name, selection)
            value = self[name]
            return value if selection is None else value[selection]
        return _path_get(self, key) if isinstance(key, str) and "/" in key else super().__getitem__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, IndexError, TypeError):
            return default


def _apply_pipe(row_data: Mapping[str, Any], pipe: list[Any], reader: Reader, path: Path, *, scoped: bool = False) -> Mapping[str, Any] | bool:
    row_data = _PipeRow(row_data, reader, path)
    last_was_filter = False
    for step in pipe:
        result = _apply_pipe(deepcopy(row_data), step, reader, path, scoped=True) if isinstance(step, list) else step(row_data)
        if result is None: last_was_filter = False
        elif isinstance(result, Mapping): row_data, last_was_filter = _PipeRow(result, reader, path), False
        elif isinstance(result, bool):
            if not result: return False
            last_was_filter = True
        else: raise TypeError("Pipe callbacks must return a mapping, bool, or None.")
    if scoped and not last_was_filter: raise TypeError("Scoped pipe blocks must end with a filter.")
    return True if scoped else row_data

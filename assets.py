"""Physical assets accessed through the existing file readers."""

import os
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from .readers import Reader, require_reader


class _ReaderAccessor:
    """Bracket syntax for a reader bound to a lazily resolved physical path."""

    def __init__(self, reader: Reader, path: Callable[[], Path]):
        self._reader, self._path = reader, path

    def load(self) -> Mapping[str, Any]:
        return self._reader.load(self._path())

    def get(self, key: str, selection: Any | None = None) -> Any:
        """Read a physical field, following Reader.get's strict semantics."""
        return self._reader.get(self._path(), key, selection)

    def items(self) -> Iterable[tuple[str, bool, Any]]:
        return self._reader.items(self._path())

    def __getitem__(self, keys: Any) -> Any:
        if isinstance(keys, slice) and keys == slice(None):
            return self.load()
        if isinstance(keys, str):
            return self.get(keys)
        if isinstance(keys, tuple):
            if len(keys) < 2 or not isinstance(keys[0], str):
                raise TypeError("Tuple access must start with a field name and a selection.")
            selection = keys[1] if len(keys) == 2 else keys[1:]
            return self.get(keys[0], selection)
        if isinstance(keys, list) and all(isinstance(key, str) for key in keys):
            return {key: self.get(key) for key in keys}
        raise TypeError("Keys must be str, tuple[str, selection], list[str], or [:]")


class AssetsCollection:
    """Bind physical files to readers and look them up by relative path or alias.

    ``filepaths`` accepts an iterable of paths or a single path. Repeated paths
    are registered once. Iteration yields the registered relative paths.

    ``readers`` supplies collection-local readers, taking precedence over the
    registered readers. ``aliases`` maps additional names to registered paths.
    Asset paths must be relative to the directory ``relative_to``, which defaults
    to the working directory at construction. Construction does not load files.
    """

    def __init__(
        self,
        filepaths: Iterable[str | Path] | str | Path,
        readers: Reader | Iterable[Reader] | None = None,
        *,
        aliases: Mapping[str, str | Path] | None = None,
        relative_to: str | Path | None = None,
    ):
        self.relative_to = Path(relative_to).resolve() if relative_to is not None else Path.cwd()
        self._readers: dict[Path, Reader] = {}
        self._aliases: dict[str, Path] = {}

        local_readers = (readers,) if isinstance(readers, Reader) else readers or ()
        suffix_readers = {}
        for reader in local_readers:
            if not isinstance(reader, Reader):
                raise TypeError("readers must contain Reader instances.")
            for suffix in reader.suffixes:
                suffix = suffix.lower()
                if not suffix.startswith("."):
                    raise ValueError("Reader suffixes must start with '.'.")
                if suffix in suffix_readers:
                    raise ValueError(f"Multiple local readers for {suffix!r}.")
                suffix_readers[suffix] = reader

        if isinstance(filepaths, (str, Path)):
            filepaths = [filepaths]
        if isinstance(filepaths, Mapping):
            raise TypeError("filepaths must contain paths; use aliases to name them.")
        for path in filepaths:
            path = self._relative(path)
            if path not in self._readers:
                suffixes = path.suffixes
                reader = next((
                    suffix_readers[suffix]
                    for start in range(len(suffixes))
                    if (suffix := "".join(suffixes[start:]).lower()) in suffix_readers
                ), None)
                if reader is None:
                    reader = require_reader(path)
                self._readers[path] = reader

        for alias, target in (aliases or {}).items():
            if not isinstance(alias, str):
                raise TypeError("Asset aliases must be strings.")
            path = self._relative(target)
            if path not in self._readers:
                raise KeyError(f"Unknown asset: {target!r}")
            name_path = self._relative(alias)
            if name_path in self._readers and name_path != path:
                raise ValueError(f"Asset alias {alias!r} conflicts with a registered path.")
            self._aliases[alias] = path

    @staticmethod
    def _relative(path: str | Path) -> Path:
        if not isinstance(path, (str, Path)):
            raise TypeError("Asset paths must be strings or Path objects.")
        if Path(path).is_absolute():
            raise ValueError("Asset paths must be relative to relative_to.")
        return Path(os.path.normpath(path))

    def _resolve(self, reference: str | Path) -> Path:
        if isinstance(reference, str) and reference in self._aliases:
            return self._aliases[reference]
        path = self._relative(reference)
        if path in self._readers:
            return path
        raise KeyError(f"Unknown asset: {reference!r}")

    def __call__(self, reference: str | Path) -> _ReaderAccessor:
        path = self._resolve(reference)
        return _ReaderAccessor(self._readers[path], lambda: (self.relative_to / path).resolve())

    def __iter__(self) -> Iterator[Path]:
        return iter(self._readers)

    def __len__(self) -> int:
        return len(self._readers)

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(self._readers)

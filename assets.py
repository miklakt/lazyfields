"""A thin read-only filesystem binding registered physical files to readers."""

import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .filesystem import AssetNode, PhysicalLocator, _MountIndex
from .readers import NodeInfo, Reader, require_reader


class AssetsCollection:
    """Read-only namespace over registered files and reader-defined infile paths.

    Registration and node creation do not read contents. Local readers override
    global readers; paths are relative to ``relative_to`` (the current directory
    by default). Repeated files are mounted once. Values are not cached.
    """

    def __init__(
        self,
        filepaths: Iterable[str | Path] | str | Path,
        readers: Reader | Iterable[Reader] | None = None,
        *,
        relative_to: str | Path | None = None,
    ):
        self.relative_to = Path(relative_to).resolve() if relative_to is not None else Path.cwd()
        self._readers: dict[Path, Reader] = {}

        local_readers = (readers,) if isinstance(readers, Reader) else readers or ()
        suffix_readers = {}
        for reader in local_readers:
            if not isinstance(reader, Reader):
                raise TypeError("readers must contain Reader instances.")
            for suffix in reader.suffixes:
                suffix = suffix.lower()
                if not suffix.startswith(".") or suffix in suffix_readers:
                    raise ValueError(f"Invalid or duplicate reader suffix: {suffix!r}")
                suffix_readers[suffix] = reader

        if isinstance(filepaths, (str, Path)):
            filepaths = [filepaths]
        if isinstance(filepaths, Mapping):
            raise TypeError("filepaths must contain paths.")
        suffixes = sorted(suffix_readers, key=len, reverse=True)
        for path in filepaths:
            path = self._relative(path)
            if path not in self._readers:
                filename = path.name.lower()
                suffix = next((suffix for suffix in suffixes if filename.endswith(suffix)), None)
                self._readers[path] = suffix_readers.get(suffix) or require_reader(path)

        self._mounts = _MountIndex(self._readers)

    @staticmethod
    def _relative(path: str | Path) -> Path:
        if not isinstance(path, (str, Path)):
            raise TypeError("Asset paths must be strings or Path objects.")
        if Path(path).is_absolute():
            raise ValueError("Asset paths must be relative to relative_to.")
        path = Path(os.path.normpath(path))
        if path == Path(".") or path.parts[0] == "..":
            raise ValueError("Asset paths must identify files below relative_to.")
        return path

    def node(self, path: str | Path = "") -> AssetNode:
        """Bind a lazy handle, checking mounts but not infile contents."""
        path = self._mounts.normalize(path)
        self._mounts.split(path)
        return AssetNode(self, path)

    def locate(self, path: str | Path, selection: Any | None = None) -> PhysicalLocator:
        """Resolve a namespace path to a file/path/selection triple without I/O."""
        path = self._mounts.normalize(path)
        file_path, infile_path = self._mounts.split(path)
        if file_path is None:
            raise IsADirectoryError(f"Physical directory has no single locator: {path!r}")
        if selection is not None and infile_path is None:
            raise ValueError("selection requires an infile path.")
        return PhysicalLocator(file_path, infile_path, selection)

    def read(self, path: str | Path = "", selection: Any | None = None) -> Any:
        """Read a node or materialize a directory's registered children.

        Dataset selections are forwarded unchanged to the mounted reader.
        """
        path = self._mounts.normalize(path)
        file_path, infile_path = self._mounts.split(path)
        if file_path is None:
            if selection is not None:
                raise ValueError("selection requires an infile path.")
            return {name: self.read(f"{path}/{name}") for name in self._mounts.directories[path]}
        return self._readers[file_path].get(self.relative_to / file_path, infile_path, selection)

    def listdir(self, path: str | Path = "") -> tuple[str, ...]:
        """List child names through the index or reader's metadata/fallback."""
        info = self._info(path)
        if info.kind != "group":
            raise NotADirectoryError(str(path))
        return info.children

    def _info(self, path: str | Path, *, list_children: bool = True) -> NodeInfo:
        path = self._mounts.normalize(path)
        file_path, infile_path = self._mounts.split(path)
        if file_path is None:
            return NodeInfo("group", tuple(self._mounts.directories[path]))
        if infile_path is None and not list_children:
            return NodeInfo("group")
        return self._readers[file_path].info(self.relative_to / file_path, infile_path)

    def __iter__(self) -> Iterator[Path]:
        return iter(self._readers)

    def __len__(self) -> int:
        return len(self._readers)

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(self._readers)

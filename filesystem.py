"""Editable filesystems over lazy, read-only structured data sources."""

from __future__ import annotations

import os
from copy import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from .readers import NodeInfo, NodeKind, Reader, require_reader

if TYPE_CHECKING:
    from .merge import _MergedDataset

__all__ = ["FileSystem", "Node", "SourceLocator"]


def _normalize_path(path: str | Path) -> str:
    if not isinstance(path, (str, Path)):
        raise TypeError("Filesystem paths must be strings or Path objects.")
    parts = [part for part in str(path).split("/") if part not in ("", ".")]
    if ".." in parts:
        raise ValueError("Filesystem paths cannot contain '..'.")
    return "/".join(parts)


def _relative_file_path(path: str | Path) -> Path:
    """Normalize a registered filename while keeping it below the source root."""
    if not isinstance(path, (str, Path)):
        raise TypeError("File paths must be strings or Path objects.")
    if Path(path).is_absolute():
        raise ValueError("File paths must be relative to relative_to.")
    path = Path(os.path.normpath(path))
    if path == Path(".") or path.parts[0] == "..":
        raise ValueError("File paths must identify files below relative_to.")
    return path


def _reader_overrides(readers: Reader | Iterable[Reader] | None) -> list[tuple[str, Reader]]:
    """Validate local readers and order their suffixes from most to least specific."""
    readers = (readers,) if isinstance(readers, Reader) else readers or ()
    overrides = {}
    for reader in readers:
        if not isinstance(reader, Reader):
            raise TypeError("readers must contain Reader instances.")
        for suffix in reader.suffixes:
            suffix = suffix.lower()
            if not suffix.startswith(".") or suffix in overrides:
                raise ValueError(f"Invalid or duplicate reader suffix: {suffix!r}")
            overrides[suffix] = reader
    return sorted(overrides.items(), key=lambda item: len(item[0]), reverse=True)


@dataclass(frozen=True)
class SourceLocator:
    """An absolute filename, infile path, and selection independent of the namespace."""

    file_path: Path
    infile_path: str | None = None
    selection: Any | None = None


@dataclass(frozen=True)
class _FileSource:
    """Physical source adapter; selections pass directly to the reader."""

    is_merged = False

    reader: Reader
    origin: SourceLocator

    def read(self, selection=None):
        return self.reader.get(self.origin.file_path, self.origin.infile_path, selection)

    def info(self, *, list_children=True):
        if self.origin.infile_path is None and not list_children:
            return NodeInfo("group")
        return self.reader.info(self.origin.file_path, self.origin.infile_path)

    def items(self):
        return ((str(name), kind, value) for name, kind, value in
                self.reader.items(self.origin.file_path, self.origin.infile_path))

    def locate(self, selection=None):
        if selection is not None and self.origin.infile_path is None:
            raise ValueError("selection requires a dataset path.")
        return [replace(self.origin, selection=selection)]

    def descendant(self, path):
        parent = self.origin.infile_path
        origin = replace(self.origin, infile_path=f"{parent}/{path}" if parent else path)
        return FileSystem._from_source(replace(self, origin=origin))


@dataclass(frozen=True)
class Node:
    """Lazy namespace handle; subsequent edits are visible through its path."""

    _filesystem: FileSystem = field(repr=False)
    path: str

    def node(self, path: str | Path = "") -> Node:
        """Bind a descendant using a path relative to this node."""
        return self._filesystem.node(f"{self.path}/{_normalize_path(path)}")

    def read(self, selection: Any | None = None) -> Any:
        return self._filesystem.read(self.path, selection)

    def listdir(self) -> tuple[str, ...]:
        return self._filesystem.listdir(self.path)

    def items(self) -> Iterable[tuple[str, NodeKind, Any]]:
        return self._filesystem.items(self.path)

    def locate(self, selection: Any | None = None) -> list[SourceLocator]:
        return self._filesystem.locate(self.path, selection)

    @property
    def is_root(self) -> bool:
        """Whether this is the namespace root or an attached filesystem root."""
        return not self.path or self.is_mount

    @property
    def origin(self) -> SourceLocator | None:
        """Original source coordinates, retained through namespace edits."""
        return self._filesystem._lookup(self.path)._origin

    @property
    def in_file(self) -> bool:
        """Whether this node originates from file contents, including a file root."""
        return self.origin is not None

    @property
    def is_merged(self) -> bool:
        """Whether this node is a merge-created group or computed dataset."""
        return self._filesystem._lookup(self.path).is_merged

    @property
    def is_mount(self) -> bool:
        """Whether this position was explicitly attached through mount()."""
        return self._filesystem._lookup(self.path).is_mount

    @property
    def is_relocated(self) -> bool:
        """Whether this attachment relocates physical sources in the namespace."""
        return self._filesystem._lookup(self.path).is_relocated

    @property
    def kind(self) -> NodeKind:
        return self._filesystem._lookup(self.path)._info(list_children=False).kind


class FileSystem:
    """Compose reader-backed trees through mounts and independent clones.

    Namespace edits never write source files. Registration, handles, clones and
    locator translation perform no reads. Editing a source group discovers its
    children through Reader metadata or fallback reads; values are not cached.
    """

    is_root = True  # A standalone filesystem represents its own root.
    is_merged = False

    def __init__(
        self,
        filepaths: Iterable[str | Path] | str | Path = (),
        readers: Reader | Iterable[Reader] | None = None,
        *,
        relative_to: str | Path | None = None,
    ):
        self.relative_to = Path(relative_to).resolve() if relative_to is not None else Path.cwd()
        self.is_mount = False
        self.is_relocated = False
        self._namespace_path = self.relative_to
        self._source: _FileSource | _MergedDataset | None = None
        self._children: dict[str, FileSystem] | None = {}
        overrides = _reader_overrides(readers)
        if isinstance(filepaths, (str, Path)):
            filepaths = [filepaths]
        if isinstance(filepaths, Mapping):
            raise TypeError("filepaths must contain paths.")
        paths = dict.fromkeys(_relative_file_path(path) for path in filepaths)
        if any(parent in paths for path in paths for parent in path.parents):
            raise ValueError("Registered file paths cannot overlap.")
        for path in paths:
            reader = next((reader for suffix, reader in overrides if path.name.lower().endswith(suffix)), None)
            self.mount(path, self.from_file(self.relative_to / path, reader).node())

    @staticmethod
    def merge(sources: Mapping[str, Node], *, rules: Mapping) -> FileSystem:
        """Compose named source groups using per-key rules from lazyfields.merge."""
        from .merge import merge
        return merge(sources, rules=rules)

    @property
    def in_file(self) -> bool:
        return self._origin is not None

    @property
    def _origin(self) -> SourceLocator | None:
        return self._source.origin if self._source is not None else None

    @classmethod
    def _from_source(cls, source: _FileSource | _MergedDataset) -> FileSystem:
        origin = source.origin
        result = cls(relative_to=origin.file_path.parent if origin else None)
        result._source, result._children = source, None
        result.is_merged = source.is_merged
        if origin:
            result._namespace_path = origin.file_path / (origin.infile_path or "")
        return result

    @classmethod
    def from_file(cls, path: str | Path, reader: Reader | None = None) -> FileSystem:
        """Root a filesystem at a file's contents without reading them."""
        if reader is not None and not isinstance(reader, Reader):
            raise TypeError("reader must be a Reader instance.")
        path = Path(path).absolute()
        return cls._from_source(_FileSource(reader or require_reader(path), SourceLocator(path)))

    def _lookup(self, path: str | Path) -> FileSystem:
        parts = tuple(filter(None, _normalize_path(path).split("/")))
        entry = self
        for index, name in enumerate(parts):
            if entry._children is None:
                return entry._source.descendant("/".join(parts[index:]))
            entry = entry._children[name]
        return entry

    def _expand(self) -> dict[str, FileSystem]:
        """Discover child trees only when editing their parent."""
        if self._children is None:
            info = self._source.info()
            if info.kind != "group":
                raise NotADirectoryError(str(self._origin or "composed dataset"))
            self._children = {name: self._source.descendant(name) for name in info.children}
        return self._children

    def _parent(self, path: str | Path, *, create: bool = False) -> tuple[dict[str, FileSystem], str]:
        """Expand only the ancestry needed for a namespace edit."""
        path = _normalize_path(path)
        if not path:
            raise ValueError("Mount and unmount require a non-root path.")
        *parents, name = path.split("/")
        entry = self
        for part in parents:
            children = entry._expand()
            if create and part not in children:
                children[part] = FileSystem(relative_to=self.relative_to)
            entry = children[part]
        return entry._expand(), name

    def clone(self, path: str | Path = "") -> FileSystem:
        """Clone namespace structure while sharing lazy physical sources."""
        entry = self._lookup(path)
        result = copy(entry)
        if entry._children is not None:
            result._children = {name: child.clone() for name, child in entry._children.items()}
        result._namespace_path = self._namespace_path / _normalize_path(path)
        return result

    def _relocated(self, destination: Path) -> bool:
        """Compare source positions without reading or discovering children."""
        if self._origin is not None:
            return destination != self._origin.file_path / (self._origin.infile_path or "")
        return self.is_merged or any(
            child._relocated(destination / name) for name, child in self._children.items())

    def mount(self, path: str | Path, source: Node) -> None:
        """Attach an independent subtree at a vacant path; create missing parents."""
        if not isinstance(source, Node):
            raise TypeError("Mount sources must be nodes obtained with node(path).")
        entry = source._filesystem.clone(source.path)
        children, name = self._parent(path, create=True)
        if name in children:
            raise FileExistsError(str(path))
        destination = self._namespace_path / _normalize_path(path)
        entry.is_mount = True
        entry.is_relocated = entry._relocated(destination)
        entry._namespace_path = destination
        children[name] = entry

    def unmount(self, path: str | Path) -> None:
        """Remove a namespace subtree without writing its source files."""
        children, name = self._parent(path)
        del children[name]

    def node(self, path: str | Path = "") -> Node:
        path = _normalize_path(path)
        self._lookup(path)
        return Node(self, path)

    def read(self, path: str | Path = "", selection: Any | None = None) -> Any:
        entry = self._lookup(path)
        if entry._children is None:
            return entry._source.read(selection)
        if selection is not None:
            raise ValueError("selection requires a dataset path.")
        return {name: child.read() for name, child in entry._children.items()}

    def _info(self, *, list_children: bool = True) -> NodeInfo:
        if self._children is not None:
            return NodeInfo("group", tuple(self._children) if list_children else ())
        return self._source.info(list_children=list_children)

    def listdir(self, path: str | Path = "") -> tuple[str, ...]:
        info = self._lookup(path)._info()
        if info.kind != "group":
            raise NotADirectoryError(str(path))
        return info.children

    def items(self, path: str | Path = "") -> Iterable[tuple[str, NodeKind, Any]]:
        entry = self._lookup(path)
        if entry._children is None:
            yield from entry._source.items()
        else:
            for name, child in entry._children.items():
                kind = child._info(list_children=False).kind
                yield name, kind, child.read() if kind == "scalar" else None

    def locate(self, path: str | Path = "", selection: Any | None = None) -> list[SourceLocator]:
        entry = self._lookup(path)
        if entry._children is None:
            return entry._source.locate(selection)
        if selection is not None:
            raise ValueError("selection requires a dataset path.")
        return [locator for child in entry._children.values() for locator in child.locate()]

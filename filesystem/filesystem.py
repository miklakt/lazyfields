"""Editable filesystems over lazy, read-only structured data sources."""

from __future__ import annotations

import hashlib
import os
from copy import copy
from dataclasses import dataclass, field, replace
from glob import has_magic
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from ..readers import NodeInfo, NodeKind, Reader, _reader_match, require_reader

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

    def digest(self):
        # Every node inside a file changes with the file; a deleted file has no stat.
        try:
            stat = self.origin.file_path.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
        except FileNotFoundError:
            signature = None
        state = (str(self.origin.file_path), self.origin.infile_path, signature)
        return hashlib.blake2b(repr(state).encode(), digest_size=16).hexdigest()

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

    def stat(self) -> os.stat_result | None:
        return self._filesystem.stat(self.path)

    def digest(self) -> str:
        return self._filesystem.digest(self.path)

    @property
    def _entry(self) -> FileSystem:
        return self._filesystem._lookup(self.path)

    @property
    def origin(self) -> SourceLocator | None:
        """Original source coordinates, retained through namespace edits."""
        return self._entry._origin

    @property
    def is_merged(self) -> bool:
        """Whether this node is a merge-created group or computed dataset."""
        return self._entry.is_merged

    @property
    def is_mount(self) -> bool:
        """Whether this position was explicitly attached through mount()."""
        return self._entry.is_mount

    @property
    def is_relocated(self) -> bool:
        """Whether this attachment relocates physical sources in the namespace."""
        return self._entry.is_relocated

    @property
    def kind(self) -> NodeKind:
        return self._entry._info(list_children=False).kind


class FileSystem:
    """Compose reader-backed trees through mounts and independent clones.

    Namespace edits never write source files. Registration, handles, clones and
    locator translation perform no reads. Editing a source group discovers its
    children through Reader metadata or fallback reads; values are not cached.
    """

    is_merged = False
    is_mount = False
    is_relocated = False
    _source: _FileSource | _MergedDataset | None = None
    _children: dict[str, FileSystem] | None = None
    _source_children: tuple[str, ...] | None = None
    _path_patterns: tuple[tuple[str, ...], tuple[str, ...]] | None = None

    def __init__(
        self,
        filepaths: Iterable[str | Path] | str | Path = (),
        *,
        relative_to: str | Path | None = None,
    ):
        self.relative_to = Path(relative_to).resolve() if relative_to is not None else Path.cwd()
        self._namespace_path = self.relative_to
        self._children = {}
        if isinstance(filepaths, (str, Path)):
            filepaths = [filepaths]
        if isinstance(filepaths, Mapping):
            raise TypeError("filepaths must contain paths.")
        paths = dict.fromkeys(_relative_file_path(path) for path in filepaths)
        if any(parent in paths for path in paths for parent in path.parents):
            raise ValueError("Registered file paths cannot overlap.")
        for path in paths:
            self.mount(path, self.from_file(self.relative_to / path).node())

    @staticmethod
    def merge(sources: Mapping[str, Node], *, rules: Mapping) -> FileSystem:
        """Compose named source groups using rules from lazyfields.filesystem.merge."""
        from .merge import merge
        return merge(sources, rules=rules)

    @property
    def _origin(self) -> SourceLocator | None:
        return self._source.origin if self._source is not None else None

    @classmethod
    def _from_source(cls, source: _FileSource | _MergedDataset) -> FileSystem:
        """Bind a lazy source without reinitializing a physical namespace."""
        origin = source.origin
        result = cls.__new__(cls)
        result.relative_to = origin.file_path.parent if origin else Path.cwd()
        result._source = source
        result.is_merged = source.is_merged
        result._namespace_path = origin.file_path / (origin.infile_path or "") if origin else result.relative_to
        return result

    @classmethod
    def from_file(cls, path: str | Path, reader: Reader | None = None) -> FileSystem:
        """Root a filesystem at a file's contents without reading them."""
        if reader is not None and not isinstance(reader, Reader):
            raise TypeError("reader must be a Reader instance.")
        path = Path(path).absolute()
        return cls._from_source(_FileSource(reader or require_reader(path), SourceLocator(path)))

    @classmethod
    def from_directory(cls, path: str | Path, pattern: str = "*") -> FileSystem:
        """Mount matching supported files at their relative paths without reading them."""
        root = Path(path)
        if not root.is_dir():
            raise NotADirectoryError(str(root))
        return cls._from_paths(root, [pattern])

    @classmethod
    def _from_paths(cls, root: Path, paths: Iterable[str]) -> FileSystem:
        """Register literal filenames and the supported files matching glob patterns."""
        paths = [_relative_file_path(path).as_posix() for path in paths]
        if not any(map(has_magic, paths)):
            return cls(paths, relative_to=root)
        if not root.is_dir():
            raise NotADirectoryError(str(root))
        matches = sorted({
            match for path in paths
            for match in ([file.relative_to(root) for file in root.glob(path)
                           if file.is_file() and _reader_match(file)[0] is not None]
                          if has_magic(path) else [Path(path)])
        })
        result = cls(matches, relative_to=root)
        # Retain patterns so an unedited registration exports them, not its matches.
        result._path_patterns = (tuple(paths), tuple(file.as_posix() for file in matches))
        return result

    def to_spec(
        self, *, relative_to: str | Path | None = None,
        readers: Mapping[str, Reader] | None = None,
    ) -> dict:
        """Return a data-only snapshot of current composition, without source reads.

        Paths are relative to relative_to, defaulting to this filesystem's base.
        Custom readers require names in readers; supply the same names when
        restoring the specification. Source groups with retained discovery use
        overrides/unmount; other groups retain exact children.
        """
        from .specification import to_spec
        return to_spec(self, relative_to=relative_to, readers=readers)

    @classmethod
    def from_spec(
        cls, spec: Mapping, *, relative_to: str | Path | None = None,
        readers: Mapping[str, Reader] | None = None,
    ) -> FileSystem:
        """Restore relative to relative_to or cwd; overlays discover source children."""
        from .specification import from_spec
        return from_spec(cls, spec, relative_to=relative_to, readers=readers)

    def _lookup(self, path: str | Path) -> FileSystem:
        """Find a namespace entry or bind its remaining path inside a lazy source."""
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
            self._source_children = tuple(info.children)
        return self._children

    def _parent(self, path: str | Path, *, create: bool = False) -> tuple[FileSystem, str]:
        """Expand only the ancestry needed for a namespace edit."""
        path = _normalize_path(path)
        if not path:
            raise ValueError("Namespace edits require a non-root path.")
        *parents, name = path.split("/")
        entry = self
        for part in parents:
            children = entry._expand()
            if create and part not in children:
                children[part] = FileSystem(relative_to=self.relative_to)
            entry = children[part]
        entry._expand()
        return entry, name

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

    def _attachment(self, path: str | Path, source: Node) -> FileSystem:
        """Prepare a mounted snapshot before changing the destination namespace."""
        if not isinstance(source, Node):
            raise TypeError("Mount sources must be nodes obtained with node(path).")
        entry = source._filesystem.clone(source.path)
        destination = self._namespace_path / _normalize_path(path)
        entry.is_mount = True
        entry.is_relocated = entry._relocated(destination)
        entry._namespace_path = destination
        return entry

    def _replace_child(self, name: str, entry: FileSystem) -> None:
        """Commit an existing child's replacement, appended as a fresh mount."""
        children = self._children.copy()
        del children[name]
        children[name] = entry
        self._children = children

    def mount(self, path: str | Path, source: Node) -> None:
        """Attach an independent subtree at a vacant path; create missing parents."""
        entry = self._attachment(path, source)
        parent, name = self._parent(path, create=True)
        if name in parent._children:
            raise FileExistsError(str(path))
        parent._children[name] = entry

    def replace(self, path: str | Path, source: Node) -> None:
        """Atomically replace an existing subtree with an independent mounted copy.

        Preparation failures retain the old attachment. Like unmount then mount,
        replacement moves the child to the end of its parent's listing.
        """
        entry = self._attachment(path, source)
        parent, name = self._parent(path)
        parent._replace_child(name, entry)

    def unmount(self, path: str | Path) -> None:
        """Remove a namespace subtree without writing its source files."""
        parent, name = self._parent(path)
        del parent._children[name]

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

    def stat(self, path: str | Path = "") -> os.stat_result | None:
        origin = self._lookup(path)._origin
        return origin.file_path.stat() if origin is not None else None

    def digest(self, path: str | Path = "") -> str:
        """Change digest from file stats and namespace structure, without source reads."""
        entry = self._lookup(path)
        if entry._children is None:
            return entry._source.digest()
        state = [(name, child.digest()) for name, child in entry._children.items()]
        return hashlib.blake2b(repr(state).encode(), digest_size=16).hexdigest()

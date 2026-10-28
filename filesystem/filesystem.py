"""Editable filesystems over lazy, read-only structured data sources."""

from __future__ import annotations

import hashlib
import os
from copy import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from ..readers import MAPPING, NodeInfo, NodeKind, View, _reader_match, registered_view, require_reader, value_info

if TYPE_CHECKING:
    from .merge import _MergedDataset, _MergedGroup

__all__ = ["FileSystem", "Node", "SourceLocator"]


def _normalize_path(path: str | Path) -> str:
    if not isinstance(path, (str, Path)):
        raise TypeError("Filesystem paths must be strings or Path objects.")
    parts = [part for part in str(path).split("/") if part not in ("", ".")]
    if ".." in parts:
        raise ValueError("Filesystem paths cannot contain '..'.")
    return "/".join(parts)


def _join(*keys: str | None) -> str | None:
    """Keys joined by "/", skipping empty ones; None when all are."""
    return "/".join(filter(None, keys)) or None


def _is_name(name: Any) -> bool:
    return isinstance(name, str) and name not in ("", ".", "..") and "/" not in name


def _digest(state: Any) -> str:
    return hashlib.blake2b(repr(state).encode(), digest_size=16).hexdigest()


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

    @property
    def position(self) -> Path:
        return self.file_path / (self.infile_path or "")


@dataclass(frozen=True, eq=False)
class _Source:
    """An object interpreted by a view at a key: a file's path, an in-memory value, or a node to interpret."""

    merged = None

    view: View
    obj: Any
    key: str | None = None

    @property
    def origin(self) -> SourceLocator | None:
        if isinstance(self.obj, FileSystem):  # Positions continue below the interpreted node.
            outer = self.obj._origin
            return outer and replace(outer, infile_path=_join(outer.infile_path, self.key))
        return SourceLocator(self.obj, self.key) if isinstance(self.obj, Path) else None

    @property
    def _object(self):
        """What the view interprets; an interpreted node's value is read for each request."""
        return self.obj.read() if isinstance(self.obj, FileSystem) else self.obj

    def read(self, selection=None):
        return self.view.get(self._object, self.key, selection)

    @property
    def format(self):
        return self.view.format if self.key is None else None

    def info(self, *, full=True):
        if self.key is None and not full and isinstance(self.obj, (Path, FileSystem)):
            return NodeInfo("group")  # A file's root, or an interpreted node, is a group; saying so reads nothing.
        info = self.view.info(self._object, self.key)
        # Views without metadata describe a dataset only by loading it.
        return value_info(self.read()) if full and info.kind == "dataset" and info.dtype is None else info

    def items(self):
        return ((str(name), kind, value) for name, kind, value in self.view.items(self._object, self.key))

    def locate(self, selection=None):
        if selection is not None and self.key is None:
            raise ValueError("selection requires a dataset path.")
        origin = self.origin
        return [replace(origin, selection=selection)] if origin is not None else []

    def digest(self):
        origin = self.origin
        if origin is None:  # An in-memory object changes without a trace; only its identity counts.
            return _digest((id(self.obj), self.key))
        try:
            stat = origin.file_path.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
        except FileNotFoundError:
            signature = None
        return _digest((str(origin.file_path), origin.infile_path, signature))

    def descendant(self, path):
        return FileSystem._from_source(replace(self, key=_join(self.key, path)))


@dataclass(frozen=True)
class _ChildrenSource:
    """A group answering from its namespace children."""

    children: dict[str, FileSystem]

    def read(self, selection=None):
        if selection is not None:
            raise ValueError("selection requires a dataset path.")
        return {name: child.read() for name, child in self.children.items()}

    def info(self, *, full=True):
        return NodeInfo("group", tuple(self.children) if full else ())

    def items(self):
        for name, child in self.children.items():
            kind = child._backing.info(full=False).kind
            yield name, kind, child.read() if kind == "scalar" else None

    def locate(self, selection=None):
        if selection is not None:
            raise ValueError("selection requires a dataset path.")
        return [locator for child in self.children.values() for locator in child.locate()]

    def digest(self):
        return _digest([(name, child.digest()) for name, child in self.children.items()])

    def descendant(self, path):
        name, _, rest = path.partition("/")
        return self.children[name]._lookup(rest)


@dataclass(frozen=True)
class _Composed(_ChildrenSource):
    """Children resolved once, when built; editing works on copies, keeping these for export."""

    origin = format = merged = None

    def descendant(self, path):
        name, _, rest = path.partition("/")
        return self.children[name].clone(rest)


@dataclass(frozen=True)
class _Directory(_Composed):
    """A directory with the supported files its glob patterns matched."""

    format = "directory"

    origin: SourceLocator
    patterns: tuple[str, ...] = ()


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

    def files(self) -> set[Path]:
        return {locator.file_path for locator in self.locate()}

    def digest(self) -> str:
        return self._filesystem.digest(self.path)

    def info(self, *, full: bool = True) -> NodeInfo:
        return self._filesystem.info(self.path, full=full)

    @property
    def origin(self) -> SourceLocator | None:
        """Original source coordinates, retained through namespace edits."""
        return self._filesystem._lookup(self.path)._origin

    @property
    def attrs(self) -> dict[str, Any]:
        return self._filesystem.attrs(self.path)


class FileSystem:
    """Compose lazy data trees through mounts and independent clones.

    Namespace edits leave source files unchanged.
    """

    _source: _Source | _Directory | _MergedDataset | _MergedGroup | None = None
    _children: dict[str, FileSystem] | None = None
    _source_children: tuple[str, ...] | None = None  # The source's child names when expanded, for export.
    _attrs: Mapping[str, Any] = {}  # From the specification; replaced, never mutated.

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
    def merge(sources: Mapping[str, Node], *, rules) -> FileSystem:
        """Merge named sources with a rule from lazyfields.filesystem.merge, or key by key with a mapping of them."""
        from .merge import merge
        return merge(sources, rules=rules)

    @property
    def _origin(self) -> SourceLocator | None:
        return getattr(self._source, "origin", None)

    @property
    def _merged(self) -> str | None:
        return getattr(self._source, "merged", None)

    def _pristine(self) -> FileSystem | None:
        """This entry's source alone, without namespace edits or attributes."""
        return FileSystem._from_source(self._source) if self._source is not None else None

    @property
    def _backing(self) -> _Source | _MergedDataset | _ChildrenSource:
        """Namespace children once expanded, else the source, as the view its "view" attribute names interprets it."""
        if self._children is not None:
            return _ChildrenSource(self._children)
        name = self._attrs.get("view")
        return self._source if name is None else _Source(registered_view(name), self._pristine())

    @classmethod
    def _from_source(cls, source: _Source | _Directory | _MergedDataset | _MergedGroup) -> FileSystem:
        """Bind a lazy source without reinitializing a physical namespace."""
        origin = source.origin
        result = cls.__new__(cls)
        result.relative_to = origin.file_path.parent if origin else Path.cwd()
        result._source = source
        result._namespace_path = origin.position if origin else result.relative_to
        return result

    @classmethod
    def from_file(cls, path: str | Path, reader: View | None = None) -> FileSystem:
        """Root a filesystem at a file's contents without reading them; reader is chosen by suffix if not given."""
        if reader is not None and not isinstance(reader, View):
            raise TypeError("reader must be a View, such as a Reader.")
        path = Path(path).absolute()
        return cls._from_source(_Source(reader or require_reader(path), path))

    @classmethod
    def from_value(cls, value: Any, *, view: View = MAPPING, attrs: Mapping[str, Any] | None = None) -> FileSystem:
        """Root a filesystem at an in-memory object, interpreted by view; by default a mapping of results.

        attrs, if given, become its root's attributes.
        """
        if not isinstance(view, View):
            raise TypeError("view must be a View instance.")
        result = cls._from_source(_Source(view, value))
        result._attrs = dict(attrs or {})
        return result

    @classmethod
    def from_directory(cls, path: str | Path, *patterns: str) -> FileSystem:
        """Bind a directory, mounting the supported files matching glob patterns at their relative paths."""
        root = Path(path).resolve()
        if not root.is_dir():
            raise NotADirectoryError(str(root))
        patterns = tuple(_relative_file_path(pattern).as_posix() for pattern in patterns)
        matches = sorted({file.relative_to(root) for pattern in patterns for file in root.glob(pattern)
                          if file.is_file() and _reader_match(file)[0] is not None})
        result = cls._from_source(_Directory(cls(matches, relative_to=root)._children, SourceLocator(root), patterns))
        result.relative_to = root
        return result

    def to_spec(
        self, *, relative_to: str | Path | None = None,
        readers: Mapping[str, View] | None = None,
    ) -> dict:
        """Export the composition as a data-only specification.

        Paths are relative to relative_to, defaulting to this filesystem's base.
        Custom readers require names in readers; supply the same names when
        restoring the specification.
        """
        from .specification import to_spec
        return to_spec(self, relative_to=relative_to, readers=readers)

    @classmethod
    def from_spec(
        cls, spec: Mapping, *, relative_to: str | Path | None = None,
        readers: Mapping[str, View] | None = None,
    ) -> FileSystem:
        """Restore a specification with source paths relative to relative_to or cwd."""
        from .specification import from_spec
        return from_spec(cls, spec, relative_to=relative_to, readers=readers)

    def _lookup(self, path: str | Path) -> FileSystem:
        """Find a namespace entry or bind its remaining path inside a lazy source."""
        path = _normalize_path(path)
        return self._backing.descendant(path) if path else self

    def _backing_at(self, path: str | Path):
        return self._lookup(path)._backing

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

    def _moved(self, position: Path) -> bool:
        """Whether this entry's source position differs from its namespace position."""
        origin = self._origin
        return origin is not None and origin.position != position

    @staticmethod
    def _snapshot(source: Node) -> FileSystem:
        if not isinstance(source, Node):
            raise TypeError("Mount sources must be nodes obtained with node(path).")
        return source._filesystem.clone(source.path)

    def mount(self, path: str | Path, source: Node) -> None:
        """Attach an independent subtree at a vacant path; create missing parents."""
        entry = self._snapshot(source)
        parent, name = self._parent(path, create=True)
        if name in parent._children:
            raise FileExistsError(str(path))
        parent._children[name] = entry

    def replace(self, path: str | Path, source: Node) -> None:
        """Atomically replace a subtree, keeping its place in its parent's listing."""
        entry = self._snapshot(source)
        parent, name = self._parent(path)
        if name not in parent._children:
            raise KeyError(str(path))
        parent._children[name] = entry

    def unmount(self, path: str | Path) -> None:
        """Remove a namespace subtree without writing its source files."""
        parent, name = self._parent(path)
        del parent._children[name]

    def node(self, path: str | Path = "") -> Node:
        path = _normalize_path(path)
        self._lookup(path)
        return Node(self, path)

    def read(self, path: str | Path = "", selection: Any | None = None) -> Any:
        return self._backing_at(path).read(selection)

    def attrs(self, path: str | Path = "") -> dict[str, Any]:
        """Metadata given to a node; facts derived from its source are in info()."""
        return dict(self._lookup(path)._attrs)

    def info(self, path: str | Path = "", *, full: bool = True) -> NodeInfo:
        """Derived facts with the node's attrs; full adds child names, dtype and shape, which may read metadata."""
        path = _normalize_path(path)
        entry, position = self._lookup(path), self._namespace_path / path
        source = entry._source
        file_format = source.format if source is not None else "directory" if position.is_dir() else None
        # Only where the namespace departs from its sources: below a moved or merged parent, nothing is.
        parent = self._lookup(path.rpartition("/")[0]) if path else None
        relocated = entry._moved(position) and not (parent and (parent._merged or parent._moved(position.parent)))
        return replace(entry._backing.info(full=full), format=file_format, relocated=relocated,
                       merged=entry._merged, attrs=dict(entry._attrs))

    def listdir(self, path: str | Path = "") -> tuple[str, ...]:
        info = self._backing_at(path).info()
        if info.kind != "group":
            raise NotADirectoryError(str(path))
        return info.children

    def items(self, path: str | Path = "") -> Iterable[tuple[str, NodeKind, Any]]:
        return self._backing_at(path).items()

    def locate(self, path: str | Path = "", selection: Any | None = None) -> list[SourceLocator]:
        return self._backing_at(path).locate(selection)

    def digest(self, path: str | Path = "") -> str:
        """Change digest from file stats and namespace structure, without source reads."""
        return self._backing_at(path).digest()

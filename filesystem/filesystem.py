"""Editable filesystems over lazy, read-only structured data sources."""

from __future__ import annotations

import hashlib
import os
import sys
from collections import OrderedDict
from copy import copy
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePath
from threading import Lock
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from ..readers import (MAPPING, NodeInfo, NodeKind, View, _reader_match, registered_view,
                       require_reader, value_info)

if TYPE_CHECKING:
    from .merge import _MergedDataset, _MergedGroup

__all__ = ["FileSystem", "Node", "SourceLocator"]


def _normalize_path(path: str | PurePath) -> str:
    if not isinstance(path, (str, PurePath)):
        raise TypeError("Filesystem paths must be strings or Path objects.")
    path = path.as_posix() if isinstance(path, PurePath) else path
    parts = [part for part in path.split("/") if part not in ("", ".")]
    if ".." in parts:
        raise ValueError("Filesystem paths cannot contain '..'.")
    return "/".join(parts)


def _join(*keys: str | None) -> str | None:
    """Keys joined by "/", skipping empty ones; None when all are."""
    return "/".join(filter(None, keys)) or None


def _is_name(name: Any) -> bool:
    return isinstance(name, str) and name not in ("", ".", "..") and "/" not in name


def _split_key(key: Any) -> tuple[str, Any]:
    """[:], [sel], ["a/b"] and ["a/b", sel] as a normalized relative path and a selection, None for everything.

    Paths are strings or PurePath objects, so x[PurePosixPath("a") / "b"] reads like x["a/b"].
    """
    if isinstance(key, tuple) and key and isinstance(key[0], (str, PurePath)):
        return _normalize_path(key[0]), key[1] if len(key) == 2 else key[1:] or None
    if isinstance(key, (str, PurePath)):
        return _normalize_path(key), None
    return "", None if isinstance(key, slice) and key == slice(None) else key


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


_SCALAR = NodeInfo("scalar")  # Most nodes are scalars; they share one info.


class _Descriptions:
    """Files' nodes, described once per file version and kept within a memory budget: metadata, never data."""

    def __init__(self, budget: int):
        self._files: OrderedDict[tuple, tuple[dict, int]] = OrderedDict()
        self._size, self._budget, self._lock = 0, budget, Lock()

    def __call__(self, view: View, path: Path) -> dict[str | None, tuple[NodeInfo, Any]]:
        stat = path.stat()
        version = (view, path, stat.st_size, stat.st_mtime_ns)
        with self._lock:
            if version in self._files:
                self._files.move_to_end(version)
                return self._files[version][0]
        nodes, size = {}, 0
        for key, info, value in view.nodes(path):
            info = _SCALAR if info == _SCALAR else info
            nodes[key] = info, value
            # An estimate: the entry, its key and value, and an info of its own with its child names.
            size += 256 + sys.getsizeof(key) + sys.getsizeof(value)
            size += 0 if info is _SCALAR else 512 + 64 * len(info.children)
        with self._lock:
            if version not in self._files:
                self._files[version] = nodes, size
                self._size += size
                while self._size > self._budget and len(self._files) > 1:
                    self._size -= self._files.popitem(last=False)[1][1]
        return nodes


_described = _Descriptions(64 * 1024 * 1024)


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
        return self.obj[:] if isinstance(self.obj, FileSystem) else self.obj

    def read(self, selection=None):
        return self.view.get(self._object, self.key, selection)

    @property
    def format(self):
        return self.view.format if self.key is None else None

    def info(self, *, full=True):
        if self.key is None and not full and isinstance(self.obj, (Path, FileSystem)):
            return NodeInfo("group")  # A file's root, or an interpreted node, is a group; saying so reads nothing.
        nodes = self._nodes()
        info = nodes[self.key][0]
        if full and info.kind == "dataset" and info.dtype is None:  # Left for the value to tell, and remembered.
            info = value_info(self.read())
            nodes[self.key] = info, None
        return info

    def items(self):
        nodes = self._nodes()
        info = nodes[self.key][0]
        if info.kind != "group":
            raise NotADirectoryError(self.key)
        children = [nodes[_join(self.key, name)] for name in info.children]  # Names are path segments.
        return iter([(name, child.kind, value) for name, (child, value) in zip(info.children, children)])

    def scalar(self):
        """This scalar node's value, as described with it."""
        return self._nodes()[self.key][1]

    def _nodes(self):
        """Nodes by key: all of a file's, described once per version; another object's at this key, each time."""
        if isinstance(self.obj, Path):
            return _described(self.view, self.obj)
        return {key: (info, value) for key, info, value in self.view.nodes(self._object, self.key, depth=1)}

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

    origin = format = merged = None

    children: dict[str, FileSystem]

    def read(self, selection=None):
        if selection is not None:
            raise ValueError("selection requires a dataset path.")
        return {name: child[:] for name, child in self.children.items()}

    def info(self, *, full=True):
        return NodeInfo("group", tuple(self.children) if full else ())

    def items(self):
        for name, child in self.children.items():
            backing = child._backing
            kind = backing.info(full=False).kind
            yield name, kind, backing.scalar() if kind == "scalar" else None  # Only a source's node is a scalar.

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
class _Directory(_ChildrenSource):
    """A directory with the supported files its glob patterns matched."""

    format = "directory"

    origin: SourceLocator
    patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class Node:
    """Lazy namespace handle; subsequent edits are visible through its path.

    Indexing reads values: node[:] everything, node[sel] a selection, node["a/b"] a child path,
    node["a/b", sel] a selection at a child path. Dividing navigates without reading: node / "a" / "b".
    """

    _filesystem: FileSystem = field(repr=False)
    path: str
    # Attributes of this handle alone: never stored in the filesystem, its specification or a pickle.
    _runtime_attrs: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def with_runtime_attrs(self, **attrs: Any) -> Node:
        """This handle with attributes of its own, kept as long as it is; the filesystem is unchanged.

        Descendants, clones and mounts do not inherit them.
        """
        return replace(self, _runtime_attrs={**self._runtime_attrs, **attrs})

    @property
    def runtime_attrs(self) -> dict[str, Any]:
        return dict(self._runtime_attrs)

    def node(self, path: str | PurePath = "") -> Node:
        """Bind a descendant using a path relative to this node."""
        return self._filesystem.node(f"{self.path}/{_normalize_path(path)}")

    __truediv__ = node  # node / "a" / "b": a handle, nothing read.

    def __getitem__(self, key: Any) -> Any:
        path, selection = _split_key(key)
        return self._filesystem[f"{self.path}/{path}", selection]

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
        info = self._filesystem.info(self.path, full=full)
        return replace(info, runtime_attrs=self.runtime_attrs) if self._runtime_attrs else info

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
    def from_value(cls, value: Any, *, view: View = MAPPING) -> FileSystem:
        """Root a filesystem at an in-memory object, interpreted by view; by default a mapping of results."""
        if not isinstance(view, View):
            raise TypeError("view must be a View instance.")
        return cls._from_source(_Source(view, value))

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
            self._children = {name: self._source.descendant(name).clone() for name in info.children}
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

    def mount(self, path: str | Path, source: Node, *, overwrite: bool = False) -> None:
        """Attach a copy of source's subtree at path, creating missing parents; source is left as it was.

        An occupied path raises FileExistsError unless overwrite, which rebinds it in its place in the listing.
        """
        if not isinstance(source, Node):
            raise TypeError("Mount sources must be nodes obtained with node(path).")
        entry = source._filesystem.clone(source.path)
        parent, name = self._parent(path, create=True)
        if name in parent._children and not overwrite:
            raise FileExistsError(str(path))
        parent._children[name] = entry

    def unmount(self, path: str | Path) -> None:
        """Remove a namespace subtree without writing its source files."""
        parent, name = self._parent(path)
        del parent._children[name]

    def node(self, path: str | PurePath = "") -> Node:
        path = _normalize_path(path)
        self._lookup(path)
        return Node(self, path)

    __truediv__ = node  # fs / "a" / "b": a handle, nothing read; fs["a/b"] reads.

    def __getitem__(self, key: Any) -> Any:
        """Read as Node does, with paths from the root."""
        path, selection = _split_key(key)
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

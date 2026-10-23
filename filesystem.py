"""Editable filesystems over lazy, read-only structured data sources."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from .readers import NodeInfo, NodeKind, Reader, require_reader

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
class _ReaderTree:
    """Expose a reader's file contents as a lazy tree."""

    reader: Reader
    locator: SourceLocator

    def descendant(self, path: str) -> _ReaderTree:
        """Bind an internal path relative to this source."""
        parent = self.locator.infile_path
        return replace(self, locator=replace(self.locator, infile_path=f"{parent}/{path}" if parent else path))

    def read(self, selection: Any | None = None) -> Any:
        return self.reader.get(self.locator.file_path, self.locator.infile_path, selection)

    def info(self, *, list_children: bool = True) -> NodeInfo:
        if self.locator.infile_path is None and not list_children:
            return NodeInfo("group")
        return self.reader.info(self.locator.file_path, self.locator.infile_path)


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

    def locate(self, selection: Any | None = None) -> list[SourceLocator]:
        return self._filesystem.locate(self.path, selection)

    @property
    def kind(self) -> NodeKind:
        return self._filesystem._lookup(self.path)._info(list_children=False).kind


class FileSystem:
    """Compose reader-backed trees through mounts and independent clones.

    Namespace edits never write source files. Registration, handles, clones and
    locator translation perform no reads. Editing a source group discovers its
    children through Reader metadata or fallback reads; values are not cached.
    """

    def __init__(
        self,
        filepaths: Iterable[str | Path] | str | Path = (),
        readers: Reader | Iterable[Reader] | None = None,
        *,
        relative_to: str | Path | None = None,
    ):
        self.relative_to = Path(relative_to).resolve() if relative_to is not None else Path.cwd()
        self._tree: _ReaderTree | dict[str, FileSystem] = {}
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

    @classmethod
    def _from_tree(cls, tree: _ReaderTree | dict[str, FileSystem], relative_to: Path) -> FileSystem:
        result = cls(relative_to=relative_to)
        result._tree = tree
        return result

    @classmethod
    def from_file(cls, path: str | Path, reader: Reader | None = None) -> FileSystem:
        """Root a filesystem at a file's contents without reading them."""
        if reader is not None and not isinstance(reader, Reader):
            raise TypeError("reader must be a Reader instance.")
        path = Path(path).absolute()
        return cls._from_tree(_ReaderTree(reader or require_reader(path), SourceLocator(path)), path.parent)

    def _lookup(self, path: str | Path) -> FileSystem:
        parts = tuple(filter(None, _normalize_path(path).split("/")))
        entry = self
        for index, name in enumerate(parts):
            if isinstance(entry._tree, _ReaderTree):
                return self._from_tree(entry._tree.descendant("/".join(parts[index:])), self.relative_to)
            entry = entry._tree[name]
        return entry

    def _expand(self) -> dict[str, FileSystem]:
        """Discover child trees only when editing their parent."""
        if isinstance(self._tree, _ReaderTree):
            info = self._tree.info()
            if info.kind != "group":
                raise NotADirectoryError(self._tree.locator.infile_path or str(self._tree.locator.file_path))
            self._tree = {
                name: self._from_tree(self._tree.descendant(name), self.relative_to)
                for name in info.children
            }
        return self._tree

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
        tree = self._lookup(path)._tree
        if isinstance(tree, dict):
            tree = {name: child.clone() for name, child in tree.items()}
        return self._from_tree(tree, self.relative_to)

    def mount(self, path: str | Path, source: Node) -> None:
        """Attach an independent subtree at a vacant path; create missing parents."""
        if not isinstance(source, Node):
            raise TypeError("Mount sources must be nodes obtained with node(path).")
        entry = source._filesystem.clone(source.path)
        children, name = self._parent(path, create=True)
        if name in children:
            raise FileExistsError(str(path))
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
        tree = self._lookup(path)._tree
        if isinstance(tree, _ReaderTree):
            return tree.read(selection)
        if selection is not None:
            raise ValueError("selection requires a dataset path.")
        return {name: child.read() for name, child in tree.items()}

    def _info(self, *, list_children: bool = True) -> NodeInfo:
        if isinstance(self._tree, _ReaderTree):
            return self._tree.info(list_children=list_children)
        return NodeInfo("group", tuple(self._tree) if list_children else ())

    def listdir(self, path: str | Path = "") -> tuple[str, ...]:
        info = self._lookup(path)._info()
        if info.kind != "group":
            raise NotADirectoryError(str(path))
        return info.children

    def locate(self, path: str | Path = "", selection: Any | None = None) -> list[SourceLocator]:
        tree = self._lookup(path)._tree
        if selection is not None and (isinstance(tree, dict) or tree.locator.infile_path is None):
            raise ValueError("selection requires a dataset path.")
        if isinstance(tree, _ReaderTree):
            return [replace(tree.locator, selection=selection)]
        return [locator for child in tree.values() for locator in child.locate()]

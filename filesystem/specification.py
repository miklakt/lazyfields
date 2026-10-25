"""Data-only filesystem snapshots and source overlays, without edit history.

All physical paths are relative to one external base directory. An absent
``children`` field delegates to a source; a present mapping is the exact namespace
of an expanded group. Merge groups have already resolved prefix/prefer rules.
Node types follow their fields and parent source; an explicit ``type`` is only
needed to distinguish a virtual group beneath a file-backed parent.
``overrides`` and ``unmount`` instead inherit the source namespace on restoration.
``attrs`` adds attributes to any node, over the dtype and shape datasets have by default.
"""

import os
from dataclasses import replace
from glob import has_magic
from pathlib import Path
from typing import Mapping

from .filesystem import SourceLocator, _FileSource
from .merge import _MergedDataset, collect, concatenate, hstack, stack, vstack
from ..readers import Reader, _reader_match, require_reader


_FLAGS = ("is_mount", "is_relocated", "is_merged")
_COMMON = {"type", "namespace_path", "attrs", *_FLAGS}
_FIELDS = {
    "file": {"path", "infile_path", "reader", "children", "overrides", "unmount"},
    "files": {"paths"},
    "group": {"children"},
    "merge": {"op", "axis", "sources"},
}
_RULES = {"collect": collect, "stack": stack, "concatenate": concatenate,
          "vstack": vstack, "hstack": hstack}


def _readers(readers):
    readers = dict(readers or {})
    if any(not isinstance(name, str) or not name or not isinstance(reader, Reader)
           for name, reader in readers.items()):
        raise TypeError("readers must map nonempty names to Reader instances.")
    return readers


def _inherited_source(parent, name):
    """The natural file binding of a child; no metadata discovery is needed."""
    if parent is None or not isinstance(parent._source, _FileSource):
        return None
    source = parent._source
    infile = source.origin.infile_path
    return replace(source, origin=replace(source.origin, infile_path=f"{infile}/{name}" if infile else name))


def _same_source(source, inherited):
    return (isinstance(source, _FileSource) and inherited is not None and
            source.origin == inherited.origin and source.reader is inherited.reader)


def _default_mount(entry, parent, name):
    if parent is None or parent.is_merged:
        return False
    return (not _same_source(entry._source, _inherited_source(parent, name)) and
            (entry._source is not None or entry.is_merged))


def _node_type(spec, parent):
    if "type" in spec:
        return spec["type"]
    if "paths" in spec:
        return "files"
    if "path" in spec:
        return "file"
    if "op" in spec or "sources" in spec:
        return "merge"
    return "file" if parent is not None and isinstance(parent._source, _FileSource) else "group"


def _validate_node(data, parent):
    """Validate node fields and both namespace forms before binding sources."""
    kind = _node_type(data, parent)
    if not isinstance(kind, str) or kind not in _FIELDS:
        raise ValueError(f"Unknown filesystem specification type: {kind!r}")
    unknown = data.keys() - (_COMMON | _FIELDS[kind])
    if unknown:
        raise ValueError(f"Unknown fields for {kind}: {sorted(unknown, key=str)}")
    if "namespace_path" in data and parent is not None:
        raise ValueError("Only a standalone root can specify namespace_path.")
    for flag in _FLAGS:
        if flag in data and type(data[flag]) is not bool:
            raise TypeError(f"{flag} must be a boolean.")
    attrs = data.get("attrs", {})
    if not isinstance(attrs, Mapping) or any(not isinstance(name, str) for name in attrs):
        raise TypeError("attrs must map attribute names to values.")
    if "children" in data and ("overrides" in data or "unmount" in data):
        raise ValueError("Use either children or overrides/unmount at each node.")
    unmounted = data.get("unmount", [])
    if not isinstance(unmounted, list):
        raise TypeError("unmount must be a list of child names.")
    for key in ("children", "overrides", "unmount"):
        if key not in data:
            continue
        names = data[key]
        if key != "unmount" and not isinstance(names, Mapping):
            raise TypeError(f"{key} must map child names to specifications.")
        for name in names:
            if not isinstance(name, str) or not name or "/" in name or name in (".", ".."):
                raise ValueError(f"Invalid child name: {name!r}")
    if len(set(unmounted)) != len(unmounted):
        raise ValueError("unmount must contain distinct child names.")
    if set(data.get("overrides", {})) & set(unmounted):
        raise ValueError("A child cannot appear in both overrides and unmount.")
    return kind


def _registered_paths(spec, prefix=""):
    """Recognize an ordinary file registration tree without losing empty groups."""
    if set(spec) == {"path"}:
        return [prefix] if spec["path"] == prefix else None
    if set(spec) != {"children"} or not spec["children"]:
        return None
    paths = []
    for name, child in spec["children"].items():
        found = _registered_paths(child, f"{prefix}/{name}" if prefix else name)
        if found is None:
            return None
        paths.extend(found)
    return paths


def to_spec(filesystem, *, relative_to=None, readers=None):
    """Infer compact source edits or exact children, without source reads."""
    base = Path(relative_to).resolve() if relative_to is not None else filesystem.relative_to
    readers = _readers(readers)

    def path(value):
        return Path(os.path.relpath(value, base)).as_posix()

    def visit(entry, parent=None, name=None, namespace=None):
        source = entry._source
        if isinstance(source, _FileSource):
            origin = source.origin
            inherited = _inherited_source(parent, name)
            result = {}
            if not _same_source(source, inherited):
                result["path"] = path(origin.file_path)
                if origin.infile_path is not None:
                    result["infile_path"] = origin.infile_path
                reader_name = next((key for key, reader in readers.items()
                                    if reader is source.reader), None)
                if reader_name is not None:
                    result["reader"] = reader_name
                elif _reader_match(origin.file_path)[0] is not source.reader:
                    raise ValueError(f"Supply a reader name for {origin.file_path!s} in readers.")
        elif isinstance(source, _MergedDataset):
            rule = source.rule
            result = {
                "op": rule.operation,
                "sources": [visit(child) for child in source.sources],
            }
            if rule.operation in ("stack", "concatenate") and rule.argument != 0:
                result["axis"] = rule.argument
        elif source is None:
            result = {"type": "group"} if _node_type({}, parent) == "file" else {}
        else:
            raise TypeError(f"Unsupported filesystem source: {type(source).__name__}")
        if entry._attrs:
            result["attrs"] = dict(entry._attrs)

        if namespace is None:
            namespace = (source.origin.file_path / (source.origin.infile_path or "")
                         if isinstance(source, _FileSource) else entry.relative_to)
            if entry._namespace_path != namespace:
                namespace = entry._namespace_path
                result["namespace_path"] = path(namespace)
        if entry._children is not None:
            children = {
                child_name: visit(child, entry, child_name, namespace / child_name)
                for child_name, child in entry._children.items()
            }
            if isinstance(source, _FileSource) and entry._source_children is not None:
                overrides = {key: child for key, child in children.items() if child}
                unmounted = [key for key in entry._source_children if key not in children]
                if overrides:
                    result["overrides"] = overrides
                if unmounted:
                    result["unmount"] = unmounted
            else:
                result["children"] = children
        if entry.is_mount != _default_mount(entry, parent, name):
            result["is_mount"] = entry.is_mount
        relocated = entry.is_mount and entry._relocated(namespace)
        if entry.is_relocated != relocated:
            result["is_relocated"] = entry.is_relocated
        if entry.is_merged != isinstance(source, _MergedDataset):
            result["is_merged"] = entry.is_merged
        return result

    spec = visit(filesystem)
    paths = _registered_paths(spec)
    patterns = filesystem._path_patterns
    # Patterns are relative to the registration base, so only that base can export them.
    if patterns is not None and filesystem.relative_to == base:
        if spec == {"children": {}}:
            paths = []
        if paths is not None and tuple(paths) == patterns[1]:
            return {"paths": list(patterns[0])}
    # Literal glob characters in filenames must remain explicit file bindings.
    return {"paths": paths} if paths is not None and not any(map(has_magic, paths)) else spec


def from_spec(cls, spec, *, relative_to=None, readers=None):
    """Rebind exact snapshots directly; discover source children for overlays."""
    base = Path(relative_to).resolve() if relative_to is not None else Path.cwd()
    readers = _readers(readers)
    active = set()

    def path(value):
        if not isinstance(value, str):
            raise TypeError("Physical paths must be strings.")
        if Path(value).is_absolute():
            raise ValueError("Specification paths must be relative to relative_to or cwd.")
        # Physical paths may contain '..'; logical child names may not.
        return Path(os.path.abspath(base / value))

    def visit(data, parent=None, name=None, namespace=None):
        if not isinstance(data, Mapping):
            raise TypeError("Each filesystem specification must be a mapping.")
        if id(data) in active:
            raise ValueError("Cyclic filesystem specification.")
        active.add(id(data))
        try:
            kind = _validate_node(data, parent)
            overlay = "overrides" in data or "unmount" in data
            unmounted = data.get("unmount", [])
            if kind == "file":
                inherited = _inherited_source(parent, name)
                if "path" not in data and inherited is None:
                    raise ValueError("A file needs path unless it inherits its parent's file binding.")
                filename = path(data["path"]) if "path" in data else inherited.origin.file_path
                infile = data.get("infile_path", inherited.origin.infile_path
                                  if "path" not in data else None)
                if infile is not None and (not isinstance(infile, str) or
                                          any(part == ".." for part in infile.split("/"))):
                    raise ValueError("infile_path must be a string without '..' segments.")
                if "reader" in data:
                    reader_name = data["reader"]
                    if not isinstance(reader_name, str) or reader_name not in readers:
                        raise ValueError(f"Unknown reader name: {reader_name!r}")
                    reader = readers[reader_name]
                else:
                    reader = inherited.reader if "path" not in data else require_reader(filename)
                entry = cls._from_source(_FileSource(reader, SourceLocator(filename, infile)))
            elif kind == "files":
                paths = data.get("paths")
                if not isinstance(paths, list) or any(not isinstance(item, str) for item in paths):
                    raise TypeError("paths must be a list of relative filenames.")
                entry = cls._from_paths(base, paths)
            elif kind == "merge":
                operation = data.get("op")
                if not isinstance(operation, str) or operation not in _RULES:
                    raise ValueError(f"Unknown dataset merge operation: {operation!r}")
                if operation in ("stack", "concatenate"):
                    axis = data.get("axis", 0)
                    if type(axis) is not int:
                        raise TypeError("axis must be an integer.")
                    rule = _RULES[operation](axis=axis)
                else:
                    if "axis" in data:
                        raise ValueError(f"{operation} does not accept axis.")
                    rule = _RULES[operation]()
                sources = data.get("sources")
                if not isinstance(sources, list) or not sources:
                    raise ValueError("Merge sources must be a nonempty list.")
                entry = cls._from_source(_MergedDataset(tuple(visit(item) for item in sources), rule))
            else:
                entry = cls(relative_to=base)

            entry.relative_to = base
            if "attrs" in data:
                entry._attrs = dict(data["attrs"])
            if namespace is not None:
                entry._namespace_path = namespace
            elif kind != "file":
                entry._namespace_path = base
            if "namespace_path" in data:
                entry._namespace_path = path(data["namespace_path"])
            entry.is_merged = data.get("is_merged", entry.is_merged)
            entry.is_mount = data.get("is_mount", _default_mount(entry, parent, name))
            if "children" in data:
                entry._children = {}
            elif overlay:
                children = entry._expand()
                for child_name in unmounted:
                    children.pop(child_name, None)
                # Retain tombstones even if the current source lacks those names.
                entry._source_children = tuple(dict.fromkeys((*entry._source_children, *unmounted)))
            for child_name, child in data.get("children", data.get("overrides", {})).items():
                replacement = visit(child, entry, child_name, entry._namespace_path / child_name)
                if overlay and replacement.is_mount and child_name in entry._children:
                    entry._replace_child(child_name, replacement)
                else:
                    entry._children[child_name] = replacement
            entry.is_relocated = data.get("is_relocated", entry.is_mount and entry._relocated(entry._namespace_path))
            return entry
        finally:
            active.remove(id(data))

    return visit(spec)

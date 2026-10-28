"""Export and restore filesystem specifications.

A node is bound by "path" (a directory, a file, or a place inside a file), by "merge" of
the children of its "sources", or by "mount" glob rules in its directory; otherwise it is a
virtual group. Its "children" change what the binding gives: null removes a child, a patch
changes or rebinds one in place, and a new name adds one. Paths are relative to the parent's
position, or to the base when they start with "/". Captures, conditions, placement with "at",
definitions and pattern selectors are not supported.
"""

import json
import os
from glob import has_magic
from pathlib import Path
from typing import Mapping

from .filesystem import FileSystem, _Directory, _Source, _is_name
from .merge import _MergedDataset, _MergedGroup, collect, concatenate, hstack, merge, prefer, prefix, stack, vstack
from ..readers import View, _reader_match, require_reader

_BINDING = ("path", "reader", "merge", "axis", "sources", "mount")
_FIELDS = {*_BINDING, "attrs", "children", "$comment"}
_RULES = {rule.__name__: rule for rule in (collect, prefer, prefix, stack, concatenate, vstack, hstack)}


def _readers(readers):
    readers = dict(readers or {})
    if any(not isinstance(name, str) or not name or not isinstance(reader, View)
           for name, reader in readers.items()):
        raise TypeError("readers must map nonempty names to views, such as Readers.")
    return readers


def _default_reader(file, parent):
    """The reader of a file unless named: its parent's inside the same file, otherwise by suffix."""
    source = parent._source if parent is not None else None
    if isinstance(source, _Source) and source.origin is not None and source.origin.file_path == file:
        return source.view
    return _reader_match(file)[0]


def _pristine(entry):
    """entry's binding alone, without namespace edits."""
    return FileSystem._from_source(entry._source) if entry._source is not None else None


def _rule_spec(rule):
    return {"merge": rule.operation, "axis": rule.axis} if rule.axis else rule.operation


def _parse_rule(value, axis=None):
    if isinstance(value, Mapping):
        if value.keys() - {"merge", "axis"}:
            raise ValueError(f"A merge rule has only merge and axis: {dict(value)!r}")
        value, axis = value.get("merge"), value.get("axis")
    if not isinstance(value, str) or value not in _RULES:
        raise ValueError(f"Unknown merge rule: {value!r}")
    return _RULES[value](**({} if axis is None else {"axis": axis}))


def _updated(attrs, changes):
    """attrs with changes applied name by name; null removes a name."""
    return {name: value for name, value in {**attrs, **changes}.items() if value is not None}


def to_spec(filesystem, *, relative_to=None, readers=None):
    """Describe bindings, attributes and changed children, without reading sources."""
    base = Path(relative_to).resolve() if relative_to is not None else filesystem.relative_to
    names = {id(reader): name for name, reader in _readers(readers).items()}

    def path(target, position):
        """target from position, or from the base as "/..." where that would climb out."""
        relative = Path(os.path.relpath(target, position))
        if relative.parts[:1] == ("..",) and target.is_relative_to(base):
            return "/" + target.relative_to(base).as_posix()
        return relative.as_posix()

    def binding(entry, position, parent):
        """Fields binding entry below parent, at position, and the position of its children."""
        source = entry._source
        if isinstance(source, _Source):
            origin = source.origin
            if origin is None:
                raise ValueError("An in-memory object cannot be described by a specification.")
            fields = {"path": path(origin.position, position)}
            if source.view is not _default_reader(origin.file_path, parent):
                if id(source.view) not in names:
                    raise ValueError(f"Supply a reader name for {origin.file_path!s} in readers.")
                fields["reader"] = names[id(source.view)]
            return fields, origin.position
        if isinstance(source, _Directory):
            directory = source.origin.position
            fields = {} if directory == position else {"path": path(directory, position)}
            if source.patterns:
                fields["mount"] = [{"glob": pattern} for pattern in source.patterns]
            return fields, directory
        if isinstance(source, _MergedDataset):
            fields = {"merge": source.rule.operation, **({"axis": source.rule.axis} if source.rule.axis else {})}
        elif isinstance(source, _MergedGroup):
            fields = {"merge": {key: _rule_spec(rule) for key, rule in source.rules.items()}}
        else:
            return {}, position
        inputs = {name: describe(entry, position, None) for name, entry in source.sources.items()}
        return {**fields, "sources": {"children": inputs}}, position

    def describe(entry, position, parent):
        """The full description of entry below parent, at position."""
        fields, inner = binding(entry, position, parent)
        if entry._attrs:
            fields["attrs"] = dict(entry._attrs)
        children = changes(entry, _pristine(entry), inner)
        if children:
            fields["children"] = children
        return fields

    def patch(entry, given, position, parent):
        """The patch turning given, a child of parent's binding, into entry."""
        fields, inner = binding(entry, position, parent)
        given_fields = binding(given, position, parent)[0]
        if fields == given_fields:
            fields, baseline = {}, given
        else:  # Rebinding: fields the new binding lacks are cleared.
            fields, baseline = {**dict.fromkeys(given_fields), **fields}, _pristine(entry)
        attrs = {name: entry._attrs.get(name) for name in {**given._attrs, **entry._attrs}
                 if entry._attrs.get(name) != given._attrs.get(name)}
        if attrs:
            fields["attrs"] = attrs
        children = changes(entry, baseline, inner)
        if children:
            fields["children"] = children
        return fields

    def changes(entry, baseline, position):
        """Children entries turning the children of baseline into entry's."""
        if entry._children is None:
            return {}
        if baseline is None:
            given = {}
        elif baseline._children is not None:
            given = baseline._children
        else:
            given = {name: baseline._source.descendant(name) for name in entry._source_children or ()}
        result = {name: None for name in given if name not in entry._children}
        for name, child in entry._children.items():
            if name not in given:
                result[name] = describe(child, position, entry)
            elif changed := patch(child, given[name], position, entry):
                result[name] = changed
        return result

    return describe(filesystem, base, None)


def from_spec(cls, spec, *, relative_to=None, readers=None):
    """Build the described filesystem; merges and mount rules inspect their sources while building."""
    base = Path(relative_to).resolve() if relative_to is not None else Path.cwd()
    readers = _readers(readers)

    def check(data):
        if not isinstance(data, Mapping):
            raise TypeError("A node description must be an object.")
        unknown = data.keys() - _FIELDS
        if unknown:
            raise ValueError(f"Unsupported node fields: {sorted(unknown, key=str)}")
        attrs = data.get("attrs", {})
        if not isinstance(attrs, Mapping) or not all(isinstance(name, str) for name in attrs):
            raise TypeError("attrs must map attribute names to values.")
        if data.get("reader") is not None and data.get("path") is None:
            raise ValueError("reader needs path beside it.")
        if (data.get("axis") is not None or data.get("sources") is not None) and data.get("merge") is None:
            raise ValueError("axis and sources need merge beside them.")

    def resolve(value, position):
        if not isinstance(value, str) or not value:
            raise TypeError("path must be a nonempty string.")
        return Path(os.path.normpath(base / value[1:] if value.startswith("/") else position / value))

    def bind(position, name, parent):
        """Bind the file containing position, at the place inside it."""
        file = next((path for path in (position, *position.parents) if path.is_file()), position)
        if name is not None and name not in readers:
            raise ValueError(f"Unknown reader name: {name!r}")
        reader = readers[name] if name is not None else _default_reader(file, parent) or require_reader(file)
        key = position.relative_to(file).as_posix() if file != position else None
        return cls._from_source(_Source(reader, file, key))

    def globs(rules):
        if not isinstance(rules, list) or any(
                not isinstance(rule, Mapping) or rule.keys() - {"glob", "$comment"}
                or not isinstance(rule.get("glob"), str) for rule in rules):
            raise ValueError('Only mount rules of the form {"glob": pattern} are supported.')
        return [rule["glob"] for rule in rules]

    def merged(data, position):
        inputs = build(data["sources"], position)
        nodes = {name: inputs.node(name) for name in inputs.listdir()}
        if isinstance(data["merge"], Mapping):
            if "axis" in data:
                raise ValueError("A key-by-key merge gives each rule its own axis.")
            return merge(nodes, rules={key: _parse_rule(rule) for key, rule in data["merge"].items()})
        return merge(nodes, rules=_parse_rule(data["merge"], data.get("axis")))

    def build(data, position, parent=None):
        check(data)
        if data.get("path") is not None:
            position = resolve(data["path"], position)
        if data.get("merge") is not None:
            entry = merged(data, position)
        elif data.get("mount") is not None or data.get("path") is not None and position.is_dir():
            entry = cls.from_directory(position, *globs(data.get("mount") or []))
        elif data.get("path") is not None:
            entry = bind(position, data.get("reader"), parent)
        else:
            entry = cls(relative_to=position)
        entry._attrs = _updated({}, data.get("attrs", {}))
        apply(entry, data.get("children", {}), position)
        return entry

    def apply(entry, children, position):
        """Apply children entries in order: null removes, binding fields rebind in place, others patch."""
        if isinstance(children, Mapping):
            entries = children.items()
        elif isinstance(children, list) and all(isinstance(item, Mapping) and len(item) == 1 for item in children):
            entries = [next(iter(item.items())) for item in children]
        else:
            raise TypeError("children must be an object or an array of single-entry objects.")
        for name, patch in entries:
            if not _is_name(name) or has_magic(name) or "${" in name:
                raise ValueError(f"Unsupported children selector: {name!r}")
            existing = entry._expand()
            child = existing.get(name)
            if patch is None:
                existing.pop(name, None)
                continue
            check(patch)
            if any(key in patch for key in _BINDING):
                replacement = build(patch, position, entry)
                if child is not None:
                    replacement._attrs = _updated(child._attrs, patch.get("attrs", {}))
                existing[name] = replacement
            elif child is not None:
                child._attrs = _updated(child._attrs, patch.get("attrs", {}))
                apply(child, patch.get("children", {}), child._origin.position if child._origin else position)
            elif patch.get("children"):  # An annotation alone waits until the name exists.
                existing[name] = build(patch, position, entry)

    root = build(spec, base)
    root.relative_to = base
    return root


def add_spec_arguments(parser):
    """Add a specification, as a JSON file or inline JSON, and --relative-to for its paths to parser."""
    parser.add_argument("spec", help="specification JSON file, or inline JSON such as "
                                     '\'{"mount": [{"glob": "*.h5"}]}\'')
    parser.add_argument("--relative-to", type=Path,
                        help="base directory of specification paths "
                             "(default: the file's directory, or the working directory for inline JSON)")


def filesystem_from_arguments(args):
    """Restore the filesystem specified by arguments that add_spec_arguments parsed."""
    if args.spec.lstrip().startswith("{"):
        spec, base = json.loads(args.spec), Path.cwd()
    else:
        spec, base = json.loads(Path(args.spec).read_text(encoding="utf-8")), Path(args.spec).resolve().parent
    return FileSystem.from_spec(spec, relative_to=args.relative_to or base)

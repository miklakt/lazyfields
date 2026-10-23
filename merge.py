"""Lazy, per-key composition of structured filesystem nodes.

Array rules materialize their inputs before applying the logical selection.
Their locators conservatively report complete inputs, in source order.
"""

from dataclasses import dataclass
from functools import cached_property
from operator import index
from typing import Mapping

from .filesystem import FileSystem, Node, _normalize_path
from .readers import NodeInfo

__all__ = ["merge", "collect", "prefer", "prefix", "stack", "concatenate", "vstack", "hstack"]


@dataclass(frozen=True)
class _Rule:
    operation: str
    argument: object = None


def collect():
    """Collect scalar occurrences into a Python list, in source order."""
    return _Rule("collect")


def prefer(*sources: str):
    """Choose the first named source containing the key."""
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("prefer requires distinct source names in priority order.")
    return _Rule("prefer", sources)


def prefix():
    """Expose each occurrence under source_key, preserving its subtree."""
    return _Rule("prefix")


def stack(axis=0):
    """Stack equal-shaped inputs along a new axis."""
    return _Rule("stack", index(axis))


def concatenate(axis=0):
    """Concatenate inputs along an existing axis."""
    return _Rule("concatenate", index(axis))


def vstack():
    return _Rule("vstack")


def hstack():
    return _Rule("hstack")


@dataclass(frozen=True)
class _Value:
    origin = None

    sources: tuple[Node, ...]
    rule: _Rule

    def info(self, *, list_children=True):
        return NodeInfo("dataset")

    def descendant(self, path):
        raise NotADirectoryError(path)

    def read(self, selection=None):
        values = [node.read() for node in self.sources]
        operation, axis = self.rule.operation, self.rule.argument
        if operation == "collect":
            result = values
        else:
            import numpy as np
            function = getattr(np, operation)
            result = function(values, axis=axis) if axis is not None else function(values)
        return result if selection is None else result[selection]

    def locate(self, selection=None):
        # Selection addresses the assembled value, never each input independently.
        return [locator for node in self.sources for locator in node.locate()]


@dataclass(frozen=True)
class _Group:
    origin = None

    sources: tuple[tuple[str, Node], ...]
    rules: dict[str, _Rule]

    @cached_property
    def filesystem(self):
        occurrences = {}
        for name, source in self.sources:
            for key in source.listdir():
                occurrences.setdefault(key, {})[name] = source.node(key)
        unknown = self.rules.keys() - occurrences.keys()
        if unknown:
            raise KeyError(f"Merge rules reference missing keys: {sorted(unknown)}")
        result = FileSystem()

        def add(key, node):
            if key in result._children:
                raise ValueError(f"Merge output collision: {key!r}")
            result._children[key] = node._filesystem.clone(node.path)

        for key, nodes in occurrences.items():
            rule = self.rules.get(key)
            if rule is None:
                if len(nodes) != 1:
                    raise ValueError(f"Repeated key {key!r} requires a merge rule.")
                add(key, next(iter(nodes.values())))
            elif rule.operation == "prefix":
                for name, node in nodes.items():
                    add(f"{name}_{key}", node)
            elif rule.operation == "prefer":
                chosen = next((nodes[name] for name in rule.argument if name in nodes), None)
                if chosen is None:
                    raise KeyError(f"No preferred source contains {key!r}")
                add(key, chosen)
            else:
                if len(nodes) != len(self.sources):
                    raise KeyError(f"Merge key {key!r} must exist in every source.")
                expected = "scalar" if rule.operation == "collect" else "dataset"
                if any(node.kind != expected for node in nodes.values()):
                    raise TypeError(f"Merge key {key!r} requires {expected} inputs.")
                add(key, FileSystem._from_source(_Value(tuple(nodes.values()), rule)).node())
        return result

    def info(self, *, list_children=True):
        return self.filesystem._info() if list_children else NodeInfo("group")

    def descendant(self, path):
        return self.filesystem.clone(path)

    def read(self, selection=None):
        return self.filesystem.read(selection=selection)

    def locate(self, selection=None):
        return self.filesystem.locate(selection=selection)


def merge(sources: Mapping[str, Node], *, rules: Mapping[str, _Rule]) -> FileSystem:
    """Return a lazy filesystem from named group nodes and explicit per-key rules.

    Namespace structures are snapshotted; source values remain live. Construction
    performs no reads. Output names and kinds are resolved once on first access;
    subsequent operations reuse that namespace without caching values.
    Subgroups require prefix/prefer when repeated; recursive merging is explicit.
    Locating a merged group discovers its output keys, which can require metadata
    reads, but never evaluates the composition operations.
    """
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("sources must be a nonempty mapping of names to nodes.")
    for name, node in sources.items():
        if not isinstance(name, str) or not name or "/" in name or name in (".", ".."):
            raise ValueError("Source names must be nonempty path segments.")
        if not isinstance(node, Node):
            raise TypeError("Merge sources must be Node handles.")
    rules = dict(rules)
    for key, rule in rules.items():
        if not isinstance(key, str) or not key or _normalize_path(key) != key or "/" in key:
            raise ValueError("Rule keys must be top-level field names.")
        if not isinstance(rule, _Rule):
            raise TypeError("Rules must be created with the merge rule helpers.")
        if rule.operation == "prefer" and any(name not in sources for name in rule.argument):
            raise ValueError(f"Unknown source in priority rule for {key!r}")
    snapshots = tuple((name, node._filesystem.clone(node.path).node()) for name, node in sources.items())
    return FileSystem._from_source(_Group(snapshots, rules))

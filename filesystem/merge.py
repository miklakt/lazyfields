"""Per-key namespace merging with lazy dataset evaluation.

Array rules materialize their inputs before applying the logical selection.
Their locators conservatively report complete inputs, in source order.
"""

from dataclasses import dataclass
from operator import index
from typing import Mapping

from .filesystem import FileSystem, Node, _digest, _is_name
from ..readers import NodeInfo

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
class _MergedDataset:
    origin = None
    is_merged = True

    sources: tuple[FileSystem, ...]
    rule: _Rule

    def info(self, *, list_children=True):
        return NodeInfo("dataset")

    def items(self):
        raise NotADirectoryError("Merged dataset has no children.")

    def descendant(self, path):
        raise NotADirectoryError(path)

    def read(self, selection=None):
        operation, axis = self.rule.operation, self.rule.argument
        key, at = (selection if isinstance(selection, tuple) else (selection,)), axis or 0
        if (selection is not None and (operation == "collect" or operation == "stack" and at >= 0)
                and len(key) > at and all(type(part) in (int, slice) for part in key[:at + 1])):
            # The merge axis enumerates sources: read only the selected ones, each
            # with the rest of the selection, so no other source is loaded.
            pick, inner = key[at], key[:at] + key[at + 1:] or None
            if type(pick) is int:
                return self.sources[pick].read("", inner)
            parts = [source.read("", inner) for source in self.sources[pick]]
            if operation == "collect":
                return parts
            if parts:
                import numpy as np
                # Integer indices before the merge axis remove dimensions, shifting it left.
                return np.stack(parts, axis=at - sum(type(part) is int for part in key[:at]))
        values = [source.read() for source in self.sources]
        if operation == "collect":
            result = values
        else:
            import numpy as np
            function = getattr(np, operation)
            result = function(values, axis=axis) if axis is not None else function(values)
        return result if selection is None else result[selection]

    def digest(self):
        return _digest((self.rule, [source.digest() for source in self.sources]))

    def locate(self, selection=None):
        # Selection addresses the assembled value, never each input independently.
        return [locator for source in self.sources for locator in source.locate()]


def merge(sources: Mapping[str, Node], *, rules: Mapping[str, _Rule]) -> FileSystem:
    """Build an ordinary namespace from immediate source keys and per-key rules.

    Construction inspects keys and kinds once, without traversing subgroups.
    Readers without metadata inspection may load their files for discovery.
    Source namespaces are snapshotted; values and array operations remain lazy.
    Repeated subgroups require prefix/prefer; recursive merging is explicit.
    """
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("sources must be a nonempty mapping of names to nodes.")
    for name, node in sources.items():
        if not _is_name(name):
            raise ValueError("Source names must be nonempty path segments.")
        if not isinstance(node, Node):
            raise TypeError("Merge sources must be Node handles.")
    rules = dict(rules)
    for key, rule in rules.items():
        if not _is_name(key):
            raise ValueError("Rule keys must be top-level field names.")
        if not isinstance(rule, _Rule):
            raise TypeError("Rules must be created with the merge rule helpers.")
        if rule.operation == "prefer" and any(name not in sources for name in rule.argument):
            raise ValueError(f"Unknown source in priority rule for {key!r}")
    occurrences = {}
    for name, source in sources.items():
        for key, kind, _ in source.items():
            occurrences.setdefault(key, {})[name] = (source._filesystem.clone(f"{source.path}/{key}"), kind)
    unknown = rules.keys() - occurrences.keys()
    if unknown:
        raise KeyError(f"Merge rules reference missing keys: {sorted(unknown)}")
    result = FileSystem()
    result.is_merged = True

    def add(key, subtree):
        if key in result._children:
            raise ValueError(f"Merge output collision: {key!r}")
        result._children[key] = subtree

    for key, found in occurrences.items():
        subtrees = {name: subtree for name, (subtree, _) in found.items()}
        rule = rules.get(key)
        if rule is None:
            if len(subtrees) != 1:
                raise ValueError(f"Repeated key {key!r} requires a merge rule.")
            add(key, next(iter(subtrees.values())))
        elif rule.operation == "prefix":
            for name, subtree in subtrees.items():
                add(f"{name}_{key}", subtree)
        elif rule.operation == "prefer":
            chosen = next((subtrees[name] for name in rule.argument if name in subtrees), None)
            if chosen is None:
                raise KeyError(f"No preferred source contains {key!r}")
            add(key, chosen)
        else:
            if len(subtrees) != len(sources):
                raise KeyError(f"Merge key {key!r} must exist in every source.")
            expected = "scalar" if rule.operation == "collect" else "dataset"
            if any(kind != expected for _, kind in found.values()):
                raise TypeError(f"Merge key {key!r} requires {expected} inputs.")
            add(key, FileSystem._from_source(_MergedDataset(tuple(subtrees.values()), rule)))
    return result

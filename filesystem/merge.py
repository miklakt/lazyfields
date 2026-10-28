"""Merging named sources into one lazy dataset, or into a group key by key.

Array rules materialize their inputs before applying the logical selection.
Their locators conservatively report complete inputs, in source order.
"""

from dataclasses import dataclass
from operator import index
from typing import Mapping

from .filesystem import FileSystem, Node, _Composed, _digest, _is_name
from ..readers import NodeInfo, value_info

__all__ = ["merge", "collect", "prefer", "prefix", "stack", "concatenate", "vstack", "hstack"]


@dataclass(frozen=True)
class _Rule:
    operation: str
    axis: int | None = None


def collect():
    """Collect scalar occurrences into a Python list, in source order."""
    return _Rule("collect")


def prefer():
    """Choose the key from the first source containing it."""
    return _Rule("prefer")


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
    """Inputs combined by one rule, computed when read."""

    origin = format = None

    sources: dict[str, FileSystem]
    rule: _Rule

    @property
    def merged(self):
        return self.rule.operation

    def info(self, *, full=True):
        """Infer merged dataset metadata from source metadata."""
        if not full:
            return NodeInfo("dataset", axis=self.rule.axis)
        if self.rule.operation == "collect":  # Scalars, cheap to read.
            return value_info(self.read())
        import numpy as np
        inputs = [source.info() for source in self.sources.values()]
        stand_ins = [np.empty(each.shape, dtype="V0") for each in inputs]  # Their shapes, without data.
        return NodeInfo("dataset", dtype=str(np.result_type(*(each.dtype for each in inputs))),
                        shape=self._combine(stand_ins).shape, type="numpy.ndarray", axis=self.rule.axis)

    def items(self):
        raise NotADirectoryError("Merged dataset has no children.")

    def descendant(self, path):
        raise NotADirectoryError(path)

    def read(self, selection=None):
        operation, axis, sources = self.rule.operation, self.rule.axis, list(self.sources.values())
        key, at = (selection if isinstance(selection, tuple) else (selection,)), axis or 0
        if (selection is not None and (operation == "collect" or operation == "stack" and at >= 0)
                and len(key) > at and all(type(part) in (int, slice) for part in key[:at + 1])):
            # The merge axis enumerates sources: read only the selected ones, each
            # with the rest of the selection, so no other source is loaded.
            pick, inner = key[at], key[:at] + key[at + 1:] or None
            if type(pick) is int:
                return sources[pick].read("", inner)
            parts = [source.read("", inner) for source in sources[pick]]
            if operation == "collect":
                return parts
            if parts:
                import numpy as np
                # Integer indices before the merge axis remove dimensions, shifting it left.
                return np.stack(parts, axis=at - sum(type(part) is int for part in key[:at]))
        result = self._combine([source.read() for source in sources])
        return result if selection is None else result[selection]

    def _combine(self, values):
        """Combine source values according to the merge rule."""
        operation, axis = self.rule.operation, self.rule.axis
        if operation == "collect":
            return values
        import numpy as np
        function = getattr(np, operation)
        return function(values, axis=axis) if axis is not None else function(values)

    def digest(self):
        return _digest((self.rule, [source.digest() for source in self.sources.values()]))

    def locate(self, selection=None):
        # Selection addresses the assembled value, never each input independently.
        return [locator for source in self.sources.values() for locator in source.locate()]


@dataclass(frozen=True)
class _MergedGroup(_Composed):
    """Keys of the sources, resolved by rules when merged."""

    merged = "keys"

    sources: dict[str, FileSystem]
    rules: dict[str, _Rule]


def merge(sources: Mapping[str, Node], *, rules: _Rule | Mapping[str, _Rule]) -> FileSystem:
    """Merge sources into one dataset by a rule, or into a group by a mapping of keys to rules.

    "*" sets the rule for unlisted keys; without it, such a key must occur in one source.
    Construction inspects immediate keys and kinds once, without traversing subgroups.
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
    inputs = {name: node._filesystem.clone(node.path) for name, node in sources.items()}
    if isinstance(rules, _Rule):
        if rules.operation in ("prefer", "prefix"):
            raise ValueError(f"{rules.operation} merges groups key by key; it needs a mapping of rules.")
        return FileSystem._from_source(_MergedDataset(inputs, rules))
    rules = dict(rules)
    for key, rule in rules.items():
        if not (_is_name(key) or key == "*"):
            raise ValueError("Rule keys must be top-level field names or '*'.")
        if not isinstance(rule, _Rule):
            raise TypeError("Rules must be created with the merge rule helpers.")
    occurrences = {}
    for name, source in inputs.items():
        for key, kind, _ in source.items():
            occurrences.setdefault(key, {})[name] = (source.clone(key), kind)
    unknown = rules.keys() - occurrences.keys() - {"*"}
    if unknown:
        raise KeyError(f"Merge rules reference missing keys: {sorted(unknown)}")
    children = {}

    def add(key, subtree):
        if key in children:
            raise ValueError(f"Merge output collision: {key!r}")
        children[key] = subtree

    for key, found in occurrences.items():
        subtrees = {name: subtree for name, (subtree, _) in found.items()}
        rule = rules.get(key, rules.get("*"))
        if rule is None and len(subtrees) != 1:
            raise ValueError(f"Repeated key {key!r} requires a merge rule.")
        if rule is None or rule.operation == "prefer":
            add(key, next(iter(subtrees.values())))
        elif rule.operation == "prefix":
            for name, subtree in subtrees.items():
                add(f"{name}_{key}", subtree)
        else:
            if len(subtrees) != len(sources):
                raise KeyError(f"Merge key {key!r} must exist in every source.")
            expected = "scalar" if rule.operation == "collect" else "dataset"
            if any(kind != expected for _, kind in found.values()):
                raise TypeError(f"Merge key {key!r} requires {expected} inputs.")
            add(key, FileSystem._from_source(_MergedDataset(subtrees, rule)))
    return FileSystem._from_source(_MergedGroup(children, inputs, rules))

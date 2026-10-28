"""Pickle files read as data: structure from opcodes that are never run, values from a fixed set of data types.

Listing, kinds, and NumPy dtypes and shapes come from scanning the opcodes, skipping large byte payloads.
Loading builds only data types. Any other object becomes a record of what it is made of, its type, arguments
and state, so the data inside it stays readable while its code is never imported or run.
"""

import _compat_pickle
import pickle
import pickletools
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from . import NodeInfo, NodeKind, Reader, _group_entries, _path_get, value_info, value_kind

_ALLOWED = frozenset({
    *(("builtins", name) for name in (
        "bool", "bytearray", "bytes", "complex", "dict", "float", "frozenset", "int", "list", "range", "set",
        "slice", "str", "tuple")),
    ("collections", "OrderedDict"), ("collections", "defaultdict"), ("collections", "deque"),
    ("datetime", "date"), ("datetime", "datetime"), ("datetime", "time"), ("datetime", "timedelta"),
    ("datetime", "timezone"), ("decimal", "Decimal"),
    ("_codecs", "encode"),  # Protocols 0-2 store bytes as text encoded by this.
    ("numpy", "dtype"), ("numpy", "ndarray"), ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy.core.numeric", "_frombuffer"),
})
_MAPPINGS = {("builtins", "dict"), ("collections", "OrderedDict"), ("collections", "defaultdict")}
_SCALARS = {("builtins", "bytes"), ("builtins", "complex"), ("datetime", "date"), ("datetime", "datetime"),
            ("datetime", "time"), ("datetime", "timedelta"), ("decimal", "Decimal"),
            ("numpy.core.multiarray", "scalar")}
# Opcodes whose argument is the value they push.
_VALUES = {"INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4", "FLOAT", "BINFLOAT",
           "STRING", "BINSTRING", "SHORT_BINSTRING", "UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8",
           "SHORT_BINBYTES", "BINBYTES", "BINBYTES8", "BYTEARRAY8"}
# Byte payloads, by the size of their length prefix; the scan seeks past those above _INLINE bytes.
_PAYLOADS, _INLINE = {"bytes1": 1, "bytes4": 4, "bytes8": 8, "bytearray8": 8}, 1024


def _qualified(module: str, name: str) -> tuple[str, str]:
    # Python 2 names, such as __builtin__, map as the unpickler maps them; NumPy 2 moved numpy.core to numpy._core.
    module, name = _compat_pickle.NAME_MAPPING.get((module, name), (module, name))
    return _compat_pickle.IMPORT_MAPPING.get(module, module).replace("numpy._core", "numpy.core"), name


def _positional(items) -> dict[str, Any]:
    return {str(index): item for index, item in enumerate(items)}


class _Record(Mapping):
    """An object of a type that is not data, as its parts instead of the object: browsable, never run.

    Pickling calls the type, or its constructor function, with args, then gives the object its state and items.
    The record keeps each of those as given; tuples become positional mappings, so their contents stay reachable.
    """

    type: str

    def __new__(cls, *args, **kwargs):
        record = super().__new__(cls)
        record._parts = {"type": cls.type, "args": _positional(args)} | ({"kwargs": kwargs} if kwargs else {})
        return record

    def __init__(self, *args, **kwargs):
        pass

    def __getitem__(self, key):
        return self._parts[key]

    def __iter__(self):
        return iter(self._parts)

    def __len__(self):
        return len(self._parts)

    def __setstate__(self, state):
        self._parts["state"] = _positional(state) if isinstance(state, tuple) else state

    def __setitem__(self, key, value):
        self._parts.setdefault("items", {})[key] = value

    def extend(self, values):
        items = self._parts.setdefault("items", {})
        items.update(_positional([*items.values(), *values]))

    def append(self, value):
        self.extend([value])

    def __call__(self, *args, **kwargs):  # Calling a recorded object is recorded too.
        return _record_type(f"{self.type}()")(*args, **kwargs)


@lru_cache(maxsize=None)
def _record_type(name: str) -> type[_Record]:
    """A stand-in for the global name: calling it records the call, and nothing named is imported."""
    return type(name.rpartition(".")[2], (_Record,), {"type": name})


def _resolved(module: str, name: str):
    """A data type's global as a reference for the scan, or the record type standing in for any other."""
    name = _qualified(module, name)
    return _Ref(name) if name in _ALLOWED else _record_type(".".join(name))


class _DataUnpickler(pickle.Unpickler):
    """Builds data types; other globals become record types. Persistent IDs and out-of-band buffers stay refused."""

    def find_class(self, module, name):
        resolved = _resolved(module, name)
        return super().find_class(*resolved.name) if isinstance(resolved, _Ref) else resolved


def pickle_load(path: str | Path) -> Any:
    """A pickle file's value: data types built, other objects as records of their parts."""
    with open(path, "rb") as file:
        return _DataUnpickler(file).load()


@dataclass(eq=False)
class _Ref:
    """A global, or what calling it builds, as the scan records it without building; args is None until called."""

    name: tuple[str, str] | None
    args: Any = None
    state: Any = None


def _operations(file) -> Iterable[tuple[str, Any]]:
    """Opcode names and arguments; large byte payloads are skipped, their contents never read."""
    while code := file.read(1):
        operation = pickletools.code2op[code.decode("latin-1")]
        descriptor = operation.arg
        if descriptor is not None and descriptor.name in _PAYLOADS:
            size = int.from_bytes(file.read(_PAYLOADS[descriptor.name]), "little")
            if size <= _INLINE:
                argument = file.read(size)
            else:
                file.seek(size, 1)
                argument = _Ref(("builtins", "bytes"), ())
        else:
            argument = descriptor and descriptor.reader(file)
        yield operation.name, argument
    raise ValueError("Pickle ends without STOP.")


def _call(function, args, kwargs=None):
    """What calling function with args builds, so far as scanning can know without running it."""
    if isinstance(function, (type, _Record)):  # Record types, and records, record the call as loading does.
        return function(*args, **(kwargs or {}))
    name = function.name if isinstance(function, _Ref) and function.args is None else None
    if name in _MAPPINGS:  # Rebuilt as dicts, so their items fill in and paths reach them.
        return {}
    if name == ("_codecs", "encode") and all(isinstance(arg, str) for arg in args):
        return args[0].encode(*args[1:])  # How protocols 0-2 store bytes; encoding text runs nothing.
    return b"" if name == ("builtins", "bytes") and not args else _Ref(name, args)


_CONSTANTS = {"NONE": lambda: None, "NEWTRUE": lambda: True, "NEWFALSE": lambda: False, "EMPTY_DICT": dict,
              "EMPTY_LIST": list, "EMPTY_TUPLE": tuple, "EMPTY_SET": lambda: _Ref(("builtins", "set"), ())}


def _scan(file):
    """The pickled structure: containers and plain values, with references for anything to build."""
    stack, memo, marks = [], {}, []

    def take(count=None):
        """The top count items, or those above the last mark, removed from the stack."""
        start = marks.pop() if count is None else len(stack) - count
        items = stack[start:]
        del stack[start:]
        return items

    for name, argument in _operations(file):
        if name == "STOP":
            return stack.pop()
        if name in _VALUES:
            stack.append(argument)
        elif name in _CONSTANTS:
            stack.append(_CONSTANTS[name]())
        elif name == "MARK":
            marks.append(len(stack))
        elif name in ("POP", "POP_MARK"):
            take(1 if name == "POP" else None)
        elif name == "DUP":
            stack.append(stack[-1])
        elif name in ("PUT", "BINPUT", "LONG_BINPUT", "MEMOIZE"):
            memo[len(memo) if name == "MEMOIZE" else argument] = stack[-1]
        elif name in ("GET", "BINGET", "LONG_BINGET"):
            stack.append(memo[argument])
        elif name in ("TUPLE1", "TUPLE2", "TUPLE3", "TUPLE"):
            stack.append(tuple(take(int(name[-1]) if name[-1].isdigit() else None)))
        elif name in ("LIST", "DICT"):
            items = take()
            stack.append(items if name == "LIST" else dict(zip(items[::2], items[1::2])))
        elif name in ("APPEND", "APPENDS"):
            items = take(1 if name == "APPEND" else None)
            if isinstance(stack[-1], (list, _Record)):
                stack[-1].extend(items)
        elif name in ("SETITEM", "SETITEMS"):
            items = take(2 if name == "SETITEM" else None)
            if isinstance(stack[-1], (dict, _Record)):
                for key, value in zip(items[::2], items[1::2]):
                    stack[-1][key] = value
        elif name in ("FROZENSET", "ADDITEMS"):
            take()
            if name == "FROZENSET":
                stack.append(_Ref(("builtins", "frozenset"), ()))
        elif name in ("GLOBAL", "STACK_GLOBAL"):
            stack.append(_resolved(*(argument.split(" ", 1) if name == "GLOBAL" else take(2))))
        elif name in ("REDUCE", "NEWOBJ", "NEWOBJ_EX"):
            stack.append(_call(*take(3 if name == "NEWOBJ_EX" else 2)))
        elif name in ("INST", "OBJ"):
            items = take()
            function = _resolved(*argument.split(" ", 1)) if name == "INST" else items.pop(0)
            stack.append(_call(function, tuple(items)))
        elif name == "BUILD":
            state, = take(1)
            if isinstance(stack[-1], _Record):
                stack[-1].__setstate__(state)
            elif isinstance(stack[-1], _Ref):
                stack[-1].state = state
        elif name == "NEXT_BUFFER":
            stack.append(_Ref(("builtins", "bytes"), ()))
        elif name not in ("PROTO", "FRAME", "READONLY_BUFFER"):
            raise ValueError(f"Pickle opcode {name} is not supported; it may need objects outside the file.")
    raise ValueError("Pickle ends without STOP.")


@lru_cache(maxsize=16)
def _scanned(path: str, size: int, modified: int):
    with open(path, "rb") as file:
        return _scan(file)


def _structure(path, key):
    """The structure at key, scanned once per file version."""
    stat = Path(path).stat()
    return _path_get(_scanned(str(path), stat.st_size, stat.st_mtime_ns), key or "")


def _parts(node) -> Iterable:
    if isinstance(node, _Ref):
        return node.args, node.state
    if isinstance(node, Mapping):
        return node.values()
    return node if isinstance(node, (list, tuple)) else ()


def _plain(node) -> bool:
    """Whether node holds only values the scan read, with nothing left to build."""
    return not isinstance(node, _Ref) and all(map(_plain, _parts(node)))


def _kind(node) -> NodeKind:
    if isinstance(node, _Ref):
        return "scalar" if node.name in _SCALARS else "dataset"
    return value_kind(node)


def _dtype(node) -> str | None:
    if not isinstance(node, _Ref) or node.name != ("numpy", "dtype") or not isinstance(node.state, tuple):
        return None
    code, order, fields = node.args[0], node.state[1], node.state[3]
    if fields is not None:  # Structured dtypes need their fields built.
        return None
    import numpy as np
    return str(np.dtype(order + code if order in "<>" else code))


def _array(node) -> NodeInfo | None:
    """The dtype and shape of a NumPy array the pickle would build."""
    if not isinstance(node, _Ref):
        return None
    if node.name == ("numpy.core.multiarray", "_reconstruct") and isinstance(node.state, tuple):
        shape, dtype = node.state[1], _dtype(node.state[2])
    elif node.name == ("numpy.core.numeric", "_frombuffer"):
        shape, dtype = node.args[2], _dtype(node.args[1])
    else:
        return None
    # Object arrays are described by the dtype their loaded items share, as in-memory values are.
    return None if dtype in (None, "object") else NodeInfo("dataset", dtype=dtype, shape=tuple(shape),
                                                          type="numpy.ndarray")


def _inspect_node(path, key=None) -> NodeInfo:
    node = _structure(path, key)
    kind = _kind(node)
    if kind == "group":
        return NodeInfo(kind, tuple(str(name) for name, _ in _group_entries(node)))
    if kind == "scalar":
        return NodeInfo(kind)
    return _array(node) or (value_info(node) if _plain(node) else NodeInfo(kind))


def _inspect(path, key=None) -> Iterable[tuple[str, NodeKind, Any]]:
    node = _structure(path, key)
    if _kind(node) != "group":
        raise NotADirectoryError(key)
    loaded = None
    for name, child in _group_entries(node):
        kind = _kind(child)
        if kind == "scalar" and not _plain(child):  # Dates, decimals and skipped bytes show their loaded value.
            loaded = _path_get(pickle_load(path), key or "") if loaded is None else loaded
            child = _path_get(loaded, str(name))
        yield str(name), kind, child if kind == "scalar" else None


PICKLE = Reader(pickle_load, inspect=_inspect, inspect_node=_inspect_node, suffixes=(".pkl", ".pickle"),
                format="pickle")

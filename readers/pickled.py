"""Pickle files read as data: structure from opcodes that are never run, trusted types unpickled for real.

A value built only from types that are _ALLOWED or explicitly trusted - see register_view's and
register_viewer's trust= argument - comes back for real, and from then on is just a value: browsed
through value_kind and value_info exactly like any other in-memory value, nothing pickle-specific about
it. Anything else is never imported or run: it is an opaque leaf naming only its type, with nothing
underneath it to open.
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
# Glue many libraries pickle their bound callbacks with: trusted only once something else is, since a bound
# method reduces to (getattr, (instance, name)), which *runs* getattr on whatever instance the pickle names -
# safe once that instance's own class is itself trusted to be real, but not a default risk for untrusted data.
_GLUE = frozenset({("builtins", "object"), ("builtins", "getattr"), ("functools", "partial")})
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


_TRUSTED: set[tuple[str, str] | str] = set()


def _allow(trust) -> None:
    """Trust register_view's and register_viewer's trust= argument to unpickle for real and run its code.

    Not for general use: register_view and register_viewer are the only callers, so trust is only
    ever granted alongside something that can actually use the real object.

    A target is a class (trusted exactly), an exact (module, name) pair, a module or package prefix
    string such as "matplotlib" (trusts everything under it), or a list of any of these. A complex
    object such as a Figure nests many classes in its pickled state; every one of them needs to be
    trusted too, or it loads as an opaque leaf instead, so a library-wide prefix is usually what makes
    such an object come back whole. This is as safe, or as unsafe, as pickle.load is for whatever is
    trusted: only reachable from code that trusts the pickles it will be given.
    """
    if trust is None:
        return
    if isinstance(trust, list):
        for target in trust:
            _allow(target)
    elif isinstance(trust, tuple):
        _TRUSTED.add(trust)
    elif isinstance(trust, type):
        _TRUSTED.add((trust.__module__, trust.__qualname__))
    elif isinstance(trust, str):
        _TRUSTED.add(trust)
    else:
        raise TypeError("trust needs a class, a module prefix string, a (module, name) pair, or a list of these.")


def _buildable(qualified: tuple[str, str]) -> bool:
    """Whether a global is safe to build for real: a baseline data type, or explicitly trusted."""
    if qualified in _ALLOWED or qualified in _TRUSTED:
        return True
    if _TRUSTED and qualified in _GLUE:
        return True
    module, _ = qualified
    return any(isinstance(prefix, str) and (module == prefix or module.startswith(f"{prefix}."))
               for prefix in _TRUSTED)


@dataclass(frozen=True)
class _Opaque:
    """What scanning finds in place of anything not built for real: its type, with nothing underneath."""

    type: str

    def __repr__(self):
        return f"<{self.type}>"


def _global(module: str, name: str):
    """A GLOBAL reference during scanning: a reference to build later, or already-known opaque."""
    qualified = _qualified(module, name)
    return _Ref(qualified) if _buildable(qualified) else _Opaque(".".join(qualified))


_touched: list[str] = []  # Opaque leaf types stood in for since _leaf_error last read this, for a clear error.


def _leaf_error() -> Exception | None:
    """A clear error naming the last opaque leaf stood in for since this was last called, if any."""
    if not _touched:
        return None
    leaf, _touched[:] = _touched[-1], []
    return RuntimeError(f"{leaf} was never built for real, so using it here failed: trust it through "
                        f"register_view's or register_viewer's trust= argument.")


class _Unbuilt:
    """Stands in for any untrusted global while loading for real: holds nothing of what it is given.

    __getattr__ returns itself for plain names so a bound method elsewhere in the same pickle - one
    pickles as (getattr, (instance, name)) - can resolve against an unrelated opaque instance without
    crashing the whole load. Dunder lookups still raise, so value_kind's shape-based scalar check (see
    the class attribute below) still recognizes this as a leaf with nothing to open, not a dataset.
    Every fallback also records itself in _touched, so code reading the real value afterwards - a view,
    say - can turn a confusing failure into a clear one instead, once it actually tries to use this leaf.
    """

    shape = ()

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass

    def append(self, value):
        pass

    def extend(self, values):
        pass

    def __setitem__(self, key, value):
        pass

    def __call__(self, *args, **kwargs):
        _touched.append(self.type)
        return self

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        _touched.append(self.type)
        return self

    def __repr__(self):
        return f"<{self.type}>"


@lru_cache(maxsize=None)
def _unbuilt(name: str) -> type:
    """A class standing in for an untrusted global while loading for real: builds nothing but its name."""
    return type(name.rpartition(".")[2], (_Unbuilt,), {"type": name})


class _DataUnpickler(pickle.Unpickler):
    """Builds trusted types for real; anything else becomes an _Unbuilt instance naming only its type."""

    def find_class(self, module, name):
        qualified = _qualified(module, name)
        return super().find_class(*qualified) if _buildable(qualified) else _unbuilt(".".join(qualified))


def pickle_load(path: str | Path) -> Any:
    """A pickle file's value: trusted types built for real, anything else an opaque, unopenable leaf."""
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
    if isinstance(function, _Opaque):  # An untrusted call stays untrusted, whatever it is given.
        return function
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
            if isinstance(stack[-1], list):
                stack[-1].extend(items)
        elif name in ("SETITEM", "SETITEMS"):
            items = take(2 if name == "SETITEM" else None)
            if isinstance(stack[-1], dict):
                for key, value in zip(items[::2], items[1::2]):
                    stack[-1][key] = value
        elif name in ("FROZENSET", "ADDITEMS"):
            take()
            if name == "FROZENSET":
                stack.append(_Ref(("builtins", "frozenset"), ()))
        elif name in ("GLOBAL", "STACK_GLOBAL"):
            stack.append(_global(*(argument.split(" ", 1) if name == "GLOBAL" else take(2))))
        elif name in ("REDUCE", "NEWOBJ", "NEWOBJ_EX"):
            stack.append(_call(*take(3 if name == "NEWOBJ_EX" else 2)))
        elif name in ("INST", "OBJ"):
            items = take()
            function = _global(*argument.split(" ", 1)) if name == "INST" else items.pop(0)
            stack.append(_call(function, tuple(items)))
        elif name == "BUILD":
            state, = take(1)
            if isinstance(stack[-1], _Ref):
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
    if isinstance(node, _Opaque):
        return "scalar"
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
    if kind == "scalar":
        return NodeInfo(kind, type=node.type if isinstance(node, _Opaque) else None)
    if kind == "group":
        return NodeInfo(kind, tuple(str(name) for name, _ in _group_entries(node)))
    array = _array(node)
    if array is not None:
        return array
    if _plain(node):
        return value_info(node)
    return NodeInfo(kind, type=".".join(node.name) if isinstance(node, _Ref) else None)


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

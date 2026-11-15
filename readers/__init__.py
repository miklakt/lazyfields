from __future__ import annotations

import hashlib, logging, shutil, sys, tarfile, warnings, zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, time, timedelta
from decimal import Decimal
from numbers import Number
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from typing import Any, Callable, Iterable, Iterator, Literal, Mapping

try:
    import orjson as json
except ImportError:
    import json


def _json_load(path: str | Path) -> Mapping[str, Any]: return json.loads(Path(path).read_bytes())


NodeKind = Literal["scalar", "dataset", "group"]


def _group_entries(value: Any) -> Iterable[tuple[Any, Any]]:
    # Lists of mappings are groups whose children are named by index.
    return value.items() if isinstance(value, Mapping) else ((str(index), item) for index, item in enumerate(value))


def value_kind(value: Any) -> NodeKind:
    if isinstance(value, Mapping): return "group"
    if isinstance(value, (list, tuple)) and value and all(isinstance(item, Mapping) for item in value):
        return "group"
    if value is None or isinstance(value, (str, bytes, Number, Decimal, date, time, timedelta)):
        return "scalar"
    # Array scalars expose shape but have no container length; 0-D arrays remain datasets.
    if getattr(value, "shape", None) == () and not hasattr(value, "__len__"):
        return "scalar"
    return "dataset"


def as_array(value: Any) -> Any:
    """Convert a value to a NumPy array, inferring a common dtype where possible."""
    import numpy as np

    try:
        array = np.asarray(value)
    except ValueError:
        array = np.asarray(value, dtype=object)
    if array.dtype == object:
        try:
            array = np.asarray(array.tolist())
        except (TypeError, ValueError):
            pass
    return array


def value_info(value: Any) -> NodeInfo:
    """Classify a value and describe its children, or a dataset's dtype, shape and type."""
    kind = value_kind(value)
    if kind == "group":
        return NodeInfo(kind, tuple(str(name) for name, _ in _group_entries(value)))
    if kind == "dataset":
        array = value if hasattr(value, "shape") and getattr(value, "dtype", object) != object else as_array(value)
        return NodeInfo(kind, dtype=str(array.dtype), shape=tuple(array.shape),
                        type=f"{type(value).__module__}.{type(value).__qualname__}")
    return NodeInfo(kind)


def _child_key(key: str | None, name: Any) -> str:
    return f"{key}/{name}" if key else str(name)


def _describe_value(value: Any, key: str | None = None, depth: int | None = None):
    """(key, info, scalar value or None) for value and its descendants, depth levels down or all.

    Nodes at the depth limit are given their kind only; a group containing itself is not entered again.
    """
    def walk(value, key, depth, ancestors):
        info = value_info(value) if depth != 0 else NodeInfo(value_kind(value))
        yield key, info, value if info.kind == "scalar" else None
        if info.kind == "group" and depth != 0 and id(value) not in ancestors:
            for name, item in _group_entries(value):
                yield from walk(item, _child_key(key, name), depth and depth - 1, ancestors | {id(value)})

    return walk(value, key, depth, frozenset())


@dataclass(frozen=True)
class NodeInfo:
    """Facts derived from a node's source; attrs holds the metadata given to it, runtime_attrs the
    attributes of the node handle it was asked through."""

    kind: NodeKind
    children: tuple[str, ...] = ()
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    # The loaded value's type, known once a value is classified.
    type: str | None = None
    # The root's format from its view, such as a file's "hdf5", or "directory".
    format: str | None = None
    # Whether a namespace edit moved this subtree away from its source position.
    relocated: bool = False
    # The merge rule that computed this node, such as "stack", or "keys" for a group, and its axis.
    merged: str | None = None
    axis: int | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)
    runtime_attrs: Mapping[str, Any] = field(default_factory=dict)


def _h5():
    try: import h5py
    except ImportError as exc: raise ImportError("h5py is required to load HDF5-backed rows.") from exc
    return h5py


def _normalize(value: Any) -> Any:
    if isinstance(value, bytes): return value.decode("utf-8")
    return value.item() if getattr(value, "shape", None) == () else value


def _dataset_value(node: Any, selection: Any = ()) -> Any:
    """Read a dataset selection; strings of any HDF5 string type come back as str."""
    # An empty dataspace holds no strings; h5py's asstr() cannot read it.
    if node.shape is not None and _h5().check_string_dtype(node.dtype) is not None:
        try:
            # ASCII is a subset of UTF-8, so this also reads UTF-8 written with an ASCII label.
            value = node.asstr("utf-8")[selection]
            return value.astype(str) if getattr(value, "dtype", None) == object else value
        except UnicodeDecodeError:
            pass  # Not UTF-8: fall back to the stored bytes.
    return _normalize(node[selection])


HDF5_LOCK_TIMEOUT = 5.0  # seconds to wait for a writer to release a growing file


def _open_hdf5(path: str | Path) -> Any:
    """Open read-only, waiting while another process holds the file for writing.

    HDF5 locks a file while a writer has it open; a writer that opens the file only to
    append (such as a running simulation) releases it within moments.
    """
    deadline = monotonic() + HDF5_LOCK_TIMEOUT
    while True:
        try:
            return _h5().File(path, "r")
        except BlockingIOError:
            if monotonic() >= deadline:
                raise
            sleep(0.05)


@contextmanager
def _hdf5_node(path: str | Path, key: str | None = None) -> Iterator[Any]:
    """Open an HDF5 file and yield the selected node."""
    with _open_hdf5(path) as file:
        yield file[key] if key else file


def _read(node: Any) -> Any:
    if isinstance(node, _h5().Dataset): return _dataset_value(node)
    return {key: _read(node[key]) for key in node.keys()}


def _hdf5_load(path: str | Path) -> Mapping[str, Any]:
    with _hdf5_node(path) as file: return _read(file)


def hdf5_get(path: str | Path, key: str, selection: Any | None = None) -> Any:
    with _hdf5_node(path, key) as node:
        if selection is None:
            return _read(node)
        if isinstance(node, _h5().Dataset):
            try:
                return _dataset_value(node, selection)
            except (TypeError, ValueError):
                pass
        return _normalize(_read(node)[selection])


def _hdf5_kind(node: Any) -> NodeKind:
    return "group" if isinstance(node, _h5().Group) else "scalar" if node.shape == () else "dataset"


def hdf5_describe(path: str | Path) -> Iterable[tuple[str | None, NodeInfo, Any]]:
    def walk(node, key, ancestors):
        kind = _hdf5_kind(node)
        if kind == "group":
            yield key, NodeInfo(kind, tuple(node.keys())), None
            if node.id not in ancestors:  # A hard link back to an ancestor is listed, not entered.
                for name, child in node.items():
                    yield from walk(child, _child_key(key, name), ancestors | {node.id})
        elif kind == "dataset":
            yield key, NodeInfo(kind, dtype=str(node.dtype), shape=node.shape), None
        else:
            yield key, NodeInfo(kind), _dataset_value(node)

    with _open_hdf5(path) as file:
        yield from walk(file, None, frozenset())


def _path_get(value: Any, key: str) -> Any:
    for part in filter(None, key.split("/")):
        quoted = len(part) >= 2 and part[0] == part[-1] and part[0] in {"'", '"'}
        if quoted: part = part[1:-1]
        try:
            index = int(part)
        except ValueError:
            index = part
        if isinstance(value, Mapping):
            if quoted and isinstance(index, int):
                # Quoting distinguishes intentional string keys from numeric path indices.
                warnings.warn(f"Quoted path segment {part!r} is treating the segment as a string key.", stacklevel=5)
            if not quoted and part not in value and index in value: part = index
        elif not quoted: part = index
        value = value[part]
    return value


def _check_key(key: Any) -> None:
    if key is not None and not isinstance(key, str): raise TypeError("key must be a string or None.")


@dataclass(frozen=True)
class View:
    """Interprets one object: its value, and the kinds, children and scalar values of the nodes in it."""

    # Returns the object's complete value
    load: Callable[..., Any]
    # Optional read(obj, key, selection): one key or selection directly, without loading everything
    read: Callable[..., Any] | None = None
    # Optional describe(obj): every node in one pass, as nodes() gives them, without loading the data.
    describe: Callable[..., Iterable[tuple[str | None, NodeInfo, Any]]] | None = field(default=None, kw_only=True)
    # Format of the object's root, such as "pdb".
    format: str | None = field(default=None, kw_only=True, compare=False)

    def get(self, obj: Any, key: str | None = None, selection: Any | None = None) -> Any:
        """Read the object or one key, forwarding selection to a direct read."""
        _check_key(key)
        if selection is not None and key is None:
            raise ValueError("selection requires a single key.")
        if key is None: return self.load(obj)
        if self.read is not None: return self.read(obj, key, selection)
        value = _path_get(self.load(obj), key)
        if isinstance(selection, tuple) and not (hasattr(value, "shape") and getattr(value, "dtype", object) != object):
            # Only a real array already supports indexing by a tuple of slices, one per axis; anything
            # else - a JSON list, a DataFrame missing a single dtype - needs converting to one first.
            value = as_array(value)
        return value if selection is None else value[selection]

    def nodes(self, obj: Any) -> Iterable[tuple[str | None, NodeInfo, Any]]:
        """Every node as (key, info, scalar value or None): the root as key None, each parent before its children.

        A dataset's info may leave dtype and shape out, for reading its value to tell.
        """
        return self.describe(obj) if self.describe is not None else _describe_value(self.load(obj))


@dataclass(frozen=True)
class Reader(View):
    """A view of files: its object is a file's path, and suffixes select it."""

    # File suffixes handled by this reader
    suffixes: tuple[str, ...] = field(default=(), kw_only=True)
    format: str = field(default="file", kw_only=True, compare=False)


# Mappings, and lists of mappings, as groups: the view of in-memory results and of JSON and pickle data.
MAPPING = View(lambda value: value)


@dataclass(frozen=True)
class DebugReader(Reader):
    """A Reader whose backend requests are logged to stdout with timestamps; of(reader) decorates one."""

    @classmethod
    def of(cls, reader: Reader) -> DebugReader:
        logger = logging.getLogger("lazyfields.DebugReader")
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        def logged(command, callback):
            if callback is None: return None
            def call(*args, **kwargs):
                logger.info("DebugReader %s %s args=%r kwargs=%r", reader.suffixes, command, args, kwargs)
                return callback(*args, **kwargs)
            return call

        hooks = {name: logged(name, getattr(reader, name)) for name in ("load", "read", "describe")}
        return cls(**hooks, suffixes=reader.suffixes, format=reader.format)


_ARCHIVE_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".txz", ".zip")
_CACHE, _READERS = TemporaryDirectory(prefix="lazyfields-"), {}


def register_reader(reader: Reader) -> Reader:
    suffixes = tuple(suffix.lower() for suffix in reader.suffixes)
    if any(not suffix.startswith(".") for suffix in suffixes): raise ValueError("Reader suffixes must start with '.'.")
    duplicate = next((suffix for suffix in suffixes if suffix in _READERS), None)
    if duplicate is not None: raise ValueError(f"A reader for {duplicate!r} is already registered.")
    _READERS.update(dict.fromkeys(suffixes, reader))
    return reader


_VIEWS: dict[str, View] = {}


def register_view(name: str, view: View) -> View:
    """Name a view; a node whose "view" attribute names it is browsed as the object view interprets."""
    if not isinstance(name, str) or not name or not isinstance(view, View):
        raise TypeError("register_view needs a nonempty name and a View.")
    _VIEWS[name] = view
    return view


def registered_view(name: str) -> View:
    try:
        return _VIEWS[name]
    except KeyError:
        raise KeyError(f"No view is registered as {name!r}.") from None


def debug_readers() -> None:
    """Enable debug logging for registered readers; call before creating filesystems."""
    for reader in _READERS.values():
        if not isinstance(reader, DebugReader):
            debug = DebugReader.of(reader)
            for suffix in reader.suffixes:
                _READERS[suffix.lower()] = debug


def _reader_match(path: str | Path, archives: bool = True) -> tuple[Reader | None, str | None]:
    suffixes = Path(path).suffixes
    for start in range(len(suffixes)):
        suffix = "".join(suffixes[start:]).lower()
        reader = _READERS.get(suffix)
        if reader is not None and (archives or suffix not in _ARCHIVE_SUFFIXES):
            return reader, suffix
    return None, None


def require_reader(path: str | Path) -> Reader:
    reader = _reader_match(path)[0]
    if reader is None: raise ValueError(f"Could not infer file reader from '{Path(path).name}'.")
    return reader


def archive_suffix(path: str | Path) -> str | None:
    return next((suffix for suffix in _ARCHIVE_SUFFIXES if Path(path).name.lower().endswith(suffix)), None)


def _archive_member(path: str | Path, files: list[tuple[Any, str]], kind: str) -> tuple[Any, str]:
    supported = [(member, filename, _reader_match(filename, False)[1]) for member, filename in files]
    supported = [item for item in supported if item[2] is not None]
    if len(supported) > 1: raise ValueError(f"{kind} archives must contain exactly one supported file.")
    if supported:
        member, filename, suffix = supported[0]
    else:
        archive = archive_suffix(path)
        _, suffix = _reader_match(Path(path).name[:-len(archive)] if archive else "", False)
        if len(files) != 1 or suffix is None: raise ValueError(f"{kind} archives must contain one supported file.")
        member, filename = files[0]
    return member, _inner_filename(filename, suffix)


def _inner_filename(filename: str, suffix: str) -> str:
    name = Path(filename).name
    return name if name and name.lower().endswith(suffix) else f"inner_file{suffix}"


def _copy_member(source: Any, inner_path: Path) -> Path:
    tmp_path = inner_path.with_name(f"{inner_path.name}.tmp")
    try:
        with source, tmp_path.open("wb") as target:
            shutil.copyfileobj(source, target)
        tmp_path.replace(inner_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return inner_path


@contextmanager
def unpacked_archive_member(path: str | Path) -> Iterator[Path]:
    path = Path(path)
    suffix = archive_suffix(path)
    if suffix is None: raise ValueError(f"'{path.name}' is not a supported archive.")
    stat = path.stat()
    key = hashlib.sha256(f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()).hexdigest()[:16]
    cache_dir = Path(_CACHE.name) / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = [file for file in cache_dir.iterdir() if file.is_file() and not file.name.endswith(".tmp")]
    if len(files) == 1:
        yield files[0]; return
    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True)
    if suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            files = [(info, info.filename) for info in archive.infolist() if not info.is_dir()]
            info, inner_name = _archive_member(path, files, "Zip")
            yield _copy_member(archive.open(info), cache_dir / inner_name)
    else:
        with tarfile.open(path, "r:*") as archive:
            files = [(info, info.name) for info in archive.getmembers() if info.isfile()]
            info, inner_name = _archive_member(path, files, "Tar")
            source = archive.extractfile(info)
            if source is None: raise ValueError(f"Could not read '{info.name}' from '{Path(path).name}'.")
            yield _copy_member(source, cache_dir / inner_name)


def _archive_load(path: str | Path) -> Mapping[str, Any]:
    with unpacked_archive_member(path) as inner_path: return require_reader(inner_path).load(inner_path)


def archive_get(path: str | Path, key: str, selection: Any | None = None) -> Any:
    with unpacked_archive_member(path) as inner_path: return require_reader(inner_path).get(inner_path, key, selection)


def archive_describe(path: str | Path) -> Iterable[tuple[str | None, NodeInfo, Any]]:
    with unpacked_archive_member(path) as inner_path: yield from require_reader(inner_path).nodes(inner_path)


from .pickled import PICKLE, pickle_reader  # noqa: E402  Builds on the names above.

for _reader in (
    PICKLE,
    Reader(_json_load, suffixes=(".json",), format="json"),
    Reader(_hdf5_load, hdf5_get, describe=hdf5_describe, suffixes=(".h5", ".hdf5"), format="hdf5"),
    Reader(_archive_load, archive_get, describe=archive_describe, suffixes=_ARCHIVE_SUFFIXES,
           format="archive")
    ): register_reader(_reader)

import hashlib, json, pickle, shutil, tarfile, warnings, zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterable, Iterator, Mapping

import pandas as pd


def pickle_load(path: str | Path) -> Mapping[str, Any]: return pickle.loads(Path(path).read_bytes())
def json_load(path: str | Path) -> Mapping[str, Any]: return json.loads(Path(path).read_text())


def _h5():
    try: import h5py
    except ImportError as exc: raise ImportError("h5py is required to load HDF5-backed rows.") from exc
    return h5py


def _normalize(value: Any) -> Any:
    if isinstance(value, bytes): return value.decode("utf-8")
    return value.item() if getattr(value, "shape", None) == () else value


def _read(node: Any, h5py: Any) -> Any:
    if isinstance(node, h5py.Dataset): return _normalize(node[()])
    return {key: _read(node[key], h5py) for key in node.keys()}


def hdf5_load(path: str | Path) -> Mapping[str, Any]:
    h5py = _h5()
    with h5py.File(path, "r") as fh: return {key: _read(fh[key], h5py) for key in fh.keys()}


def hdf5_get(path: str | Path, key: str, selection: Any | None = None) -> Any:
    h5py = _h5()
    with h5py.File(path, "r") as fh:
        node = fh
        for part in key.split("/"):
            if part:
                node = node[part]
        if selection is None:
            return _read(node, h5py)
        if isinstance(node, h5py.Dataset):
            try:
                return _normalize(node[selection])
            except (TypeError, ValueError):
                pass
        return _normalize(_read(node, h5py)[selection])


def hdf5_inspect(path: str | Path) -> Iterable[tuple[str, bool, Any]]:
    h5py = _h5()
    with h5py.File(path, "r") as fh:
        for key, node in fh.items():
            scalar = isinstance(node, h5py.Dataset) and node.shape == ()
            value = _normalize(node[()]) if scalar else None
            yield key, scalar and pd.api.types.is_scalar(value), value


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


@dataclass(frozen=True)
class Reader:
    # File suffixes handled by this reader
    suffixes: tuple[str, ...]
    # Loads one file and returns its complete key-value data
    load: Callable[..., Mapping[str, Any]]
    # Optionally reads one key or selection directly, without loading everything
    read: Callable[..., Any] | None = None
    # Optionally provide when keys and scalars can be inspected without loading large values
    # HDF5 supports this, while pickle and JSON must load fully.
    inspect: Callable[..., Iterable[tuple[str, bool, Any]]] | None = None
    # Exceptions that indicate a requested key or selection is missing.
    missing_errors: tuple[type[Exception], ...] = (KeyError, TypeError, IndexError)

    def get(self, path: str | Path, key: str, selection: Any | None = None) -> Any:
        if self.read is not None: return self.read(path, key, selection)
        value = _path_get(self.load(path), key)
        return value if selection is None else value[selection]

    def items(self, path: str | Path) -> Iterable[tuple[str, bool, Any]]:
        if self.inspect is not None: return self.inspect(path)
        values = self.load(path)
        if not isinstance(values, Mapping): raise TypeError("expected a mapping row dictionary.")
        return ((key, pd.api.types.is_scalar(value), value) for key, value in values.items())


_ARCHIVE_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".txz", ".zip")
_CACHE, _READERS = TemporaryDirectory(prefix="lazyfields-"), {}


def register_reader(reader: Reader) -> Reader:
    suffixes = tuple(suffix.lower() for suffix in reader.suffixes)
    if any(not suffix.startswith(".") for suffix in suffixes): raise ValueError("Reader suffixes must start with '.'.")
    duplicate = next((suffix for suffix in suffixes if suffix in _READERS), None)
    if duplicate is not None: raise ValueError(f"A reader for {duplicate!r} is already registered.")
    _READERS.update(dict.fromkeys(suffixes, reader))
    return reader


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


def archive_load(path: str | Path) -> Mapping[str, Any]:
    with unpacked_archive_member(path) as inner_path: return require_reader(inner_path).load(inner_path)


def archive_get(path: str | Path, key: str, selection: Any | None = None) -> Any:
    with unpacked_archive_member(path) as inner_path: return require_reader(inner_path).get(inner_path, key, selection)


def archive_inspect(path: str | Path) -> Iterable[tuple[str, bool, Any]]:
    with unpacked_archive_member(path) as inner_path: yield from require_reader(inner_path).items(inner_path)


for _reader in (
    Reader((".pkl", ".pickle"), pickle_load),
    Reader((".json",), json_load),
    Reader((".h5", ".hdf5"), hdf5_load, hdf5_get, hdf5_inspect, (KeyError, TypeError)),
    Reader(_ARCHIVE_SUFFIXES, archive_load, archive_get, archive_inspect, (KeyError, TypeError, IndexError, ValueError))
    ): register_reader(_reader)

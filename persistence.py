import atexit
import hashlib
import json
import pickle
import shutil
import tarfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


def pickle_load(path: str | Path) -> Mapping[str, Any]: return pickle.loads(Path(path).read_bytes())
def json_load(path: str | Path) -> Mapping[str, Any]: return json.loads(Path(path).read_text())


def _h5():
    try: import h5py
    except ImportError as exc: raise ImportError("h5py is required to load HDF5-backed rows.") from exc
    return h5py


def _normalize(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value.item() if getattr(value, "shape", None) == () else value


def _select(value: Any, selection: Any | None) -> Any:
    if selection is None:
        return value
    return _normalize(value[selection])


def _node(root: Any, key: str) -> Any:
    node = root
    for part in key.split("/"):
        if part:
            node = node[part]
    return node


def _read(node: Any, h5py: Any) -> Any:
    if isinstance(node, h5py.Dataset):
        return _normalize(node[()])
    return {key: _read(node[key], h5py) for key in node.keys()}


def hdf5_load(path: str | Path) -> Mapping[str, Any]:
    h5py = _h5()
    with h5py.File(path, "r") as fh: return {key: _read(fh[key], h5py) for key in fh.keys()}


def _read_selected(node: Any, h5py: Any, selection: Any | None) -> Any:
    if selection is None:
        return _read(node, h5py)
    if isinstance(node, h5py.Dataset):
        try:
            return _normalize(node[selection])
        except (TypeError, ValueError):
            pass
    # Fall back to normal Python indexing when the node is not a dataset or
    # h5py cannot apply the requested selection directly.
    return _select(_read(node, h5py), selection)


def hdf5_get(path: str | Path, key: str, selection: Any | None = None) -> Any:
    h5py = _h5()
    with h5py.File(path, "r") as fh:
        return _read_selected(_node(fh, key), h5py, selection)


_DIRECT_LOADERS_BY_SUFFIX = {
    ".pkl": pickle_load,
    ".pickle": pickle_load,
    ".h5": hdf5_load,
    ".hdf5": hdf5_load,
    ".json": json_load,
}

_ARCHIVE_SUFFIXES = (".tar.gz2", ".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".txz", ".zip")
_ARCHIVE_LOADER_SUFFIXES = (".zip", ".tgz", ".tbz2", ".txz", ".gz", ".gz2", ".bz2", ".xz")
_CACHE_DIR_NAME = "__lazyfields__"
_CACHE_ROOTS: set[Path] = set()


def archive_suffix(path: str | Path) -> str | None:
    name = Path(path).name.lower()
    return next((suffix for suffix in _ARCHIVE_SUFFIXES if name.endswith(suffix)), None)


def _wrapped_suffix(path: str | Path) -> str:
    suffix = archive_suffix(path)
    return "" if suffix is None else Path(Path(path).name[:-len(suffix)]).suffix.lower()


def is_supported_storage_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in _DIRECT_LOADERS_BY_SUFFIX or archive_suffix(path) is not None


def _archive_member(path: str | Path, files: list[tuple[Any, str]], kind: str) -> tuple[Any, str]:
    supported = [
        (member, Path(filename).suffix.lower())
        for member, filename in files
        if Path(filename).suffix.lower() in _DIRECT_LOADERS_BY_SUFFIX
    ]
    if len(supported) == 1:
        return supported[0]
    if len(supported) > 1:
        raise ValueError(f"{kind} archives must contain exactly one supported storage file.")

    suffix = _wrapped_suffix(path)
    if len(files) == 1 and suffix in _DIRECT_LOADERS_BY_SUFFIX:
        return files[0][0], suffix
    raise ValueError(f"{kind} archives must contain one .pkl, .pickle, .json, .h5, or .hdf5 file.")


def _inner_filename(filename: str, suffix: str) -> str:
    name = Path(filename).name
    return name if name and Path(name).suffix.lower() == suffix else f"inner_file{suffix}"


def _cleanup_caches() -> None:
    for root in list(_CACHE_ROOTS):
        shutil.rmtree(root, ignore_errors=True)


atexit.register(_cleanup_caches)


def _archive_key(path: Path) -> str:
    stat = path.stat()
    data = f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()
    return hashlib.sha256(data).hexdigest()[:16]


def _cache_root(path: Path) -> Path:
    root = path.parent / _CACHE_DIR_NAME
    if root not in _CACHE_ROOTS:
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(exist_ok=True)
        _CACHE_ROOTS.add(root)
    return root


def _archive_cache_dir(path: str | Path) -> Path:
    path = Path(path)
    cache_dir = _cache_root(path) / "unpacked_archive_member" / _archive_key(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _cached_archive_file(path: str | Path) -> Path | None:
    cache_dir = _archive_cache_dir(path)
    files = [file for file in cache_dir.iterdir() if file.is_file() and not file.name.endswith(".tmp")]
    if len(files) != 1:
        shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
    return files[0] if len(files) == 1 else None


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
    suffix = archive_suffix(path)
    if suffix is None:
        raise ValueError(f"'{Path(path).name}' is not a supported archive.")
    cached = _cached_archive_file(path)
    if cached is not None:
        yield cached
        return
    cache_dir = _archive_cache_dir(path)
    if suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            files = [(info, info.filename) for info in archive.infolist() if not info.is_dir()]
            info, inner_suffix = _archive_member(path, files, "Zip")
            yield _copy_member(archive.open(info), cache_dir / _inner_filename(info.filename, inner_suffix))
    else:
        with tarfile.open(path, "r:*") as archive:
            files = [(info, info.name) for info in archive.getmembers() if info.isfile()]
            info, inner_suffix = _archive_member(path, files, "Tar")
            source = archive.extractfile(info)
            if source is None:
                raise ValueError(f"Could not read '{info.name}' from '{Path(path).name}'.")
            yield _copy_member(source, cache_dir / _inner_filename(info.name, inner_suffix))


def archive_load(path: str | Path) -> Mapping[str, Any]:
    with unpacked_archive_member(path) as inner_path:
        return _DIRECT_LOADERS_BY_SUFFIX[inner_path.suffix.lower()](inner_path)


def zip_load(path: str | Path) -> Mapping[str, Any]:
    return archive_load(path)


LOADERS_BY_SUFFIX = {
    **_DIRECT_LOADERS_BY_SUFFIX,
    **dict.fromkeys(_ARCHIVE_LOADER_SUFFIXES, archive_load),
}

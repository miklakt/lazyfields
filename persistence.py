import hashlib
import json
import pickle
import shutil
import tarfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
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


def _read(node: Any, h5py: Any) -> Any:
    if isinstance(node, h5py.Dataset):
        return _normalize(node[()])
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


_DIRECT_LOADERS_BY_SUFFIX = {
    ".pkl": pickle_load,
    ".pickle": pickle_load,
    ".h5": hdf5_load,
    ".hdf5": hdf5_load,
    ".json": json_load,
}

_ARCHIVE_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".txz", ".zip")
_ARCHIVE_LOADER_SUFFIXES = (".zip", ".tgz", ".tbz2", ".txz", ".gz", ".bz2", ".xz")
_CACHE = TemporaryDirectory(prefix="lazyfields-")


def archive_suffix(path: str | Path) -> str | None:
    name = Path(path).name.lower()
    return next((suffix for suffix in _ARCHIVE_SUFFIXES if name.endswith(suffix)), None)


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

    archive = archive_suffix(path)
    suffix = "" if archive is None else Path(Path(path).name[:-len(archive)]).suffix.lower()
    if len(files) == 1 and suffix in _DIRECT_LOADERS_BY_SUFFIX:
        return files[0][0], suffix
    raise ValueError(f"{kind} archives must contain one .pkl, .pickle, .json, .h5, or .hdf5 file.")


def _inner_filename(filename: str, suffix: str) -> str:
    name = Path(filename).name
    return name if name and Path(name).suffix.lower() == suffix else f"inner_file{suffix}"


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
    if suffix is None:
        raise ValueError(f"'{path.name}' is not a supported archive.")
    stat = path.stat()
    key = hashlib.sha256(f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()).hexdigest()[:16]
    cache_dir = Path(_CACHE.name) / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = [file for file in cache_dir.iterdir() if file.is_file() and not file.name.endswith(".tmp")]
    if len(files) == 1:
        yield files[0]
        return
    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True)
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


LOADERS_BY_SUFFIX = {
    **_DIRECT_LOADERS_BY_SUFFIX,
    **dict.fromkeys(_ARCHIVE_LOADER_SUFFIXES, archive_load),
}

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
    if not name or Path(name).suffix.lower() != suffix:
        return f"inner_file{suffix}"
    return name


def _copy_member(source: Any, temp_dir: str, filename: str, suffix: str) -> Path:
    inner_path = Path(temp_dir) / _inner_filename(filename, suffix)
    with source, inner_path.open("wb") as target:
        shutil.copyfileobj(source, target)
    return inner_path


@contextmanager
def unpacked_archive_member(path: str | Path) -> Iterator[Path]:
    suffix = archive_suffix(path)
    if suffix is None:
        raise ValueError(f"'{Path(path).name}' is not a supported archive.")
    with TemporaryDirectory() as temp_dir:
        if suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                files = [(info, info.filename) for info in archive.infolist() if not info.is_dir()]
                info, inner_suffix = _archive_member(path, files, "Zip")
                yield _copy_member(archive.open(info), temp_dir, info.filename, inner_suffix)
        else:
            with tarfile.open(path, "r:*") as archive:
                files = [(info, info.name) for info in archive.getmembers() if info.isfile()]
                info, inner_suffix = _archive_member(path, files, "Tar")
                source = archive.extractfile(info)
                if source is None:
                    raise ValueError(f"Could not read '{info.name}' from '{Path(path).name}'.")
                yield _copy_member(source, temp_dir, info.name, inner_suffix)


def archive_load(path: str | Path) -> Mapping[str, Any]:
    with unpacked_archive_member(path) as inner_path:
        return _DIRECT_LOADERS_BY_SUFFIX[inner_path.suffix.lower()](inner_path)


def zip_load(path: str | Path) -> Mapping[str, Any]:
    return archive_load(path)


LOADERS_BY_SUFFIX = {
    **_DIRECT_LOADERS_BY_SUFFIX,
    **dict.fromkeys(_ARCHIVE_LOADER_SUFFIXES, archive_load),
}

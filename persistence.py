import json
import pickle
from pathlib import Path
from typing import Any, Mapping


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


LOADERS_BY_SUFFIX = {
    ".pkl": pickle_load,
    ".pickle": pickle_load,
    ".h5": hdf5_load,
    ".hdf5": hdf5_load,
    ".json": json_load,
}

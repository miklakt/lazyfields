import json
from pathlib import Path
from typing import Any, Mapping
import pickle


def pickle_load(path: str | Path) -> Mapping[str, Any]: return pickle.loads(Path(path).read_bytes())
def json_load(path: str | Path) -> Mapping[str, Any]: return json.loads(Path(path).read_text())


def _h5():
    try: import h5py
    except ImportError as exc: raise ImportError("h5py is required to load HDF5-backed rows.") from exc
    return h5py


def _read(node: Any, h5py: Any) -> Any:
    if isinstance(node, h5py.Dataset):
        value = node[()]
        if isinstance(value, bytes): return value.decode("utf-8")
        return value.item() if getattr(value, "shape", None) == () else value
    return {key: _read(node[key], h5py) for key in node.keys()}


def hdf5_load(path: str | Path) -> Mapping[str, Any]:
    h5py = _h5()
    with h5py.File(path, "r") as fh: return {key: _read(fh[key], h5py) for key in fh.keys()}


def hdf5_get(path: str | Path, key: str) -> Any:
    h5py = _h5()
    with h5py.File(path, "r") as fh:
        node: Any = fh
        for part in key.split("/"):
            if part: node = node[part]
        return _read(node, h5py)


LOADERS_BY_SUFFIX = {
    ".pkl": pickle_load,
    ".pickle": pickle_load,
    ".h5": hdf5_load,
    ".hdf5": hdf5_load,
    ".json": json_load,
}

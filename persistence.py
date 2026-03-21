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


class HDF5DatasetRef:
    def __init__(self, path: str | Path, key: str):
        self.path = str(path)
        self.key = key

    def __getitem__(self, selection: Any) -> Any:
        return hdf5_get(self.path, self.key, selection=selection)

    def __repr__(self) -> str:
        return f"HDF5DatasetRef(path={self.path!r}, key={self.key!r})"


def _read(node: Any, h5py: Any) -> Any:
    if isinstance(node, h5py.Dataset):
        value = node[()]
        if isinstance(value, bytes): return value.decode("utf-8")
        return value.item() if getattr(value, "shape", None) == () else value
    return {key: _read(node[key], h5py) for key in node.keys()}


def hdf5_load(path: str | Path) -> Mapping[str, Any]:
    h5py = _h5()
    with h5py.File(path, "r") as fh: return {key: _read(fh[key], h5py) for key in fh.keys()}


def _read_selected(node: Any, h5py: Any, selection: Any | None) -> Any:
    if selection is None:
        return _read(node, h5py)
    if not isinstance(node, h5py.Dataset):
        raise TypeError("HDF5 slices can only be applied to datasets.")
    value = node[selection]
    if isinstance(value, bytes): return value.decode("utf-8")
    return value.item() if getattr(value, "shape", None) == () else value


def hdf5_get(path: str | Path, key: str, selection: Any | None = None) -> Any:
    h5py = _h5()
    with h5py.File(path, "r") as fh:
        node: Any = fh
        for part in key.split("/"):
            if part: node = node[part]
        if selection is not None:
            return _read_selected(node, h5py, selection)
        if isinstance(node, h5py.Dataset):
            return _read(node, h5py) if getattr(node, "shape", None) == () else HDF5DatasetRef(path, key)
        return _read(node, h5py)


LOADERS_BY_SUFFIX = {
    ".pkl": pickle_load,
    ".pickle": pickle_load,
    ".h5": hdf5_load,
    ".hdf5": hdf5_load,
    ".json": json_load,
}

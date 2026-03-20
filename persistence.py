from pathlib import Path
from typing import Any, Mapping
import pickle


def pickle_load(path: str | Path) -> Mapping[str, Any]: return pickle.loads(Path(path).read_bytes())


def hdf5_load(path: str | Path) -> Mapping[str, Any]:
    try: import h5py
    except ImportError as exc: raise ImportError("h5py is required to load HDF5-backed rows.") from exc

    def read_node(node: Any) -> Any:
        if isinstance(node, h5py.Dataset):
            value = node[()]
            if isinstance(value, bytes): return value.decode("utf-8")
            if getattr(value, "shape", None) == ():
                try: return value.item()
                except ValueError: return value
            return value
        return {key: read_node(node[key]) for key in node.keys()}

    with h5py.File(path, "r") as fh: return {key: read_node(fh[key]) for key in fh.keys()}


LOADERS_BY_SUFFIX = {
    ".pkl": pickle_load,
    ".pickle": pickle_load,
    ".h5": hdf5_load,
    ".hdf5": hdf5_load,
}

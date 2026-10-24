from zipfile import ZipFile

import numpy as np
from numpy.lib import format as npy_format

from . import NodeInfo, Reader


def _npz_headers(path):
    with ZipFile(path) as archive:
        for member in archive.infolist():
            if not member.filename.endswith(".npy"): continue
            with archive.open(member) as file:
                version = npy_format.read_magic(file)
                read_header = npy_format.read_array_header_1_0 if version == (1, 0) else npy_format.read_array_header_2_0
                shape, _, dtype = read_header(file)
            if dtype.hasobject: raise ValueError("NPZ object arrays are not supported.")
            yield member.filename[:-4], "scalar" if shape == () else "dataset"


def _npz_value(value):
    return value.item() if value.shape == () else value


def _npz_load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: _npz_value(archive[key]) for key in archive.files}


def _npz_read(path, key, selection=None):
    with np.load(path, allow_pickle=False) as archive:
        value = archive[key]
        if selection is not None: value = value[selection]
        return _npz_value(value)


def _npz_inspect(path, key=None):
    if key is not None:
        _npz_info(path, key)
        raise NotADirectoryError(key)
    with np.load(path, allow_pickle=False) as archive:
        for key, kind in _npz_headers(path):
            yield key, kind, _npz_value(archive[key]) if kind == "scalar" else None


def _npz_info(path, key=None):
    if key is None:
        return NodeInfo("group", tuple(name for name, _ in _npz_headers(path)))
    for name, kind in _npz_headers(path):
        if name == key:
            return NodeInfo(kind)
    raise KeyError(key)


NUMPY_ARCHIVE = Reader((".npz",), _npz_load, _npz_read, _npz_inspect, inspect_node=_npz_info)

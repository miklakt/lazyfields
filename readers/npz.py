from zipfile import ZipFile

import numpy as np
from numpy.lib import format as npy_format

from . import NodeInfo, Reader, _normalize


def _npz_describe(path):
    """Every array's dtype and shape from its header; only scalars are read."""
    entries = []
    with ZipFile(path) as archive:
        for member in archive.infolist():
            if not member.filename.endswith(".npy"): continue
            with archive.open(member) as file:
                version = npy_format.read_magic(file)
                header = npy_format.read_array_header_1_0 if version == (1, 0) else npy_format.read_array_header_2_0
                shape, _, dtype = header(file)
                if dtype.hasobject: raise ValueError("NPZ object arrays are not supported.")
                if shape == ():
                    file.seek(0)
                    entries.append((member.filename[:-4], NodeInfo("scalar"), _normalize(npy_format.read_array(file))))
                else:
                    entries.append((member.filename[:-4], NodeInfo("dataset", dtype=str(dtype), shape=shape), None))
    yield None, NodeInfo("group", tuple(name for name, _, _ in entries)), None
    yield from entries


def _npz_load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: _normalize(archive[key]) for key in archive.files}


def _npz_read(path, key, selection=None):
    with np.load(path, allow_pickle=False) as archive:
        value = archive[key]
        if selection is not None: value = value[selection]
        return _normalize(value)


NUMPY_ARCHIVE = Reader(_npz_load, _npz_read, describe=_npz_describe, suffixes=(".npz",),
                       format="npz")

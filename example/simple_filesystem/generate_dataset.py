"""Generate the benchmark dataset in three physical format directories."""

import json
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ROW_COUNT = 3
FLOAT_COUNT = 13_107_200
ARRAY2D_SHAPE = (3_200, 4_096)
ARRAY3D_SHAPE = (200, 256, 256)
OBJECT_COUNT = 3_200
OBJECT_ITEM_BYTES = 32 * 1024


def write_json_array(fh, array):
    """Write numeric arrays in chunks without building a full Python list."""
    fh.write("[")
    if array.ndim == 1:
        for start in range(0, len(array), 8192):
            if start:
                fh.write(",")
            fh.write(json.dumps(array[start:start + 8192].tolist(), separators=(",", ":"))[1:-1])
    else:
        for index, child in enumerate(array):
            if index:
                fh.write(",")
            write_json_array(fh, child)
    fh.write("]")


def generate_dataset(directory, file_format, *, overwrite=True):
    """Write benchmark rows in one format; reused by the benchmark scripts."""
    suffix = {"h5": ".h5", "json": ".json", "pickle": ".pkl"}[file_format]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    missing = [index for index in range(ROW_COUNT)
               if overwrite or not (directory / f"row_{index:02d}{suffix}").exists()]
    if not missing:
        return
    base = np.arange(FLOAT_COUNT, dtype=np.float64)
    for index in missing:
        prefix = f"file-{index:02d}:"
        objects = np.full(OBJECT_COUNT, "x" * OBJECT_ITEM_BYTES, dtype=object)
        objects[0] = prefix + "y" * (OBJECT_ITEM_BYTES - len(prefix))
        row = {
            "x_int": 100 + index,
            "x_float": 10.5 + index / 10.0,
            "x_bool": bool(index % 2),
            "x_str": f"row-{index:02d}",
            "variant_id": index,
            "array1d": base,
            "array2d": base.reshape(ARRAY2D_SHAPE),
            "array3d": base.reshape(ARRAY3D_SHAPE),
            "array1d_objects": objects,
        }
        path = directory / f"row_{index:02d}{suffix}"
        if file_format == "h5":
            import h5py
            with h5py.File(path, "w") as fh:
                for key, value in row.items():
                    if key == "array1d_objects":
                        fh.create_dataset(key, data=value, dtype=h5py.string_dtype("utf-8"))
                    else:
                        fh.create_dataset(key, data=value)
        elif file_format == "pickle":
            with path.open("wb") as fh:
                pickle.dump(row, fh, protocol=4)
        else:
            temp_path = path.with_suffix(".json.tmp")
            with temp_path.open("w") as fh:
                fh.write("{")
                for position, (key, value) in enumerate(row.items()):
                    if position:
                        fh.write(",")
                    fh.write(json.dumps(key) + ":")
                    if key == "array1d_objects":
                        json.dump(value.tolist(), fh, separators=(",", ":"))
                    elif isinstance(value, np.ndarray):
                        write_json_array(fh, value)
                    else:
                        json.dump(value, fh)
                fh.write("}")
            temp_path.replace(path)


def main():
    for file_format in ("h5", "json", "pickle"):
        generate_dataset(ROOT / file_format, file_format)
        print(f"Created benchmark rows in {file_format}/", flush=True)


if __name__ == "__main__":
    main()

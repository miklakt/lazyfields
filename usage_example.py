import pickle
from pathlib import Path

try:
    import h5py
except ImportError as exc:
    raise ImportError("Install h5py to run the mixed pickle/HDF5 example.") from exc

import lazyfields as lf


# %%
root_dir = Path(__file__).resolve().parent
data_dir = root_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)
for path in data_dir.iterdir():
    if path.is_file():
        path.unlink()


# %%
sample_rows = [
    {
        "some_int_key": 1,
        "some_float_key1": 1.25,
        "some_float_key2": 10.5,
        "some_string_key": "run-a",
        "some_image_key": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "some_tensor_key": [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
    },
    {
        "some_int_key": 2,
        "some_float_key1": 2.5,
        "some_float_key2": 20.0,
        "some_string_key": "run-b",
        "some_image_key": [[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]],
        "some_tensor_key": [[[9, 10], [11, 12]], [[13, 14], [15, 16]]],
    },
    {
        "some_int_key": 3,
        "some_float_key1": 3.75,
        "some_float_key2": 30.25,
        "some_string_key": "run-c",
        "some_image_key": [[2.1, 2.2, 2.3], [2.4, 2.5, 2.6]],
        "some_tensor_key": [[[17, 18], [19, 20]], [[21, 22], [23, 24]]],
    },
    {
        "some_int_key": 4,
        "some_float_key1": 4.5,
        "some_float_key2": 40.0,
        "some_string_key": "run-d",
        "some_nested_object_key": {"a": [2.1, 2.2, 2.3], "b": [2.4, 2.5, 2.6]},
    },
]


# %%
def write_h5(path: Path, row_data: dict) -> None:
    def write_node(group, key, value):
        if isinstance(value, dict):
            sub = group.create_group(key)
            for sub_key, sub_value in value.items():
                write_node(sub, sub_key, sub_value)
        else:
            group.create_dataset(key, data=value)

    with h5py.File(path, "w") as fh:
        for key, value in row_data.items():
            write_node(fh, key, value)


# %%
for idx, row_data in enumerate(sample_rows, start=1):
    if (idx + 1) % 2:
        with open(data_dir / f"result_{idx:03d}.pkl", "wb") as fh:
            pickle.dump(row_data, fh, protocol=4)
    else:
        write_h5(data_dir / f"result_{idx:03d}.h5", row_data)


# %%
# Build a reference table from mixed pickle and HDF5 files.
reference_df = lf.create_reference_table(data_dir)
display(reference_df)


# %%
# Load the backing file path for a single row.
reference_df.iloc[0].storage_file


# %%
# Load a lazy field that is sparse across rows.
reference_df.store["some_image_key"]


# %%
# Load a lazy field with nested HDF5 hierarchy data.
reference_df.store["some_nested_object_key"]


# %%
# Load the full stored row for a single series row.
reference_df.iloc[-1].store[:]

import json
import pickle
from pathlib import Path

try:
    import h5py
except ImportError as exc:
    raise ImportError("Install h5py to run the mixed pickle/HDF5/JSON example.") from exc

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
        "some_bool_key": True,
        "some_2darray_key": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "some_3darray_key": [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
    },
    {
        "some_int_key": 2,
        "some_float_key1": 2.5,
        "some_float_key2": 20.0,
        "some_string_key": "run-b",
        "some_list_key": ["red", "green", "blue"],
        "some_2darray_key": [[1.1, 1.2], [1.3, 1.4], [1.5, 1.6]],
        "some_3darray_key": [[[9, 10], [11, 12], [13, 14]]],
    },
    {
        "some_int_key": 3,
        "some_float_key1": 3.75,
        "some_float_key2": 30.25,
        "some_string_key": "run-c",
        "some_nested_object_key": {"a": [2.1, 2.2, 2.3], "b": [2.4, 2.5, 2.6]},
        "some_tags_key": ["json", "row", "three"],
        "some_matrix_key": [[1, 0], [0, 1]],
        "some_1darray_key": [0.25, 0.5, 0.75, 1.0],
    },
    {
        "some_int_key": 4,
        "some_float_key1": 4.5,
        "some_float_key2": 40.0,
        "some_string_key": "run-d",
        "some_nested_object_key": {"a": [3.1, 3.2], "b": [3.3, 3.4]},
        "some_optional_key": None,
        "some_vector_key": [4, 5, 6, 7],
        "some_1darray_key": [10, 20, 30],
    },
    {
        "some_int_key": 5,
        "some_float_key1": 5.25,
        "some_float_key2": 50.5,
        "some_string_key": "run-e",
        "some_2darray_key": [[2.0, 2.5, 3.0], [3.5, 4.0, 4.5]],
        "some_3darray_key": [[[1, 0], [0, 1]], [[2, 0], [0, 2]]],
        "some_metadata_key": {"source": "sim", "batch": 2},
    },
    {
        "some_int_key": 6,
        "some_float_key1": 6.75,
        "some_float_key2": 60.0,
        "some_string_key": "run-f",
        "some_nested_object_key": {"left": {"a": [1, 2]}, "right": {"b": [3, 4]}},
        "some_comment_key": "final json row",
        "some_2darray_key": [[9.1, 9.2], [9.3, 9.4]],
        "some_1darray_key": [100, 200],
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


def write_json(path: Path, row_data: dict) -> None:
    path.write_text(json.dumps(row_data))


# %%
for idx, row_data in enumerate(sample_rows, start=1):
    if idx % 3 == 1:
        with open(data_dir / f"result_{idx:03d}.pkl", "wb") as fh:
            pickle.dump(row_data, fh, protocol=4)
    elif idx % 3 == 2:
        write_h5(data_dir / f"result_{idx:03d}.h5", row_data)
    else:
        write_json(data_dir / f"result_{idx:03d}.json", row_data)


# %%
# Build a reference table from mixed pickle and HDF5 files.
reference_df = lf.create_reference_table(data_dir)
display(reference_df)


# %%
# Load the backing file path for a single row.
reference_df.iloc[0].storage_file


# %%
# Materialize a 2d array field across rows.
reference_df.store["some_2darray_key"]


# %%
# Load a nested JSON field through the same path syntax.
reference_df.iloc[4].store["some_metadata_key"]


# %% [markdown]
# ### HDF5 tuple-key selection
#
# `reference_df.iloc[1].store["some_2darray_key", 0:1]` works in one accessor call:
#
# 1. `reference_df.iloc[1]` is handled by pandas first, producing a Series row
#    object.
# 2. `row.store["some_2darray_key", 0:1]` calls `StoreAccessor.__getitem__`
#    with both the field key and the selection.
# 3. The accessor resolves the row's `storage_file` column and asks the loader
#    for the field and selection together.
# 4. For HDF5 datasets, `hdf5_get(..., selection=slice(0, 1))` applies the
#    hyperslab read and only loads the selected slice.


# %%
# Read only a hyperslab from the HDF5 dataset through tuple-key indexing.
reference_df.iloc[1].store["some_2darray_key", 0:1]


# %%
# Load the full stored row for a single reference row.
reference_df.iloc[-1].store[:]


# %%
# The same rule applies to a row slice.
reference_df.iloc[1:3].store[:]


# %%
# Use a pandas-style lambda filter to keep only rows whose stored file contains
# a specific deferred key.
rows_with_2darrays = reference_df[
    reference_df["non_scalar_keys"].apply(lambda keys: "some_2darray_key" in keys)
]
rows_with_2darrays[["some_string_key", "non_scalar_keys"]]

# %%

#%%
import json
import pickle
from pathlib import Path

import h5py
import lazyfields as lf


# %%
root_dir = Path(__file__).parent
data_dir = root_dir / "data"
reference_path = root_dir / "reference_table.pkl"
data_dir.mkdir(parents=True, exist_ok=True)
nested_data_dir = data_dir / "subdirectory"
for path in sorted(data_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
    if path.is_file():
        path.unlink()
    elif path.is_dir():
        path.rmdir()


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
        "some_2darray_key": [[9.1, 9.2], [9.3, 9.4]],
        "some_1darray_key": [100, 200],
    },
    {
        "some_int_key": 7,
        "some_float_key1": 7.05,
        "some_float_key2": 70.0,
        "some_string_key": "run-g",
        "some_nested_object_key": [{"a": [1, 2], "b": [3, 4]}, {"c": [5, 6, 7]}],
        "some_2darray_key": [[9.1, 9.2], [9.3, 9.4]],
        "some_1darray_key": [100, 200],
        "some_comment_key": "final json row",
    },
]


# %%
def write_h5_node(group, key, value):
    if isinstance(value, dict):
        sub = group.create_group(key)
        for sub_key, sub_value in value.items():
            write_h5_node(sub, sub_key, sub_value)
    else:
        group.create_dataset(key, data=value)


# %%
for idx, row_data in enumerate(sample_rows, start=1):
    if idx % 3 == 1:
        with open(data_dir / f"result_{idx:03d}.pkl", "wb") as fh:
            pickle.dump(row_data, fh, protocol=4)
    elif idx % 3 == 2:
        with h5py.File(data_dir / f"result_{idx:03d}.h5", "w") as fh:
            for key, value in row_data.items():
                write_h5_node(fh, key, value)
    else:
        target_dir = nested_data_dir if idx == len(sample_rows) - 1 else data_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / f"result_{idx:03d}.json").write_text(json.dumps(row_data))


# %%
def unnest_a_0(row: dict) -> None:
    nested = row.get("some_nested_object_key")
    if isinstance(nested, dict) and "a" in nested and len(nested["a"]) > 0:
        row["a_0"] = nested["a"][0]


def keep_a_0_greater_than(value):
    def f(row: dict) -> bool:
        return row.get("a_0", 0) > value

    return f


def add_a_sum(row: dict) -> None:
    nested = row.get("some_nested_object_key")
    if isinstance(nested, dict) and "a" in nested:
        row["a_sum"] = sum(nested["a"])


pipe = [[unnest_a_0, keep_a_0_greater_than(2.0)], add_a_sum]

reference_df = lf.create_reference_table(
    data_dir,
    # Keep paths to the indexed files relative to the reference table when it is outside the current working directory.
    relative_to=reference_path,
    search_subdirectories=True,
    pipe=pipe,
)


# %%
@reference_df.store.register_method
def length(row, key):
    try:
        return len(row.store[key])
    except (KeyError, TypeError):
        return None


# %%
reference_df.to_pickle(reference_path)

# %%

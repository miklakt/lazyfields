# lazyfields

`lazyfields` is a small pandas helper layer for working with result files through a reference table.

It is meant for data that naturally lives as a set of files, such as results from a numerical simulation or other batch process.
You can inspect and filter the results through a lightweight pandas DataFrame without loading everything into RAM at once.

Each indexed file must contain one dictionary.
Scalar values are copied into the DataFrame.
Other values stay in the original file until requested through `row.store[...]` or `reference_df.store[...]`.

> **Note:** "Scalar" follows `pandas.api.types.is_scalar(...)`: numbers, strings, booleans, timestamps, and `None` are scalar; lists, dictionaries, and arrays are not.

## Quick Start

Requirements:

- Python 3.10+
- `pandas`
- `h5py` if you want to read or write HDF5 files

Install from the repository root:

```bash
python3 -m pip install -e .
```

Install with HDF5 support:

```bash
python3 -m pip install -e '.[hdf5]'
```

Create the example data and reference table, then run the usage examples:

```bash
python3 example/create_reference.py
python3 example/use_reference.py
```

## How It Works

```python
import lazyfields as lf

reference_df = lf.create_reference_table("path_to_data")
```

`create_reference_table(...)` finds supported files and creates one table row per file.
Pass `file_pattern` to narrow the search or `search_subdirectories=True` to include nested folders.
Supported file extensions are `.pkl`, `.pickle`, `.json`, `.h5`, `.hdf5`, `.zip`, `.tar.gz`, `.tar.bz2`, `.tar.xz`, `.tgz`, `.tbz2`, and `.txz`.
An archive must contain exactly one supported data file.

> **Implementation note:** Extracted archive contents are reused from a temporary directory while the Python process is running.

The returned DataFrame contains copied values plus three columns:

- `storage_file` identifies the original file.
- `non_scalar_keys` lists the names of lists, dictionaries, arrays, and other nested values kept in that file.
- `creation_time` contains the file creation time.

Use the stored in the indexed files data through `.store`:

- `row.store["key"]` loads one value from one row.
- `reference_df.store["key"]` loads that value for every row and returns a Series.
  Missing values are returned as `pd.NA`; a `KeyError` is raised only when the key is absent from every row.
- `row.store["group/subkey"]` loads a nested value.
- `row.store["dataset", slice(...)]` loads only the requested part of an HDF5 array.
- `row.store[:]` loads all data for one row.

> **Performance note:** An HDF5 slice is read directly, so the rest of the array is not loaded into memory.

You can register methods for use on one row or a selection of rows.
Each method receives a row as its first argument.
On a DataFrame, its result is a Series with the same index.

```python
@reference_df.store.register_method
def length(row, key):
    try:
        return len(row.store[key])
    except (KeyError, TypeError):
        return None

lengths = reference_df.store.length("some_2darray_key")
```

A registered method is evaluated on demand for one row or for every row in a selection.

> **Implementation note:** A method's source code is stored in `DataFrame.attrs`.
> It must be a named function defined in a Python file.
> Put any imports it needs inside the function so it still works after loading.

Paths are relative to the folder where Python is run unless `relative_to` is provided.
Pass the reference-table filename when creating the table so the paths to the indexed files remain valid after loading.
For now, reference tables support only pickle serialization:

```python
import pandas as pd

destination = "reference.pkl"
reference_df = lf.create_reference_table("data", relative_to=destination)
reference_df.to_pickle(destination)
reference_df = pd.read_pickle(destination)
```

Only load trusted pickle files because they can execute code.

Use `pipe=[...]` to run functions on each dictionary while the reference table is created, not on demand.
The functions run in the listed order.

```python
def unnest_a_0(row):
    # {"a": [2.1, 2.2], "b": [2.3, 2.4]} in row["nested_object_key"]
    # Copy the first value from the nested "a" array into the reference table.
    nested = row.get("nested_object_key")
    if isinstance(nested, dict) and "a" in nested and len(nested["a"]) > 0:
        row["a_0"] = nested["a"][0]

def keep_a_0_greater_than(value):
    # Create a filter for the extracted value.
    def f(row):
        return row.get("a_0", 0) > value
    return f

def add_a_sum(row):
    # Add the sum of the nested "a" array to the reference table.
    nested = row.get("nested_object_key")
    if isinstance(nested, dict) and "a" in nested:
        row["a_sum"] = sum(nested["a"])

# Keep only rows whose indexed file has nested_object_key["a"][0] greater than 2.0.
reference_df = lf.create_reference_table(
    "data",
    pipe=[unnest_a_0, keep_a_0_greater_than(2.0), add_a_sum],
    # Because these steps are not inside a nested list, a_0 and a_sum remain in the reference table and can be used for indexing.
    # otherwise use
    # pipe=[[unnest_a_0, keep_a_0_greater_than(2.0)], add_a_sum]
)
```

- A function augments the input data in place and returns `None`, or returns a new dictionary that replaces it for the following steps and the reference-table row.
- A filter returns `True` to continue or `False` to filter out the row.
- A nested list runs functions followed by a filter on a temporary copy, so its changes do not appear in the output data.

```text
input_data -> function -> filter -> [function -> filter] -> function -> output_data -> reference table
```

Applying the pipe only affect the reference table.
The original indexed files remain unchanged, and `.store[...]` still reads their original data.

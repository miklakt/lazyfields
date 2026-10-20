# lazyfields

`lazyfields` is a lightweight pandas extension for indexing collections of structured data files and loading their contents on demand

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
python3 example/generate_dataset.py
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

- `file_path` identifies the original file.
- `non_scalar_keys` lists the names of lists, dictionaries, arrays, and other nested values kept in that file.
- `creation_time` contains the file creation time.

Use the stored in the indexed files data through `.store`:

- `row.store["key"]` loads one value from one row.
- `reference_df.store["key"]` loads that value for every row and returns a Series.
  Missing values are returned as `pd.NA`; a `KeyError` is raised only when the key is absent from every row.
- `row.store["group/subkey"]` loads a nested value.
- `row.store["dataset", slice(...)]` loads only the requested part of an HDF5 array.
- `row.store[:]` loads all data for one row.

Keys containing `/` are rejected when creating a reference table because `/` separates nested path segments.

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

Use `bind_method` as a helper to adapt an existing free function to a row-wise method.
Calling `bind_method(...)(abc)` creates an adapted function
without changing `abc`, so the original remains independently callable.
Using `@bind_method(...)` instead reassigns the function name to the adapted version:

```python
def abc(a, b, c):
    return a + b + c

reference_df.store.register_method(
    lf.bind_method(a='_["a"]', b='_.store["b"]')(abc)
)

abc(1, 2, 10)                         # ordinary function call: 13
results = reference_df.store.abc(10)  # _["a"] + _.store["b"] + 10
```

The example above stores this generated source in
`reference_df.attrs["store_methods"]["abc"]`:

```python
def abc(_, c):
    a = _['a']
    b = _.store['b']
    return a + b + c
```

Expressions may also call previously registered methods, for example:

```python
reference_df.store.register_method(
    lf.bind_method(a='_["a"]', b='_.store.length("b")')(abc)
)
```

This source is saved with the table when pickled and reconstructed when the
method is first called after loading.

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

Use `pipe=[...]` to process each file as the reference table is built, compute additional values, and add them to the index.
The functions run in the listed order.
Pipe functions can read nested values with slash paths, such as
`row["key/subkey/0"]`; numeric segments follow the same rules as `.store[...]`.
They can also index or slice a resolved value with `row["key/subkey", 1:5]`.
Use `row.get(("key/subkey", 1:5), default)` for a selection that may be missing;
the tuple holds the selection, and the second `get` argument is the fallback.
Assign extracted values to top-level columns, for example
`row["indexing_object"] = row["key/subkey/0"]`.
Pipe errors skip the affected file by default and print a one-line error.
Pass `pipe_error_skip=False` to `create_reference_table` or `update_reference_table`
to raise the original exception instead.

```python
def unnest_a_0(row):
    # For example, row["some_nested_object_key"] may be {"a": [2.1, 2.2], "b": [2.3, 2.4]}.
    # Read the first value of the nested "a" array, or use 0 if it is missing.
    row["a_0"] = row.get(("some_nested_object_key/a", 0), 0)

def keep_a_0_greater_than(value):
    # Create a filter for the extracted value.
    def f(row):
        return row.get("a_0", 0) > value
    return f

def add_a_sum(row):
    # Add the sum of the nested "a" array to the reference table.
    row["a_sum"] = sum(row.get("some_nested_object_key/a", []))

# Keep files whose first nested "a" value is greater than 2.0.
# The nested list uses a temporary row, so only a_sum is added to the table.
reference_df = lf.create_reference_table(
    "data",
    pipe=[[unnest_a_0, keep_a_0_greater_than(2.0)], add_a_sum],
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

## Extra readers

Extra readers for formats such as CSV files with a metadata header containing key-value pairs and NumPy `.npz` archives are available but still in development, so their APIs may change. Register the readers you need before creating a reference table:

```python
from lazyfields import register_reader
from lazyfields.extra_readers import CSV_WITH_HEADER, NUMPY_ARCHIVE

register_reader(CSV_WITH_HEADER())  # Header metadata lines start with "#" by default.
register_reader(NUMPY_ARCHIVE)
```

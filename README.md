# lazyfields

`lazyfields` is a lightweight pandas extension for indexing structured filesystem groups and loading their contents on demand.

It is meant for data that naturally lives as a set of files, such as results from a numerical simulation or other batch process.
You can inspect and filter the results through a lightweight pandas DataFrame without loading everything into RAM at once.

Each immediate group beneath the supplied filesystem root becomes one row.
Scalar values are copied into the DataFrame.
Other values stay in the backing filesystem until requested through `row.store[...]` or `reference_df.store[...]`.

> **Note:** Numbers, strings, booleans, dates, times, and `None` are scalar; lists, dictionaries, and arrays are not.

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

Install with faster JSON decoding:

```bash
python3 -m pip install -e '.[json]'
```

JSON readers automatically use `orjson` when installed, otherwise Python's `json`.
`orjson` expects UTF-8 JSON and rejects nonstandard values such as `NaN` and `Infinity`.

Create the example data and reference table, then run the usage examples:

```bash
python3 example/quickstart/generate_dataset.py
python3 example/quickstart/create_reference.py
python3 example/quickstart/use_reference.py
```

## How It Works

```python
from pathlib import Path
import lazyfields as lf
from lazyfields.filesystem import FileSystem

directory = Path("path_to_data")
filesystem = FileSystem(
    sorted(path.name for path in directory.glob("*.h5")),
    relative_to=directory,
)
reference_df = lf.create_reference_table(filesystem)
```

`create_reference_table(filesystem)` copies immediate scalar children of each root
group into one table row. Datasets and nested groups remain accessible through
`.store[path, selection]`, relative to each row group. Non-group root children are
skipped. Select files when constructing the filesystem; use `filesystem.clone("h5")`
to index the HDF5 subtree of `example/simple_filesystem`.

`reference_df.store.filesystem` exposes the backing filesystem. Mounted and merged
groups work the same way as file roots. `update_reference_table(reference_df, filesystem)`
reindexes the current groups, retaining registered methods. There is no fingerprint
cache or automatic change tracking; calling update explicitly reinspects each row.

Supported file extensions are `.pkl`, `.pickle`, `.json`, `.h5`, `.hdf5`, `.zip`, `.tar.gz`, `.tar.bz2`, `.tar.xz`, `.tgz`, `.tbz2`, and `.txz`.
An archive must contain exactly one supported data file.

>  Extracted archive contents are reused from a temporary directory while the Python process is running.

The returned DataFrame contains copied values plus three columns:

- `file_path` identifies the row group within the filesystem.
- `non_scalar_keys` lists the names of lists, dictionaries, arrays, and other nested values kept in that file.
- `creation_time` contains the latest physical source creation time, or `NaT` when unknown.

Use the stored in the indexed files data through `.store`:

- `row.store["key"]` loads one value from one row.
- `reference_df.store["key"]` loads that value for every row and returns a Series.
  Missing values are returned as `pd.NA`; a `KeyError` is raised only when the key is absent from every row.
- `row.store["group/subkey"]` loads a nested value.
- `row.store["dataset", slice(...)]` loads only the requested part of an HDF5 array.
- `row.store[:]` loads all data for one row.

Keys containing `/` are rejected when creating a reference table because `/` separates nested path segments.

> An HDF5 slice is read directly, so the rest of the array is not loaded into memory.

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

Remove a registered method by name:

```python
reference_df.store.unregister_method("length")
```

Use `bind_method` as a helper to adapt an existing free function to a row-wise method.
The second positional argument selects `"source"` (the default) or
`"wrap"`.
Use a column name directly (`a="a"` or `a='["a"]'`) and `.store["b"]` for a value kept in the original file.
Calling `bind_method(abc, ...)` creates an adapted function without changing
`abc`, so the original remains independently callable:

```python
def abc(a, b, c):
    return a + b + c

reference_df.store.register_method(
    lf.bind_method(abc, a="a", b='.store["b"]')
)

abc(1, 2, 10)                        # ordinary function call: 13
results = reference_df.store.abc(10)  # adds column a, value b from the original file, and 10
```

The example above stores this generated source in
`reference_df.attrs["store_methods"]["abc"]`:

```python
def abc(__row, c):
    a = __row['a']
    b = __row.store['b']
    return a + b + c
```

The row reference is supplied automatically. Use `.store.method(...)` to call
an already registered method as an alternative binding:

```python
bound_abc = lf.bind_method(abc, a="a", b='.store.length("b")')
```

With `"source"`, `bind_method` adds imports for modules and importable
functions or classes referenced by the body.
Global constants and other closure values are not saved.

The generated source is saved with the table when pickled and reconstructed
when the method is first called after loading.

Pass `"wrap"` when the original function lives in an importable
module and should remain a dependency. This option does not inspect or copy its
source; the original function keeps its module globals and defaults:

```python
from my_analysis import abc

reference_df.store.register_method(
    lf.bind_method(abc, "wrap", a="a", b='.store["b"]')
)
results = reference_df.store.abc(10)
```

The stored source is a small forwarding wrapper:

```python
def abc(__row, c):
    from my_analysis import abc as __method
    return __method(__row['a'], __row.store['b'], c)
```

The wrapper's signature is generated once when binding, so calls forward arguments
directly. Defaults reference the original function's objects when the wrapper is
defined or reconstructed after loading. The
`my_analysis` module must remain available after loading; changes to its `abc`
function affect calls through the wrapper. Keep that function unchanged when
registering: create a separate wrapper rather than replacing the original function.
Functions defined only in `__main__` or a local scope cannot be used this way.

A registered method is evaluated on demand for one row or for every row in a selection.

> A method's source code is stored in `DataFrame.attrs`.
> It must be a named function defined in a Python file.
> Put any imports it needs inside the function so it still works after loading.

One shared context in `DataFrame.attrs` holds the backing filesystem. Rows,
filtered tables, and copies share that same filesystem; pandas does not copy its
namespace. Use `filesystem.clone()` when an independent namespace is needed.
Pickle round trips retain this sharing within the loaded table. Physical source paths are absolute;
moving the source files requires constructing and indexing a new filesystem.
Existing tables created with the directory-based API must be rebuilt.
Register custom readers before loading a saved table, as with the CSV example.

```python
import pandas as pd

destination = "reference.pkl"
reference_df = lf.create_reference_table(filesystem)
reference_df.to_pickle(destination)
reference_df = pd.read_pickle(destination)
```

Only load trusted pickle files because they can execute code.

Use `pipe=[...]` to process each row group as the reference table is built, compute additional values, and add them to the index.
Callbacks receive ordinary pandas Series containing the indexed scalars and row
metadata. Read or assign indexed and computed fields with `row["field"]`, and read
source data through the existing accessor: `row.store["group/array", :10]`.
HDF5 selections read only the requested part of the dataset. Assigning a computed
field changes the reference row; `.store[...]` continues to read source values.
Pipe errors skip the affected file by default and print a one-line error.
Pass `pipe_error_skip=False` to `create_reference_table` or `update_reference_table`
to raise the original exception instead.

```python
def unnest_a_0(row):
    # For example, row.store["some_nested_object_key"] may be {"a": [2.1, 2.2], "b": [2.3, 2.4]}.
    # Read the first value of the nested "a" array, or use 0 if it is missing.
    try:
        row["a_0"] = row.store["some_nested_object_key/a", 0]
    except (KeyError, IndexError, TypeError):
        row["a_0"] = 0

def keep_a_0_greater_than(value):
    # Create a filter for the extracted value.
    def f(row):
        return row.get("a_0", 0) > value
    return f

def add_a_sum(row):
    # Add the sum of the nested "a" array to the reference table.
    try:
        row["a_sum"] = sum(row.store["some_nested_object_key/a"])
    except (KeyError, IndexError, TypeError):
        row["a_sum"] = 0

# Keep files whose first nested "a" value is greater than 2.0.
# The nested list uses a temporary row, so only a_sum is added to the table.
reference_df = lf.create_reference_table(
    filesystem,
    pipe=[[unnest_a_0, keep_a_0_greater_than(2.0)], add_a_sum],
)
```

- A function modifies the Series in place and returns `None`, or returns a new Series or dictionary that replaces its indexed fields. Source access through `.store` is retained.
- A filter returns `True` to continue or `False` to filter out the row.
- A nested list runs functions followed by a filter on a copy of the Series values, so its changes do not appear in the output data. The filesystem stays shared.

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

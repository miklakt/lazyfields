# lazyfields

`lazyfields` is a lightweight pandas extension for indexing collections of structured data files and loading their contents on demand.

It is meant for data that naturally lives as a set of files, such as results from a numerical simulation or other batch process.
You can inspect and filter the results through a lightweight pandas DataFrame without loading everything into RAM at once.

By default, each file contains a dictionary of named fields and becomes one row in
the DataFrame. Scalar fields become columns you can use to inspect and filter the
data. Larger or nested values, such as arrays and dictionaries, stay in the source
files and are loaded only when requested through the DataFrame's `.store` accessor,
for example `reference_df.store["path/to/value"]`.

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
import lazyfields as lf

reference_df = lf.create_reference_table("path_to_data")
```

`create_reference_table(...)` finds the supported files directly inside the folder and creates one table row per file.
File paths in the reference table are stored relative to `relative_to`.
If it is omitted, they are relative to the current working directory.
Pass an absolute base folder, such as the folder that holds the saved table, so the table keeps working when loaded from another working directory:

```python
from pathlib import Path

base = Path(__file__).parent
reference_df = lf.create_reference_table(base / "path_to_data", relative_to=base)
```

`update_reference_table(reference_df)` searches the folder again and updates the table, keeping registered methods:
new files are added, deleted files are removed, and rows whose files changed are read again.
Other rows are kept without reading their files, so changes to the pipe or columns are not picked up;
use `create_reference_table` to reindex everything.
Supported file extensions are `.pkl`, `.pickle`, `.json`, `.h5`, `.hdf5`, `.zip`, `.tar.gz`, `.tar.bz2`, `.tar.xz`, `.tgz`, `.tbz2`, and `.txz`.
An archive must contain exactly one supported data file.
Support for other formats can be added by providing a custom reader.

>  Extracted archive contents are reused from a temporary directory while the Python process is running.

### Indexing and data access

Indexing builds a lightweight table describing your data: each row represents a
record, and its scalar fields become columns. For example, a simulation's
temperature, run number, and status can be used to find, filter, sort, or group
runs with ordinary pandas operations, without loading their arrays or nested data.
Pipes can extract or compute additional scalar columns, such as an array's mean.
These columns form the reference table. Larger simulation results, such as particle
trajectories over time or a velocity field on a spatial grid, stay in the source
files until requested.

The table also includes four metadata columns: `lf_path`, `lf_store_keys`,
`lf_mtime`, and `lf_digest`, which identify source data and track changes.

Use ordinary pandas column access, such as `reference_df["temperature"]`, to work
with indexed fields. Use the additional `.store` accessor to load source data for
the rows in your selection:

```python
selected = reference_df[reference_df["temperature"] > 300]
trajectories = selected.store["trajectory"]  # Load the trajectory for each selected run.

row = selected.iloc[0]
trajectory = row.store["trajectory"]        # Load the trajectory for one run.
```

On a DataFrame, `.store["key"]` returns a Series with the same index, containing one
value per row. Missing values are returned as `pd.NA`; a `KeyError` is raised only
when the key is absent from every row. On a single row, it returns the value itself.
The key is a path within each row's source group:

- `row.store["group/subkey"]` follows a nested path to load one value.
- `row.store["dataset", slice(...)]` selects the requested part of a stored value.
- `row.store[:]` loads all source data for that row.

Keys containing `/` are rejected when creating a reference table because `/` separates nested path segments.
Indexed fields named like the four columns above are rejected as well.

> An HDF5 slice is read directly, so the rest of the array is not loaded into memory.

### Registered methods

You can register methods for use on one row or a selection of rows.
Each method receives a row as its first argument, like a pipe callback: a dict of the
row's columns, backed by its group, so other fields are read on demand with `row["key"]`.
On a DataFrame, its result is a Series with the same index.

```python
@reference_df.store.register_method
def length(row, key):
    try:
        return len(row[key])
    except (KeyError, TypeError):
        return None

lengths = reference_df.store.length("some_2darray_key")
```

Inside a method, other registered methods are called on the row, such as `row.length("key")`.
Method names cannot shadow the row's own attributes, such as `get`, `items`, or `node`.

Remove a registered method by name:

```python
reference_df.store.unregister_method("length")
```

Use `bind_method` as a helper to adapt an existing free function to a row-wise method.
The second positional argument selects `"source"` (the default) or
`"wrap"`.
Bind a field by name (`a="a"`, `b="group/b"`), whether it is a column or kept in the original file,
or by a row expression such as `b='["b", 0]'`.
Calling `bind_method(abc, ...)` creates an adapted function without changing
`abc`, so the original remains independently callable:

```python
def abc(a, b, c):
    return a + b + c

reference_df.store.register_method(
    lf.bind_method(abc, a="a", b="b")
)

abc(1, 2, 10)                        # ordinary function call: 13
results = reference_df.store.abc(10)  # adds column a, value b from the original file, and 10
```

The example above stores this generated source in
`reference_df.attrs["store_methods"]["abc"]`:

```python
def abc(__row, c):
    a = __row['a']
    b = __row['b']
    return a + b + c
```

The row reference is supplied automatically. A row expression selects part of a
value or calls an already registered method:

```python
bound_abc = lf.bind_method(abc, a="a", b='["b", 0]')
bound_abc = lf.bind_method(abc, a="a", b='.length("b")')
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
    lf.bind_method(abc, "wrap", a="a", b="b")
)
results = reference_df.store.abc(10)
```

The stored source is a small forwarding wrapper:

```python
def abc(__row, c):
    from my_analysis import abc as __method
    return __method(__row['a'], __row['b'], c)
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

### Pipes and processing

Use `pipe=[...]` to process each file as the reference table is built, compute additional values, and add them to the index.
Callbacks receive a dict of the row's indexed scalars, backed by its group: any
other field is read on demand, including nested paths and selections such as
`row["group/array", :10]`. HDF5 selections read only the requested part of the
dataset. Assigning a field changes the reference row and shadows a source field of
the same name.
Pipe errors skip the affected file by default and print a one-line error.
Pass `pipe_error_skip=False` to `create_reference_table` or `update_reference_table`
to raise the original exception instead.

```python
def unnest_a_0(row):
    # For example, row["some_nested_object_key"] may be {"a": [2.1, 2.2], "b": [2.3, 2.4]}.
    # Read the first value of the nested "a" array, or use 0 if it is missing.
    try:
        row["a_0"] = row["some_nested_object_key/a", 0]
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
        row["a_sum"] = sum(row["some_nested_object_key/a"])
    except (KeyError, IndexError, TypeError):
        row["a_sum"] = 0

# Keep files whose first nested "a" value is greater than 2.0.
# The nested list uses a temporary row, so only a_sum is added to the table.
reference_df = lf.create_reference_table(
    "path_to_data",
    pipe=[[unnest_a_0, keep_a_0_greater_than(2.0)], add_a_sum],
)
```

- A function modifies the row in place and returns `None`, or returns a dictionary that replaces its fields. Source fields remain readable.
- A filter returns `True` to continue or `False` to filter out the row.
- A nested list runs functions followed by a filter on a copy of the row, so its changes do not appear in the output data.

```text
input_data -> function -> filter -> [function -> filter] -> function -> output_data -> reference table
```

Applying the pipe only affect the reference table.
The original indexed files remain unchanged, and `.store[...]` still reads their original data.

### Saving and loading

Save the reference table with pandas and load it later.
Stored values are read from the original files again after loading;
the copied scalar columns remain a snapshot from when the table was built.
Register custom readers before accessing stored data, as in the CSV example.

```python
import pandas as pd

reference_df = lf.create_reference_table(base / "path_to_data", relative_to=base)
reference_df.to_pickle(base / "reference.pkl")
reference_df = pd.read_pickle(base / "reference.pkl")
```

Only load trusted pickle files because they can execute code.

To move the data, pass the new base folder and rebuild the index:

```python
reference_df = lf.update_reference_table(reference_df, relative_to="/new/base")
```

## Browsing the table

Open a reference table in the graphical browser to inspect its indexed fields and
explore the source data for a selected row:

```python
from lazyfields.browser import browse_table

browse_table(reference_df)
```

You can also open a saved table from the command line:

```bash
lf-browse-table reference.pkl
```

Enter a pandas query such as `temperature > 300` in the filter box and press Enter,
or click a column heading to sort. Selecting a row shows its source group below
the table, where you can explore nested fields and inspect values. Right-click
selected rows to run a registered method and view its result.

The browser requires Tkinter.

## Extra readers

Extra readers for formats such as CSV files with a metadata header containing key-value pairs and NumPy `.npz` archives are available but still in development, so their APIs may change. Register the readers you need before creating a reference table:

```python
from lazyfields import register_reader
from lazyfields.readers.csv import CSV_WITH_HEADER
from lazyfields.readers.npz import NUMPY_ARCHIVE

register_reader(CSV_WITH_HEADER())  # Header metadata lines start with "#" by default.
register_reader(NUMPY_ARCHIVE)
```

## Filesystem specifications (experimental)

For power users: instead of a folder, `create_reference_table` accepts a filesystem
specification, a JSON-compatible dictionary describing which files to mount and how
to compose them. This API is experimental and may change.

The main requirement is that the filesystem root contains a reasonably homogeneous
collection of groups: each group represents one logical file and becomes one row
in the reference table. When `subtree` is set, this requirement applies to that
group's children instead. Groups should describe comparable records with similar
fields, though their contents need not be identical.

Scalar leaves directly inside each group are used immediately as index columns.
Pipes can extract or compute additional scalars from nested fields and larger
values and add them to the reference table. Other data remains available on demand.

A logical file need not correspond to one physical file. A group can represent a
whole file, part of a file, or a combination of files. For example, a large file
containing several runs can be split into one logical group per run. Conversely,
separate files holding different aspects of the same run can be combined into one
logical group. The filesystem specification defines this logical organization
over the physical files.

`example/composed_filesystem/README.md` describes filesystems and the specification
format. `example/large_arrays/README.md` shows how to browse a specification in the
OS file manager with `lf-mount`.

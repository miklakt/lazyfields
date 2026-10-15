# lazyfields

`lazyfields` is a small helper built on pandas for working with stored result
files through an indexed, lazy reference table.

It is meant for data that naturally lives as a set of files, such as results
from a numerical simulation or other batch process. You can inspect and filter
the results through a lightweight pandas DataFrame without loading everything
into RAM at once.

Each stored file is expected to contain one mapping. Scalar values are copied
into the DataFrame, while non-scalar values stay in the backing file and remain
available through lazy access.

Here, "scalar" follows `pandas.api.types.is_scalar(...)`: numbers, strings,
booleans, timestamps, and `None` count as scalar, while lists, dicts, arrays,
and other nested containers do not.

Importing `lazyfields` adds lazy field access through `row.store[...]` and
`reference_df.store[...]`.

Register custom row-wise methods to use them on one row or any DataFrame
selection. The function receives a reference row as its first argument.
On a DataFrame, the result is a Series with the same index.
Registered methods can call one another through `row.store`:

```python
import lazyfields as lf

reference_df = lf.create_reference_table("data")

@reference_df.store.register_method
def load_array(row, field):
    return row.store[field]

@reference_df.store.register_method
def array_length(row, field):
    return len(row.store.load_array(field))

lengths = reference_df.iloc[1:4].store.array_length("some_string_key")
```

Methods belong to the reference table where they were registered. Pandas
selections that retain the table's attributes retain its methods; registering a
method on another table does not add it to this one.
The table stores each method's source code in `attrs`, so methods are available
again after saving and loading the table with pandas. Registered functions must
have inspectable source code. Put imports and other dependencies inside the
function if it must work after reloading. Only load tables from trusted sources,
since calling a saved method executes its stored code.

## Quick Start

Requirements:

- Python 3.10+
- `pandas`
- `h5py` if you want to read or write HDF5-backed rows or run the mixed-format
  example

Install from the repository root:

```bash
python3 -m pip install -e .
```

Install with HDF5 support:

```bash
python3 -m pip install -e '.[hdf5]'
```

Run the bundled example:

```bash
python3 usage_example.py
```

The example script clears files in `data/` before regenerating sample rows.

## How It Works

- `create_reference_table(...)` builds an in-memory pandas DataFrame that points
  to stored row files. Each file must deserialize to one mapping-like row
  object.
- Supported suffixes are `.pkl`, `.pickle`, `.json`, `.h5`, `.hdf5`, `.zip`,
  `.tar.gz`, `.tar.bz2`, `.tar.xz`, `.tgz`, `.tbz2`, and `.txz`.
- Archive members are memoized in one process-local temporary directory.
- The returned DataFrame contains copied scalar fields plus
  `storage_file`, `non_scalar_keys`, and `creation_time`
- `non_scalar_keys` lists the top-level fields left in the backing file.
- `creation_time` stores the source file creation timestamp for each row.
- `row.store["field"]` loads one field from one stored row.
- `reference_df.store["field"]` loads that field across all rows as a pandas
  Series.
- `row.store["group/subkey"]` uses slash-separated path access for nested
  values.
- `row.store["dataset", slice(...)]` lets HDF5-backed rows forward that
  selection to the final dataset without loading the whole array first.
- `row.store[:]` loads the full stored row for one reference row.

```python
import lazyfields as lf

reference_df = lf.create_reference_table("data")
```

`create_reference_table(...)` scans matching files, infers the loader from the
file suffix. Pass `file_pattern` if you want to narrow the scan, or
`search_subdirectories=True` to include matching files below nested folders.
Paths are relative to the creation working directory unless `relative_to` is
provided; both it and `set_relative_to` take the reference-table filename.
Set it before saving and again after loading with pandas:

```python
reference_df.store.set_relative_to(destination).to_pickle(destination)
reference_df = pd.read_pickle(destination).store.set_relative_to(destination)
```

You can also customize reference-table construction with `pipe=[...]`, a list
of row-wise callbacks applied in order.

- a callback may mutate the loaded row in place and return `None`
- a callback may return a new mapping to replace the current row view
- a callback may return `True` or `False` to keep or skip the row
- a nested list defines a scoped filter block; its last step must be a filter,
  and any preprocessing inside that block is temporary

These hooks only affect the reference table. The stored files remain unchanged,
and `.store[...]` still reads the original row data from `storage_file`.


HDF5 can read a single stored field directly without loading the whole backing
file.

For DataFrame-level access, `reference_df.store["field"]` returns a pandas
Series and raises `KeyError` only if that field is missing for every row. If a
field exists in some rows but not others, the missing values come back as
`pd.NA`.

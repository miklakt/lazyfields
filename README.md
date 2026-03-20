# lazyfields

`lazyfields` is a small helper built on pandas for working with stored result
files through an indexed, lazy reference table.

It is meant for data that naturally lives as a set of files, such as results
from a numerical simulation or other batch process. You can inspect and filter
the results through a lightweight pandas DataFrame without loading everything
into RAM at once.

Each stored file is expected to contain one mapping, meaning a key-to-value
structure. When you build a reference table:

- scalar values are copied into the DataFrame,
- non-scalar values stay in the stored row file,
- the table gets a `storage_file` column pointing to the row file,
- the table gets a `non_scalar_keys` column listing deferred fields.

Importing `lazyfields` registers the `.store` pandas accessors. The accessor
gives pandas rows and tables a mapping-like interface for retrieving persisted
fields by key or nested path.

## Main API

```python
import lazyfields as lf

reference_df = lf.create_reference_table("data")
```

`create_reference_table(...)` scans matching files, infers the loader from the
file suffix, and returns a pandas DataFrame. It will pick up `.pkl`, `.json`,
and `.h5` rows in the same directory. Pass `file_pattern` if you want to
narrow the scan.

## Lazy Access

```python
row = reference_df.iloc[0]

full_row = row.store[:]
some_array = row.store["some_array_key"]
all_arrays = reference_df.store["some_array_key"]
```

- `row.store[:]` loads the full stored row
- `row.store["field"]` loads one field from one row
- `row.store["group/subkey"]` loads a nested field by path
- `reference_df.store["field"]` loads one field for all rows
- HDF5 can read a single field directly without loading the full row.

## Minimal Example

A runnable example lives in `usage_example.py`.

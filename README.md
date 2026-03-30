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

Importing `lazyfields` registers the `.store` pandas accessors.

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
- supported suffixes are `.pkl`, `.pickle`, `.json`, `.h5`, and `.hdf5`
- the returned DataFrame contains copied scalar fields plus:
  `storage_file`, `non_scalar_keys`, and `creation_time`
- `storage_file` is a normal DataFrame column, not a `.store` field
- `non_scalar_keys` lists the top-level fields left in the backing file
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
`search_subdirectories=True` to include matching files below nested folders. If
you need to save or reload a reference table elsewhere, pass
`reference_path=...` so relative `storage_file` values still resolve correctly
through `.store[...]`.

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

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

Importing `lazyfields` registers the `.store` pandas accessors.

## How It Works

- `create_reference_table(...)` builds an in-memory pandas DataFrame that points
  to the stored files.
- `storage_file` tells pandas where each row's backing file lives.
- `storage_file` is a normal DataFrame column, not a `.store` field.
- `non_scalar_keys` lists the top-level fields that were left in the file
  instead of being copied into the DataFrame.
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
file suffix, and returns a pandas DataFrame. It will pick up `.pkl`, `.json`,
and `.h5` rows in the same directory. Pass `file_pattern` if you want to
narrow the scan.

You can also customize reference-table construction with `pipe=[...]`, a list
of row-wise callbacks applied in order.

- a callback may mutate the loaded row in place and return `None`
- a callback may return a new mapping to replace the current row view
- a callback may return `True` or `False` to keep or skip the row
- a nested list defines a scoped filter block; its last step must be a filter,
  and any preprocessing inside that block is temporary

These hooks only affect the reference table. The stored files remain unchanged,
and `.store[...]` still reads the original row data from `storage_file`.

```python
reference_df = lf.create_reference_table("data")

def unnest_a_0(row_data):
    nested = row_data.get("some_nested_object_key")
    if isinstance(nested, dict) and "a" in nested and len(nested["a"]) > 0:
        row_data["a_0"] = nested["a"][0]

def keep_a_0(row_data):
    return row_data.get("a_0", 0) >= 3.0

reference_with_a_0_df = lf.create_reference_table("data", pipe=[unnest_a_0])
reference_filtered_df = lf.create_reference_table(
    "data",
    pipe=[unnest_a_0, keep_a_0],
)
reference_scoped_filter_df = lf.create_reference_table(
    "data",
    pipe=[[unnest_a_0, keep_a_0]],
)

row = reference_df.iloc[0]

stored_row = row.store[:]
some_array = row.store["some_array_key"]
some_slice = row.store["some_array_key", 0:10]
all_arrays = reference_df.store["some_array_key"]
```

In `pipe=[[unnest_a_0, keep_a_0]]`, `unnest_a_0` is only used to evaluate the
filter and does not add `a_0` to the final reference table.

HDF5 can read a single stored field directly without loading the whole backing
file.

## Minimal Example

A runnable example lives in `usage_example.py`.

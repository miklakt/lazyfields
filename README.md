# lazyfields

`lazyfields` is a small pandas-oriented helper for working with directories of
persisted row dictionaries.

Each stored row file is expected to contain one mapping. When you build a
reference table:

- scalar values are copied into the DataFrame,
- non-scalar values stay in the stored row file,
- the table gets a `storage_file` column pointing to the row file,
- the table gets a `non_scalar_keys` column listing deferred fields.

Importing `lazyfields` registers the `.store` pandas accessors.

## Main API

```python
import lazyfields as lf

reference_df = lf.create_reference_table("data")
```

`create_reference_table(...)` scans matching files, infers the loader from the
file suffix, and returns a pandas DataFrame. It will pick up both `.pkl` and
`.h5` rows in the same directory. Pass `file_pattern` if you want to narrow the
scan.

## Lazy Access

```python
row = reference_df.iloc[0]

full_row = row.store[:]
some_array = row.store["some_array_key"]
all_arrays = reference_df.store["some_array_key"]
```

- `row.store[:]` loads the full stored row
- `row.store["field"]` loads one field from one row
- `reference_df.store["field"]` loads one field for all rows

## Minimal Example

A runnable example lives in `usage_example.py`.

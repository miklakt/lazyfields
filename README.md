# lazyfields

`lazyfields` is a small pandas-oriented helper for working with directories of
pickled row dictionaries.

Each `.pkl` file is expected to contain one mapping. When you build a reference
table:

- scalar values are copied into the DataFrame,
- non-scalar values stay in the pickle file,
- the table gets a `pickle_file` column pointing to the row file,
- the table gets a `non_scalar_keys` column listing deferred fields.

Importing `lazyfields` registers the `.pkl` pandas accessors.

## Main API

```python
import lazyfields as lf

reference_df = lf.create_reference_table("data")
```

`create_reference_table(...)` scans a directory of `.pkl` files and returns a
pandas DataFrame.

## Lazy Access

```python
row = reference_df.iloc[0]

full_row = row.pkl[:]
some_array = row.pkl["some_array_key"]
all_arrays = reference_df.pkl["some_array_key"]
```

- `row.pkl[:]` loads the full stored row
- `row.pkl["field"]` loads one field from one row
- `reference_df.pkl["field"]` loads one field for all rows

## Minimal Example

A runnable example lives in `example_json_results.py`.

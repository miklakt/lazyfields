from pathlib import Path

import lazyfields
import pandas as pd

try:
    from IPython import get_ipython
    from IPython.display import display
    show = display if get_ipython() else print
except ImportError:
    show = print


# %%
reference_path = Path(__file__).parent / "reference_table.pkl"
reference_df = pd.read_pickle(reference_path)
show(reference_df)


# %%
# Methods saved with the reference table are ready to use after loading.
show(reference_df.store.get_registered_methods())

# Run a saved method for one row.
show(reference_df.iloc[0].store.length("some_1darray_key"))
# Run the same method for every row. The result is a Series.
show(reference_df.store.length("some_1darray_key"))


# %%
# The reference table includes values calculated from nested data when it was created.
show(reference_df[["some_string_key", "a_sum"]])


# %%
# Show the data file referenced by the first row.
show(reference_df.iloc[0]["file_path"])


# %%
# Load a 1D array for every row.
show(reference_df.store["some_1darray_key"])


# %%
# Load the nested data stored for the first row.
show(reference_df.iloc[0].store["some_nested_object_key"])


# %%
# Load only the first part of an array instead of reading the whole array into memory.
show(reference_df.iloc[1].store["some_1darray_key", 0:1])


# %%
# Load all the data referenced by the last row.
show(reference_df.iloc[-1].store[:])

# Load all the data for a selection of rows.
show(reference_df.iloc[1:3].store[:])
# %%
# Keep only rows that contain a particular stored value.
rows_with_1darrays = reference_df[
    reference_df["non_scalar_keys"].apply(lambda keys: "some_1darray_key" in keys)
]
rows_with_1darrays[["some_string_key", "non_scalar_keys"]]


# %%
# Access a nested value by separating each key with a slash.
reference_df.store["some_nested_object_key/a"]


# %%
# Use a number in the path to select an item from a nested list.
reference_df.store["some_nested_object_key/a/1"]

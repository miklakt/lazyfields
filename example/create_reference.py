# %%
from pathlib import Path

import lazyfields as lf


# %%
root_dir = Path(__file__).parent
data_dir = root_dir / "data"
reference_path = root_dir / "reference_table.pkl"
if not data_dir.is_dir() or not any(path.is_file() for path in data_dir.rglob("*")):
    raise FileNotFoundError("Example data is missing. Run example/generate_dataset.py first.")


# %%
def unnest_a_0(row: dict) -> None:
    row["a_0"] = row.get(("some_nested_object_key/a", 0), 0)


def keep_a_0_greater_than(value):
    def f(row: dict) -> bool:
        return row.get("a_0", 0) > value

    return f


def add_a_sum(row: dict) -> None:
    row["a_sum"] = sum(row.get("some_nested_object_key/a", []))


pipe = [[unnest_a_0, keep_a_0_greater_than(2.0)], add_a_sum]

reference_df = lf.create_reference_table(
    data_dir,
    # Keep paths to the indexed files relative to the reference table when it is outside the current working directory.
    relative_to=reference_path,
    search_subdirectories=True,
    pipe=pipe,
)


# %%
@reference_df.store.register_method
def length(row, key):
    """Finds the length of the array provided by the key"""
    try:
        return len(row.store[key])
    except (KeyError, TypeError):
        return None


# %%
reference_df.to_pickle(reference_path)

# %%

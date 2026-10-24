# %%
from pathlib import Path

import lazyfields as lf
from lazyfields.filesystem import FileSystem


# %%
root_dir = Path(__file__).parent
data_dir = root_dir / "data"
reference_path = root_dir / "reference_table.pkl"
if not data_dir.is_dir() or not any(path.is_file() for path in data_dir.rglob("*")):
    raise FileNotFoundError("Example data is missing. Run example/generate_dataset.py first.")


# %%
def unnest_a_0(row) -> None:
    try:
        row["a_0"] = row.store["some_nested_object_key/a", 0]
    except (KeyError, IndexError, TypeError):
        row["a_0"] = 0


def keep_a_0_greater_than(value):
    def f(row) -> bool:
        return row.get("a_0", 0) > value

    return f


def add_a_sum(row) -> None:
    try:
        row["a_sum"] = sum(row.store["some_nested_object_key/a"])
    except (KeyError, IndexError, TypeError):
        row["a_sum"] = 0


pipe = [[unnest_a_0, keep_a_0_greater_than(2.0)], add_a_sum]

filesystem = FileSystem(relative_to=data_dir)
for path in sorted(data_dir.rglob("*")):
    if path.is_file():
        filesystem.mount(path.name, FileSystem.from_file(path).node())


reference_df = lf.create_reference_table(
    filesystem.to_spec(), relative_to=data_dir.resolve(), pipe=pipe,
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

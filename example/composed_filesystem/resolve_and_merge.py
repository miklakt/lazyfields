"""Expose partitions at root with referenced metadata and merged block files."""

import json
import sys
from pathlib import Path
from typing import Literal

import pandas as pd

import lazyfields as lf
from lazyfields.readers import debug_readers
from lazyfields.filesystem import FileSystem
from lazyfields.filesystem.merge import collect, prefer, prefix, stack
from lazyfields.browser import ARRAY_ACTIONS, DEFAULT_METHODS, EXTRA_VIEWERS, browse_table


def compose(root: Path) -> FileSystem:
    source = Path("partitions.json")
    fs = FileSystem(source, relative_to=root)
    view = fs.clone(f"{source}/partitions")
    # Resolve stored references to source nodes.
    for name, partition in fs.read(f"{source}/partitions").items():
        target = FileSystem.from_file(root / source.parent / partition["file_ref"])
        view.replace(f"{name}/file_ref", target.node())
        sources = {
            Path(reference).stem: FileSystem.from_file(root / source.parent / reference).node()
            for reference in partition["file_refs"]
        }
        blocks = FileSystem.merge(sources, rules={
            "description": prefix(),
            "block_index": collect(),
            "coordinates": prefer(),
            "values": stack(axis=0),
        })
        view.replace(f"{name}/file_refs", blocks.node())
    return view


def add_to_index(path):
    """Pipe step: copy a scalar buried at path into a column named after its last key."""
    def step(row) -> None:
        row[path.rsplit("/", 1)[-1]] = row[path]
    return step


def weighted_vector(row, factor=1.0):
    """Scale the partition vector by its indexed metadata weight and an optional factor.

    :param factor: Multiplier applied on top of the weight.
    """
    import numpy as np
    return np.asarray(row["vector"]) * row["weight"] * factor


def block_values(row, block: Literal[0, 1] = 0):
    """Select one 2D block from the lazily stacked block values.

    :param block: Block position; choose a suggestion or enter another index, such as -1.
    """
    return row["file_refs/values", block]


def block_statistic(row,
                    statistic: Literal["mean", "median", "max", "min", "sum", "std", "var", "ptp", "size"] = "mean",
                    axis: Literal[0, 1, 2, None] = 0):
    """Reduce the stacked block values with a numpy statistic.

    :param statistic: Name of the numpy function; type to filter, e.g. m for mean, median, max, min.
    :param axis: Axis to reduce; 0 combines the blocks into one heatmap, None gives a single number.
    """
    import numpy as np
    return getattr(np, statistic)(np.asarray(row["file_refs/values"]), axis=axis)


def register_row_methods(reference_df):
    """Install the persistent row methods; return whether their saved definitions changed."""
    before = (reference_df.attrs.get("store_methods", {}).copy(),
              reference_df.attrs.get("store_method_categories", {}).copy())
    registered = set(reference_df.store.get_registered_methods())
    for method in (weighted_vector, block_values, block_statistic):
        if method.__name__ in registered:
            reference_df.store.unregister_method(method.__name__)
        reference_df.store.register_method(method, category="Analysis/Plots")
    after = (reference_df.attrs.get("store_methods", {}),
             reference_df.attrs.get("store_method_categories", {}))
    return before != after


def main() -> None:
    methods = DEFAULT_METHODS | EXTRA_VIEWERS | ARRAY_ACTIONS
    debug_readers()
    root = Path(__file__).resolve().parent / "data"
    spec_path = Path(__file__).with_suffix(".json")
    reference_path = Path(__file__).with_suffix(".pkl")
    rebuild = "--rebuild" in sys.argv
    if spec_path.exists() and not rebuild:
        view = FileSystem.from_spec(json.loads(spec_path.read_text(encoding="utf-8")), relative_to=root)
    else:
        view = compose(root)
        spec_path.write_text(json.dumps(view.to_spec(), indent=2) + "\n", encoding="utf-8")
    save_reference = rebuild or not reference_path.exists()
    if not save_reference:
        reference_df = pd.read_pickle(reference_path)
    else:
        # file_ref/label lands in row["label"], file_ref/weight in row["weight"].
        pipe = [add_to_index("file_ref/label"), add_to_index("file_ref/weight")]
        reference_df = lf.create_reference_table(view.to_spec(), relative_to=root, pipe=pipe)
    # Re-register after loading too, so this script always demonstrates its current method bodies.
    methods_changed = register_row_methods(reference_df)
    if save_reference or methods_changed:
        reference_df.to_pickle(reference_path)
    browse_table(reference_df, methods=methods)


if __name__ == "__main__":
    main()

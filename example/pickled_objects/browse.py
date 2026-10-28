"""Browse a pickle holding a DataFrame and a custom object, as records and through a view of the table's columns."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from lazyfields.browser import browse_filesystem
from lazyfields.filesystem import FileSystem
from lazyfields.readers import View, register_view

DATA_DIR = Path(__file__).resolve().parent / "data"


def unwrapped(array):
    """An array's values; a pandas extension array, such as strings, is a record keeping them in its state."""
    return array["state"]["1"] if isinstance(array, Mapping) else array


def labels(index):
    """The labels in a pandas Index record, read from its parts."""
    data = index["args"]["1"]
    if "data" not in data:  # A RangeIndex stores only its range.
        return np.arange(data["start"], data["stop"], data["step"])
    return unwrapped(data["data"])


def columns(table):
    """A DataFrame record as its index and named columns, rearranged from its blocks without running pandas.

    Follows the pickled layout of pandas 2 and 3: a BlockManager whose arguments are the blocks, each holding
    some columns' values at its placement, and the axes, the column and row labels.
    """
    manager = table["state"]["_mgr"]
    blocks, axes = manager["args"]["0"], manager["args"]["1"]
    names = labels(axes[0])
    result = {"index": labels(axes[1])}
    for block in blocks:
        values, placement = block["args"]["0"], block["args"]["1"]
        if isinstance(values, Mapping):  # An extension array, with its column as a row, such as dates.
            values = np.atleast_2d(unwrapped(values))
        for position, column in zip(np.arange(len(names))[placement], values):
            result[str(names[position])] = column
    return result


def main():
    register_view("dataframe-columns", View(columns))
    # The data directory with results.pkl as recorded, and the file again with its table read through the view.
    fs = FileSystem.from_spec({"mount": [{"glob": "*.pkl"}], "children": {
        "readable": {"path": "results.pkl", "children": {"table": {"attrs": {"view": "dataframe-columns"}}}},
    }}, relative_to=DATA_DIR)
    print("results.pkl/table/type:", fs.read("results.pkl/table/type"))
    print("results.pkl/run/state/started:", fs.read("results.pkl/run/state/started"))
    print("readable/table:", fs.listdir("readable/table"))
    print("readable/table/energy[1:]:", fs.read("readable/table/energy", slice(1, None)))
    browse_filesystem(fs)


if __name__ == "__main__":
    main()

"""Browse a pickle holding a DataFrame and a custom object: untrusted leaves, and a trusted, real view."""

from pathlib import Path

from lazyfields.browser import browse_filesystem
from lazyfields.filesystem import FileSystem
from lazyfields.readers import View, register_view

DATA_DIR = Path(__file__).resolve().parent / "data"


def columns(table):
    """A DataFrame's index and columns, as plain values; ordinary pandas code once pandas is trusted."""
    return dict(table.items()) | {"index": list(table.index)}


def main():
    register_view("dataframe-columns", View(columns), trust="pandas")
    # The data directory with results.pkl as it is, and the file again with its table read through the view.
    fs = FileSystem.from_spec({"mount": [{"glob": "*.pkl"}], "children": {
        "readable": {"path": "results.pkl", "children": {"table": {"attrs": {"view": "dataframe-columns"}}}},
    }}, relative_to=DATA_DIR)
    browse_filesystem(fs)


if __name__ == "__main__":
    main()

"""Browse a pickle holding a DataFrame and a custom object: untrusted leaves and trusted objects."""

from pathlib import Path

from lazyfields.browser import DEFAULT_METHODS, FIGURE, PLOT, SCATTER, TABLE, browse_filesystem
from lazyfields.filesystem import FileSystem
from lazyfields.readers import pickle_reader

DATA_DIR = Path(__file__).resolve().parent / "data"


def main():
    trusted = pickle_reader(trust=["pandas", "matplotlib"])
    naive = pickle_reader(trust=True)  # Ordinary pickle.load semantics: every class runs, for real.
    # Three mounts of the same file, side by side: the safe default, explicitly trusted, and naive.
    fs = FileSystem.from_spec({"children": {
        "untrusted": {"path": "results.pkl"},
        "trusted": {"path": "results.pkl", "reader": "trusted-pickle"},
        "naive": {"path": "results.pkl", "reader": "naive-pickle"},
    }}, relative_to=DATA_DIR, readers={"trusted-pickle": trusted, "naive-pickle": naive})
    # Table carries its own priority (it must win over Browser for a reconstructed DataFrame, see
    # extra_methods.py), so placing it is only a matter of its menu category here.
    methods = (DEFAULT_METHODS
               | FIGURE
               | SCATTER.set(category="Display/Plots")
               | PLOT.set(category="Display/Plots")
               | TABLE.set(category="Display/Tables"))
    browse_filesystem(fs, methods=methods)


if __name__ == "__main__":
    main()

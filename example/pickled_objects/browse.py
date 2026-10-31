"""Browse a pickle holding a DataFrame and a custom object: untrusted leaves, and a trusted, real view."""

from dataclasses import replace
from pathlib import Path

from lazyfields.browser import browse_filesystem, register_viewer
from lazyfields.browser.extra_viewers import FIGURE, PLOT, SCATTER, TABLE
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
    register_viewer(FIGURE)  # No category: shown directly in the context menu.
    register_viewer(SCATTER, category="View/Plots")
    register_viewer(PLOT, category="View/Plots")
    # specific: a reconstructed DataFrame still reports its pickle internals too, so it also matches
    # Browser's deliberately loose "any group" catch-all - Table should win that tie, not lose it.
    register_viewer(replace(TABLE, specific=True), category="View/Tables")
    browse_filesystem(fs)


if __name__ == "__main__":
    main()

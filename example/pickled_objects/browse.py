"""Browse a pickle holding a DataFrame and a custom object: untrusted leaves, and a trusted, real view."""

from pathlib import Path

from lazyfields.browser import browse_filesystem
from lazyfields.filesystem import FileSystem
from lazyfields.readers import pickle_reader

DATA_DIR = Path(__file__).resolve().parent / "data"


def main():
    trusted = pickle_reader(trust=["pandas", "matplotlib"])
    # Two mounts of the same file, side by side: one through the safe default, one explicitly trusted.
    fs = FileSystem.from_spec({"children": {
        "untrusted": {"path": "results.pkl"},
        "trusted": {"path": "results.pkl", "reader": "trusted-pickle"},
    }}, relative_to=DATA_DIR, readers={"trusted-pickle": trusted})
    browse_filesystem(fs)


if __name__ == "__main__":
    main()

"""Explore files mounted at their physical relative paths."""

from pathlib import Path

from lazyfields.filesystem import FileSystem

from lazyfields.filesystem.explorer import explore
from lazyfields.readers import debug_readers


def main():
    debug_readers()
    root = Path(__file__).resolve().parent
    paths = sorted(
        path.relative_to(root)
        for folder in ("h5", "json", "pickle")
        for path in (root / folder).glob("row_*")
        if path.is_file()
    )
    fs = FileSystem(paths, relative_to=root)
    explore(fs)


if __name__ == "__main__":
    main()

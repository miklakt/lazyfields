"""Explore files mounted at their physical relative paths."""

import json
import sys
from pathlib import Path

from lazyfields.readers import debug_readers
from lazyfields.filesystem import FileSystem
from lazyfields.browser import browse_filesystem


def load_filesystem(*, rebuild: bool = False) -> FileSystem:
    """Load the folder mount, rebuilding its specification when requested."""
    root = Path(__file__).resolve().parent
    spec_path = Path(__file__).with_suffix(".json")
    if spec_path.exists() and not rebuild:
        fs = FileSystem.from_spec(json.loads(spec_path.read_text(encoding="utf-8")), relative_to=root)
    else:
        fs = FileSystem.from_directory(root, "*/row_*")
        spec_path.write_text(json.dumps(fs.to_spec(), indent=2) + "\n", encoding="utf-8")
    return fs


def main() -> None:
    debug_readers()
    browse_filesystem(load_filesystem(rebuild="--rebuild" in sys.argv))


if __name__ == "__main__":
    main()
